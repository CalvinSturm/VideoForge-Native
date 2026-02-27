"""
CUDA Stream Pipeline — overlaps CPU↔GPU transfers with model inference.

Uses two CUDA streams (transfer + compute) to hide ~30% of transfer latency
by copying frame N+1 to GPU while frame N is still being inferred.

Requires pinned (page-locked) memory for async DMA transfers; the caller
must supply a PinnedStagingBuffers instance.

Architecture:
    transfer_stream:  [H2D_0][       ][H2D_1][       ][H2D_2] ...
    compute_stream:          [INFER_0]       [INFER_1]       [INFER_2] ...
    transfer_stream:                  [D2H_0]         [D2H_1]         [D2H_2] ...

    Event synchronization ensures compute waits for its H2D, and D2H waits
    for compute to finish.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from shm_worker import PinnedStagingBuffers


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

@dataclass
class StreamTelemetry:
    """Lightweight counters for stream pipeline performance logging."""
    frames_submitted: int = 0
    frames_collected: int = 0
    batches_completed: int = 0


# ---------------------------------------------------------------------------
# CudaStreamPipeline
# ---------------------------------------------------------------------------

class CudaStreamPipeline:
    """Double-buffered CUDA stream pipeline for overlapping transfers and inference.

    Usage (within _process_batch):
        pipeline.begin_batch()
        for img in inputs:
            pipeline.submit(img)
        results = pipeline.drain()
    """

    # Maximum in-flight frames (pinned buffer slots used for double-buffering)
    MAX_DEPTH = 2

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        half: bool,
        adapter,
        pinned_staging: "PinnedStagingBuffers",
        logger: logging.Logger,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(
                "CudaStreamPipeline requires a CUDA device, "
                f"got {device.type}"
            )

        self.model = model
        self.device = device
        self.half = half
        self.adapter = adapter
        self.pinned = pinned_staging
        self.log = logger

        # Precision
        self._use_fp16 = half or (
            # Read global precision mode from shm_worker if available
            globals().get("_PRECISION_MODE", "fp32") == "fp16"
        )
        self._dtype = torch.float16 if self._use_fp16 else torch.float32

        # CUDA streams
        self._transfer_stream = torch.cuda.Stream(device=device)
        self._compute_stream = torch.cuda.Stream(device=device)

        # Event pool: one event per in-flight slot
        self._h2d_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(self.MAX_DEPTH)
        ]
        self._compute_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(self.MAX_DEPTH)
        ]
        self._d2h_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(self.MAX_DEPTH)
        ]

        # In-flight queue: list of (slot_idx, gpu_output_tensor | None)
        self._inflight: List[Optional[int]] = []
        # Per-slot GPU output tensors (filled during compute)
        self._gpu_outputs: List[Optional[torch.Tensor]] = [
            None
        ] * self.MAX_DEPTH

        self.telemetry = StreamTelemetry()
        self.log.info(
            "CudaStreamPipeline initialized "
            f"(depth={self.MAX_DEPTH}, dtype={self._dtype})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_batch(self) -> None:
        """Reset in-flight state for a new batch of frames."""
        self._inflight.clear()
        for i in range(self.MAX_DEPTH):
            self._gpu_outputs[i] = None

    def submit(self, img_rgb: np.ndarray) -> None:
        """Stage one frame and launch async H2D + inference.

        If the pipeline is full (MAX_DEPTH in-flight), this blocks until
        the oldest frame's D2H is complete, effectively back-pressuring.
        """
        # Back-pressure: if pipeline is full, collect the oldest result
        if len(self._inflight) >= self.MAX_DEPTH:
            self._collect_oldest()

        slot = len(self._inflight) % self.MAX_DEPTH
        self._inflight.append(slot)

        # Stage input into pinned slot (CPU-side, fast memcpy)
        self.pinned.stage_input(slot, img_rgb)

        # --- Async H2D on transfer stream ---
        with torch.cuda.stream(self._transfer_stream):
            gpu_input = self.pinned.input_pinned[slot:slot + 1].to(
                device=self.device, dtype=self._dtype, non_blocking=True
            )
            self._h2d_events[slot].record(self._transfer_stream)

        # --- Inference on compute stream (waits for H2D) ---
        with torch.cuda.stream(self._compute_stream):
            self._compute_stream.wait_event(self._h2d_events[slot])

            with torch.no_grad():
                if self.adapter is not None:
                    if self._use_fp16:
                        with torch.autocast("cuda", dtype=torch.float16):
                            output = self.adapter.forward(gpu_input)
                    else:
                        output = self.adapter.forward(gpu_input)
                else:
                    if self._use_fp16:
                        with torch.autocast("cuda", dtype=torch.float16):
                            output = self.model(gpu_input)
                    else:
                        output = self.model(gpu_input)

            # Clamp and store on GPU (still on compute stream)
            output = output.squeeze(0).float().clamp_(0, 1)
            self._gpu_outputs[slot] = output

            self._compute_events[slot].record(self._compute_stream)

        # --- Async D2H on transfer stream (waits for compute) ---
        with torch.cuda.stream(self._transfer_stream):
            self._transfer_stream.wait_event(self._compute_events[slot])

            pinned_out = self.pinned.get_output_slice(slot)
            if pinned_out is not None and pinned_out.shape == output.shape:
                pinned_out.copy_(self._gpu_outputs[slot], non_blocking=True)
            # else: we'll do a synchronous .cpu() in collect

            self._d2h_events[slot].record(self._transfer_stream)

        self.telemetry.frames_submitted += 1

    def drain(self) -> List[np.ndarray]:
        """Collect all in-flight frames in submission order.

        Returns a list of numpy arrays [H, W, 3] uint8.
        """
        results: List[np.ndarray] = []
        while self._inflight:
            results.append(self._collect_oldest())
        self.telemetry.batches_completed += 1
        return results

    def update_model(
        self,
        model: torch.nn.Module,
        adapter=None,
        half: bool = False,
    ) -> None:
        """Swap the model/adapter after a model reload."""
        # Synchronize before swapping to avoid races
        self._transfer_stream.synchronize()
        self._compute_stream.synchronize()
        self.model = model
        self.adapter = adapter
        self.half = half
        self._use_fp16 = half or (
            globals().get("_PRECISION_MODE", "fp32") == "fp16"
        )
        self._dtype = torch.float16 if self._use_fp16 else torch.float32
        self.log.info("CudaStreamPipeline model updated")

    def shutdown(self) -> None:
        """Synchronize all streams and clear state."""
        self._transfer_stream.synchronize()
        self._compute_stream.synchronize()
        self._inflight.clear()
        self._gpu_outputs = [None] * self.MAX_DEPTH
        self.log.info(
            f"CudaStreamPipeline shutdown — "
            f"{self.telemetry.frames_submitted} frames submitted, "
            f"{self.telemetry.batches_completed} batches"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_oldest(self) -> np.ndarray:
        """Wait for the oldest in-flight frame and return its numpy result."""
        if not self._inflight:
            raise RuntimeError("No in-flight frames to collect")

        slot = self._inflight.pop(0)

        # Wait for the D2H copy to finish
        self._d2h_events[slot].synchronize()

        # Read from pinned output buffer
        pinned_out = self.pinned.get_output_slice(slot)
        if (
            pinned_out is not None
            and self._gpu_outputs[slot] is not None
            and pinned_out.shape == self._gpu_outputs[slot].shape
        ):
            cpu_out = pinned_out.numpy()
        else:
            # Fallback: synchronous copy
            gpu_out = self._gpu_outputs[slot]
            if gpu_out is None:
                raise RuntimeError(f"GPU output for slot {slot} is None")
            cpu_out = gpu_out.cpu().numpy()

        # Clear GPU reference
        self._gpu_outputs[slot] = None

        # Convert (C, H, W) → (H, W, C), denormalize
        frame = cpu_out.transpose(1, 2, 0)
        frame = (frame * 255.0).round().astype(np.uint8)

        self.telemetry.frames_collected += 1
        return frame
