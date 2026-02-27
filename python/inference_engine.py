"""
VideoForge Inference Engine — core inference functions and buffer management.

Extracted from shm_worker.py (Task #9) so inference logic can be profiled,
tested, and optimised independently of the IPC/SHM frame-loop machinery.

Public API:
    - get_precision_mode() / set_precision_mode()  — global precision state
    - configure_precision(mode)                    — set torch backend flags
    - enforce_deterministic_mode(log, enabled)     — seed + backend guardrails
    - inference(model, img_rgb, device, ...)        — single-frame inference
    - inference_batch(model, imgs_rgb, device, ...) — multi-frame batched inference
    - PreallocBuffers                               — reusable GPU input/output tensors
    - PinnedStagingBuffers                          — pinned CPU staging for async DMA
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

log = logging.getLogger("videoforge")

# =============================================================================
# PRECISION STATE
# =============================================================================

# Module-level precision mode — set by configure_precision(), read by inference()
_PRECISION_MODE: str = "fp32"


def get_precision_mode() -> str:
    """Return the current global precision mode."""
    return _PRECISION_MODE


def set_precision_mode(mode: str) -> None:
    """Set the global precision mode directly (no torch flags changed)."""
    global _PRECISION_MODE
    _PRECISION_MODE = mode


def configure_precision(mode: str = "fp32") -> None:
    """
    Configure PyTorch backend flags for the selected precision mode.

    Must be called BEFORE any model loading or CUDA kernel launch.

    Modes:
      fp32          — TF32 enabled, cuDNN deterministic, no autocast.
                      Best balance of speed and quality.
      fp16          — TF32 enabled, cuDNN deterministic, autocast float16.
                      ~2× throughput, minor quality trade-off.
      deterministic — TF32 disabled, strict deterministic algorithms,
                      cuDNN deterministic, no autocast, batch_size=1.
                      Same GPU + driver + CUDA → bit-exact output.
    """
    global _PRECISION_MODE
    mode = mode.lower().strip()
    if mode not in ("fp32", "fp16", "deterministic"):
        raise ValueError(f"Unknown precision mode: {mode!r}. Use fp32/fp16/deterministic.")
    _PRECISION_MODE = mode

    # --- Common: always keep cuDNN deterministic, never auto-tune ---
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if mode == "deterministic":
        # Strictest reproducibility: disable TF32 and enable deterministic algorithms
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        log.info(f"Precision: DETERMINISTIC (TF32=off, strict_deterministic=on)")
    else:
        # fp32 / fp16: enable TF32 for 2-4× speedup on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)
        tag = "FP16 (autocast)" if mode == "fp16" else "FP32"
        log.info(f"Precision: {tag} (TF32=on, cuDNN_deterministic=on)")


def enforce_deterministic_mode(log: logging.Logger, enabled: bool) -> None:
    """Apply deterministic guardrails when explicitly requested."""
    if not enabled:
        return

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    log.info(
        "deterministic mode enabled: seed=0 cudnn_deterministic=True "
        "cudnn_benchmark=False tf32=False batch_size=1"
    )


# =============================================================================
# UNIFIED INFERENCE FUNCTION
# =============================================================================

def inference(
    model: torch.nn.Module,
    img_rgb: np.ndarray,
    device: torch.device,
    half: bool = False,
    adapter=None,
    pinned_input: Optional[torch.Tensor] = None,
    pinned_output: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Unified inference function for both image and video processing.

    DETERMINISM GUARANTEES:
    - No randomness in this function
    - torch.no_grad() context ensures no gradient computation
    - Model must be in eval() mode (caller responsibility)
    - Input normalization is deterministic (simple division)
    - Output denormalization is deterministic (simple multiplication + clamp)

    Precision behaviour:
    - fp32:          Standard float32 inference (TF32 used on Ampere+ via backend flags)
    - fp16:          torch.autocast wraps the forward pass in float16
    - deterministic: Standard float32, strictest backend flags

    Args:
        model: The SR model (must be in eval mode)
        img_rgb: Input image as numpy array, RGB order, uint8 [0-255]
        device: Torch device (cuda or cpu)
        half: Whether to use FP16 precision (legacy flag, overridden by _PRECISION_MODE)
        adapter: Optional BaseAdapter for pre/post processing (window padding,
                 output clamping). Required for transformer models (SwinIR, HAT, DAT).
        pinned_output: Optional pinned CPU tensor [1,3,oH,oW] for async GPU→CPU
                       copy.  When provided, avoids a pageable .cpu() allocation.

    Returns:
        Upscaled image as numpy array, RGB order, uint8 [0-255]
    """
    # Validate input
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {img_rgb.shape}")

    # Determine dtype from precision mode
    use_fp16 = (_PRECISION_MODE == "fp16") or half
    dtype = torch.float16 if use_fp16 else torch.float32

    # Build input tensor — use pre-allocated pinned buffer when available to
    # avoid a CUDA malloc on every frame (~1-2ms savings at 4K).
    if pinned_input is not None:
        arr = img_rgb.astype(np.float32) / 255.0
        pinned_input[0].copy_(torch.from_numpy(arr.transpose(2, 0, 1)))
        tensor = pinned_input.to(device=device, dtype=dtype, non_blocking=True)
    else:
        img_float = img_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)

    # Use adapter if available — handles window padding, output cropping, etc.
    if adapter is not None:
        if use_fp16 and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                output = adapter.forward(tensor)
        else:
            output = adapter.forward(tensor)
    else:
        # CRITICAL: No gradient computation - ensures determinism and saves memory
        with torch.no_grad():
            if use_fp16 and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(tensor)
            else:
                output = model(tensor)

    # Convert back to numpy: (1, C, H, W) -> (H, W, C)
    # Use pinned output staging when available for async DMA transfer.
    # NOTE: clamp (not clamp_) because adapter.forward() returns an inference
    # tensor; in-place ops on inference tensors outside InferenceMode raise.
    output = output.squeeze(0).float().clamp(0, 1)
    if pinned_output is not None and pinned_output.shape == output.shape:
        pinned_output.copy_(output, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.current_stream().synchronize()
        cpu_out = pinned_output.numpy()
    else:
        cpu_out = output.cpu().numpy()
    cpu_out = cpu_out.transpose(1, 2, 0)

    # Denormalize to [0, 255] uint8 - DETERMINISTIC
    cpu_out = (cpu_out * 255.0).round().astype(np.uint8)

    return cpu_out


def inference_batch(
    model: torch.nn.Module,
    imgs_rgb: list,
    device: torch.device,
    half: bool = False,
    adapter=None,
    pinned_staging: Optional["PinnedStagingBuffers"] = None,
) -> list:
    """
    Batched inference: process multiple frames in a single GPU forward pass.

    Requires all images to have identical (H, W) dimensions (guaranteed for
    video frames from the same SHM ring buffer).

    Returns a list of numpy arrays in the same order as the input.

    Falls back to sequential inference if the batch forward pass fails
    (e.g. OOM on very large frames).
    """
    # Import Config here to avoid circular import (Config lives in shm_worker)
    from shm_worker import Config

    if not imgs_rgb:
        return []
    if len(imgs_rgb) == 1:
        _po = pinned_staging.get_output_slice(0) if pinned_staging is not None else None
        return [inference(model, imgs_rgb[0], device, half=half, adapter=adapter, pinned_output=_po)]

    # Validate: all frames must have same shape
    shape0 = imgs_rgb[0].shape
    for img in imgs_rgb[1:]:
        if img.shape != shape0:
            # Shape mismatch — fall back to sequential
            return [inference(model, img, device, half=half, adapter=adapter) for img in imgs_rgb]

    # Stack into batch tensor: list of (H,W,3) -> (N,3,H,W)
    # Use pinned staging buffers when available for async DMA input transfer.
    use_fp16 = (_PRECISION_MODE == "fp16") or half
    dtype = torch.float16 if use_fp16 else torch.float32
    n = len(imgs_rgb)

    if pinned_staging is not None:
        for i, img in enumerate(imgs_rgb):
            pinned_staging.stage_input(i, img)
        batch_tensor = pinned_staging.input_pinned[:n].to(
            device=device, dtype=dtype, non_blocking=True
        )
    else:
        batch_float = np.stack([img.astype(np.float32) / 255.0 for img in imgs_rgb], axis=0)
        batch_tensor = torch.from_numpy(batch_float.transpose(0, 3, 1, 2))  # (N,3,H,W)
        batch_tensor = batch_tensor.to(device=device, dtype=dtype, non_blocking=True)

    try:
        if adapter is not None:
            if use_fp16 and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    output = adapter.forward(batch_tensor)
            else:
                output = adapter.forward(batch_tensor)
        else:
            with torch.no_grad():
                if use_fp16 and device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        output = model(batch_tensor)
                else:
                    output = model(batch_tensor)
    except RuntimeError as e:
        # OOM or other failure — fall back to sequential
        if "out of memory" in str(e).lower():
            old_max = Config.MAX_BATCH_SIZE
            Config.MAX_BATCH_SIZE = max(1, len(imgs_rgb) // 2)
            log.warning(
                f"Batch OOM (N={len(imgs_rgb)}), reducing MAX_BATCH_SIZE "
                f"{old_max} → {Config.MAX_BATCH_SIZE}, falling back to sequential"
            )
            torch.cuda.empty_cache()
        else:
            log.info(f"Batch forward failed: {e}, falling back to sequential")
        return [inference(model, img, device, half=half, adapter=adapter) for img in imgs_rgb]

    # Split batch output back to list of numpy arrays
    # Use pinned staging for async GPU→CPU copy when available.
    # NOTE: clamp (not clamp_) — same reason as single-frame path above.
    output = output.float().clamp(0, 1)
    if pinned_staging is not None and pinned_staging.output_pinned is not None:
        out_pinned = pinned_staging.output_pinned[:output.shape[0]]
        if out_pinned.shape == output.shape:
            out_pinned.copy_(output, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            cpu_batch = out_pinned
        else:
            cpu_batch = output.cpu()
    else:
        cpu_batch = output.cpu()
    results = []
    for i in range(cpu_batch.shape[0]):
        frame = cpu_batch[i].numpy().transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
        frame = (frame * 255.0).round().astype(np.uint8)
        results.append(frame)

    return results


# =============================================================================
# BUFFER CLASSES
# =============================================================================

class PreallocBuffers:
    """Reusable input/output tensors for per-frame inference.

    Keeps one input tensor [1,3,H,W] and one output tensor [1,3,H*s,W*s].
    Reallocates only when shape/dtype/device changes.
    """

    def __init__(self, logger: logging.Logger):
        self.log = logger
        self.input_gpu: Optional[torch.Tensor] = None
        self.output_gpu: Optional[torch.Tensor] = None
        self._h: Optional[int] = None
        self._w: Optional[int] = None
        self._scale: Optional[int] = None
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None

    def ensure(self, h: int, w: int, scale: int, dtype: torch.dtype, device: torch.device) -> None:
        need_alloc = (
            self.input_gpu is None
            or self.output_gpu is None
            or self._h != h
            or self._w != w
            or self._scale != scale
            or self._dtype != dtype
            or self._device != device
        )
        if not need_alloc:
            return

        had_buffers = self.input_gpu is not None and self.output_gpu is not None
        self.input_gpu = torch.empty((1, 3, h, w), dtype=dtype, device=device)
        self.output_gpu = torch.empty((1, 3, h * scale, w * scale), dtype=dtype, device=device)
        self._h = h
        self._w = w
        self._scale = scale
        self._dtype = dtype
        self._device = device

        msg = (
            f"prealloc enabled: allocating input [1,3,{h},{w}] "
            f"output [1,3,{h * scale},{w * scale}] dtype={dtype} device={device}"
        )
        if had_buffers:
            self.log.warning(msg)
        else:
            self.log.info(msg)

    def copy_in_from_cpu(self, cpu_tensor_or_numpy) -> torch.Tensor:
        if self.input_gpu is None:
            raise RuntimeError("PreallocBuffers.ensure() must be called before copy_in_from_cpu()")

        if isinstance(cpu_tensor_or_numpy, np.ndarray):
            arr = cpu_tensor_or_numpy.astype(np.float32) / 255.0
            cpu = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        else:
            cpu = cpu_tensor_or_numpy

        src = cpu.to(device=self.input_gpu.device, dtype=self.input_gpu.dtype, non_blocking=True)
        self.input_gpu.copy_(src)
        return self.input_gpu

    def get_output(self) -> torch.Tensor:
        if self.output_gpu is None:
            raise RuntimeError("PreallocBuffers.ensure() must be called before get_output()")
        return self.output_gpu

    def clear(self) -> None:
        self.input_gpu = None
        self.output_gpu = None
        self._h = None
        self._w = None
        self._scale = None
        self._dtype = None
        self._device = None


class PinnedStagingBuffers:
    """CUDA page-locked (pinned) CPU staging tensors for async DMA transfers.

    Maintains reusable pinned-memory input and output tensors sized for the
    current SHM ring buffer layout.  Pinned memory enables the GPU driver to
    use DMA engines for CPU↔GPU copies, roughly doubling transfer throughput
    compared to pageable allocations.

    Allocate once in create_shm() and reuse across all frames/batches.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.log = logger
        self.input_pinned: Optional[torch.Tensor] = None
        self.output_pinned: Optional[torch.Tensor] = None
        self._max_batch: int = 0
        self._h: int = 0
        self._w: int = 0
        self._scale: int = 1

    def ensure(
        self,
        max_batch: int,
        h: int,
        w: int,
        scale: int,
    ) -> None:
        """Allocate / reallocate pinned buffers when shape changes."""
        if (
            self.input_pinned is not None
            and self._max_batch == max_batch
            and self._h == h
            and self._w == w
            and self._scale == scale
        ):
            return  # shape unchanged

        self.input_pinned = torch.empty(
            (max_batch, 3, h, w), dtype=torch.float32
        ).pin_memory()
        self.output_pinned = torch.empty(
            (max_batch, 3, h * scale, w * scale), dtype=torch.float32
        ).pin_memory()
        self._max_batch = max_batch
        self._h = h
        self._w = w
        self._scale = scale
        self.log.info(
            f"Pinned staging allocated: input [{max_batch},3,{h},{w}] "
            f"output [{max_batch},3,{h * scale},{w * scale}]"
        )

    def stage_input(self, idx: int, img_rgb: np.ndarray) -> None:
        """Normalize uint8 numpy frame and copy into pinned slot *idx*."""
        if self.input_pinned is None:
            raise RuntimeError("PinnedStagingBuffers.ensure() must be called first")
        arr = img_rgb.astype(np.float32) / 255.0
        self.input_pinned[idx].copy_(
            torch.from_numpy(arr.transpose(2, 0, 1))
        )

    def get_output_slice(self, idx: int) -> Optional[torch.Tensor]:
        """Return a view of the pinned output tensor for frame *idx*."""
        if self.output_pinned is None:
            return None
        return self.output_pinned[idx]

    def clear(self) -> None:
        """Release all pinned buffers."""
        self.input_pinned = None
        self.output_pinned = None
        self._max_batch = 0
        self._h = 0
        self._w = 0
        self._scale = 1
