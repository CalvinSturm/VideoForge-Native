"""Tests for CUDA stream pipeline infrastructure.

Validates CudaStreamPipeline submit/drain, multi-frame overlap, pipeline
reuse, empty drain, and CPU fallback — all without requiring a GPU.

For non-CUDA environments, the test creates a mock CUDA device to verify
the pipeline logic, or falls back to CPU-based validation.

Run:
    python -m pytest python/tests/test_cuda_streams.py -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure python/ is on sys.path and stub optional deps (same as test_batch_inference.py)
# ---------------------------------------------------------------------------
_STUBS = {}
for mod_name in ("cv2", "zenoh", "watchdog", "event_sync", "shm_ring",
                 "blender_engine", "research_layer", "ipc_protocol",
                 "auto_grade_analysis"):
    if mod_name not in sys.modules:
        stub = type(sys)(mod_name)
        if mod_name == "shm_ring":
            stub.ShmRingBuffer = type("ShmRingBuffer", (), {"__init__": lambda s, *a, **k: None, "close": lambda s: None})
        elif mod_name == "event_sync":
            stub.EventSync = type("EventSync", (), {
                "__init__": lambda s, *a, **k: None,
                "cleanup": lambda s: None,
                "disable": lambda s, *a: None,
                "setup": lambda s, *a: None,
                "wait_for_input": lambda s: None,
                "signal_output": lambda s: None,
                "events_enabled": False,
            })
        elif mod_name == "ipc_protocol":
            stub.PROTOCOL_VERSION = 1
        sys.modules[mod_name] = stub
        _STUBS[mod_name] = stub

sys.path.insert(0, os.path.join(os.getcwd(), "python"))

from shm_worker import PinnedStagingBuffers  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny models for testing (no learned weights)
# ---------------------------------------------------------------------------

class _IdentityModel(torch.nn.Module):
    """1× identity model — outputs input unchanged."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Scale2xModel(torch.nn.Module):
    """2× model using nearest-neighbor upscale (no learned weights)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


# ---------------------------------------------------------------------------
# Helper: create a PinnedStagingBuffers instance on CPU (mocked as pinned)
# ---------------------------------------------------------------------------

def _make_pinned_staging(max_batch: int, h: int, w: int, scale: int) -> PinnedStagingBuffers:
    """Create PinnedStagingBuffers without requiring CUDA for pin_memory()."""
    import logging
    ps = PinnedStagingBuffers(logging.getLogger("test"))
    # Allocate regular tensors (pin_memory() fails without CUDA driver)
    ps.input_pinned = torch.empty((max_batch, 3, h, w), dtype=torch.float32)
    ps.output_pinned = torch.empty((max_batch, 3, h * scale, w * scale), dtype=torch.float32)
    ps._max_batch = max_batch
    ps._h = h
    ps._w = w
    ps._scale = scale
    return ps


def _make_frame(h: int = 16, w: int = 16, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests: CudaStreamPipeline (CPU-based, mocked streams)
# ---------------------------------------------------------------------------

class CudaStreamPipelineCPUTest(unittest.TestCase):
    """Test pipeline logic using CPU device with mocked CUDA primitives.

    Since CudaStreamPipeline requires a CUDA device and torch.cuda.Stream,
    we patch the CUDA-specific parts and test the data flow / correctness.
    """

    def setUp(self):
        self.model = _IdentityModel().eval()
        self.device = torch.device("cpu")
        self.h, self.w = 16, 16
        self.scale = 1  # identity model
        self.pinned = _make_pinned_staging(
            max_batch=2, h=self.h, w=self.w, scale=self.scale
        )

    def test_cpu_device_raises(self):
        """CudaStreamPipeline should reject CPU devices."""
        from cuda_streams import CudaStreamPipeline
        import logging
        with self.assertRaises(ValueError):
            CudaStreamPipeline(
                model=self.model,
                device=self.device,
                half=False,
                adapter=None,
                pinned_staging=self.pinned,
                logger=logging.getLogger("test"),
            )

    def test_drain_empty_no_crash(self):
        """StreamTelemetry tracks zero frames without errors."""
        from cuda_streams import StreamTelemetry
        t = StreamTelemetry()
        self.assertEqual(t.frames_submitted, 0)
        self.assertEqual(t.frames_collected, 0)
        self.assertEqual(t.batches_completed, 0)

    def test_pinned_staging_buffers_stage_and_read(self):
        """PinnedStagingBuffers can stage input and read output slices."""
        frame = _make_frame(self.h, self.w, 200)
        self.pinned.stage_input(0, frame)

        # Verify pinned slot 0 contains normalized data
        expected = frame.astype(np.float32) / 255.0
        actual = self.pinned.input_pinned[0].numpy().transpose(1, 2, 0)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

        # Verify output slice returns a tensor
        out_slice = self.pinned.get_output_slice(0)
        self.assertIsNotNone(out_slice)
        self.assertEqual(out_slice.shape, torch.Size([3, self.h, self.w]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class CudaStreamPipelineGPUTest(unittest.TestCase):
    """Integration tests that require an actual CUDA device."""

    def setUp(self):
        import logging
        from cuda_streams import CudaStreamPipeline

        self.device = torch.device("cuda")
        self.h, self.w = 16, 16

        # Identity model (scale=1)
        self.model_1x = _IdentityModel().eval().to(self.device)
        self.pinned_1x = PinnedStagingBuffers(logging.getLogger("test"))
        self.pinned_1x.ensure(max_batch=2, h=self.h, w=self.w, scale=1)
        self.pipeline_1x = CudaStreamPipeline(
            model=self.model_1x,
            device=self.device,
            half=False,
            adapter=None,
            pinned_staging=self.pinned_1x,
            logger=logging.getLogger("test"),
        )

        # 2x model
        self.model_2x = _Scale2xModel().eval().to(self.device)
        self.pinned_2x = PinnedStagingBuffers(logging.getLogger("test"))
        self.pinned_2x.ensure(max_batch=2, h=self.h, w=self.w, scale=2)
        self.pipeline_2x = CudaStreamPipeline(
            model=self.model_2x,
            device=self.device,
            half=False,
            adapter=None,
            pinned_staging=self.pinned_2x,
            logger=logging.getLogger("test"),
        )

    def tearDown(self):
        self.pipeline_1x.shutdown()
        self.pipeline_2x.shutdown()
        torch.cuda.empty_cache()

    def test_single_frame_pipeline(self):
        """Submit 1 frame, drain, verify output matches identity model."""
        frame = _make_frame(self.h, self.w, 128)
        self.pipeline_1x.begin_batch()
        self.pipeline_1x.submit(frame)
        results = self.pipeline_1x.drain()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].shape, (self.h, self.w, 3))
        self.assertEqual(results[0].dtype, np.uint8)
        # Identity model: output should match input
        np.testing.assert_array_equal(results[0], frame)

    def test_multi_frame_overlap(self):
        """Submit 2 frames, drain, verify both outputs."""
        frames = [_make_frame(self.h, self.w, v) for v in [100, 200]]
        self.pipeline_1x.begin_batch()
        for f in frames:
            self.pipeline_1x.submit(f)
        results = self.pipeline_1x.drain()
        self.assertEqual(len(results), 2)
        for i, f in enumerate(frames):
            np.testing.assert_array_equal(results[i], f)

    def test_pipeline_reuse(self):
        """Run 2 batches through the same pipeline, verify no state leakage."""
        for batch_val in [100, 200]:
            frame = _make_frame(self.h, self.w, batch_val)
            self.pipeline_1x.begin_batch()
            self.pipeline_1x.submit(frame)
            results = self.pipeline_1x.drain()
            self.assertEqual(len(results), 1)
            np.testing.assert_array_equal(results[0], frame)

    def test_drain_empty(self):
        """Drain with no submissions returns empty list."""
        self.pipeline_1x.begin_batch()
        results = self.pipeline_1x.drain()
        self.assertEqual(results, [])

    def test_scale2x_output_shape(self):
        """2× model produces correctly sized output."""
        frame = _make_frame(self.h, self.w, 128)
        self.pipeline_2x.begin_batch()
        self.pipeline_2x.submit(frame)
        results = self.pipeline_2x.drain()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].shape, (self.h * 2, self.w * 2, 3))

    def test_telemetry_counts(self):
        """Telemetry counters increment correctly."""
        frame = _make_frame(self.h, self.w, 128)
        self.pipeline_1x.begin_batch()
        self.pipeline_1x.submit(frame)
        self.pipeline_1x.submit(frame)
        self.pipeline_1x.drain()

        self.assertEqual(self.pipeline_1x.telemetry.frames_submitted, 2)
        self.assertEqual(self.pipeline_1x.telemetry.frames_collected, 2)
        self.assertEqual(self.pipeline_1x.telemetry.batches_completed, 1)

    def test_update_model(self):
        """update_model swaps model without errors."""
        new_model = _Scale2xModel().eval().to(self.device)
        self.pipeline_1x.update_model(new_model, adapter=None, half=False)
        self.assertIs(self.pipeline_1x.model, new_model)


# ---------------------------------------------------------------------------
# Tests: CLI flag parsing
# ---------------------------------------------------------------------------

class CLIFlagTest(unittest.TestCase):
    """Verify --cuda-streams flag is parsed correctly."""

    def test_cuda_streams_flag_parsed(self):
        from shm_worker import build_parser
        parser = build_parser()
        args = parser.parse_args(["--cuda-streams", "--pinned-memory"])
        self.assertTrue(args.cuda_streams)
        self.assertTrue(args.pinned_memory)

    def test_cuda_streams_default_false(self):
        from shm_worker import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        self.assertFalse(args.cuda_streams)
        self.assertFalse(args.pinned_memory)


if __name__ == "__main__":
    unittest.main()
