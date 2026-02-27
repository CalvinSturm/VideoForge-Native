"""Tests for batch inference infrastructure in shm_worker.

Validates inference_batch(), _collect_ready_slots(), dynamic OOM recovery,
and the --batch-size CLI argument — all without requiring a GPU or model weights.

Some imports (cv2, zenoh, etc.) may not be available in every environment.
We stub them before importing shm_worker to avoid sys.exit(1) on import.
"""

import logging
import os
import sys
import types
import unittest

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Pre-stub optional dependencies that shm_worker imports at module level.
# This lets tests run in any environment that has torch + numpy.
# ---------------------------------------------------------------------------
_STUBS = {}
for mod_name in ("cv2", "zenoh", "watchdog", "event_sync", "shm_ring",
                 "blender_engine", "research_layer", "ipc_protocol",
                 "auto_grade_analysis"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        # Provide minimal attributes that shm_worker tries to access
        if mod_name == "cv2":
            stub.resize = lambda *a, **kw: None
            stub.INTER_LANCZOS4 = 4
        elif mod_name == "watchdog":
            stub.start_watchdog = lambda pid: None
        elif mod_name == "event_sync":
            stub.EventSync = type("EventSync", (), {
                "__init__": lambda self, **kw: None,
                "cleanup": lambda self: None,
            })
        elif mod_name == "shm_ring":
            stub.ShmRingBuffer = type("ShmRingBuffer", (), {
                "__init__": lambda self, cfg: None,
                "close": lambda self: None,
            })
        elif mod_name == "blender_engine":
            stub.PredictionBlender = type("PredictionBlender", (), {})
            stub.clear_temporal_buffers = lambda: None
        elif mod_name == "ipc_protocol":
            stub.PROTOCOL_VERSION = 1
        sys.modules[mod_name] = stub
        _STUBS[mod_name] = stub

# Also ensure the python dir is on sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "python"))

from shm_worker import (
    Config,
    inference,
    inference_batch,
    build_parser,
    PinnedStagingBuffers,
)


# ---------------------------------------------------------------------------
# Tiny models for testing (no learned weights)
# ---------------------------------------------------------------------------
class _IdentityModel(nn.Module):
    """1× identity model — outputs input unchanged."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Scale2xModel(nn.Module):
    """2× model using nearest-neighbor upscale (no learned weights)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, scale_factor=2, mode="nearest")


# ---------------------------------------------------------------------------
# Tests: inference_batch
# ---------------------------------------------------------------------------
class InferenceBatchTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = _IdentityModel().eval()
        self._orig_max_batch = Config.MAX_BATCH_SIZE

    def tearDown(self):
        Config.MAX_BATCH_SIZE = self._orig_max_batch

    def _make_frame(self, h: int = 16, w: int = 16, val: int = 128) -> np.ndarray:
        return np.full((h, w, 3), val, dtype=np.uint8)

    def test_single_frame_delegates_to_inference(self):
        """Batch of 1 should produce the same result as single inference."""
        frame = self._make_frame(val=100)
        single = inference(self.model, frame, self.device)
        [batched] = inference_batch(self.model, [frame], self.device)
        np.testing.assert_array_equal(single, batched)

    def test_multi_frame_returns_correct_count(self):
        """Batch of 3 same-shape frames returns 3 results."""
        frames = [self._make_frame(val=v) for v in (50, 100, 200)]
        results = inference_batch(self.model, frames, self.device)
        self.assertEqual(len(results), 3)
        for res, frame in zip(results, frames):
            self.assertEqual(res.shape, frame.shape)

    def test_multi_frame_values_correct(self):
        """Each output should match its corresponding input for identity model."""
        frames = [self._make_frame(val=v) for v in (10, 128, 250)]
        results = inference_batch(self.model, frames, self.device)
        for res, frame in zip(results, frames):
            np.testing.assert_array_equal(res, frame)

    def test_shape_mismatch_falls_back(self):
        """Mixed (H,W) shapes fall back to sequential without error."""
        f1 = self._make_frame(h=16, w=16)
        f2 = self._make_frame(h=32, w=32)
        results = inference_batch(self.model, [f1, f2], self.device)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].shape, (16, 16, 3))
        self.assertEqual(results[1].shape, (32, 32, 3))

    def test_empty_list_returns_empty(self):
        results = inference_batch(self.model, [], self.device)
        self.assertEqual(results, [])

    def test_oom_reduces_max_batch_size(self):
        """OOM during forward pass should reduce Config.MAX_BATCH_SIZE."""
        Config.MAX_BATCH_SIZE = 4

        class OOMModel(nn.Module):
            """Raises OOM on batched (N>1) input, works on single frames."""
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.shape[0] > 1:
                    raise RuntimeError("CUDA out of memory. Tried to allocate ...")
                return x

        model = OOMModel().eval()
        frames = [self._make_frame() for _ in range(4)]

        # Should fall back to sequential after OOM
        results = inference_batch(model, frames, self.device)
        self.assertEqual(len(results), 4)
        # MAX_BATCH_SIZE should have been reduced
        self.assertLess(Config.MAX_BATCH_SIZE, 4)

    def test_scale2x_multi_frame(self):
        """2× model with batched inference produces correct output shapes."""
        model = _Scale2xModel().eval()
        frames = [self._make_frame(h=8, w=8, val=v) for v in (50, 150)]
        results = inference_batch(model, frames, self.device)
        self.assertEqual(len(results), 2)
        for res in results:
            self.assertEqual(res.shape, (16, 16, 3))


# ---------------------------------------------------------------------------
# Tests: _collect_ready_slots
# ---------------------------------------------------------------------------
class CollectReadySlotsTest(unittest.TestCase):
    def _make_worker_stub(self, ring_size: int, slot_states: dict):
        """Create a minimal stub with _collect_ready_slots behavior."""
        from shm_worker import AIWorker

        worker = object.__new__(AIWorker)
        worker.ring_size = ring_size
        worker.mmap = None
        worker.global_header_size = 36
        worker.header_region_size = ring_size * Config.SLOT_HEADER_SIZE

        def _read_state(slot_idx):
            return slot_states.get(slot_idx, Config.SLOT_EMPTY)

        worker._read_slot_state = _read_state
        return worker

    def setUp(self):
        self._orig_max_batch = Config.MAX_BATCH_SIZE

    def tearDown(self):
        Config.MAX_BATCH_SIZE = self._orig_max_batch

    def test_collects_consecutive_ready(self):
        Config.MAX_BATCH_SIZE = 4
        states = {0: Config.SLOT_READY_FOR_AI, 1: Config.SLOT_READY_FOR_AI, 2: Config.SLOT_EMPTY}
        worker = self._make_worker_stub(6, states)
        self.assertEqual(worker._collect_ready_slots(0), [0, 1])

    def test_stops_at_non_ready(self):
        Config.MAX_BATCH_SIZE = 4
        states = {0: Config.SLOT_READY_FOR_AI, 1: Config.SLOT_AI_PROCESSING}
        worker = self._make_worker_stub(6, states)
        self.assertEqual(worker._collect_ready_slots(0), [0])

    def test_empty_when_start_not_ready(self):
        Config.MAX_BATCH_SIZE = 4
        states = {0: Config.SLOT_EMPTY, 1: Config.SLOT_READY_FOR_AI}
        worker = self._make_worker_stub(6, states)
        self.assertEqual(worker._collect_ready_slots(0), [])

    def test_respects_max_batch_size(self):
        Config.MAX_BATCH_SIZE = 2
        states = {i: Config.SLOT_READY_FOR_AI for i in range(6)}
        worker = self._make_worker_stub(6, states)
        self.assertEqual(len(worker._collect_ready_slots(0)), 2)

    def test_wraps_around_ring(self):
        Config.MAX_BATCH_SIZE = 3
        states = {4: Config.SLOT_READY_FOR_AI, 5: Config.SLOT_READY_FOR_AI, 0: Config.SLOT_READY_FOR_AI}
        worker = self._make_worker_stub(6, states)
        self.assertEqual(worker._collect_ready_slots(4), [4, 5, 0])


# ---------------------------------------------------------------------------
# Tests: CLI --batch-size and --pinned-memory arguments
# ---------------------------------------------------------------------------
class BatchSizeCLITest(unittest.TestCase):
    def test_parser_accepts_batch_size(self):
        parser = build_parser()
        args = parser.parse_args(["--batch-size", "2"])
        self.assertEqual(args.batch_size, 2)

    def test_parser_batch_size_default_none(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.batch_size)

    def test_parser_batch_size_one_disables(self):
        parser = build_parser()
        args = parser.parse_args(["--batch-size", "1"])
        self.assertEqual(args.batch_size, 1)

    def test_parser_pinned_memory_flag(self):
        """--pinned-memory flag should be parsed as boolean."""
        parser = build_parser()
        args = parser.parse_args(["--pinned-memory"])
        self.assertTrue(args.pinned_memory)

    def test_parser_pinned_memory_default_false(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertFalse(args.pinned_memory)


# ---------------------------------------------------------------------------
# Tests: PinnedStagingBuffers
# ---------------------------------------------------------------------------
class PinnedStagingBuffersTest(unittest.TestCase):
    """Unit tests for the PinnedStagingBuffers class.

    All shape/logic tests run on CPU.  Pinned-memory-specific assertions
    (is_pinned()) are gated behind CUDA availability.
    """

    def setUp(self):
        self.log = logging.getLogger("test")

    def _make_frame(self, h=8, w=8, val=128):
        return np.full((h, w, 3), val, dtype=np.uint8)

    def test_ensure_allocates_buffers(self):
        """ensure() should allocate input and output tensors of the right shape."""
        staging = PinnedStagingBuffers(self.log)
        staging.ensure(max_batch=3, h=16, w=16, scale=2)
        self.assertIsNotNone(staging.input_pinned)
        self.assertIsNotNone(staging.output_pinned)
        self.assertEqual(staging.input_pinned.shape, (3, 3, 16, 16))
        self.assertEqual(staging.output_pinned.shape, (3, 3, 32, 32))

    def test_ensure_reuses_on_same_shape(self):
        """Calling ensure() with same params should not reallocate."""
        staging = PinnedStagingBuffers(self.log)
        staging.ensure(max_batch=2, h=8, w=8, scale=1)
        ptr_in = staging.input_pinned.data_ptr()
        ptr_out = staging.output_pinned.data_ptr()
        staging.ensure(max_batch=2, h=8, w=8, scale=1)
        self.assertEqual(staging.input_pinned.data_ptr(), ptr_in)
        self.assertEqual(staging.output_pinned.data_ptr(), ptr_out)

    def test_ensure_reallocates_on_shape_change(self):
        """ensure() with different dimensions should produce new tensors."""
        staging = PinnedStagingBuffers(self.log)
        staging.ensure(max_batch=2, h=8, w=8, scale=1)
        ptr_in = staging.input_pinned.data_ptr()
        staging.ensure(max_batch=2, h=16, w=16, scale=1)
        self.assertNotEqual(staging.input_pinned.data_ptr(), ptr_in)
        self.assertEqual(staging.input_pinned.shape, (2, 3, 16, 16))

    def test_stage_input_normalizes(self):
        """stage_input should normalize uint8 [0-255] to float32 [0-1]."""
        staging = PinnedStagingBuffers(self.log)
        staging.ensure(max_batch=1, h=4, w=4, scale=1)
        frame = np.full((4, 4, 3), 255, dtype=np.uint8)
        staging.stage_input(0, frame)
        # All values should be 1.0
        self.assertTrue(torch.allclose(
            staging.input_pinned[0],
            torch.ones(3, 4, 4, dtype=torch.float32),
        ))

    def test_get_output_slice(self):
        staging = PinnedStagingBuffers(self.log)
        staging.ensure(max_batch=3, h=4, w=4, scale=2)
        s0 = staging.get_output_slice(0)
        s1 = staging.get_output_slice(1)
        self.assertEqual(s0.shape, (3, 8, 8))
        self.assertEqual(s1.shape, (3, 8, 8))
        # Slices should be views into the same storage
        self.assertEqual(s0.storage().data_ptr(), s1.storage().data_ptr())

    def test_clear_releases_buffers(self):
        staging = PinnedStagingBuffers(self.log)
        staging.ensure(max_batch=2, h=4, w=4, scale=1)
        staging.clear()
        self.assertIsNone(staging.input_pinned)
        self.assertIsNone(staging.output_pinned)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for pinned memory check")
    def test_ensure_creates_pinned_tensors(self):
        """On CUDA systems, tensors should be page-locked."""
        staging = PinnedStagingBuffers(self.log)
        staging.ensure(max_batch=2, h=8, w=8, scale=2)
        self.assertTrue(staging.input_pinned.is_pinned())
        self.assertTrue(staging.output_pinned.is_pinned())


# ---------------------------------------------------------------------------
# Tests: inference_batch with pinned staging
# ---------------------------------------------------------------------------
class PinnedInferenceBatchTest(unittest.TestCase):
    """End-to-end batch inference using PinnedStagingBuffers on CPU."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = _IdentityModel().eval()
        self.log = logging.getLogger("test")

    def _make_frame(self, h=16, w=16, val=128):
        return np.full((h, w, 3), val, dtype=np.uint8)

    def test_batch_with_pinned_staging_correct_results(self):
        """Batched inference with pinned staging produces same results as without."""
        staging = PinnedStagingBuffers(self.log)
        staging.ensure(max_batch=3, h=16, w=16, scale=1)

        frames = [self._make_frame(val=v) for v in (10, 128, 250)]

        results_plain = inference_batch(self.model, frames, self.device)
        results_pinned = inference_batch(
            self.model, frames, self.device, pinned_staging=staging
        )

        self.assertEqual(len(results_pinned), 3)
        for rp, rn in zip(results_pinned, results_plain):
            np.testing.assert_array_equal(rp, rn)

    def test_single_frame_with_pinned_staging(self):
        """Single-frame batch with staging should match plain inference."""
        staging = PinnedStagingBuffers(self.log)
        staging.ensure(max_batch=1, h=16, w=16, scale=1)

        frame = self._make_frame(val=200)
        [result_plain] = inference_batch(self.model, [frame], self.device)
        [result_pinned] = inference_batch(
            self.model, [frame], self.device, pinned_staging=staging
        )
        np.testing.assert_array_equal(result_pinned, result_plain)


if __name__ == "__main__":
    unittest.main()
