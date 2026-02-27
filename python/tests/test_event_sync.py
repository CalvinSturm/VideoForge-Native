"""Tests for event_sync.py graceful degradation and metrics.

Validates EventSyncMetrics, degradation tracking, counter increments,
get_metrics() shape, and periodic log_metrics_summary() — all without
requiring Windows or a GPU.
"""

import logging
import os
import sys
import time
import types
import unittest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Pre-stub optional dependencies so we can import event_sync cleanly, and
# also import shm_worker for AIWorker metrics integration tests.
# ---------------------------------------------------------------------------
_STUBS = {}
for mod_name in ("cv2", "zenoh", "watchdog", "shm_ring",
                 "blender_engine", "research_layer", "ipc_protocol",
                 "auto_grade_analysis"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        if mod_name == "cv2":
            stub.resize = lambda *a, **kw: None
            stub.INTER_LANCZOS4 = 4
        elif mod_name == "watchdog":
            stub.start_watchdog = lambda pid: None
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

# Ensure the python dir is on sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "python"))

from event_sync import EventSync, EventSyncMetrics  # noqa: E402


# ---------------------------------------------------------------------------
# Tests: EventSyncMetrics dataclass
# ---------------------------------------------------------------------------
class EventSyncMetricsTest(unittest.TestCase):
    """Unit tests for the EventSyncMetrics dataclass."""

    def test_initial_state(self):
        """Fresh metrics should have all zeros and no degradation."""
        m = EventSyncMetrics()
        self.assertFalse(m.is_degraded)
        self.assertIsNone(m.degraded_since)
        self.assertIsNone(m.degradation_reason)
        self.assertEqual(m.total_waits, 0)
        self.assertEqual(m.total_signals, 0)
        self.assertEqual(m.polling_waits, 0)
        self.assertEqual(m.event_timeouts, 0)
        self.assertEqual(m.event_errors, 0)

    def test_degraded_duration(self):
        """degraded_duration_s should be > 0 when degraded, 0.0 otherwise."""
        m = EventSyncMetrics()
        self.assertEqual(m.degraded_duration_s, 0.0)

        m.degraded_since = time.monotonic() - 2.0
        self.assertGreater(m.degraded_duration_s, 1.5)

    def test_to_dict_shape(self):
        """to_dict() should return all expected keys."""
        m = EventSyncMetrics()
        d = m.to_dict()
        expected_keys = {
            "is_degraded", "degraded_since", "degraded_duration_s",
            "degradation_reason", "total_waits", "total_signals",
            "polling_waits", "event_timeouts", "event_errors",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_reset_clears_counters_not_degradation(self):
        """reset() should zero counters but preserve degradation state."""
        m = EventSyncMetrics()
        m.degraded_since = time.monotonic()
        m.degradation_reason = "test"
        m.total_waits = 100
        m.polling_waits = 50

        m.reset()
        self.assertEqual(m.total_waits, 0)
        self.assertEqual(m.polling_waits, 0)
        # Degradation state preserved
        self.assertTrue(m.is_degraded)
        self.assertEqual(m.degradation_reason, "test")


# ---------------------------------------------------------------------------
# Tests: EventSync degradation behavior
# ---------------------------------------------------------------------------
class EventSyncDisableTest(unittest.TestCase):
    """Test that disable() records degradation info."""

    def test_disable_with_reason_records_degradation(self):
        """disable(reason) should set degraded_since and degradation_reason."""
        sync = EventSync(use_events=False)
        self.assertFalse(sync.metrics.is_degraded)

        sync.disable("permissions denied")
        self.assertTrue(sync.metrics.is_degraded)
        self.assertEqual(sync.metrics.degradation_reason, "permissions denied")
        self.assertEqual(sync.metrics.event_errors, 1)

    def test_disable_none_does_not_record_degradation(self):
        """disable(None) should not set degradation reason or increment errors."""
        sync = EventSync(use_events=False)
        sync.disable(None)

        self.assertFalse(sync.metrics.is_degraded)
        self.assertIsNone(sync.metrics.degradation_reason)
        self.assertEqual(sync.metrics.event_errors, 0)

    def test_first_degradation_timestamp_preserved(self):
        """Multiple disable() calls should keep the FIRST degraded_since timestamp."""
        sync = EventSync(use_events=False)

        sync.disable("first error")
        first_ts = sync.metrics.degraded_since

        time.sleep(0.01)
        sync.disable("second error")
        # Timestamp should be the original one
        self.assertEqual(sync.metrics.degraded_since, first_ts)
        # But reason should be updated
        self.assertEqual(sync.metrics.degradation_reason, "second error")
        self.assertEqual(sync.metrics.event_errors, 2)


# ---------------------------------------------------------------------------
# Tests: EventSync counter increments
# ---------------------------------------------------------------------------
class EventSyncCounterTest(unittest.TestCase):
    """Test that wait/signal increment the right counters."""

    def test_wait_increments_total_and_polling(self):
        """wait_for_input() with no handle should increment total_waits and polling_waits."""
        sync = EventSync(use_events=False)
        sync.wait_for_input()
        sync.wait_for_input()
        sync.wait_for_input()

        self.assertEqual(sync.metrics.total_waits, 3)
        self.assertEqual(sync.metrics.polling_waits, 3)

    def test_signal_increments_total_signals(self):
        """signal_output() should increment total_signals even when no handle."""
        sync = EventSync(use_events=False)
        sync.signal_output()
        sync.signal_output()

        self.assertEqual(sync.metrics.total_signals, 2)

    def test_get_metrics_returns_live_values(self):
        """get_metrics() dict should reflect current counter values."""
        sync = EventSync(use_events=False)
        sync.wait_for_input()
        sync.signal_output()

        d = sync.get_metrics()
        self.assertEqual(d["total_waits"], 1)
        self.assertEqual(d["total_signals"], 1)
        self.assertEqual(d["polling_waits"], 1)
        self.assertFalse(d["is_degraded"])


# ---------------------------------------------------------------------------
# Tests: EventSync setup-disable-setup cycle
# ---------------------------------------------------------------------------
class EventSyncSetupCycleTest(unittest.TestCase):
    """Test that metrics survive / reset across setup() cycles."""

    def test_setup_clears_degradation_state(self):
        """setup() should clear degraded_since and degradation_reason."""
        sync = EventSync(use_events=False)
        sync.disable("test error")
        self.assertTrue(sync.metrics.is_degraded)

        # setup() calls disable(None) then clears degradation
        sync.setup({})
        self.assertFalse(sync.metrics.is_degraded)
        self.assertIsNone(sync.metrics.degradation_reason)

    def test_counters_persist_across_setup(self):
        """Operational counters should persist across setup() calls."""
        sync = EventSync(use_events=False)
        sync.wait_for_input()
        sync.signal_output()
        self.assertEqual(sync.metrics.total_waits, 1)

        sync.setup({})
        # Counters still there
        self.assertEqual(sync.metrics.total_waits, 1)
        self.assertEqual(sync.metrics.total_signals, 1)


# ---------------------------------------------------------------------------
# Tests: log_metrics_summary interval
# ---------------------------------------------------------------------------
class MetricsLogIntervalTest(unittest.TestCase):
    """Test that log_metrics_summary() respects the interval."""

    def test_no_log_before_interval(self):
        """Should not emit a log line until interval is reached."""
        sync = EventSync(use_events=False, metrics_interval=5)

        with patch.object(logging.getLogger("videoforge"), "warning") as mock_warn:
            with patch.object(logging.getLogger("videoforge"), "debug") as mock_debug:
                for _ in range(4):
                    sync.log_metrics_summary()
                mock_warn.assert_not_called()
                mock_debug.assert_not_called()

    def test_logs_at_interval(self):
        """Should emit exactly one log line when interval is reached."""
        sync = EventSync(use_events=False, metrics_interval=3)

        with patch.object(logging.getLogger("videoforge"), "debug") as mock_debug:
            for _ in range(3):
                sync.log_metrics_summary()
            self.assertEqual(mock_debug.call_count, 1)
            # Check it contains the tag
            self.assertIn("EVENT_SYNC_METRICS", mock_debug.call_args[0][0])

    def test_logs_degraded_as_warning(self):
        """When degraded, should log at WARNING level."""
        sync = EventSync(use_events=False, metrics_interval=2)
        sync.disable("test degradation")

        with patch.object(logging.getLogger("videoforge"), "warning") as mock_warn:
            # 2 calls: first from disable(), then from log_metrics_summary interval
            mock_warn.reset_mock()
            for _ in range(2):
                sync.log_metrics_summary()
            self.assertEqual(mock_warn.call_count, 1)
            msg = mock_warn.call_args[0][0]
            self.assertIn("DEGRADED", msg)
            self.assertIn("test degradation", msg)


# ---------------------------------------------------------------------------
# Tests: AIWorker._event_metrics integration
# (Requires torch for shm_worker import — skipped when unavailable)
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch required for AIWorker integration tests")
class AIWorkerEventMetricsTest(unittest.TestCase):
    """Test that AIWorker tracks event metrics through its inline methods."""

    def _make_worker_stub(self):
        """Create a minimal AIWorker stub (no Zenoh, no model)."""
        from shm_worker import AIWorker, Config

        worker = object.__new__(AIWorker)
        worker.events_enabled = False
        worker.event_in_name = None
        worker.event_out_name = None
        worker._event_in_handle = None
        worker._event_out_handle = None
        worker._event_wait_timeout_ms = 50
        worker._event_warned = False
        worker._event_metrics = EventSyncMetrics()
        worker._event_metrics_interval = 500
        worker._event_metrics_counter = 0
        worker.log = logging.getLogger("videoforge")
        return worker

    def test_disable_records_degradation(self):
        worker = self._make_worker_stub()
        worker._disable_event_sync("handle limit reached")

        m = worker._event_metrics
        self.assertTrue(m.is_degraded)
        self.assertEqual(m.degradation_reason, "handle limit reached")
        self.assertEqual(m.event_errors, 1)

    def test_wait_increments_polling(self):
        worker = self._make_worker_stub()
        worker._wait_for_input_event()

        m = worker._event_metrics
        self.assertEqual(m.total_waits, 1)
        self.assertEqual(m.polling_waits, 1)

    def test_signal_increments_counter(self):
        worker = self._make_worker_stub()
        worker._signal_output_event()

        m = worker._event_metrics
        self.assertEqual(m.total_signals, 1)

    def test_setup_clears_degradation(self):
        worker = self._make_worker_stub()
        worker.use_events = False  # skip actual Win32 calls
        worker._disable_event_sync("test error")
        self.assertTrue(worker._event_metrics.is_degraded)

        worker._setup_event_sync({})
        self.assertFalse(worker._event_metrics.is_degraded)


if __name__ == "__main__":
    unittest.main()
