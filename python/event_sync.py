"""
VideoForge Win32 Named Event Synchronization.

Provides an EventSync class that wraps Windows named events for
low-latency IPC signaling between Rust and Python processes.
Falls back to polling if events are unavailable or fail.

When degradation occurs (event creation fails due to permissions,
handle limits, etc.), metrics are recorded and logged periodically
so that latency spikes can be diagnosed.
"""

import ctypes
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("videoforge")

# Windows constants
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
SYNCHRONIZE = 0x00100000
EVENT_MODIFY_STATE = 0x0002


@dataclass
class EventSyncMetrics:
    """Observability counters for Win32 event sync degradation.

    Tracks when and why the system fell back to polling, plus
    operational counters so operators can correlate latency spikes
    with degraded event sync.
    """

    # Degradation state
    degraded_since: Optional[float] = None  # monotonic timestamp
    degradation_reason: Optional[str] = None

    # Operational counters
    total_waits: int = 0
    total_signals: int = 0
    polling_waits: int = 0  # waits with no event handle (polling fallback)
    event_timeouts: int = 0  # WaitForSingleObject returned WAIT_TIMEOUT
    event_errors: int = 0  # errors before disable

    @property
    def is_degraded(self) -> bool:
        return self.degraded_since is not None

    @property
    def degraded_duration_s(self) -> float:
        """Seconds spent in degraded mode (0.0 if not degraded)."""
        if self.degraded_since is None:
            return 0.0
        return time.monotonic() - self.degraded_since

    def reset(self) -> None:
        """Reset all counters (not degradation state)."""
        self.total_waits = 0
        self.total_signals = 0
        self.polling_waits = 0
        self.event_timeouts = 0
        self.event_errors = 0

    def to_dict(self) -> dict:
        """Snapshot for programmatic access / IPC reporting."""
        return {
            "is_degraded": self.is_degraded,
            "degraded_since": self.degraded_since,
            "degraded_duration_s": round(self.degraded_duration_s, 2),
            "degradation_reason": self.degradation_reason,
            "total_waits": self.total_waits,
            "total_signals": self.total_signals,
            "polling_waits": self.polling_waits,
            "event_timeouts": self.event_timeouts,
            "event_errors": self.event_errors,
        }


class EventSync:
    """Win32 named-event synchronization for SHM frame signaling.

    Usage:
        sync = EventSync(use_events=True, wait_timeout_ms=50)
        sync.setup(payload)         # Open named events from IPC payload
        sync.wait_for_input()       # Block until Rust signals input ready
        sync.signal_output()        # Signal Rust that output is ready
        sync.disable("reason")      # Fall back to polling
        sync.cleanup()              # Close handles

    Metrics:
        sync.metrics                # EventSyncMetrics dataclass
        sync.get_metrics()          # dict snapshot
        sync.log_metrics_summary()  # periodic one-line diagnostic
    """

    def __init__(
        self,
        use_events: bool = False,
        wait_timeout_ms: int = 50,
        metrics_interval: int = 500,
    ):
        self.use_events = use_events
        self.events_enabled = False
        self.event_in_name: Optional[str] = None
        self.event_out_name: Optional[str] = None
        self._event_in_handle = None
        self._event_out_handle = None
        self._event_wait_timeout_ms = wait_timeout_ms
        self._event_warned = False

        # Observability
        self.metrics = EventSyncMetrics()
        self._metrics_interval = max(1, metrics_interval)
        self._metrics_log_counter = 0

    def _close_handle(self, handle) -> None:
        if handle is None:
            return
        try:
            ctypes.windll.kernel32.CloseHandle(handle)
        except Exception as e:
            log.warning(f"CloseHandle failed: {e}")

    def disable(self, reason: Optional[str]) -> None:
        """Disable event sync and fall back to polling."""
        if reason:
            log.warning(f"EVENT_SYNC_DEGRADED: {reason}; falling back to polling")
            self.metrics.event_errors += 1
            # Only record the first degradation timestamp
            if self.metrics.degraded_since is None:
                self.metrics.degraded_since = time.monotonic()
            self.metrics.degradation_reason = reason
        self.events_enabled = False
        self.event_in_name = None
        self.event_out_name = None
        if self._event_in_handle is not None:
            self._close_handle(self._event_in_handle)
            self._event_in_handle = None
        if self._event_out_handle is not None:
            self._close_handle(self._event_out_handle)
            self._event_out_handle = None

    def setup(self, payload: dict) -> None:
        """Open named events from an IPC create_shm payload."""
        self.disable(None)
        # Clear degradation state on fresh setup
        self.metrics.degraded_since = None
        self.metrics.degradation_reason = None

        if not self.use_events:
            return
        if os.name != "nt":
            if not self._event_warned:
                log.warning("--use-events is Windows-only; falling back to polling")
                self._event_warned = True
            return

        in_name = payload.get("event_in_name")
        out_name = payload.get("event_out_name")
        if not in_name or not out_name:
            self.disable("Event names missing from create_shm payload")
            return

        try:
            open_event = ctypes.windll.kernel32.OpenEventW
            open_event.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
            open_event.restype = ctypes.c_void_p

            in_handle = open_event(SYNCHRONIZE, 0, in_name)
            out_handle = open_event(EVENT_MODIFY_STATE, 0, out_name)
            if not in_handle or not out_handle:
                self._close_handle(in_handle)
                self._close_handle(out_handle)
                self.disable(
                    f"OpenEventW failed for in={in_name!r} out={out_name!r}"
                )
                return

            self._event_in_handle = in_handle
            self._event_out_handle = out_handle
            self.event_in_name = in_name
            self.event_out_name = out_name
            self.events_enabled = True
            log.info(
                f"Win32 event sync enabled: in={self.event_in_name} out={self.event_out_name} "
                f"timeout_ms={self._event_wait_timeout_ms}"
            )
        except Exception as e:
            self.disable(f"Win32 event setup failed: {e}")

    def wait_for_input(self) -> None:
        """Block until the input-ready event is signaled (or timeout)."""
        self.metrics.total_waits += 1

        if not self.events_enabled or self._event_in_handle is None:
            self.metrics.polling_waits += 1
            return

        try:
            wait_result = ctypes.windll.kernel32.WaitForSingleObject(
                self._event_in_handle, self._event_wait_timeout_ms
            )
            if wait_result == WAIT_OBJECT_0:
                return
            if wait_result == WAIT_TIMEOUT:
                self.metrics.event_timeouts += 1
                return
            self.disable(
                f"WaitForSingleObject returned unexpected code {wait_result}"
            )
        except Exception as e:
            self.disable(f"WaitForSingleObject failed: {e}")

    def signal_output(self) -> None:
        """Signal the output-ready event."""
        self.metrics.total_signals += 1

        if not self.events_enabled or self._event_out_handle is None:
            return
        try:
            set_event = ctypes.windll.kernel32.SetEvent
            set_event.argtypes = [ctypes.c_void_p]
            set_event.restype = ctypes.c_int
            if not set_event(self._event_out_handle):
                self.disable("SetEvent(out_ready) failed")
        except Exception as e:
            self.disable(f"SetEvent(out_ready) failed: {e}")

    def log_metrics_summary(self) -> None:
        """Log a one-line diagnostic summary every `metrics_interval` calls.

        Call this from the hot loop.  Only emits a log line every N calls
        to avoid flooding.
        """
        self._metrics_log_counter += 1
        if self._metrics_log_counter < self._metrics_interval:
            return
        self._metrics_log_counter = 0

        m = self.metrics
        if m.is_degraded:
            log.warning(
                f"EVENT_SYNC_METRICS [DEGRADED {m.degraded_duration_s:.1f}s]: "
                f"reason={m.degradation_reason!r} "
                f"waits={m.total_waits} signals={m.total_signals} "
                f"polling={m.polling_waits} timeouts={m.event_timeouts} "
                f"errors={m.event_errors}"
            )
        else:
            log.debug(
                f"EVENT_SYNC_METRICS [OK]: "
                f"waits={m.total_waits} signals={m.total_signals} "
                f"timeouts={m.event_timeouts}"
            )

    def get_metrics(self) -> dict:
        """Return a dict snapshot of current metrics."""
        return self.metrics.to_dict()

    def cleanup(self) -> None:
        """Close all event handles."""
        self.disable(None)
