"""
VideoForge Process Watchdog — Suicide pact with parent process.

Monitors the parent PID and terminates this process if the parent dies.
Windows-specific implementation using ctypes kernel32.
"""

import ctypes
import logging
import os
import threading
import time

log = logging.getLogger("videoforge")

# Windows constants
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
SYNCHRONIZE = 0x00100000
EVENT_MODIFY_STATE = 0x0002


def is_pid_alive(pid: int) -> bool:
    """Check if PID is alive on Windows using ctypes kernel32"""
    try:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False

        exit_code = ctypes.c_ulong()
        if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            ctypes.windll.kernel32.CloseHandle(handle)
            return exit_code.value == 259  # STILL_ACTIVE

        ctypes.windll.kernel32.CloseHandle(handle)
        return False
    except Exception as e:
        log.warning(f"PID check failed: {e}")
        return False


def watchdog_loop(parent_pid: int, check_interval: float = 2.0) -> None:
    """Monitor parent process. If it dies, we die."""
    log.info(f"Watchdog started for Parent PID: {parent_pid}")
    while True:
        if not is_pid_alive(parent_pid):
            log.info(f"Parent {parent_pid} died. Committing seppuku...")
            os._exit(0)
        time.sleep(check_interval)


def start_watchdog(parent_pid: int, check_interval: float = 2.0) -> None:
    if parent_pid <= 0:
        return
    t = threading.Thread(
        target=watchdog_loop, args=(parent_pid, check_interval), daemon=True
    )
    t.start()
