"""
VideoForge SHM Ring Buffer — Shared memory ring buffer for video frame IPC.

Manages the creation, layout, and slot state transitions for the mmap-based
ring buffer shared between the Rust frontend and Python sidecar.
"""

import logging
import mmap
import os
import struct
import tempfile
from typing import Optional, Tuple

log = logging.getLogger("videoforge")


class ShmRingBuffer:
    """Manages the shared-memory ring buffer for video frame IPC.

    Layout (default header = SHM_VERSION, optional extended header = SHM_VERSION+1):
        [ Global Header 36 bytes: magic|version|header_size|slot_count|W|H|S|fmt ]
        [ SlotHeader × ring_size (ring_size × 16 bytes) ]
        [ Slot 0: input (W×H×3) | output (sW×sH×3) ]
        [ Slot 1: input | output ]
        ...  (6 or 8 slots total)
    """

    def __init__(self, config) -> None:
        """
        Args:
            config: Config class with SHM constants (GLOBAL_HEADER_SIZE,
                    SLOT_HEADER_SIZE, SHM_MAGIC, SHM_VERSION, etc.)
        """
        self.config = config
        self.shm_file = None
        self.mmap: Optional[mmap.mmap] = None
        self.shm_path: Optional[str] = None
        self.global_header_size: int = 0
        self.header_region_size: int = 0
        self.ring_size: int = config.RING_SIZE
        self.input_size: int = 0
        self.output_size: int = 0
        self.slot_byte_size: int = 0

    @property
    def is_open(self) -> bool:
        return self.mmap is not None

    def create(
        self,
        width: int,
        height: int,
        scale: int,
        ring_size: int,
        shm_proto_v2: bool,
    ) -> str:
        """Create the shared memory ring buffer.

        Returns:
            shm_path: Filesystem path to the SHM file.

        Raises:
            ValueError: If total size exceeds SHM_MAX_BYTES.
        """
        self.ring_size = ring_size
        self.input_size = width * height * 3
        self.output_size = (width * scale) * (height * scale) * 3
        self.slot_byte_size = self.input_size + self.output_size
        self.global_header_size = (
            self.config.GLOBAL_HEADER_SIZE_V2
            if shm_proto_v2
            else self.config.GLOBAL_HEADER_SIZE
        )
        shm_header_version = (
            self.config.SHM_VERSION + 1 if shm_proto_v2 else self.config.SHM_VERSION
        )
        self.header_region_size = (
            self.global_header_size + self.config.SLOT_HEADER_SIZE * self.ring_size
        )
        total_size = self.header_region_size + self.slot_byte_size * self.ring_size
        if total_size > self.config.SHM_MAX_BYTES:
            raise ValueError(
                f"SHM_SIZE_TOO_LARGE: requested_bytes={total_size} exceeds "
                f"cap={self.config.SHM_MAX_BYTES}. Reduce resolution/scale/ring size."
            )

        fd, self.shm_path = tempfile.mkstemp(prefix="vf_buffer_", suffix=".bin")
        self.shm_file = os.fdopen(fd, "wb+")
        self.shm_file.write(b"\0" * total_size)
        self.shm_file.flush()
        self.shm_file.seek(0)
        self.mmap = mmap.mmap(self.shm_file.fileno(), total_size)

        # Write global header
        if shm_proto_v2:
            global_header = struct.pack(
                "<8sIIIIIIII",
                self.config.SHM_MAGIC,
                shm_header_version,
                self.header_region_size,
                self.ring_size,
                width,
                height,
                scale,
                self.config.PIXEL_FORMAT_RGB24,
                self.config.SHM_PROTOCOL_VERSION,
            )
        else:
            global_header = struct.pack(
                "<8sIIIIIII",
                self.config.SHM_MAGIC,
                shm_header_version,
                self.header_region_size,
                self.ring_size,
                width,
                height,
                scale,
                self.config.PIXEL_FORMAT_RGB24,
            )
        self.mmap[0 : self.global_header_size] = global_header

        log.info(
            f"SHM created: {total_size} bytes "
            f"(global_header={self.global_header_size}, "
            f"header_region={self.header_region_size}, "
            f"{self.ring_size} slots x {self.slot_byte_size}), "
            f"magic=VFSHM001 version={shm_header_version}"
        )
        return self.shm_path

    def validate_header(self) -> None:
        """Validate the SHM global header. Raises ValueError if malformed."""
        min_header = self.global_header_size or self.config.GLOBAL_HEADER_SIZE
        if not self.mmap or len(self.mmap) < min_header:
            raise ValueError(
                f"SHM too small for global header ({min_header} bytes)"
            )
        magic = bytes(self.mmap[0:8])
        if magic != self.config.SHM_MAGIC:
            raise ValueError(
                f"SHM magic mismatch: expected {self.config.SHM_MAGIC!r}, got {magic!r}"
            )
        version = struct.unpack_from("<I", self.mmap, 8)[0]
        if version not in (self.config.SHM_VERSION, self.config.SHM_VERSION + 1):
            raise ValueError(
                f"SHM version mismatch: expected {self.config.SHM_VERSION} or "
                f"{self.config.SHM_VERSION + 1}, got {version}"
            )

    # ── Slot state helpers ─────────────────────────────────────────────

    def slot_state_offset(self, slot_idx: int) -> int:
        """Byte offset of the state field for a given slot header."""
        return (
            self.global_header_size
            + slot_idx * self.config.SLOT_HEADER_SIZE
            + self.config.STATE_FIELD_OFFSET
        )

    def read_slot_state(self, slot_idx: int) -> int:
        """Read the u32 state of a slot from the mmap header."""
        off = self.slot_state_offset(slot_idx)
        return struct.unpack_from("<I", self.mmap, off)[0]

    def write_slot_state(self, slot_idx: int, state: int) -> None:
        """Write the u32 state of a slot into the mmap header."""
        off = self.slot_state_offset(slot_idx)
        struct.pack_into("<I", self.mmap, off, state)

    def slot_data_base(self, slot_idx: int) -> int:
        """Byte offset of the start of data for a given slot (after headers)."""
        return self.header_region_size + slot_idx * self.slot_byte_size

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the mmap and file, optionally removing the file."""
        if self.mmap:
            try:
                self.mmap.close()
            except Exception as e:
                log.warning(f"mmap close failed: {e}")
            self.mmap = None
        if self.shm_file:
            try:
                self.shm_file.close()
                if self.shm_path and os.path.exists(self.shm_path):
                    os.unlink(self.shm_path)
            except Exception as e:
                log.warning(f"SHM file cleanup failed: {e}")
            self.shm_file = None
