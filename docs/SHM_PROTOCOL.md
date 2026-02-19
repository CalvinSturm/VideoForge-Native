# VideoForge Shared Memory Protocol

Version: **1** (`SHM_VERSION = 1`)
Magic: `VFSHM001` (8 ASCII bytes)

---

## Overview

VideoForge uses a memory-mapped file (mmap) as a zero-copy ring buffer for
transferring decoded video frames from the Rust orchestrator to the Python AI
worker, and upscaled frames back.  The SHM file is created by Python
(`create_shm` command) and opened by Rust (`VideoShm::open`).

Both sides synchronize via **atomic u32 state fields** using
`SeqCst` ordering — no mutexes, no per-frame Zenoh round-trips.

---

## File Layout

```
Byte offset   Size    Field
─────────────────────────────────────────────────────────────────────────────
 0            8       magic           = b"VFSHM001"
 8            4       version         = 1  (u32 little-endian)
12            4       header_size     = 84 (= GLOBAL_HEADER_SIZE + slot headers)
16            4       slot_count      = 3  (= RING_SIZE)
20            4       width           = frame width in pixels
24            4       height          = frame height in pixels
28            4       scale           = upscale factor (e.g. 4 for 4×)
32            4       pixel_format    = 1  (RGB24: 3 bytes/pixel, interleaved)
─────────── 36 bytes total global header ────────────────────────────────────

─── Slot 0 header (at offset 36) ────────────────────────────────────────────
36            4       write_index     frame counter set by Rust decoder (u32)
40            4       read_index      frame counter set by Rust encoder (u32)
44            4       state           atomic state machine value (u32, SeqCst)
48            4       frame_bytes     number of valid bytes in input slot (u32)

─── Slot 1 header (at offset 52) ────────────────────────────────────────────
52–67                 (same layout as slot 0)

─── Slot 2 header (at offset 68) ────────────────────────────────────────────
68–83                 (same layout as slot 0)

─── 84 bytes total header region ────────────────────────────────────────────

─── Data region (starts at offset 84) ──────────────────────────────────────
 Slot 0 input:   width × height × 3 bytes  (RGB24)
 Slot 0 output:  (width×scale) × (height×scale) × 3 bytes  (RGB24)
 Slot 1 input:   ...
 Slot 1 output:  ...
 Slot 2 input:   ...
 Slot 2 output:  ...
```

Total file size:
```
84 + 3 × (W×H×3 + W×S×H×S×3)
```

---

## Pixel Format Table

| ID | Name   | Layout                             | Bytes/pixel |
|----|--------|------------------------------------|-------------|
|  1 | RGB24  | Interleaved R G B, row-major       | 3           |

---

## Slot State Machine

Each slot cycles through 6 atomic states:

```
EMPTY (0)
  │  Rust decoder acquires free slot
  ▼
RUST_WRITING (1)
  │  Rust finishes writing input frame data
  ▼
READY_FOR_AI (2)
  │  Python polling loop sees this state, claims the slot
  ▼
AI_PROCESSING (3)
  │  Python GPU inference writes to output region
  ▼
READY_FOR_ENCODE (4)
  │  Rust polling loop sees this, reads output data
  ▼
ENCODING (5)
  │  Rust encoder finishes consuming output
  ▼
EMPTY (0)  ← loop
```

**Atomic ordering**: All state reads and writes use `SeqCst`.  No fencing
beyond the atomic operation is required because the state machine guarantees
that data writes complete before the state transitions that make data visible.

---

## Cross-Process Handshake

1. **Rust** sends `create_shm` command over Zenoh.
2. **Python** creates the temp file, writes the global header, and zeros data.
3. **Python** responds with `SHM_CREATED` and the file path.
4. **Rust** opens the file, validates the global header (magic + version + header_size).
5. **Rust** calls `reset_all_slots()` to zero all slot state fields.
6. **Rust** sends `start_frame_loop` — Python enters its polling loop.
7. Pipeline begins: decode → SHM → AI → SHM → encode.

---

## Validation Rules

| Field        | Rule                                              | Error on violation               |
|--------------|---------------------------------------------------|----------------------------------|
| `magic`      | Must equal `b"VFSHM001"`                         | `SHM magic mismatch`             |
| `version`    | Must equal `SHM_VERSION` (1)                     | `SHM version mismatch`           |
| `header_size`| Must equal `HEADER_REGION_SIZE` (84)             | `SHM header_size mismatch`       |
| File size    | Must be ≥ expected total size for W/H/S/RING_SIZE | `SHM file too small`             |

Both Rust (`VideoShm::open`) and Python (`AIWorker._validate_shm_header`)
perform these checks and return structured errors on failure.

---

## Adding a New Global Header Field (Migration Guide)

1. Increment `SHM_VERSION` in both `shm.rs` and `shm_worker.py`.
2. Append the new field at the end of the global header (before slot headers).
3. Update `GLOBAL_HEADER_SIZE` in both files.
4. Update `HEADER_REGION_SIZE` (which controls `header_size` validation).
5. Update this document.
6. Old clients will reject the new file (version mismatch) — upgrade both
   sides atomically per deployment.
