# VideoForge Performance Audit

## Executive Summary

The production pipeline has **three major bottlenecks**: (1) CPU-mediated frame transfer through shared memory, (2) FFmpeg CLI subprocess overhead for decode/encode, and (3) Python GIL contention during inference. The engine-v2 crate addresses all three but is not yet integrated.

---

## Bottleneck Analysis

### 🔴 Critical: CPU-Mediated Frame Transfer

**Location**: `shm.rs` ↔ `shm_worker.py`

The shared memory ring buffer transfers raw RGB24 pixel data through host memory:

```
GPU (decoder) → CPU (FFmpeg stdout) → CPU (SHM mmap) → CPU (Python read) → GPU (torch tensor)
→ GPU (inference) → CPU (torch result) → CPU (SHM mmap) → CPU (FFmpeg stdin) → GPU (encoder)
```

**Impact**: For 4K RGB24 frames (3840×2160×3 = ~24 MB/frame):

- 2 CPU↔GPU copies per frame (decode output + encode input) via FFmpeg
- 2 CPU↔SHM copies per frame (write input + read output) in Python
- Total: ~96 MB of data movement per frame through CPU
- At 30 fps: **~2.9 GB/s** of unnecessary CPU memory bandwidth

**Evidence**: The engine-v2 pipeline eliminates this entirely with GPU-resident NV12 buffers, suggesting the team has already identified this as the primary bottleneck.

---

### 🔴 Critical: FFmpeg Subprocess Overhead

**Location**: `video_pipeline.rs` (`VideoDecoder`, `VideoEncoder`)

FFmpeg runs as a child process with stdio piping:

- **Decoder**: `ffmpeg -i input -f rawvideo -pix_fmt rgb24 pipe:1` — even when using NVDEC, frames are copied from GPU to CPU by FFmpeg, then piped to Rust.
- **Encoder**: `ffmpeg -f rawvideo -pix_fmt rgb24 -i pipe:0 -c:v h264_nvenc output` — frames go CPU→GPU again inside FFmpeg for NVENC.

**Impact**: Process spawn latency (~50-100ms), pipe buffer limits (65KB default on Windows), and double GPU↔CPU transfers negate the benefit of hardware acceleration.

---

### 🟡 Moderate: Python GIL + Frame Loop Polling

**Location**: `shm_worker.py` (`AIWorker._frame_loop`)

The frame loop polls SHM slot states in a tight loop with `time.sleep(0.0005)` (500µs):

```python
while self._frame_loop_active:
    # Read slot state via struct.unpack from mmap
    state = self._read_slot_state(current_slot)
    if state == SLOT_READY_FOR_AI:
        # process...
    else:
        time.sleep(0.0005)
```

**Issues**:

- Polling with `struct.unpack` + `time.sleep` instead of OS-level synchronization (semaphore/futex)
- GIL prevents true parallelism between the frame loop thread and the Zenoh subscriber thread
- `_process_slot` does numpy↔torch conversions on every frame

---

### 🟡 Moderate: Per-Frame Tensor Allocation

**Location**: `shm_worker.py` (`_process_slot`, `inference`)

Each frame triggers:

1. `np.frombuffer()` from SHM → numpy array (zero-copy, OK)
2. `torch.from_numpy().permute().float().div_(255.0)` → new CUDA tensor (allocation)
3. `model(tensor)` → inference output tensor (allocation)
4. `.cpu().numpy()` → copy back to CPU (copy)
5. `struct.pack_into()` → write to SHM (copy)

**Opportunity**: Pre-allocate persistent CUDA tensors for input/output and reuse across frames. The blender engine already does this for color-space matrices (`_ensure_matrix_on`).

---

### 🟢 Low: Micro-Batching Overhead

**Location**: `shm_worker.py` (`_collect_ready_slots`, `_process_batch`)

Micro-batching (collecting consecutive READY_FOR_AI slots) is well-implemented but limited:

- 3-slot ring buffer caps batch size to 3 (usually 1-2 in practice)
- Batch collection requires sequential slot readiness
- Deterministic mode forces `MAX_BATCH_SIZE = 1`

---

## Performance Characteristics by Component

### Decode Stage

| Metric | Value | Notes |
|--------|-------|-------|
| NVDEC availability | Probed at runtime | Falls back to CPU libx264 |
| Frame format | RGB24 via rawvideo pipe | 3 bytes/pixel, uncompressed |
| Pipe bandwidth | ~65KB buffer (Windows) | Potential stall on large frames |

### Inference Stage

| Metric | Value | Notes |
|--------|-------|-------|
| Default precision | FP32 | FP16 supported via `--precision fp16` |
| Batch size | 1-3 (ring slots) | 1 in deterministic mode |
| Model load time | ~2-5s (PyTorch) | Cached in registry after first load |
| Framework | PyTorch CUDA | No TensorRT in production |

### Encode Stage

| Metric | Value | Notes |
|--------|-------|-------|
| Default codec | h264_nvenc | Falls back to libx264 |
| Pixel format | RGB24 → yuv420p (FFmpeg internal) | Extra conversion step |
| Quality | CRF 18 (default) | Configurable |

---

## Memory Profile

### Host Memory

| Consumer | Estimate | Notes |
|----------|----------|-------|
| SHM mmap (3 slots, 4K×4 upscale) | ~2.1 GB | `3 × (24MB_in + 384MB_out)` |
| FFmpeg decoder process | ~100-500 MB | Depends on codec |
| FFmpeg encoder process | ~100-500 MB | NVENC session memory |
| Python process base | ~200-500 MB | PyTorch framework |

### GPU Memory

| Consumer | Estimate | Notes |
|----------|----------|-------|
| RCAN model (FP32) | ~300-500 MB | 10 groups × 20 RCAB |
| EDSR model (FP32) | ~500-800 MB | 32 ResBlocks × 256 features |
| RealESRGAN model | ~100-300 MB | Depends on variant |
| Inference workspace | ~500 MB-2 GB | Proportional to resolution |
| Research layer tensors | ~100-500 MB | HF masks, blending buffers |

---

## Latency Breakdown (Estimated, 1080p, RCAN_x4)

| Stage | Estimated Time | Bottleneck |
|-------|---------------|-----------|
| FFmpeg decode (NVDEC) | ~1-3 ms | Pipe transfer dominates |
| SHM write (Rust→mmap) | ~2-5 ms | 6 MB memcpy |
| SHM read (Python←mmap) | ~2-5 ms | 6 MB memcpy + numpy conversion |
| Torch tensor creation | ~1-2 ms | GPU allocation + H2D transfer |
| Model inference | **~50-200 ms** | **GPU compute (dominant)** |
| Research post-process | ~5-20 ms | HF analysis, blending |
| Torch→numpy→SHM write | ~5-10 ms | D2H transfer + memcpy |
| SHM→FFmpeg pipe | ~2-5 ms | 24 MB (4x upscaled) |
| FFmpeg encode (NVENC) | ~2-5 ms | H2D + NVENC |
| **Total per frame** | **~70-255 ms** | **~4-14 FPS** |

---

## Recommendations

### Immediate Wins (No Architecture Change)

1. **Pre-allocate CUDA tensors** in `_process_slot` — reuse input/output tensors across frames instead of creating new ones per frame.
2. **Increase ring buffer size** from 3 to 5-8 slots — better pipelining overlap between decode/AI/encode.
3. **Use `torch.cuda.Stream`** for async H2D/D2H copies that overlap with inference.
4. **Replace polling sleep** with a proper synchronization primitive (e.g., named semaphore for cross-process signaling).

### Medium-Term (Moderate Refactoring)

1. **Replace FFmpeg CLI with FFmpeg library binding** (e.g., `ffmpeg-next` Rust crate) — eliminate subprocess overhead and enable GPU-resident decode.
2. **Enable TensorRT in production** — convert PyTorch models to ONNX, use TRT for 3-5× inference speedup.
3. **Implement NV12 path** — avoid RGB24 conversion in decode/encode stages.

### Long-Term (Architecture Migration)

1. **Integrate engine-v2** — the GPU-native pipeline eliminates all CPU-mediated frame transfers. See `REFACTOR_OR_MIGRATION_PLAN.md`.
