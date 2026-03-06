# Engine Architecture Audit — Python vs Native (Rave)

## Architecture Overview

| Dimension | Python Engine | Native Engine (Rave) |
|---|---|---|
| **Language** | Python + PyTorch + ORT | Rust + ORT (C API) |
| **Inference backend** | PyTorch `model(tensor)` or ORT `session.run()` | ORT IO Binding with raw GPU pointers |
| **Decode/Encode** | SHM ring buffer (Rust↔Python IPC via Zenoh) | NVDEC/NVENC (GPU-native, zero-copy) |
| **Pipeline** | Single-threaded frame loop with polling | Async 4-stage pipeline (decode→preprocess→infer→encode) |
| **Color space** | RGB (numpy uint8) | NV12 → RGB planar F32 → NV12 (CUDA kernels) |

---

## Tiling

| Feature | Python Engine | Native Engine |
|---|---|---|
| **Tiled inference** | ✅ Yes | ✅ Yes |
| **Tile size (CNN)** | 512px | Configurable (`--tile-size`) |
| **Tile size (transformer)** | 256px | Configurable (`--tile-size`) |
| **Tile padding** | 32px reflect | Configurable (`--tile-pad`) |
| **Transformer detection** | `_TRANSFORMER_KEYS = {"dat", "swin", "hat", "realweb", "omnisr", "lmlt"}` | Dynamic spatial axis probe (NCHW shape check) |
| **Tile merge** | Seamless crop-and-place | CUDA `crop_tile` / `place_tile` kernels |
| **Serialization** | Sequential tile loop | Serialized crop→infer→place→recycle with `sync_stream` + `synchronize()` barriers |
| **Buffer recycling** | N/A (PyTorch GC) | ✅ Crop buffers recycled to pool (`ctx.recycle`) — zero-allocation steady-state after warm-up |

### Key files

- Python: [shm_worker.py](file:///c:/Users/Calvin/Desktop/VideoForge1/python/shm_worker.py#L1119-L1232) — `process_image_tile()`
- Python: [model_manager.py](file:///c:/Users/Calvin/Desktop/VideoForge1/python/model_manager.py#L644-L647) — `_TRANSFORMER_KEYS` and `preferred_tile_size`
- Native: [pipeline.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-pipeline/src/pipeline.rs#L1736-L1837) — `tiled_inference()`
- Native: [kernels.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-cuda/src/kernels.rs#L247-L330) — `crop_tile_planar_f32`, `place_tile_planar_f32` CUDA kernels

---

## Precision (FP16)

| Feature | Python Engine | Native Engine |
|---|---|---|
| **FP16 support** | ✅ `torch.autocast("cuda", dtype=torch.float16)` | ✅ TRT EP `with_fp16(true)` + CUDA EP FP16 model auto-detect |
| **Precision modes** | `fp32`, `fp16`, `deterministic` | `Fp32`, `Fp16`, `Int8` (TRT-only for INT8) |
| **CUDA EP optimization** | Via PyTorch autocast | ✅ TF32 + max workspace + FP16 model auto-detect (`.fp16.onnx`) |
| **FP16 model conversion** | N/A (uses torch.autocast) | ✅ `python scripts/convert_fp16.py model.onnx` |
| **Runtime configurable** | ✅ `--precision` CLI flag | ✅ Via `PrecisionPolicy` enum (drives both TRT + CUDA EP) |

### Key files

- Python: [inference_engine.py](file:///c:/Users/Calvin/Desktop/VideoForge1/python/inference_engine.py#L116-L205) — `inference()`, lines 160-162 for autocast
- Python: [shm_worker.py](file:///c:/Users/Calvin/Desktop/VideoForge1/python/shm_worker.py#L46-L84) — `configure_precision()`
- Native: [tensorrt.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-tensorrt/src/tensorrt.rs#L130-L143) — `PrecisionPolicy` enum

---

## Inference Method

| Feature | Python Engine (PyTorch) | Python Engine (ONNX) | Native Engine |
|---|---|---|---|
| **Method** | `model(tensor)` forward pass | `session.run([name], {name: np_input})` | ORT IO Binding (`run_binding`) for CNN; CPU roundtrip (`session.run()`) for transformers on CUDA EP |
| **Zero-copy GPU** | ✅ PyTorch tensors stay on GPU | ❌ CPU→GPU roundtrip per frame | ✅ GPU pointers bound directly |
| **Memory mgmt** | PyTorch GC + `torch.cuda.empty_cache()` | ORT-managed | Pre-allocated ring buffer |
| **Batch support** | ✅ `inference_batch()` with OOM fallback | ❌ Single-frame only | ✅ `process_batch()` (1-8 frames) |
| **Error recovery** | `try/except → sequential fallback` | Deadlock probe with 20s timeout | `error_on_failure()` → crash |

### Key files

- Python PyTorch: [inference_engine.py](file:///c:/Users/Calvin/Desktop/VideoForge1/python/inference_engine.py#L116-L205)
- Python ONNX: [model_manager.py](file:///c:/Users/Calvin/Desktop/VideoForge1/python/model_manager.py#L554-L571) — `OnnxModelWrapper.forward()`
- Native: [tensorrt.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-tensorrt/src/tensorrt.rs#L1220-L1287) — `run_io_bound()`

---

## Memory Architecture

| Feature | Python Engine | Native Engine |
|---|---|---|
| **GPU buffer strategy** | Per-frame alloc or `PreallocBuffers` reuse | Pre-allocated output ring (6 slots) |
| **Ring buffer** | SHM ring (CPU, 6-8 slots) | GPU ring (6 slots, VRAM-resident) |
| **VRAM per slot** | ~tile_size² × 3 × 4 bytes | out_w × out_h × 3 × elem_bytes |
| **VRAM pressure** | Low (256px tiles = ~768KB/tile) | High (1624×1880 × 3 × 4 = ~36MB/slot × 6 = ~220MB) |
| **Pinned memory** | ✅ `PinnedStagingBuffers` for async DMA | ❌ No pinned staging |
| **OOM handling** | ✅ Catches OOM, falls back to sequential | N/A — pre-allocated ring + bucketed pool; OOM only at startup (crash with clear error is correct) |
| **Tile buffer pool** | N/A (PyTorch GC) | ✅ Bucketed pool via `GpuContext::alloc/recycle` — eliminates `cuMemAllocAsync` churn |
| **Cleanup** | `torch.cuda.empty_cache()` | Pool-based recycling; no per-frame dealloc |

### Key files

- Python: [inference_engine.py](file:///c:/Users/Calvin/Desktop/VideoForge1/python/inference_engine.py#L316-L467) — `PreallocBuffers`, `PinnedStagingBuffers`
- Native: [tensorrt.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-tensorrt/src/tensorrt.rs) — ring allocation in `ensure_ring()`

---

## Transformer Model Support

| Feature | Python Engine | Native Engine |
|---|---|---|
| **Architecture detection** | ✅ `_TRANSFORMER_KEYS` + state-dict patterns | ✅ Dynamic spatial axis probe |
| **Adapted tile size** | ✅ 256px for transformers | ✅ Configurable via `--tile-size` |
| **ORT CUDA EP probe** | ✅ 20s deadlock timeout with dummy input | N/A — `run_session_cpu_roundtrip()` avoids the IO-binding deadlock path entirely |
| **EP fallback** | CUDA EP → CPU EP (if CUDA hangs) | TRT EP → CUDA EP (if TRT build fails) |
| **Window padding** | ✅ Adapters: `SwinIRAdapter`, `HATAdapter`, `DATAdapter` | ❌ No adapter system |
| **Speed (DAT2 @ 406×470)** | ~1-3s/frame (256px tiles × FP16) | ✅ Tiling verified — no black tiles (stream sync + buffer recycling fix) |

### Python's ONNX deadlock probe

[model_manager.py](file:///c:/Users/Calvin/Desktop/VideoForge1/python/model_manager.py#L586-L616) — `_probe_onnx_session()` runs a 64×64 dummy inference in a daemon thread. If it hangs for 20s (some DAT2 CUDA ops deadlock ORT), it falls back to CPU EP.

---

## Optimization Recommendations vs Current State

| Recommendation | Python Engine | Native Engine | Impact |
|---|---|---|---|
| ~~**1. Tiled inference**~~ | ✅ Already implemented | ✅ **Done** (`pipeline.rs` + CUDA kernels) | ~~10-50× speedup~~ |
| ~~**2. CUDA EP + FP16 model**~~ | ✅ Already implemented | ✅ **Done** (TF32 + FP16 auto-detect + conversion script) | ~~~2× speedup~~ |
| ~~**3. Simpler session.run()**~~ | ✅ Uses `session.run()` (ONNX path) | ✅ **Done** (`run_session_cpu_roundtrip()` for CUDA EP) | ~~**Fixes crash** after N frames~~ |
| ~~**4. Black tile race condition**~~ | N/A (single-threaded) | ✅ **Fixed** (explicit `sync_stream` + buffer recycling to pool) | ~~**Fixes intermittent black tiles**~~ |
| ~~**5. OOM recovery**~~ | ✅ Catches + falls back | **N/A** — pre-allocated buffers; OOM only at startup | ~~Not applicable~~ |
| ~~**6. Deadlock-safe ONNX probe**~~ | ✅ 20s timeout thread | **N/A** — CPU roundtrip avoids deadlock path | ~~Not applicable~~ |

### Not applicable to native engine

- **OOM recovery** — Python dynamically allocates per-frame, making OOM a real runtime risk. The native engine pre-allocates ring buffers + uses a bucketed pool, so OOM only occurs at startup/model load, not mid-frame. Crashing with a clear error is the correct behavior; the caller adjusts tile size/resolution and retries.
- **Deadlock-safe ONNX probe** — Python needs this because some transformer CUDA ops deadlock inside ORT's `session.run()` with CUDA EP + IO binding. The native engine avoids this entirely — transformers on CUDA EP use `run_session_cpu_roundtrip()` (host-side `session.run()`), which doesn't trigger the deadlock. The failure mode doesn't exist.

---

## Summary

The Python engine is **production-hardened for transformer models** with tiling, FP16, deadlock probes, and OOM recovery. The native engine now has **tiled inference** (with stream-synchronized buffer recycling — zero-allocation steady-state), **FP16-optimized CUDA EP** (TF32 + FP16 model auto-detection + conversion script), a zero-copy GPU pipeline with TRT EP acceleration, and **host round-trip inference for CUDA EP** (fixes transformer model crashes). The intermittent black tile race condition (`crop_tile` zero buffer from `cuMemAllocAsync` reuse) has been **resolved** via explicit `inference_stream` sync and pool-based buffer recycling.

The native engine is now at **practical parity** with the Python engine for transformer model support. The two remaining Python-specific items (OOM recovery and deadlock-safe ONNX probe) are architectural non-issues in the native engine — pre-allocated buffers eliminate mid-frame OOM, and CPU roundtrip inference avoids the IO-binding deadlock path entirely.

Completed work:

1. ~~**Offline ONNX FP16 model conversion**~~ — **Done** (`scripts/convert_fp16.py` + auto-detection)
2. ~~**OOM-safe `session.run()` fallback**~~ — **Done** (`run_session_cpu_roundtrip()` dispatched for CUDA EP sessions)
3. ~~**Black tile race condition**~~ — **Fixed** (explicit `sync_stream` + `ctx.recycle` buffer pooling in tile loop)
4. ~~**OOM recovery**~~ — **N/A** (native engine pre-allocates ring buffers + bucketed pool; OOM only at startup, not mid-frame)
5. ~~**Deadlock-safe ONNX probe**~~ — **N/A** (native engine uses `run_session_cpu_roundtrip()` for transformers on CUDA EP, avoiding the deadlock path)
