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
| **Serialization** | Sequential tile loop | Serialized crop→infer→place→drop with `synchronize()` barriers |

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
| **Method** | `model(tensor)` forward pass | `session.run([name], {name: np_input})` | ORT IO Binding (`run_binding`) |
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
| **OOM handling** | ✅ Catches OOM, falls back to sequential | ❌ Process crash |
| **Cleanup** | `torch.cuda.empty_cache()` | No explicit cleanup between frames |

### Key files

- Python: [inference_engine.py](file:///c:/Users/Calvin/Desktop/VideoForge1/python/inference_engine.py#L316-L467) — `PreallocBuffers`, `PinnedStagingBuffers`
- Native: [tensorrt.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-tensorrt/src/tensorrt.rs) — ring allocation in `ensure_ring()`

---

## Transformer Model Support

| Feature | Python Engine | Native Engine |
|---|---|---|
| **Architecture detection** | ✅ `_TRANSFORMER_KEYS` + state-dict patterns | ✅ Dynamic spatial axis probe |
| **Adapted tile size** | ✅ 256px for transformers | ✅ Configurable via `--tile-size` |
| **ORT CUDA EP probe** | ✅ 20s deadlock timeout with dummy input | ❌ No deadlock detection |
| **EP fallback** | CUDA EP → CPU EP (if CUDA hangs) | TRT EP → CUDA EP (if TRT build fails) |
| **Window padding** | ✅ Adapters: `SwinIRAdapter`, `HATAdapter`, `DATAdapter` | ❌ No adapter system |
| **Speed (DAT2 @ 406×470)** | ~1-3s/frame (256px tiles × FP16) | ⏳ Untested with tiling (was ~35s full-frame × FP32) |

### Python's ONNX deadlock probe

[model_manager.py](file:///c:/Users/Calvin/Desktop/VideoForge1/python/model_manager.py#L586-L616) — `_probe_onnx_session()` runs a 64×64 dummy inference in a daemon thread. If it hangs for 20s (some DAT2 CUDA ops deadlock ORT), it falls back to CPU EP.

---

## Optimization Recommendations vs Current State

| Recommendation | Python Engine | Native Engine | Impact |
|---|---|---|---|
| ~~**1. Tiled inference**~~ | ✅ Already implemented | ✅ **Done** (`pipeline.rs` + CUDA kernels) | ~~10-50× speedup~~ |
| ~~**2. CUDA EP + FP16 model**~~ | ✅ Already implemented | ✅ **Done** (TF32 + FP16 auto-detect + conversion script) | ~~~2× speedup~~ |
| ~~**3. Simpler session.run()**~~ | ✅ Uses `session.run()` (ONNX path) | ✅ **Done** (`run_session_cpu_roundtrip()` for CUDA EP) | ~~**Fixes crash** after N frames~~ |
| **4. OOM recovery** | ✅ Catches + falls back | ❌ Process crash | **Fixes exit code 1** |
| **5. Deadlock-safe ONNX probe** | ✅ 20s timeout thread | ❌ No timeout | **Prevents hangs** on some transformers |

---

## Summary

The Python engine is **production-hardened for transformer models** with tiling, FP16, deadlock probes, and OOM recovery. The native engine now has **tiled inference**, **FP16-optimized CUDA EP** (TF32 + FP16 model auto-detection + conversion script), a zero-copy GPU pipeline with TRT EP acceleration, and **host round-trip inference for CUDA EP** (fixes transformer model crashes). Still lacks OOM recovery and deadlock detection.

Remaining work to reach parity (in priority order):

1. ~~**Offline ONNX FP16 model conversion**~~ — **Done** (`scripts/convert_fp16.py` + auto-detection)
2. ~~**OOM-safe `session.run()` fallback**~~ — **Done** (`run_session_cpu_roundtrip()` dispatched for CUDA EP sessions)
3. **Deadlock-safe ONNX probe** (20s timeout, like Python)
