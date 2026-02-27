# HANDOFF — RAVE Native Engine

> Session: 2026-02-26 · Baseline: **52.2 FPS** 2x upscale (SPAN soft, 528×720 → 1056×1440, RTX 3080 Ti, FP16 mixed precision)

---

## Fixes Applied This Session

### 1. FFmpeg 7.1 ABI Mismatch

| File | Function |
| --- | --- |
| `src-tauri/src/rave_cli.rs` | `ensure_ffmpeg7_preflight` |
| `src-tauri/src/commands/rave.rs` | `rave_environment` |

**Problem:** Preflight check hardcoded `avcodec-61.dll` (FFmpeg 7.0). User had FFmpeg 7.1 (`avcodec-62.dll`).  
**Fix:** Accept both `avcodec-61` and `avcodec-62` DLLs. Updated directory from `third_party/ffmpeg7` → `third_party/ffmpeg`.

### 2. TensorRT DLL Shadowing

| File | Function |
| --- | --- |
| `src-tauri/src/rave_cli.rs` | `apply_ffmpeg_env`, `discovered_windows_provider_dirs` |
| `src-tauri/src/commands/rave.rs` | `runtime_search_dirs` |

**Problem:** DaVinci Resolve and Topaz Gigapixel ship lean `nvinfer_10.dll` runtimes without builder resource DLLs (`nvinfer_builder_resource_sm*.dll`). These shadowed the full TensorRT SDK on the system PATH, causing `Unable to load library: nvinfer_builder_resource_10.dll` errors.  
**Fix:** Filter DaVinci Resolve, Topaz Labs, and Topaz Gigapixel from the RAVE process PATH. Added `third_party/tensorrt/` as the first search directory for self-contained dev setup.

### 3. Self-Contained TensorRT in `third_party/tensorrt/`

**Problem:** Pre-existing `nvinfer_10.dll` in `third_party/tensorrt/` (428 MB) was a different build from the SDK (393 MB) and didn't support SM-specific builder resource loading.  
**Fix:** Replaced all DLLs with copies from `C:\tools\TensorRT-10.15.1.29\bin\` + cuDNN from its install path. The directory now contains 15 DLLs including all SM builder resources.

### 4. FP16 Input Type Mismatch

| File | Function |
| --- | --- |
| `third_party/rave-new/crates/rave-pipeline/src/pipeline.rs` | `preprocess_stage`, `inference_stage` |

**Problem:** Preprocessing converted NV12 → `RgbPlanarF16` when `precision=Fp16`, but the ONNX model (`2x_SPAN_soft.onnx`) defines inputs as `float` (float32). TRT's `with_fp16(true)` handles precision conversion *internally* — feeding F16 data caused ORT to reject the input tensor.  
**Fix:** Preprocess stage now always outputs `RgbPlanarF32` regardless of precision policy. TRT handles FP16 conversion in the compiled engine.

---

## Remaining Known Issues

### CUDA Device Enumeration Failure

```
cu_init_rc=0  cu_device_get_count_rc=3  cu_device_count=-1
libcuda=None  libnvidia_encode=None
```

`cuDeviceGetCount` returns `CUDA_ERROR_NOT_INITIALIZED` despite `cuInit` succeeding. This prevents NVDEC/NVENC from activating and forces software decode/encode fallback. See previous session: *NVENC Initialization Debugging* (conversation `1a271868`).

### NVENC Preset Negotiation

```
NVENC encode error: nvEncInitializeEncoder: all preset/tuning/version retries failed
```

Even when CUDA partially works, NVENC can't negotiate a valid preset. Falls back to libx265 software encode.

---

## Future Optimizations

### 🔴 High Impact

#### 1. Fix CUDA Device Enumeration

**Estimated gain:** 2–3x throughput (100+ FPS)  
**Effort:** Medium  
**Files:** NVENC init path in `rave-nvcodec/src/nvenc.rs`, CUDA context in `rave-core/src/context.rs`

The biggest single win. Unlocks:

- **NVDEC** — hardware H264/HEVC decode directly to GPU surfaces (zero CPU)
- **NVENC** — hardware encode from GPU surfaces, ~5x faster than libx265

All NVDEC/NVENC code is already implemented — it just needs CUDA device visibility to activate.

#### 2. Batch Inference (`max_batch > 1`)

**Estimated gain:** 20–40% throughput  
**Effort:** Medium  
**Files:** `rave-pipeline/src/stage_graph.rs` (`BatchConfig`), `rave-tensorrt/src/tensorrt.rs` (`run_io_bound`), `rave-pipeline/src/pipeline.rs` (`inference_stage`)

`BatchConfig` exists but `max_batch > 1` is rejected by validation. Implementing 2–4 frame batching would amortize TRT kernel launch overhead and better saturate SM cores. The ring buffer design already supports it.

### 🟡 Medium Impact

#### 3. CUDA Graph Capture

**Estimated gain:** ~5–10%  
**Effort:** Medium  

Wrap preprocess → inference → postprocess into a single CUDA Graph. Eliminates per-frame kernel launch overhead (~5–10μs per launch × dozens of kernels). Compounds with batching.

#### 4. Tiled Inference for Large Frames

**Estimated gain:** Enables 4K+ source without OOM  
**Effort:** High  

For high-res inputs (4K → 8K), VRAM becomes the bottleneck. Tiled inference with overlap-blending would handle arbitrarily large frames at a modest throughput cost.

#### 5. Dynamic TRT Engine Profiles

**Estimated gain:** Eliminates engine rebuild on resolution change  
**Effort:** Low–Medium  

Use TRT optimization profiles with min/opt/max dimensions for engine reuse across resolutions.

### 🟢 Lower Impact (Polish)

#### 6. Pinned Staging Buffers for Software Decode

**Estimated gain:** ~5% on software path  
**Effort:** Low  

Use `cudaHostAlloc` (pinned memory) for FFmpeg decode staging. Accelerates CPU → GPU transfer on the software decode path.

#### 7. Double-Buffered Ring Slots

**Estimated gain:** Prevents stalls at higher FPS  
**Effort:** Low  

Explicit double-buffering per stage. Current metrics show `contention=0`, but becomes relevant at higher throughput.

#### 8. INT8 Quantization

**Estimated gain:** ~1.5–2x over FP16  
**Effort:** Medium (requires calibration dataset)  
**Files:** `PrecisionPolicy::Int8` path already exists in `tensorrt.rs`

INT8 with calibration table would give another large throughput jump, but requires quality validation per model.

---

## Architecture Quick Reference

```
Decode (NVDEC/FFmpeg)
  │ NV12 GPU texture
  ▼
Preprocess (CUDA kernels)
  │ RgbPlanarF32 GPU tensor
  ▼
Inference (TensorRT via ORT IO Binding)
  │ RgbPlanarF32 upscaled GPU tensor
  ▼
Postprocess (CUDA kernels)
  │ NV12 GPU texture
  ▼
Encode (NVENC/libx265)
  │ H264/HEVC bitstream
  ▼
Mux (FFmpeg)
```

**Key design invariants:**

- Zero host copies in the hot path (enforced by `host_copy_audit`)
- All stage-to-stage transfers are GPU device pointers via `mpsc` channels
- Ring buffer (`OutputRing`) pre-allocates VRAM; no per-frame `cudaMalloc`
- CUDA streams provide inter-stage overlap (decode ‖ preprocess ‖ inference ‖ encode)

---

🔴 High Impact

1. Fix CUDA device enumeration (cu_device_get_count_rc=3) This is the biggest single win. Right now you're stuck on software decode/encode via FFmpeg. Fixing this unlocks:

NVDEC — hardware H264/HEVC decode directly to GPU surfaces (zero CPU involvement)
NVENC — hardware encode from GPU surfaces, ~5x faster than libx265
The pipeline already has full NVDEC/NVENC support built in — it's just not activating because CUDA can't see the device. This alone could push you past 100+ FPS.

1. Batch inference (max_batch > 1) The

BatchConfig
 struct exists but validates max_batch > 1 as an error. Implementing 2–4 frame batching would let TRT amortize kernel launch overhead and better saturate the SM cores. The architecture already supports it via the ring buffer design. Previous conversation estimated 20–40% throughput gain.

🟡 Medium Impact
3. CUDA Graph capture Wrap the preprocess→inference→postprocess kernels into a single CUDA Graph. Eliminates per-frame kernel launch overhead (~5–10μs per launch × dozens of kernels). On a 3080 Ti at 52 FPS, you're spending ~19ms per frame — shaving 100–200μs of launch overhead is ~1% gain, but it compounds with batching.

1. Tiled inference for large frames For high-res inputs (4K source → 8K output), VRAM can become a bottleneck. Tiled inference with overlap-blending would let you upscale arbitrarily large frames without OOM, at a modest throughput cost.

2. Dynamic TRT engine profiles Right now the TRT engine is built for one fixed input size. Using TRT's optimization profiles with min/opt/max dimensions would allow engine reuse across different resolutions without rebuilding.

🟢 Lower Impact (Polish)
6. Pinned staging buffers for software decode path When using FFmpeg software decode, frames likely go through pageable host memory. Using cudaHostAlloc (pinned) for the staging buffers would accelerate the CPU→GPU transfer on the software path.

1. Double-buffered ring slots The ring already recycles, but explicit double-buffering per stage would guarantee zero stalls on ring contention (the contention=0 metric suggests this isn't a bottleneck yet, but would matter at higher FPS).

2. INT8 quantization The PrecisionPolicy::Int8 path already exists with calibration table support. INT8 would give another ~1.5–2x throughput over FP16, but requires a calibration dataset and may have visible quality impact depending on the model.

Bottom line: Fix the CUDA device issue first — that's the biggest free win since all the NVDEC/NVENC code is already written. Then batch inference for the next big jump.
