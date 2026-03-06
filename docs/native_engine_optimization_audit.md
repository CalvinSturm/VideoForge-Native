# Native Engine Optimization Audit

Date: 2026-03-06

## Scope

This audit re-checks the current native engine codebase rather than relying on prior notes. It covers:

- `engine-v2/src/engine/pipeline.rs`
- `engine-v2/src/backends/tensorrt.rs`
- `engine-v2/src/codecs/nvdec.rs`
- `engine-v2/src/codecs/nvenc.rs`
- `engine-v2/src/core/context.rs`
- `engine-v2/src/core/kernels.rs`
- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/bin/smoke.rs`
- `tools/bench/run_bench.ps1`
- `tools/ci/check_native_smoke_perf.ps1`

The goal is to identify the next real optimization and hardening opportunities that still match the current code.

## Current Runtime Shape

The codebase has two distinct native-engine integration shapes today.

### 1. Active app-facing path

`upscale_request_native()` in `src-tauri` is the active command path. When the `native_engine` feature is enabled and `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1` is set, it currently delegates to `run_native_via_rave_cli()`, which forwards to `rave_upscale()`.

Implication:

- This is the path exercised by the current Tauri command surface.
- Native runtime behavior seen by users is not driven by `run_native_pipeline()`.

### 2. Dormant in-process path

`run_native_pipeline()` exists in `src-tauri/src/commands/native_engine.rs` and wires `engine-v2` directly into Tauri with `FileBitstreamSource`, `NvDecoder`, `TensorRtBackend`, `UpscalePipeline`, and `NvEncoder`.

Implication:

- This code is useful as a reference and may become the long-term runtime path.
- Today it is not the active command path, so optimizations inside it are not automatically user-visible.

### 3. Engine core

The `engine-v2` code itself is real and materially implemented:

- micro-batch accumulation exists in the pipeline
- FP16 preprocess is fused
- decode handoff events exist
- output-ring reuse exists
- audit/stress helpers exist

The main remaining gaps are inside execution details, not broad missing architecture.

## Executive Summary

The previous docs overstated some missing pieces and under-described others.

What is true today:

1. True TensorRT batch execution is still missing end-to-end.
2. FP16 postprocess is still unfused.
3. ORT device-tensor wrapper work still happens per frame.
4. Decode/preprocess handoff events already exist, but overlap is still partially neutralized by CPU-side synchronization and broader stream-ordering remains fragile.
5. TensorRT cache support exists but is opt-in.
6. Window-size padding for windowed transformer models still appears absent.
7. The Tauri-side full-file bitstream buffering issue exists, but only in the dormant in-process native path.
8. The benchmark story is split: the checked-in bench runner does not drive the native engine path, while the smoke/perf gate does.

## Verified Findings

### 1. True TensorRT batching is still not implemented

Priority: Critical

Why it matters:

- `PipelineConfig` and `BatchConfig` support micro-batching.
- `inference_stage()` accumulates batches and calls `backend.process_batch(...)`.
- The backend still warns that `max_batch > 1` remains single-frame.
- `TensorRtBackend::process_batch()` just loops over `process()`.

Evidence:

- `engine-v2/src/engine/pipeline.rs`
- `engine-v2/src/backends/tensorrt.rs`
- `engine-v2/src/core/backend.rs`

What this means:

- Queueing latency is paid now.
- True `N > 1` TensorRT execution benefit is not.

Implementation direction:

- Add batched shapes and bindings in `TensorRtBackend`.
- Keep batch size `1` behavior unchanged.
- Preserve frame ordering and output count checks already present in the pipeline.

### 2. FP16 postprocess is still paying the F16 -> F32 -> NV12 tax

Priority: High

Why it matters:

- FP16 preprocess already has a fused `NV12 -> RgbPlanarF16` path.
- Postprocess still promotes `RgbPlanarF16` to `RgbPlanarF32`.
- Only then does it run the `RgbPlanarF32 -> NV12` kernel.

Evidence:

- `engine-v2/src/core/kernels.rs`
- `engine-v2/src/engine/pipeline.rs`

What this means:

- One extra allocation
- One extra full-frame memory pass
- One extra kernel launch on the hot path

Implementation direction:

- Add a fused `RgbPlanarF16 -> NV12` kernel.
- Route FP16 output through it directly.

### 3. ORT wrapper overhead is still present in the inference hot path

Priority: High

Why it matters:

- `cached_mem_info` exists on `TensorRtBackend`.
- `mem_info()` builds and caches a `MemoryInfo`.
- `run_io_bound()` ignores that cached object and still calls `create_tensor_from_device_memory()` for both input and output every frame.
- `create_tensor_from_device_memory()` creates and releases raw `OrtMemoryInfo` on each call.

Evidence:

- `engine-v2/src/backends/tensorrt.rs`

What this means:

- Avoidable CPU-side overhead remains in the highest-frequency backend path.

Implementation direction:

- Rework device tensor creation so the cached memory metadata is actually reused.
- Evaluate whether binding objects or equivalent wrapper state can be safely reused per shape.

### 4. Decode/preprocess overlap is only partially realized

Priority: High

Why it matters:

- The old assumption that events were missing is no longer correct.
- `DecodedFrameEnvelope` carries `DecodeReadyEvent`.
- Preprocess explicitly waits on that event with `cuStreamWaitEvent`.
- NVDEC uses `cuMemcpy2DAsync_v2`, not synchronous `cuMemcpy2D_v2`.
- However, `NvDecoder::map_and_copy()` still calls `cuEventSynchronize(event)` before `cuvidUnmapVideoFrame64`.

Evidence:

- `engine-v2/src/codecs/nvdec.rs`
- `engine-v2/src/engine/pipeline.rs`

What this means:

- The decode thread still blocks on copy completion before handing the frame downstream.
- The cross-stream mechanism exists, but the producer-side CPU wait reduces the practical overlap win.

Implementation direction:

- Rework NVDEC surface-lifetime handling so `cuvidUnmapVideoFrame64` does not require a blocking wait on the decode thread.
- Keep the existing event-based handoff, but remove the forced CPU-side synchronization if surface correctness can be preserved safely.

### 5. Generic stream synchronization is still a no-op helper

Priority: High for correctness review, Medium for optimization ranking

Why it matters:

- `GpuContext::sync_stream()` currently returns `Ok(())` without synchronizing anything.
- The preprocess stage calls it after `prepare()`.
- The inference stage calls it after postprocess and before sending NV12 frames to encode.
- If correctness depends on those calls, the current ordering is weaker than the surrounding code and comments imply.

Evidence:

- `engine-v2/src/core/context.rs`
- `engine-v2/src/engine/pipeline.rs`

What this means:

- The code already has one explicit event-based handoff for decode -> preprocess.
- The later preprocess -> inference and postprocess -> encode boundaries deserve a fresh correctness pass before they are treated as fully hardened.

Implementation direction:

- Audit every cross-stream or cross-subsystem boundary.
- Replace fake synchronization with explicit CUDA event ordering where needed.
- Do not rely on `sync_stream()` comments as if they were real synchronization.

### 6. TensorRT engine caching exists but is opt-in

Priority: Medium

Why it matters:

- The backend supports engine cache directories.
- Cache enablement depends on `VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE`.
- The default remains disabled.

Evidence:

- `engine-v2/src/backends/tensorrt.rs`

What this means:

- Warm-start wins are available now but not standard.

Implementation direction:

- Decide whether the production runtime should enable cache by default.
- At minimum, standardize cache path selection and verification.

### 7. Window-size padding still appears missing for windowed transformer models

Priority: High for correctness, not throughput

Why it matters:

- The current Rust engine code does not show a transformer adapter or padding/crop-back layer.
- Archive docs explicitly call out the absence of the old adapter system.
- The current implementation plan should still treat this as a real compatibility gap.

Evidence:

- `engine-v2`
- `docs/archive/engine_audit.md`
- `docs/archive/Native Engine Transformer Support plan.md`

What this means:

- HAT, SwinIR, DAT, and similar models may still need explicit padding policy before parity claims are considered complete.

Implementation direction:

- Add explicit window-alignment policy plus crop-back.
- Keep the rule model-aware rather than applying padding blindly to all models.

### 8. Full-file buffering and chunk cloning exist, but only in the dormant in-process path

Priority: Low until that path becomes active

Why it matters:

- `FileBitstreamSource` reads the entire elementary stream into memory up front.
- `read_packet()` clones 1 MiB chunks into fresh `Vec<u8>` values.
- This is real overhead in `run_native_pipeline()`.
- It is not on the current app-facing command path.

Evidence:

- `src-tauri/src/commands/native_engine.rs`

What this means:

- This should not be presented as a top user-visible optimization until the in-process path is wired in.

Implementation direction:

- Treat it as dormant-path cleanup or future work unless runtime routing changes.

### 9. Validation tooling is split and the existing bench runner does not benchmark the native engine path

Priority: Medium

Why it matters:

- `tools/bench/run_bench.ps1` runs `videoforge_bench`.
- `videoforge_bench` benchmarks the Python upscale job path.
- `tools/ci/check_native_smoke_perf.ps1` is the checked-in script that actually drives `upscale_request_native()`.
- The engine also has `stress_test_synthetic()` and `AuditSuite`, but those are engine-level helpers rather than the app-facing benchmark workflow.

Evidence:

- `tools/bench/run_bench.ps1`
- `src-tauri/src/bin/videoforge_bench.rs`
- `tools/ci/check_native_smoke_perf.ps1`
- `src-tauri/src/bin/smoke.rs`
- `engine-v2/src/engine/pipeline.rs`

What this means:

- Benchmarks cited for native-engine optimization work must be described precisely.
- Today, the checked-in native gate is smoke/success/median-FPS oriented, not a full per-stage benchmark harness.

Implementation direction:

- Either add a dedicated native benchmark runner or explicitly document the current split.

## Recommended Priority Order

### If the goal is engine-v2 throughput on the real native core

1. Fix stream-ordering assumptions and remove decode-side CPU synchronization where safe
2. Implement true TensorRT batch execution
3. Fuse FP16 postprocess to NV12
4. Remove ORT per-frame wrapper churn
5. Enable or standardize TensorRT caching

### If the goal is production runtime clarity

1. Decide whether `upscale_request_native()` should remain CLI-backed or switch to `run_native_pipeline()`
2. Only optimize dormant Tauri-native buffering after that decision

### If the goal is transformer compatibility

1. Add window-size padding and crop-back
2. Re-evaluate any remaining transfer or staging work after correctness is in place

## Benchmark Guidance

Use the following distinctions explicitly.

### Native command validation

- `tools/ci/check_native_smoke_perf.ps1`
- `src-tauri/src/bin/smoke.rs --e2e-native`

This validates the current app-facing native command path.

### Engine-only audit helpers

- `stress_test_synthetic()` in `engine-v2/src/engine/pipeline.rs`
- `AuditSuite` in `engine-v2/src/engine/pipeline.rs`

These are useful for engine mechanics, pool behavior, and invariant checks.

### Current limitation

`tools/bench/run_bench.ps1` is not a native-engine benchmark runner today. It should not be presented as one.

## Conclusion

The native engine is materially implemented, but the stale docs confused three different questions:

1. what the engine core already does
2. what the active Tauri runtime actually uses
3. which performance gaps are still real

The code-grounded answer is:

- batching, FP16 postprocess, ORT wrapper churn, TensorRT cache defaults, and transformer window padding remain valid targets
- decode/preprocess overlap is still worth improving, but for a different reason than previously documented
- dormant in-process Tauri buffering work should not be prioritized as if it were active production runtime work
- stream-ordering correctness deserves explicit review before more aggressive optimization work lands
