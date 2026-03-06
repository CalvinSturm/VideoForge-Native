# Native Engine Workspace Audit

Date: 2026-03-06

## Status Update

As of the current workspace state:

- explicit direct-path smoke forcing is implemented via `smoke.rs --native-direct`
- checked-in native smoke/perf now targets `run_native_pipeline()`
- TensorRT `process_batch()` now performs one real batched execution for homogeneous inputs
- FP16 postprocess now has a direct `RgbPlanarF16 -> NV12` path
- cached ORT `MemoryInfo` is now reused by the device-tensor wrapper path
- temporary per-frame pipeline and NVDEC success-path logs were downgraded from `info!` to `debug!`
- NVDEC async copy overlap has been reintroduced only as an opt-in experiment behind `VIDEOFORGE_NVDEC_ASYNC_COPY=1`
- native runtime environment discovery is now cached once per process
- `FileBitstreamSource` now streams the elementary stream from disk instead of preloading the full file
- direct-path smoke is passing again in the current workspace
- direct-path encode now tries NVENC first and falls back to software bitstream encode if this machine rejects NVENC external CUDA input mapping
- direct mode remains opt-in; the default native path is still the CLI-backed route

## Purpose

This audit is a fresh code-grounded snapshot of the native-engine work in the current workspace. It is intended to replace stale planning assumptions and to capture the currently visible optimization surface across:

- runtime routing
- direct in-process engine-v2 integration
- engine-v2 hot-path behavior
- validation and benchmark tooling
- plausible next optimizations

This document is based on the checked-out code, not prior notes.

## Scope Reviewed

- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/commands/rave.rs`
- `src-tauri/src/bin/smoke.rs`
- `src-tauri/src/bin/videoforge_bench.rs`
- `src-tauri/Cargo.toml`
- `engine-v2/src/engine/pipeline.rs`
- `engine-v2/src/backends/tensorrt.rs`
- `engine-v2/src/codecs/nvdec.rs`
- `engine-v2/src/codecs/nvenc.rs`
- `engine-v2/src/core/context.rs`
- `engine-v2/src/core/kernels.rs`
- `engine-v2/src/core/backend.rs`
- `engine-v2/Cargo.toml`
- `tools/bench/run_bench.ps1`
- `tools/ci/check_native_smoke_perf.ps1`

## Current Runtime Shape

### 1. Compiled feature status

The repo is no longer blocked on the earlier ORT semver issue.

- `src-tauri/Cargo.toml` enables `native_engine = ["dep:videoforge-engine"]`
- `engine-v2/Cargo.toml` pins `ort = "=2.0.0-rc.11"`

Implication:

- The native engine feature is a real optional feature in the workspace now.
- Some stale source comments still describe older blocked state and should not be treated as source of truth.

### 2. App-facing command path

`upscale_request_native()` is the user-facing native command entry point.

Runtime gates:

- `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1` is required before the command will run at all.
- `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` is required to route into the true in-process engine-v2 path.

Routing behavior:

- default native command behavior: `run_native_via_rave_cli()`
- opt-in direct native behavior: `run_native_pipeline()`

Implication:

- The direct path is not dormant code in the sense of being unreachable.
- The direct path is also not the default shipped runtime path yet.

### 3. Two native shapes exist today

#### A. Default native command shape

`upscale_request_native()` -> `run_native_via_rave_cli()` -> `rave_upscale()`

This preserves the existing app contract and routes through the CLI-backed path unless direct mode is explicitly enabled.

#### B. Direct in-process native shape

`upscale_request_native()` -> `run_native_pipeline()` -> `engine-v2`

This path:

- demuxes with FFmpeg into Annex B elementary stream temp files
- decodes with NVDEC
- preprocesses on GPU
- runs inference with ORT + TensorRT EP
- postprocesses to NV12
- encodes with NVENC when the runtime accepts external CUDA input registration, otherwise falls back to FFmpeg software bitstream encode
- muxes final output with FFmpeg

Implication:

- Direct-path engine work is real and currently testable.
- Performance work inside `run_native_pipeline()` is only user-visible when direct mode is enabled or made default.

## What Is Completed

### 1. Direct-path stabilization work landed

The last significant commit hardened the direct path rather than adding a new feature set.

Completed items visible in code:

- `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` routing switch exists.
- `StreamReadyEvent` exists as an owned event wrapper.
- decode -> preprocess handoff is explicit.
- preprocess -> inference handoff is explicit.
- postprocess -> encode handoff is explicit.
- `GpuContext::sync_stream()` now uses `cuStreamSynchronize`.
- TensorRT metadata scale inference for dynamic spatial axes no longer silently defaults to `2`.
- FFmpeg mux failure handling preserves temp outputs and surfaces stderr.

### 2. The engine core is materially implemented

Already present in current code:

- bounded multi-stage pipeline
- output-ring reuse in TensorRT backend
- FP16 fused preprocess path
- engine-level stress/audit helpers
- TensorRT engine cache support
- direct CUDA event wrappers for stage boundaries

This is not a skeleton codebase. The remaining work is in execution quality, runtime strategy, compatibility, and measurement.

## What Is Partially Implemented

### 1. Batch admission and backend execution now both exist

The pipeline can accumulate micro-batches and the TensorRT backend now has a real batched execution path for homogeneous inputs.

Current caveats:

- `UpscaleBackend::process_batch()` default behavior is still sequential for backends that do not override it.
- the TensorRT batched path currently expects same width, height, and format across the batch and falls back to sequential execution otherwise.
- the batched path has been compile-checked but not yet profiled in this pass.

Implication:

- the repo now has both batch plumbing and a concrete TensorRT batch path
- the next validation gap is runtime measurement, not missing implementation

### 2. Caching support exists, standardization does not

TensorRT engine cache support is implemented behind environment variables.

Current state:

- cache disabled by default
- optional cache root is configurable
- warm-start benefits are available if enabled explicitly

Implication:

- startup optimization infrastructure exists
- product/runtime policy is not settled

### 3. Direct-path smoke now has an explicit forcing mode

`smoke.rs --e2e-native`:

- sets `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1`
- only sets `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` when `--native-direct` is passed

`tools/ci/check_native_smoke_perf.ps1` now calls `smoke.exe --e2e-native --native-direct`, so the checked-in perf gate explicitly targets `run_native_pipeline()`.

Implication:

- the direct path is testable with an explicit checked-in flag
- the default native command path and direct path are now distinguishable in validation

## Hot-Path Audit

### 1. Decode path

Current state in `nvdec.rs`:

- `get_raw_stream()` now uses `stream.stream.cast()`
- per-frame decode readiness is wrapped in `StreamReadyEvent`
- cross-stream wait helper exists
- default copy path is synchronous `cuMemcpy2D_v2`
- an opt-in async path now exists behind `VIDEOFORGE_NVDEC_ASYNC_COPY=1`
- in async mode, D2D copies run on `decode_stream`, `decode_done` is recorded on that stream, and surface unmap is deferred through `pending_unmaps`
- in default mode, surface is still unmapped immediately after synchronous copy

What changed from older assumptions:

- async overlap is no longer the default path
- async overlap can now be tested explicitly without changing the stabilized default

Implication:

- correctness is still favored by default
- overlap work is now available as an isolated experiment rather than an all-or-nothing rewrite

### 2. Preprocess path

Current state in `pipeline.rs` and `kernels.rs`:

- preprocess waits on decode-ready event on the preprocess stream
- NV12 -> RGB F32 exists
- fused NV12 -> RGB F16 exists
- preprocess records `preprocess_ready` event after kernel submission

Strengths:

- explicit decode dependency
- fused FP16 input preparation already avoids one intermediate conversion

Remaining opportunity:

- decode overlap still determines how much preprocess can overlap with NVDEC
- the new async decode-copy mode needs runtime validation before it should become default

### 3. Inference path

Current state in `tensorrt.rs`:

- ORT IO binding is used
- `run_binding()` is synchronous
- output ring reuses preallocated device buffers
- precision policy exists
- cached ORT `MemoryInfo` is now reused by the device-tensor wrapper path
- batched IO binding now exists for homogeneous batches
- `bind_to_thread()` rebinding was added after async lock points

Strengths:

- GPU-resident path is structurally real
- ring reuse is present
- provider policy prefers TensorRT and allows CUDA fallback

Remaining opportunity:

- no true batched execution
- per-call ORT device-tensor wrapper work still happens
- current provider validation is structural/log-based, not introspective

### 4. Postprocess path

Current state in `kernels.rs`:

- model output may be `RgbPlanarF32` or `RgbPlanarF16`
- only `RgbPlanarF32 -> NV12` conversion is fused
- FP16 output currently routes through `convert_f16_to_f32()` first
- pipeline then records one `postprocess_ready` event and only the first frame of the batch carries it downstream

Implication:

- FP16 output path still pays an extra allocation, conversion pass, and kernel launch

### 5. Encode path

Current state in `nvenc.rs`:

- prefers direct `nvEncRegisterResource(CUDADEVICEPTR)` path
- caches registered resources by device pointer
- falls back to a legacy CUDA staging surface if direct registration is rejected
- encode waits on `postprocess_ready` via CPU-side `event.synchronize()` before `encoder.encode()`
- the command-layer encoder wrapper now falls back to software bitstream encode if the first hardware encode attempt fails at runtime

Implication:

- encode boundary is correctness-first
- direct mode is operational again, but pure end-to-end hardware encode is still runtime-dependent on NVENC compatibility

### 6. Buffering and pooling

Current state in `context.rs`:

- bucketed pooled allocations exist
- VRAM accounting exists
- `alloc_policy` can flag steady-state pool misses
- `alloc_aligned()` and `prefetch_l2()` exist

Strengths:

- memory reuse infrastructure is already in place

Remaining opportunity:

- audit whether hot-path callers actually exploit `alloc_aligned()` or `prefetch_l2()`
- verify pool reaches expected steady state on direct-path workloads

## Direct-Path Integration Audit

### 1. Startup and environment setup

Current state in `native_engine.rs`:

- runtime scans for FFmpeg and TensorRT binaries
- Windows DLL preload exists
- TensorRT DLLs may be copied next to the executable if absent

Opportunities:

- repeated recursive filesystem scans were a real issue and are now cached once per process
- repeated DLL staging could be avoided with clearer deployment/runtime policy
- startup cost should be measured separately from steady-state throughput

### 2. Demux and mux

Current state:

- demux writes elementary stream temp files with FFmpeg
- mux captures stderr and preserves temp files on failure

Strengths:

- mux debugging is materially better than before

Opportunities:

- demux stderr is still discarded
- direct path still pays file-based demux/mux boundaries rather than streaming integration

### 3. Bitstream source

Current state in `FileBitstreamSource`:

- the source opens the elementary stream as a file
- packets are now served by streaming 1 MiB reads directly from disk
- the full file is no longer preloaded into memory up front

Implication:

- the direct path no longer pays the obvious full-file buffering cost
- any further file-boundary work is now more about architectural simplification than the worst startup waste

## Validation and Benchmark Audit

### 1. Native command validation

Current tools:

- `src-tauri/src/bin/smoke.rs`
- `tools/ci/check_native_smoke_perf.ps1`

These validate:

- command wiring
- end-to-end success
- output shape and duration
- rough FPS thresholding
- direct-path success explicitly, via `--native-direct`

Current limitation:

- native validation is still smoke/success/median-FPS oriented rather than a full per-stage benchmark harness
- the current passing direct smoke path may still be using software bitstream fallback on machines where NVENC external-input mapping is rejected

### 2. Bench runner

Current tool:

- `tools/bench/run_bench.ps1` -> `videoforge_bench`

This benchmarks:

- the Python upscale job path

It does not benchmark:

- the direct engine-v2 path

### 3. Engine-only audit helpers

Current helpers:

- `stress_test_synthetic()`
- `AuditSuite`

These are useful for:

- engine mechanics
- queue and pool behavior
- invariants

They are not a substitute for:

- direct-path runtime benchmarking through `upscale_request_native()`

## Optimization Catalogue

This section captures the currently visible optimization surface in the workspace. Items are grouped by whether they affect the direct path only, engine core generally, or runtime policy.

### Tier 1: High-value, code-supported next optimizations

#### 1. Implement true TensorRT batched execution

Why it matters:

- pipeline admission already accumulates batches
- current backend still serializes them

Expected benefit:

- meaningful throughput improvement on compatible models and hardware

Risk:

- medium to high
- requires careful output ordering, shape handling, and provider compatibility

Applies to:

- engine core
- direct path
- any future default native runtime

#### 2. Fuse FP16 postprocess to NV12

Why it matters:

- current FP16 output path still promotes to F32 before NV12 conversion

Expected benefit:

- lower GPU memory traffic
- one fewer allocation/conversion step
- fewer kernel launches

Risk:

- medium
- must preserve visual correctness and color conversion behavior

Applies to:

- engine core

#### 3. Reuse ORT device-tensor metadata and reduce wrapper churn

Why it matters:

- `cached_mem_info` exists but raw `CreateMemoryInfo` work still happens per tensor wrapper creation
- binding objects are recreated per call

Expected benefit:

- lower CPU overhead in the inference hot path
- lower dispatch overhead at higher FPS

Risk:

- medium
- must respect ORT object lifetime and shape changes

Applies to:

- engine core

#### 4. Expand the direct-path perf harness beyond smoke-level checks

Why it matters:

- the checked-in perf gate now targets the direct path explicitly, but it still only provides end-to-end smoke/perf coverage
- it does not distinguish pure NVENC encode from the new software-encode fallback path

Expected benefit:

- more trustworthy per-change optimization validation

Risk:

- low

Applies to:

- tooling
- planning correctness

### Tier 2: Likely worthwhile after Tier 1

#### 5. Revisit async NVDEC copy and overlap

Why it matters:

- current decode path is synchronous by design for stability

Expected benefit:

- improved decode/consumer overlap
- lower decode-thread blocking

Risk:

- high
- earlier direct-path crash work suggests this area is fragile

Applies to:

- engine core
- direct path

#### 6. Reduce or gate temporary boundary logging

Why it matters:

- `PIPELINE-BND` and NVDEC `info!` logging remain heavy in hot paths

Expected benefit:

- lower CPU and logging overhead
- cleaner traces for real regressions

Risk:

- low

Applies to:

- engine core
- direct path

#### 7. Standardize TensorRT cache defaults

Why it matters:

- cache support exists but default behavior is undecided

Expected benefit:

- better warm-start performance
- more predictable deployment behavior

Risk:

- low to medium
- requires cache invalidation and path policy clarity

Applies to:

- startup performance
- runtime policy

### Tier 3: Important but path-dependent

#### 8. Add earlier NVENC capability detection and encode-policy selection

Why it matters:

- direct mode now recovers by falling back to software encode after first-frame NVENC failure
- that keeps the pipeline working, but it is reactive rather than policy-driven

Expected benefit:

- clearer runtime behavior
- less wasted setup work on machines that cannot sustain the direct NVENC path
- better perf measurement separation between hardware and fallback encode modes

Risk:

- medium

Priority note:

- rises if the direct path remains strategic or becomes the default native runtime

#### 9. Stream demux/encode integration to reduce temp-file boundaries

Why it matters:

- direct path still writes/reads temporary elementary streams and encoded outputs

Expected benefit:

- lower disk IO
- lower startup/teardown overhead
- less failure surface around temp files

Risk:

- high
- broad integration work, not a narrow hot-path optimization

#### 10. Cache runtime environment discovery

Why it matters:

- `configure_native_runtime_env()` scans the repo and may copy DLLs at runtime

Expected benefit:

- lower startup cost
- lower filesystem churn

Risk:

- low to medium

### Tier 4: Correctness or compatibility work that can unlock later optimization

#### 11. Add model-aware window padding and crop-back for transformer-style models

Why it matters:

- no current implementation was found for window-alignment policy in the engine

Expected benefit:

- compatibility and correctness for HAT/SwinIR-like models on non-aligned sizes

Risk:

- medium

#### 12. Audit stale comments and contradictory docs in code

Why it matters:

- some source comments still describe blocked or older behavior
- this already led to stale planning documents

Expected benefit:

- lower chance of the next pass optimizing the wrong path for the wrong reason

Risk:

- low

## Lower-Confidence Opportunities

These are plausible but should be measurement-led before implementation:

- exploit `prefetch_l2()` in repeated tensor-access hotspots
- tune output-ring size, downstream capacity, and batch wait thresholds per target model
- tune NVENC bitrate/GOP/preset defaults for throughput vs output quality tradeoff
- reduce repeated `bind_to_thread()` cost if a safer context ownership model is possible
- reduce direct-path FFmpeg process startup overhead if startup latency matters
- audit whether optional CUDA fallback under ORT should remain enabled for production behavior

## Recommended Priority Order

### If the goal is real engine throughput on the direct path

1. Measure the now-current direct path, including whether each run stays on NVENC or falls back to software encode
2. Add earlier NVENC capability detection or explicit encode-policy selection if fallback machines are common
3. Revisit async NVDEC overlap only after the measurement baseline is stable

### If the goal is shipping the direct path as default runtime

1. Decide whether software-encode fallback is an acceptable product behavior for direct mode
2. Measure startup and fallback frequency on representative machines
3. Then decide whether direct mode should remain opt-in or become the primary native path

### If the goal is compatibility breadth

1. Add transformer window padding and crop-back
2. Re-validate metadata and dimension handling on representative model families

## Recommended Immediate Next Step

The best immediate move is measurement and policy clarification, not another broad implementation PR.

1. run repeated direct-path smoke/perf to separate stable NVENC cases from software-fallback cases
2. decide whether fallback-to-software is acceptable runtime behavior or whether direct mode should fail fast when hardware encode is unavailable
3. only after that, choose whether the next follow-on work is NVENC capability detection, more perf instrumentation, or product-path promotion

## Bottom Line

The current workspace is past the "can the direct path run at all?" stage.

It is now in this state:

- direct in-process native execution works behind an opt-in routing flag
- direct smoke currently passes in this workspace
- the direct path was recently stabilized with conservative synchronization choices
- the engine core already contains much of the needed structure
- the main remaining uncertainty is runtime behavior around NVENC compatibility and when the encoder falls back to software
- the next work should be driven by measurement and encode-policy decisions rather than another speculative optimization pass
