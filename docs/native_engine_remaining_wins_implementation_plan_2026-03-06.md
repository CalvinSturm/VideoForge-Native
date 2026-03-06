# Native Engine Remaining Wins Implementation Plan

Date: 2026-03-06

## Status Update

Completed in the current workspace:

- PR 1: explicit direct-path smoke/perf forcing
- PR 2: true batched TensorRT execution for homogeneous batches
- PR 3: fused FP16 postprocess to NV12
- PR 4: cached ORT `MemoryInfo` reuse in the hot path
- PR 5: hot-path logging trim
- PR 6: opt-in NVDEC async copy overlap experiment behind `VIDEOFORGE_NVDEC_ASYNC_COPY=1`
- PR 7: direct-path startup cleanup without changing the opt-in runtime policy

Still open:

- no major planned implementation PRs remain from this sequence
- the remaining work is measurement, runtime-policy decision, and any follow-on fixes from real perf/smoke runs

## Request Summary

This plan originally covered the remaining high-value native-engine wins visible in the workspace. The code has since advanced through PR 7 implementation work. The remaining open items are now validation, runtime-policy decision-making, and any follow-on changes driven by measurement.

## Relevant Codebase Findings

The workspace currently has two native runtime shapes:

- default native command path: `upscale_request_native()` -> `run_native_via_rave_cli()`
- opt-in direct path: `upscale_request_native()` -> `run_native_pipeline()` when `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`

Important constraints from the current code after the current implementation pass:

1. The checked-in perf gate now forces direct mode, so direct-path benchmarking is explicit.
2. The pipeline admits micro-batches and `TensorRtBackend::process_batch()` now has a real homogeneous-batch path.
3. FP16 preprocess and FP16 postprocess are both fused now.
4. ORT IO binding remains in place and cached `MemoryInfo` is now reused by the device-tensor wrapper path.
5. The direct path still defaults to synchronous NVDEC copy, but an opt-in async overlap mode now exists behind `VIDEOFORGE_NVDEC_ASYNC_COPY=1`.
6. Startup/file-boundary overhead in `run_native_pipeline()` has been reduced by runtime-env caching and streaming file reads, but broader direct-path simplification still depends on whether the direct path becomes the default product runtime.

Preserve:

1. existing `upscale_request_native()` command contract
2. direct-path routing behind `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` unless explicitly changed later
3. current output correctness, shape validation, and mux hardening behavior
4. narrow PR boundaries that separate tooling, backend execution, kernels, and riskier decode work

## Proposed Implementation Approach

Proceed in two phases.

### Phase 1: Enable trustworthy optimization work

Status: completed

Before changing engine throughput behavior, make the direct path easy to validate explicitly.

This phase should:

1. add a direct-path smoke/perf path that always sets `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`
2. document which scripts validate the default native command path versus the true in-process path
3. preserve the existing default routing behavior while making direct-path benchmarking explicit

### Phase 2: Land the remaining high-value engine wins in narrow PRs

Status: PR 2 through PR 7 completed

After the direct path has a trustworthy perf harness:

1. implement true batched TensorRT execution in `TensorRtBackend`
2. fuse FP16 postprocess directly to NV12
3. reduce ORT wrapper churn and standardize cache behavior where safe
4. clean up hot-path debug logging once the benchmark baseline is stable
5. only then consider the riskier decode-overlap rework

This order keeps the largest remaining throughput wins ahead of the highest-risk correctness-sensitive change.

## Ordered PR Breakdown

1. PR title: `Add Explicit Direct-Path Native Perf Validation`
   Primary goal: make direct-path validation and perf measurement unambiguous
   Dependency on earlier PRs, if any: none

2. PR title: `Implement True Batched TensorRT Execution`
   Primary goal: replace sequential `process_batch()` behavior with one real batched backend execution
   Dependency on earlier PRs, if any: PR 1 required

3. PR title: `Fuse FP16 Postprocess to NV12`
   Primary goal: remove the extra FP16-to-F32 conversion on the output path
   Dependency on earlier PRs, if any: PR 1 required

4. PR title: `Reduce TensorRT Hot-Path Wrapper Overhead`
   Primary goal: eliminate avoidable ORT wrapper churn and make cache behavior explicit
   Dependency on earlier PRs, if any: PR 1 required

5. PR title: `Trim Native Hot-Path Debug Logging`
   Primary goal: reduce measurement noise and runtime overhead from temporary boundary logs
   Dependency on earlier PRs, if any: PR 2 through PR 4 preferred

6. PR title: `Revisit NVDEC Async Copy and Decode Overlap`
   Primary goal: recover safe overlap only after a stable direct-path performance baseline exists
   Dependency on earlier PRs, if any: PR 1 required, PR 5 preferred

7. PR title: `Direct Path Runtime Policy and Startup Cleanup`
   Primary goal: decide whether the direct path should remain opt-in or become the primary native runtime path, then optimize startup/file boundaries accordingly
   Dependency on earlier PRs, if any: PR 1 required, PR 2 through PR 4 preferred

## Per-PR Scope Details

### Completed PRs

PR 1 through PR 6 are now implemented in the current workspace. The details below remain useful as rationale for how the work was split, but they are no longer future scope.

### PR 1: Add Explicit Direct-Path Native Perf Validation

In scope:

- add a checked-in direct-path smoke/perf script or mode that explicitly sets `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`
- keep current `--e2e-native` behavior understandable in `smoke.rs`
- document which path each validation script exercises
- ensure future throughput PRs have one obvious direct-path baseline

Out of scope:

- batching implementation
- kernel changes
- runtime routing changes

Key files or subsystems likely to change:

- `tools/ci/check_native_smoke_perf.ps1`
- `src-tauri/src/bin/smoke.rs`
- `docs/native_engine_workspace_audit_2026-03-06.md`
- any adjacent native-engine docs kept as working references

Acceptance criteria:

1. the repo has one documented perf path that always exercises `run_native_pipeline()`
2. validation wording no longer conflates default native path and direct native path
3. the current default routing remains unchanged unless explicitly requested otherwise

Test or verification expectations:

1. run the direct-path smoke/perf script against at least one known-good ONNX model
2. confirm `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` is in effect during the run
3. confirm output validation still passes

Notes on why this PR boundary is correct:

- every later throughput PR depends on knowing which path is actually under test
- this is small, low-risk, and unlocks credible measurement

### PR 2: Implement True Batched TensorRT Execution

In scope:

- add batched input and output shape construction in `TensorRtBackend`
- make `process_batch()` perform one real batched execution instead of sequentially calling `process()`
- preserve batch size `1` behavior
- preserve ordering and output count guarantees already enforced in the pipeline
- keep the pipeline-side micro-batch admission model intact unless a backend requirement forces a narrow adjustment

Out of scope:

- decode-path redesign
- FP16 postprocess fusion
- startup/runtime policy changes

Key files or subsystems likely to change:

- `engine-v2/src/backends/tensorrt.rs`
- `engine-v2/src/core/backend.rs`
- `engine-v2/src/engine/pipeline.rs`

Acceptance criteria:

1. `max_batch > 1` no longer routes through a sequential loop
2. the current warning about single-frame backend execution is removed or narrowed appropriately
3. output ordering and count remain correct
4. direct-path perf runs show real improvement at batch `2` or `4` on at least one compatible model

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml --features native_engine`
2. direct-path smoke/perf validation from PR 1
3. repeated before/after runs on a model that actually exercises batch settings
4. targeted engine-level checks if any batch-specific helper tests are added

Notes on why this PR boundary is correct:

- the batch admission logic already exists in the pipeline
- the missing work is concentrated in backend execution and should be reviewable as one backend-focused change

### PR 3: Fuse FP16 Postprocess to NV12

In scope:

- add a fused `RgbPlanarF16 -> NV12` kernel path
- route FP16 model outputs through that path directly
- preserve the current F32 output path unchanged

Out of scope:

- TensorRT batching
- decode-path rework
- color-space policy changes beyond matching existing behavior

Key files or subsystems likely to change:

- `engine-v2/src/core/kernels.rs`
- `engine-v2/src/engine/pipeline.rs`

Acceptance criteria:

1. FP16 output no longer requires `convert_f16_to_f32()` before NV12 conversion
2. visual output remains correct on representative FP16 runs
3. F32 behavior remains unchanged

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml --features native_engine`
2. direct-path smoke/perf validation using FP16 mode
3. targeted before/after run on a representative clip and model

Notes on why this PR boundary is correct:

- this is a narrow kernel-path optimization with clear review focus
- it should remain independent from backend execution redesign

### PR 4: Reduce TensorRT Hot-Path Wrapper Overhead

In scope:

- remove or reduce avoidable per-call ORT memory metadata creation
- evaluate safe reuse of binding-related wrapper state for stable shapes
- keep shape changes correct when dimensions vary
- standardize TensorRT cache defaults or, if product policy is not ready, make cache behavior explicit and testable

Out of scope:

- batch semantics
- decode-path redesign
- dormant-path buffering cleanup

Key files or subsystems likely to change:

- `engine-v2/src/backends/tensorrt.rs`
- possibly runtime config points that own TensorRT cache policy

Acceptance criteria:

1. avoidable `OrtMemoryInfo`/wrapper churn is materially reduced in the steady-state inference path
2. shape changes still behave correctly
3. warm-start cache behavior is documented and verifiable if defaults are changed
4. steady-state throughput does not regress

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml --features native_engine`
2. direct-path smoke/perf validation
3. explicit cold-start and warm-start comparison if cache defaults change

Notes on why this PR boundary is correct:

- this is inference overhead work, not a kernel change or pipeline redesign
- it benefits from the direct-path perf harness but does not depend on batching or FP16 fusion landing first

### PR 5: Trim Native Hot-Path Debug Logging

In scope:

- reduce or gate `PIPELINE-BND` logs in `pipeline.rs`
- reduce or gate temporary `info!` logging in `nvdec.rs`
- preserve enough logging for future crash diagnosis via `debug!`, feature flag, or env-gated tracing policy

Out of scope:

- behavioral changes to synchronization
- backend execution changes

Key files or subsystems likely to change:

- `engine-v2/src/engine/pipeline.rs`
- `engine-v2/src/codecs/nvdec.rs`

Acceptance criteria:

1. direct-path throughput measurements are less affected by tracing volume
2. crash-boundary logging remains recoverable through a clear debug mode
3. no functional behavior changes are mixed into the cleanup

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml --features native_engine`
2. one direct-path smoke run at normal log level
3. one debug-level run if a new gating mechanism is introduced

Notes on why this PR boundary is correct:

- this is pure measurement hygiene and cleanup
- it should not be mixed into larger behavior PRs unless a tiny follow-up is unavoidable

### PR 6: Revisit NVDEC Async Copy and Decode Overlap

In scope:

- re-audit NVDEC surface lifetime under async copy submission
- evaluate restoring async decode copy without reintroducing direct-path crashes
- preserve explicit decode dependency handoff into preprocess
- measure whether overlap gains are real after the lower-risk wins land

Out of scope:

- true batching
- FP16 postprocess fusion
- startup/runtime policy cleanup

Key files or subsystems likely to change:

- `engine-v2/src/codecs/nvdec.rs`
- `engine-v2/src/engine/pipeline.rs`
- `engine-v2/src/core/context.rs`

Acceptance criteria:

1. direct-path correctness remains stable under repeated smoke runs
2. no black-frame, corruption, or access-violation regressions are introduced
3. overlap improvement is measured, not assumed
4. if no safe improvement exists, the synchronous path remains with clear justification

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml --features native_engine`
2. repeated direct-path smoke/perf runs on at least two representative models
3. longer stability run or looped smoke execution

Notes on why this PR boundary is correct:

- this remains the riskiest optimization candidate in the current workspace
- it should only happen after easier throughput wins and clean validation are in place

### PR 7: Direct Path Runtime Policy and Startup Cleanup

In scope:

- decide whether the direct path remains opt-in or becomes the main native runtime path
- if the direct path is strategic, optimize startup/file-boundary overhead
- if the direct path is strategic, replace full-file buffering in `FileBitstreamSource` with streaming reads
- if the direct path is strategic, reduce repeated runtime environment scanning and DLL staging overhead where safe

Out of scope:

- core engine throughput work already handled in earlier PRs
- speculative broad FFmpeg integration redesign unless explicitly chosen

Key files or subsystems likely to change:

- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/bin/smoke.rs`
- adjacent runtime docs and validation scripts

Acceptance criteria:

1. the repo has one clear story for the app-facing native runtime path
2. if the direct path is promoted, its obvious startup and buffering waste is reduced
3. if the direct path remains opt-in, dormant-path startup work is clearly deprioritized and documented

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml --features native_engine`
2. direct-path smoke/perf validation
3. end-to-end run through whichever native path is declared strategic

Notes on why this PR boundary is correct:

- runtime-policy and startup work should follow, not precede, the main engine wins
- otherwise the repo risks optimizing the wrong path first

## Risks / Open Questions

1. True batching depends on provider and model support for the chosen batch shapes. This must be proven on actual target models before larger gain claims are made.
2. FP16 postprocess fusion must preserve existing color conversion behavior and avoid new edge artifacts.
3. ORT wrapper reuse work is only worth doing if it remains correct across dimension changes and does not create lifetime hazards.
4. Async NVDEC overlap is still the highest-risk optimization candidate because the direct path was just stabilized by moving away from that shape.
5. Startup/file-boundary work should stay behind a runtime-policy decision. Otherwise the repo may spend effort on a path that remains non-default.

## Recommended Immediate Next Step

Start with PR 1: `Add Explicit Direct-Path Native Perf Validation`.

Immediately after that, begin PR 2: `Implement True Batched TensorRT Execution`.

That sequence is the best fit for the current workspace because it first makes the direct path measurable, then attacks the largest remaining throughput gap with the current architecture already in place.
