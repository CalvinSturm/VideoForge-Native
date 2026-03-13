# Native Engine Implementation Plan

Date: 2026-03-06

## Request Summary

This plan replaces the earlier native-engine plan with a code-grounded version. It is based on the current engine and Tauri integration, and it separates active runtime work from dormant in-process code so optimization work lands in the right order.

## Relevant Codebase Findings

The current codebase has two important layers that must not be conflated.

### Active runtime path

The user-facing Tauri command `upscale_request_native()` currently routes to `run_native_via_rave_cli()`, which forwards to `rave_upscale()`.

Preserve:

1. the existing runtime-gating behavior
2. the current smoke/perf gate around the native command surface
3. the ability to ship native improvements without breaking the existing command contract

### Dormant in-process path

`run_native_pipeline()` directly uses `engine-v2` from Tauri, including:

- `FileBitstreamSource`
- `NvDecoder`
- `TensorRtBackend`
- `UpscalePipeline`
- `NvEncoder`

Preserve:

1. its usefulness as a reference implementation
2. its ability to become the future active runtime path if desired
3. narrow PR boundaries if it is touched

### Engine-v2 core facts

The engine already has:

- decode-ready event objects carried through the pipeline
- async NVDEC D2D copy
- micro-batch accumulation in the pipeline
- FP16 fused preprocess
- output ring reuse
- synthetic stress and audit helpers

The main confirmed gaps are:

1. true batch execution is still not implemented in the backend
2. FP16 postprocess is still unfused
3. ORT wrapper work still happens per frame
4. decode/preprocess overlap is only partially realized because the decode thread still waits on completion before unmapping
5. generic stream sync is still represented by a no-op helper
6. TensorRT cache enablement is still opt-in
7. window-size padding for windowed transformer models still appears absent

## Proposed Implementation Approach

Proceed in three tracks, in this order.

### Track 1: Runtime and validation alignment

Fix planning ambiguity before deeper optimization work.

This track should:

1. document the active runtime path accurately
2. decide whether in-process `run_native_pipeline()` is strategic or dormant
3. align benchmark language with the actual scripts and binaries in the repo

This track is small but important because it determines which later work is immediately user-visible.

### Track 2: Engine correctness and throughput

After the runtime boundary is clear, fix the confirmed engine issues in the hot path:

1. harden stream-ordering assumptions and decode overlap behavior
2. implement true TensorRT batch execution
3. fuse FP16 postprocess
4. reduce ORT per-frame wrapper churn
5. standardize TensorRT caching

This track should stay focused on `engine-v2`.

### Track 3: Compatibility and dormant-path cleanup

After the hot-path engine work lands:

1. add transformer window padding and crop-back
2. only touch `FileBitstreamSource` buffering if the in-process Tauri path is being activated or intentionally maintained

## Ordered PR Breakdown

1. PR title: `Align Native Runtime Documentation and Ownership`
   Primary goal: make the active native runtime path and validation workflow explicit
   Dependency on earlier PRs, if any: none

2. PR title: `Harden Native Stream Ordering and Decode Overlap`
   Primary goal: replace implied synchronization with real ordering and remove unnecessary decode-side CPU blocking where safe
   Dependency on earlier PRs, if any: PR 1 recommended

3. PR title: `Implement True Batched TensorRT Execution`
   Primary goal: turn configured micro-batching into actual backend batch execution
   Dependency on earlier PRs, if any: PR 2 preferred

4. PR title: `Fuse FP16 Postprocess to NV12`
   Primary goal: remove the extra FP16-to-F32 conversion on the output path
   Dependency on earlier PRs, if any: none

5. PR title: `Reduce TensorRT Hot-Path Wrapper Overhead`
   Primary goal: eliminate avoidable per-frame ORT wrapper work and standardize TensorRT cache use
   Dependency on earlier PRs, if any: none

6. PR title: `Add Transformer Window Padding`
   Primary goal: close the main remaining windowed-transformer correctness gap
   Dependency on earlier PRs, if any: none

7. PR title: `Decide and Clean Up the In-Process Tauri Native Path`
   Primary goal: either activate `run_native_pipeline()` and optimize it, or explicitly demote it from near-term optimization scope
   Dependency on earlier PRs, if any: PR 1 required

## Per-PR Scope Details

### PR 1: Align Native Runtime Documentation and Ownership

In scope:

- update the native-engine docs so they match the current code
- distinguish the active CLI-backed path from the dormant in-process path
- document which scripts validate which path
- remove stale claims about missing decode events and synchronous NVDEC copies

Out of scope:

- engine behavior changes
- batching implementation
- kernel changes

Key files or subsystems likely to change:

- `docs/archive/native_engine_optimization_audit.md`
- `docs/archive/native_engine_implementation_plan.md`
- possibly adjacent docs that still describe the wrong active path

Acceptance criteria:

1. docs no longer describe `run_native_pipeline()` as if it were the active runtime path
2. docs no longer claim decode events are absent or NVDEC uses synchronous copies
3. docs describe the current native validation workflow accurately

Test or verification expectations:

1. docs review against current code references

Notes on why this PR boundary is correct:

- this is a planning and ownership fix
- later PRs should not proceed from stale assumptions

### PR 2: Harden Native Stream Ordering and Decode Overlap

In scope:

- review every stage boundary that currently relies on comments or `sync_stream()`
- keep decode-ready event handoff intact
- reduce or remove the decode-thread `cuEventSynchronize(event)` wait if surface lifetime can remain correct
- add explicit event-based ordering where preprocess -> inference or postprocess -> encode currently depends on fake synchronization
- keep overlap telemetry meaningful

Out of scope:

- TensorRT batching
- FP16 postprocess fusion
- transformer padding

Key files or subsystems likely to change:

- `engine-v2/src/codecs/nvdec.rs`
- `engine-v2/src/engine/pipeline.rs`
- `engine-v2/src/core/context.rs`
- possibly `engine-v2/src/codecs/nvenc.rs`

Acceptance criteria:

1. no stage boundary relies on `GpuContext::sync_stream()` as if it were real synchronization
2. decode -> preprocess ordering remains correct
3. preprocess -> inference and postprocess -> encode ordering is explicit and reviewable
4. black-tile, corruption, and ordering regressions are not introduced
5. overlap telemetry and end-to-end throughput improve or, at minimum, the ordering model becomes clearly correct

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml`
2. `tools/ci/check_native_smoke_perf.ps1`
3. targeted long-run native smoke or equivalent stability run
4. engine synthetic stress/audit helpers if maintained for this path

Notes on why this PR boundary is correct:

- this isolates the riskiest correctness surface first
- batching and kernel work should not pile on top of unclear stream ordering

### PR 3: Implement True Batched TensorRT Execution

In scope:

- add batched shape construction and binding in `TensorRtBackend`
- make `process_batch()` perform one real batched execution
- preserve batch size `1` behavior
- keep frame ordering and output count validation intact

Out of scope:

- multi-GPU work
- transformer padding
- encode redesign

Key files or subsystems likely to change:

- `engine-v2/src/backends/tensorrt.rs`
- `engine-v2/src/core/backend.rs`
- `engine-v2/src/engine/pipeline.rs`

Acceptance criteria:

1. `max_batch > 1` no longer routes through a sequential loop
2. the current warning about single-frame execution is removed
3. output count and ordering remain correct
4. compatible models show real benefit at batch `2` or `4`

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml`
2. native smoke/perf validation
3. repeated before/after runs on a native path that actually exercises batch settings

Notes on why this PR boundary is correct:

- the pipeline-side batch admission already exists
- the missing work is concentrated in backend execution

### PR 4: Fuse FP16 Postprocess to NV12

In scope:

- add a fused `RgbPlanarF16 -> NV12` kernel
- route FP16 model outputs through that path
- keep the existing F32 output path unchanged

Out of scope:

- decode ordering
- TensorRT batching
- color policy changes

Key files or subsystems likely to change:

- `engine-v2/src/core/kernels.rs`
- `engine-v2/src/engine/pipeline.rs`

Acceptance criteria:

1. the FP16 output path no longer promotes to F32 before NV12 conversion
2. visual output remains correct
3. F32 behavior remains unchanged

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml`
2. native smoke/perf validation
3. targeted FP16 regression check on representative clips

Notes on why this PR boundary is correct:

- this is a narrow kernel-path optimization
- it should stay independent from backend execution changes

### PR 5: Reduce TensorRT Hot-Path Wrapper Overhead

In scope:

- stop recreating avoidable ORT memory metadata per call
- reuse cached metadata or equivalent wrapper state where safe
- evaluate reusable binding state for stable shapes
- standardize TensorRT cache behavior for the production runtime

Out of scope:

- pipeline ordering changes
- dormant Tauri-native buffering changes

Key files or subsystems likely to change:

- `engine-v2/src/backends/tensorrt.rs`
- possibly runtime configuration points that control cache defaults

Acceptance criteria:

1. avoidable per-frame `OrtMemoryInfo` churn is removed or materially reduced
2. warm-start cache behavior is documented and verifiable
3. steady-state throughput does not regress

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml`
2. native smoke/perf validation
3. explicit cold-start and warm-start timing comparison if cache defaults are changed

Notes on why this PR boundary is correct:

- this is backend overhead work, not pipeline redesign
- it should not be mixed with dormant-path cleanup

### PR 6: Add Transformer Window Padding

In scope:

- detect when window-aligned input sizing is required
- pad inputs before inference
- crop outputs back after inference
- preserve output dimensions and avoid edge artifacts

Out of scope:

- batching redesign
- dormant Tauri runtime cleanup
- speculative transfer optimizations

Key files or subsystems likely to change:

- `engine-v2/src/backends/tensorrt.rs`
- `engine-v2/src/engine/pipeline.rs`
- any small support module introduced for model policy

Acceptance criteria:

1. windowed transformer models can process non-window-aligned inputs correctly
2. output dimensions remain correct after crop-back
3. no visible border corruption is introduced

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml`
2. native smoke or targeted transformer validation
3. representative transformer compatibility checks

Notes on why this PR boundary is correct:

- this is a correctness feature, not a generic throughput change
- reviewers should be able to reason about model behavior separately

### PR 7: Decide and Clean Up the In-Process Tauri Native Path

In scope:

- choose one of two outcomes:
- either make `run_native_pipeline()` the intended runtime path and then optimize its buffering/path setup
- or explicitly mark it as dormant and stop treating its host-memory behavior as a top optimization target
- if activated, replace full-file buffering with streaming reads
- if activated, remove repeated 1 MiB chunk cloning in `FileBitstreamSource`

Out of scope:

- engine-v2 throughput work already handled in earlier PRs
- transformer padding

Key files or subsystems likely to change:

- `src-tauri/src/commands/native_engine.rs`
- possibly adjacent runtime docs or smoke wiring

Acceptance criteria:

1. the repository has one clear story for the app-facing native runtime path
2. dormant-path code is no longer described as active production behavior
3. if the in-process path is activated, its buffering and startup behavior is no longer obviously wasteful

Test or verification expectations:

1. `cargo check --manifest-path src-tauri/Cargo.toml`
2. `tools/ci/check_native_smoke_perf.ps1`
3. a direct end-to-end run through whichever native path is declared active

Notes on why this PR boundary is correct:

- this keeps dormant-path cleanup separate from engine-core hot-path work
- it prevents a grab-bag PR that mixes app routing, buffering, and backend execution

## Risks / Open Questions

1. The biggest unresolved technical risk is stream ordering beyond decode -> preprocess. The code comments often read as if synchronization exists even where the helper is a no-op.
2. True batching depends on model/provider support for the target batch shapes. That should be proven on the actual production model before promising a large gain.
3. If the product intends to stay on the CLI-backed native path for a while, some in-process `src-tauri` optimization work should move down the priority list.
4. Window padding should be model-aware. Blind padding policy risks changing behavior for models that do not need it.

## Recommended Immediate Next Step

Start with PR 1: `Align Native Runtime Documentation and Ownership`.

Immediately after that, start PR 2: `Harden Native Stream Ordering and Decode Overlap`.

That sequence is the best fit for the current repository because it:

1. removes stale planning assumptions
2. fixes the highest-risk correctness and overlap surface first
3. clears the way for batching and kernel optimizations without compounding synchronization uncertainty
