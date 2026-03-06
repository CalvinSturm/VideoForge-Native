# Native Engine Implementation Plan

## Plan Status
- Overall status: `In progress`
- Last updated: `2026-03-06`
- Scope: `engine-v2` native path and its Tauri integration in `src-tauri/src/commands/native_engine.rs`

## How To Use This File
- This file is the execution tracker for the native engine workstream.
- After each implemented section, update:
  - the phase status
  - the completed task checkboxes
  - the `Last updated` date
  - the implementation notes for that phase
- Keep scope changes explicit. If a task is deferred, mark it as `Deferred` and record why.

## Request Summary
- Turn the `engine-v2` architecture review into a concrete, phased implementation plan.
- Prioritize correctness and observability fixes first, then optimize the current architecture, then pursue larger architecture changes.
- Keep the plan reviewable in narrow PRs and structured so it can be updated after each completed section.

## Relevant Codebase Findings
- Direct native entrypoint and routing live in [`src-tauri/src/commands/native_engine.rs`](/C:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/commands/native_engine.rs).
- The native engine crate lives in [`engine-v2/`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2).
- The bounded staged pipeline lives in [`engine-v2/src/engine/pipeline.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/engine/pipeline.rs).
- The TensorRT/ORT backend lives in [`engine-v2/src/backends/tensorrt.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/backends/tensorrt.rs).
- Shared GPU resource management and VRAM accounting live in [`engine-v2/src/core/context.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/core/context.rs).
- The backend contract requires explicit cleanup via `shutdown()` and treats `Drop` as a safety net, but the current direct native host path does not call `shutdown()`.
- The current direct native path still crosses disk boundaries:
  - FFmpeg demux to temp elementary stream
  - `engine-v2` reads from disk and writes compressed output to disk
  - FFmpeg muxes final output
- Current review findings that should drive implementation:
  - explicit backend shutdown and VRAM accounting are incomplete
  - pipeline queue-depth accounting appears incorrect
  - CLI-backed native path reports incorrect audio preservation metadata
  - batched inference metrics undercount frames

## Execution Rules
- Preserve the current external command/API surface unless a phase explicitly changes it.
- Land correctness fixes before optimization work.
- Keep each PR independently reviewable and testable.
- Do not combine architecture refactors with bugfixes unless the bug cannot be fixed safely without the refactor.

## Phase Tracker

### Phase 1: Correctness And Observability Baseline
- Status: `In progress`
- Goal: Fix cleanup, accounting, and result-contract bugs before performance tuning.

#### Tasks
- [x] Call `backend.shutdown().await` explicitly from the direct native host path after pipeline completion and before teardown/mux completion.
- [x] Update `TensorRtBackend::shutdown()` so output ring teardown is explicitly accounted for in VRAM accounting.
- [x] Ensure backend drop behavior remains a safety net rather than the primary cleanup path.
- [x] Fix duplicate `preprocess` queue-depth decrement behavior in the engine pipeline.
- [x] Fix CLI-backed native result reporting so `audio_preserved` matches the requested and actual behavior.
- [x] Fix batched inference metrics so frame counts and averages are correct under `max_batch > 1`.
- [ ] Add focused tests or verifications for:
  - explicit backend shutdown path
  - queue-depth accounting invariants
  - batch metrics correctness
  - CLI fallback audio result contract

#### Key Files
- [`src-tauri/src/commands/native_engine.rs`](/C:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/commands/native_engine.rs)
- [`engine-v2/src/backends/tensorrt.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/backends/tensorrt.rs)
- [`engine-v2/src/engine/pipeline.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/engine/pipeline.rs)
- [`engine-v2/src/core/context.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/core/context.rs)
- [`engine-v2/src/core/backend.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/core/backend.rs)

#### Dependencies
- None. This phase should land first.

#### Risks
- Shutdown ordering changes can expose latent lifetime/resource bugs.
- Queue-depth changes may break any debug tooling that implicitly relied on current counters.
- If CLI-backed native behavior does not currently preserve audio in all cases, result-contract fixes may surface existing product mismatches.

#### Acceptance Criteria
- Direct native execution explicitly calls backend shutdown rather than relying on `Drop`.
- VRAM/accounting reports remain stable after clean shutdown and do not systematically over-report retained allocations from ring teardown.
- Queue-depth counters do not underflow or wrap during normal execution.
- CLI-backed native results no longer claim audio was preserved when it was not requested or not guaranteed.
- Batched inference metrics report per-frame counts accurately.
- At least one targeted verification exists for each corrected behavior.

#### Implementation Notes
- Completed on: `2026-03-06` for the first cleanup subsection
- What changed:
  - direct native host path now calls explicit backend shutdown after pipeline execution
  - shutdown and drop paths share centralized backend resource teardown
  - output ring teardown is now reflected in VRAM accounting instead of relying on untracked drop behavior
  - duplicate preprocess queue-depth decrement was removed from the inference stage
  - CLI-backed native results now report `audio_preserved` from the requested native path behavior instead of always claiming `true`
  - batched inference metrics now count frames correctly and include a unit test for frame-based averaging
- Verification:
  - code-path review completed
  - `cargo check` passed in `engine-v2`
  - `cargo test --lib` passed in `engine-v2`
  - `cargo check --features native_engine` passed in `src-tauri`
- Follow-up:
  - add deeper targeted verification for shutdown/accounting and queue-depth behavior under real pipeline execution

### Phase 2: Internal Throughput Optimization In Current Architecture
- Status: `In progress`
- Goal: Improve steady-state throughput without changing the external native host architecture.

#### Tasks
- [x] Audit device-to-device copy overhead in batched inference:
  - copy into batch input buffer
  - ORT execution
  - copy back into output ring slots
- [x] Determine whether batch output can bind closer to final output ownership to remove one copy step.
- [x] Review `OutputRing` sizing and contention behavior under realistic batch sizes and channel capacities.
- [x] Review pool warmup and `steady_state` behavior and add explicit transition points if needed.
- [x] Validate profiler and overlap instrumentation against real pipeline behavior before using it for tuning decisions.
- [x] Add benchmark or smoke verification guidance for pre/post comparisons.
- [x] Fix direct NVENC output corruption/truncation in real native runs by enforcing the intended no-reorder encoder config.

#### Key Files
- [`engine-v2/src/backends/tensorrt.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/backends/tensorrt.rs)
- [`engine-v2/src/engine/pipeline.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/engine/pipeline.rs)
- [`engine-v2/src/core/context.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/core/context.rs)

#### Dependencies
- Depends on Phase 1.
- Do not start with performance tuning while cleanup/accounting signals are still untrustworthy.

#### Risks
- Micro-optimizations can destabilize buffer ownership and reuse semantics if done before contract checks are in place.
- Stream-overlap instrumentation may not reflect true GPU overlap if events are recorded at the wrong boundaries.

#### Acceptance Criteria
- There is a documented and verified understanding of current batch copy overhead.
- Any accepted optimization measurably reduces copies, contention, or pool misses without regressing correctness.
- Pool/profiler metrics are trustworthy enough to support further architecture decisions.

#### Implementation Notes
- In progress on: `2026-03-06`
- What changed:
  - the backend batch API now borrows `&[GpuTexture]` so the pipeline no longer clones the whole batch vector before dispatch
  - `TensorRtBackend` now computes a batch-aware minimum ring size of `downstream_capacity + max_batch + 1`
  - the native host now derives `ring_size` from the configured batch size instead of using a fixed constant
  - the inference stage now reuses hot-path `Vec` allocations for batch envelopes, metadata, inputs, and outbound frames
  - `GpuContext` now exposes explicit helpers to reset pool stats, enter/reset steady state, and reset runtime telemetry
  - synthetic stress/audit runs now mark the measured phase as steady-state only after warm-up and reset profiler/overlap counters before measurement
  - `GpuTexture` now carries a `GpuBuffer` abstraction that supports either owned allocations or zero-copy views into a shared device buffer
  - the TensorRT batch path now returns per-frame GPU views into the shared batch output buffer instead of copying each frame back into per-frame ring slots
  - the batched inference path no longer allocates or touches `OutputRing`, avoiding the extra VRAM footprint for a ring that batch mode no longer needs
  - native NVENC now uses the existing custom encoder config instead of preset defaults, with `b_frames=0`, `enableLookahead=0`, `zeroReorderDelay=1`, `enableEncodeAsync=0`, and `NV12` buffer format explicitly applied
  - the first attempted fix for post-EOS draining was removed after it proved unsafe and could block indefinitely on `nvEncLockBitstream`
  - `NvEncoder` now records `packets_written`, which makes packet-count mismatches visible in logs and verification
  - the pipeline now records decode-side overlap markers on sampled frames, samples every 8 frames instead of every 60, and logs overlap sampling failures instead of silently swallowing them
- Why it matters:
  - removes avoidable CPU-side batch cloning and repeated small allocations in the inference loop
  - prevents artificial output-ring contention when `max_batch > 1`
  - makes pool-miss warnings and profiler/overlap snapshots correspond to the measured phase instead of mixed warm-up noise
  - removes one full device-to-device copy stage from batched inference output handoff
  - fixes the direct native regression where the output MP4 was truncated/corrupt because NVENC preset defaults were silently enabling B-frames and frame reordering even though the app requested `b_frames=0`
  - turns `overlap_samples=0` into a live signal that can distinguish “no data” from “positive inter-stage gap / no overlap”
- Verification:
  - `cargo test --lib` passed in `engine-v2`
  - `cargo check --features native_engine` passed in `src-tauri`
  - added a unit test for the batch-aware minimum ring-size contract
  - `cargo run --manifest-path src-tauri/Cargo.toml --features native_engine --bin videoforge_bench -- --native --input input.mp4 --output output_batch1_cfgfix.mp4 --onnx-model weights/2x_SPAN_soft.onnx --scale 2 --precision fp16 --max-batch 1 --native-direct`
  - `ffprobe` on `output_batch1_cfgfix.mp4` reported `nb_frames=61`, `duration=2.033313`, and `has_b_frames=0`
  - `cargo run --manifest-path src-tauri/Cargo.toml --features native_engine --bin videoforge_bench -- --native --input input.mp4 --output output_batch4_cfgfix.mp4 --onnx-model weights/2x_SPAN_soft.onnx --scale 2 --precision fp16 --max-batch 4 --native-direct`
  - `ffprobe` on `output_batch4_cfgfix.mp4` reported `nb_frames=61`, `duration=2.033313`, and `has_b_frames=0`
  - `cargo run --manifest-path src-tauri/Cargo.toml --features native_engine --bin videoforge_bench -- --native --input input.mp4 --output output_batch4_overlapcheck.mp4 --onnx-model weights/2x_SPAN_soft.onnx --scale 2 --precision fp16 --max-batch 4 --native-direct`
  - overlap telemetry reported `overlap_samples=7`, `overlap_avg_gap_ms=8000.63`, `overlap_min_gap_ms=5.45`, `overlap_max_gap_ms=55969.30`, `overlap_pct=0.0`
- Follow-up:
  - add targeted coverage for `GpuBuffer::View` semantics if the repo grows a native-only integration test harness
  - add real hardware before/after numbers once a representative native test clip and ONNX model are selected for the team baseline
  - if image-level corruption is still reported visually after the packet-count/duration fix, compare extracted frames from `output_batch1_cfgfix.mp4` and `output_batch4_cfgfix.mp4` to rule out a remaining batch-output ownership issue
  - if a future optimization effort needs true GPU overlap measurement rather than queue-gap sampling, replace the current decode-to-preprocess sampled timer with explicit cross-stream stage-pair telemetry
  - controlled native benchmark matrix on `input.mp4` was recorded under `artifacts/benchmarks/native_matrix/`:
    - `SPAN fp16 batch=1`: `elapsed_ms=52662`, `inference_avg_us=826930`, `encode_avg_us=1737`, `vram_peak_mb=31`, `encoder_mode=nvenc`
    - `SPAN fp16 batch=4`: `elapsed_ms=80624`, `inference_avg_us=1284791`, `encode_avg_us=1783`, `vram_peak_mb=60`, `encoder_mode=nvenc`
    - `Nomos fp32 batch=1`: `elapsed_ms=97102`, `inference_avg_us=1333370`, `encode_avg_us=30696`, `vram_peak_mb=79`, `encoder_mode=nvenc`
    - `Nomos fp32 batch=4`: `elapsed_ms=93071`, `inference_avg_us=1265306`, `encode_avg_us=45182`, `vram_peak_mb=122`, `encoder_mode=nvenc`
  - important interpretation from that matrix:
    - on this clip and runtime, CNN batching did not improve throughput and increased VRAM materially
    - the transformer `batch=4` result is not evidence of true batching speedup, because transformer-family models are currently conservatively routed to sequential execution
    - overlap telemetry still reports `overlap_pct=0.0` in every case, so it remains useful only as a gap sampler, not as proof of real cross-stream overlap
  - profiling semantics were tightened after the first matrix run so batched inference and postprocess now report both per-dispatch and per-frame-equivalent timings instead of one ambiguous average
  - rerunning `SPAN fp16 batch=4` after the profiler fix produced:
    - `elapsed_ms=85552`
    - `inference_dispatch_avg_us=5010431`
    - `inference_frame_equiv_us=1314211`
    - `inference_dispatches=16`
    - `postprocess_dispatch_avg_us=216`
    - `postprocess_frame_equiv_us=56`
  - follow-up investigation showed that the apparent `SPAN batch=4` slowdown was heavily polluted by lazy TensorRT engine/profile build on the first real inference dispatch:
    - uncached `SPAN fp16 batch=4`: `elapsed_ms=85552`
    - same model with TRT cache enabled, first cache-populating run: `elapsed_ms=80196`
    - same model with TRT cache enabled, second warm run: `elapsed_ms=1043`, `inference_dispatch_avg_us=7509`, `inference_frame_equiv_us=1969`
  - `videoforge_bench` now supports native warm benchmarking directly via `--trt-cache --warmup-runs <n>`, so future comparisons can exclude one-time TensorRT cold-start compilation noise without manual env-var setup
  - next optimization target should be chosen from measured evidence rather than assumptions; likely candidates are backend execution tuning for CNN models and profiler fidelity, not broader host-architecture changes

#### Native Verification Workflow
- Correctness smoke:
  - `cargo run --manifest-path src-tauri/Cargo.toml --features native_engine --bin smoke -- --e2e-native --native-direct --input <video> --e2e-onnx <model.onnx> --e2e-scale <scale> --precision <fp16|fp32> [--keep-temp]`
  - Purpose: verify the direct `engine-v2` path completes end-to-end on real media.
- Native benchmark dry run:
  - `cargo run --manifest-path src-tauri/Cargo.toml --features native_engine --bin videoforge_bench -- --native --input <video> --output <out.mp4> --onnx-model <model.onnx> --scale <scale> --precision <fp16|fp32> --max-batch <n> --native-direct --dry-run`
  - Purpose: validate runtime prerequisites and arguments without running the pipeline.
- Native benchmark run:
  - `cargo run --manifest-path src-tauri/Cargo.toml --features native_engine --bin videoforge_bench -- --native --input <video> --output <out.mp4> --onnx-model <model.onnx> --scale <scale> --precision <fp16|fp32> --max-batch <n> --native-direct`
  - Output shape:
    - emits JSON `start` and `done` events
    - `done` includes `elapsed_ms`, `frames_processed`, `engine`, `encoder_mode`, `native_direct`, and `max_batch`
- Comparison rules:
  - hold `input`, `onnx-model`, `scale`, `precision`, `max_batch`, and direct-vs-CLI routing constant across runs
  - prefer `--native-direct` when evaluating `engine-v2` Phase 2 changes
  - keep audio disabled for throughput-focused runs unless mux/audio behavior is part of the test goal

### Phase 3: Direct Native Host Architecture Optimization
- Status: `In progress`
- Goal: Remove avoidable file-boundary overhead around direct native processing.

#### Tasks
- [x] Replace or reduce temp-file demux boundary between FFmpeg and `engine-v2`.
- [x] Replace or reduce temp-file boundary between `engine-v2` encoded output and final mux.
- [ ] Decide whether the preferred model is:
  - FFmpeg stream -> native decoder -> native encoder -> mux stream
  - or a tighter native-only container/output path where feasible
- [ ] Re-evaluate fallback behavior once direct native reliability improves.
- [ ] Update native command error handling and temp-artifact retention rules to match the new boundary model.

#### Key Files
- [`src-tauri/src/commands/native_engine.rs`](/C:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/commands/native_engine.rs)
- [`engine-v2/src/codecs/nvdec.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/codecs/nvdec.rs)
- [`engine-v2/src/codecs/nvenc.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/codecs/nvenc.rs)
- [`engine-v2/src/engine/pipeline.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/engine/pipeline.rs)

#### Dependencies
- Depends on Phase 1.
- Strongly recommended to start after Phase 2, unless product priorities require architecture work sooner.

#### Risks
- This is the highest-risk phase because it changes process boundaries, error recovery, and operational debugging behavior.
- Removing temp files can make failure inspection harder unless explicit diagnostic capture replaces them.
- Audio/video mux behavior and container support may constrain how much file-boundary removal is practical.

#### Acceptance Criteria
- Direct native execution no longer requires a temp encoded video file between `engine-v2` and the final mux path.
- Native benchmark output remains valid under the streamed mux path, with frame count and duration matching the prior temp-file path.
- Streaming mux startup is deterministic enough that FFmpeg does not depend on stdin auto-probing luck to identify the bitstream container.

#### Implementation Notes
- Started on: `2026-03-06`
- What changed:
  - the direct native host no longer writes an intermediate `engine_output` bitstream file before final mux
  - the direct native host no longer writes a demuxed input elementary stream temp file before starting `engine-v2`
  - `probe_video_coded_geometry()` now identifies the native input codec (`H.264` vs `HEVC`) up front so the host can choose a single streamed demux path
  - `FfmpegBitstreamSource` now streams FFmpeg demux stdout directly into the NVDEC `BitstreamSource` contract
  - `StreamingMuxSink` now streams NVENC output directly into an FFmpeg mux process over stdin
  - streamed demux and mux children now keep bounded stderr tails so failure paths can report useful FFmpeg diagnostics without reintroducing blocking temp-file workflows
  - the mux sink now accepts an explicit shared codec hint, seeded from input probing and updated from the actual NVENC encoder selection, so mux startup no longer depends primarily on packet sniffing
  - the mux sink starts lazily on the first encoded packet and supplies an explicit FFmpeg stdin format instead of relying entirely on auto-probing
  - when packet sniffing cannot classify a bitstream packet, the sink logs a warning and currently falls back to `hevc`, which matches the current native runtime capability path on this machine
  - postprocess NV12 outputs now use a dedicated raw CUDA allocation path instead of pooled `CudaSlice` buffers
  - `GpuBuffer` now supports synchronous host readback from raw CUDA allocations so the software fallback path still works if a raw-backed NV12 frame crosses that boundary
  - the direct NVENC path now registers postprocess NV12 surfaces successfully on this runtime, eliminating the `nvenc_legacy_staging` fallback
- Verification:
  - `cargo check --manifest-path src-tauri/Cargo.toml --features native_engine`
  - `cargo run --manifest-path src-tauri/Cargo.toml --features native_engine --bin videoforge_bench -- --native --input input.mp4 --output output_phase3_streammux_fmt2.mp4 --onnx-model weights/2x_SPAN_soft.onnx --scale 2 --precision fp16 --max-batch 4 --native-direct`
  - `ffprobe -hide_banner -show_streams -show_format output_phase3_streammux_fmt2.mp4`
  - `ffprobe` reported a valid HEVC MP4 with `nb_frames=61`, `duration=2.033313`, and `has_b_frames=0`
  - `cargo run --manifest-path src-tauri/Cargo.toml --features native_engine --bin videoforge_bench -- --native --input input.mp4 --output output_phase3_streamin.mp4 --onnx-model weights/2x_SPAN_soft.onnx --scale 2 --precision fp16 --max-batch 4 --native-direct`
  - `ffprobe -hide_banner -show_streams -show_format output_phase3_streamin.mp4`
  - the streamed-input path reported `Pipeline finished decoded=61 preprocessed=61 inferred=61 encoded=61`
  - `ffprobe` reported a valid HEVC MP4 with `nb_frames=61`, `duration=2.033313`, and `has_b_frames=0`
  - `cargo run --manifest-path src-tauri/Cargo.toml --features native_engine --bin videoforge_bench -- --native --input input.mp4 --output output_phase3_hardened.mp4 --onnx-model weights/2x_SPAN_soft.onnx --scale 2 --precision fp16 --max-batch 4 --native-direct`
  - the hardened streamed path reported `Pipeline finished decoded=61 preprocessed=61 inferred=61 encoded=61`
  - `ffprobe` on `output_phase3_hardened.mp4` reported a valid HEVC MP4 with `nb_frames=61`, `duration=2.033313`, and `has_b_frames=0`
  - `cargo check --manifest-path engine-v2/Cargo.toml`
  - `cargo run --manifest-path src-tauri/Cargo.toml --features native_engine --bin videoforge_bench -- --native --input input.mp4 --output output_phase3_nvenc_direct.mp4 --onnx-model weights/2x_SPAN_soft.onnx --scale 2 --precision fp16 --max-batch 4 --native-direct`
  - `ffprobe -hide_banner -show_streams -show_format output_phase3_nvenc_direct.mp4`
  - the direct-registration run reported `Pipeline finished decoded=61 preprocessed=61 inferred=61 encoded=61`
  - the final benchmark event reported `encoder_mode=\"nvenc\"`, `frames_processed=61`, and `elapsed_ms=90466`
  - `ffprobe` on `output_phase3_nvenc_direct.mp4` reported a valid HEVC MP4 with `nb_frames=61`, `duration=2.033313`, and `has_b_frames=0`
- Follow-up:
  - both major temp-file boundaries have now been removed from the direct native path, but the preferred long-term contract still needs to be formalized: streamed FFmpeg demux/mux around `engine-v2` vs a tighter native-only container/output path
  - direct NVENC registration now works on this runtime, so the main remaining native-engine unknowns are profiler/overlap fidelity and whether further host-boundary simplification is worth the risk
  - known-bad transformer export `weights/4xNomos2_hq_dat2_fp32.fp16.onnx` is now filtered from normal model discovery, and explicit native use of that file returns an `Invalid ONNX artifact` error instead of a generic backend-init failure
  - transformer-family ONNX models are currently routed off the true batched inference path and back to sequential execution even when the graph advertises dynamic batch axes; this remains a conservative guard, but the original `4xNomos2_hq_dat2_fp32.onnx` `max_batch=4` concern was not reproduced in multiple players and appears to have been a VLC playback false signal rather than a confirmed engine-output corruption bug
  - the current mux-format fallback is intentionally conservative; if the runtime later selects H.264 more often, the sink should receive codec information directly from the encoder instead of inferring it from packets
  - benchmark/probe steps should not be run in parallel when validating streamed output files, because probing can race the still-running benchmark process and report a false missing-file result

#### Acceptance Criteria
- The direct native path removes at least one major temp-file boundary or replaces it with a lower-overhead streaming boundary.
- Failure handling remains diagnosable.
- Native output correctness and mux behavior remain intact for supported inputs.
- Performance improvement is demonstrated with the project’s native smoke/benchmark workflow.

#### Implementation Notes
- None yet.

## Proposed Implementation Approach
1. Stabilize the native engine contract first.
   - Make shutdown explicit.
   - Fix accounting and result-contract bugs.
   - Repair metrics so optimization data is trustworthy.
2. Optimize only after the baseline is reliable.
   - Remove unnecessary device copies in batch execution if practical.
   - Tune ring/pool behavior based on verified metrics.
3. Attack the highest-leverage architecture cost last.
   - Reduce disk boundaries around direct native execution.
   - Revisit fallback and runtime boundaries once the direct path is stable.

## Ordered PR Breakdown
1. `PR 1: Native shutdown and accounting correctness`
   - Primary goal: make cleanup deterministic and fix VRAM/accounting issues
   - Dependency: none
2. `PR 2: Pipeline observability and result-contract fixes`
   - Primary goal: fix queue-depth accounting, batch metrics, and CLI result correctness
   - Dependency: PR 1
3. `PR 3: Native engine internal throughput improvements`
   - Primary goal: reduce batch-path overhead and tune reuse behavior inside current architecture
   - Dependency: PR 2
4. `PR 4: Direct native boundary reduction`
   - Primary goal: reduce temp-file overhead around the direct native pipeline
   - Dependency: PR 2
   - Recommended dependency: PR 3

## Per-PR Scope Details

### PR 1: Native Shutdown And Accounting Correctness
- Status: `Completed`
- In scope:
  - explicit backend shutdown from the direct native host path
  - deterministic ring/accounting cleanup
  - cleanup-order verification
- Out of scope:
  - batching changes
  - routing changes
  - temp-file architecture refactors
- Key files or subsystems likely to change:
  - [`src-tauri/src/commands/native_engine.rs`](/C:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/commands/native_engine.rs)
  - [`engine-v2/src/backends/tensorrt.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/backends/tensorrt.rs)
  - [`engine-v2/src/core/context.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/core/context.rs)
- Acceptance criteria:
  - direct native host path explicitly shuts down the backend
  - shutdown path accounts for output ring memory correctly
  - cleanup works on success and on pipeline error paths
- Test or verification expectations:
  - targeted unit/integration coverage where practical
  - manual validation with direct native path if environment is available
- Why this PR boundary is correct:
  - it isolates cleanup correctness without mixing in optimization or architecture changes

### PR 2: Pipeline Observability And Result-Contract Fixes
- Status: `Completed in code; follow-up verification remains`
- In scope:
  - queue-depth accounting correction
  - batch metric correctness
  - CLI audio result correctness
- Out of scope:
  - throughput optimizations
  - host-boundary refactors
- Key files or subsystems likely to change:
  - [`engine-v2/src/engine/pipeline.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/engine/pipeline.rs)
  - [`engine-v2/src/backends/tensorrt.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/backends/tensorrt.rs)
  - [`src-tauri/src/commands/native_engine.rs`](/C:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/commands/native_engine.rs)
- Acceptance criteria:
  - queue counters remain sane during normal execution
  - frame-based metrics are correct under batch mode
  - CLI-backed native result metadata is truthful
- Test or verification expectations:
  - narrow tests for counter and metric behavior
  - result-contract verification for CLI fallback response
- Why this PR boundary is correct:
  - it keeps observability and API-contract fixes together while avoiding unrelated cleanup changes

### PR 3: Native Engine Internal Throughput Improvements
- Status: `In progress`
- In scope:
  - current-architecture batch path optimization
  - ring/pool tuning based on measured behavior
  - profiler/metric trustworthiness needed for tuning
- Out of scope:
  - direct host boundary removal
  - native routing changes
- Key files or subsystems likely to change:
  - [`engine-v2/src/backends/tensorrt.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/backends/tensorrt.rs)
  - [`engine-v2/src/engine/pipeline.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/engine/pipeline.rs)
  - [`engine-v2/src/core/context.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/core/context.rs)
- Acceptance criteria:
  - measured reduction in avoidable batch-path overhead or contention
  - no regression in correctness or cleanup behavior
- Test or verification expectations:
  - before/after benchmark or smoke comparison
  - validation under batch sizes greater than 1
- Why this PR boundary is correct:
  - it isolates optimization within the current architecture and avoids mixing with file-boundary refactors

### PR 4: Direct Native Boundary Reduction
- Status: `Not started`
- In scope:
  - reducing or replacing temp-file boundaries around direct native execution
  - matching error-handling and diagnostics to the new boundary model
- Out of scope:
  - unrelated frontend changes
  - Python engine changes
- Key files or subsystems likely to change:
  - [`src-tauri/src/commands/native_engine.rs`](/C:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/commands/native_engine.rs)
  - [`engine-v2/src/codecs/`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/codecs)
  - [`engine-v2/src/engine/pipeline.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/engine/pipeline.rs)
- Acceptance criteria:
  - at least one material temp-file boundary is removed or replaced
  - supported native outputs remain correct and diagnosable
  - performance benefit is demonstrated
- Test or verification expectations:
  - native smoke validation on supported media
  - failure-path validation to ensure diagnostics are still usable
- Why this PR boundary is correct:
  - it is the largest-risk architectural change and should be reviewed separately from internal fixes

## Risks / Open Questions
- It is still unclear how much of the temp-file architecture can be removed without broader mux/container changes.
- Native performance tuning should wait until cleanup and accounting are trustworthy, otherwise benchmark data will be noisy or misleading.
- If there are hidden operational dependencies on current temp-file retention, Phase 3 may need an explicit diagnostics design before implementation.

## Suggested Update Template
- When a phase starts, change `Status` to `In progress`.
- When a phase lands, change `Status` to `Completed` and add:

```md
#### Implementation Notes
- Completed on: `YYYY-MM-DD`
- PR / commit: `...`
- What changed:
  - ...
- Verification:
  - ...
- Follow-up:
  - ...
```

## Evidence
- Evidence: [`src-tauri/src/commands/native_engine.rs`](/C:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/commands/native_engine.rs)
- Evidence: [`engine-v2/src/backends/tensorrt.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/backends/tensorrt.rs)
- Evidence: [`engine-v2/src/engine/pipeline.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/engine/pipeline.rs)
- Evidence: [`engine-v2/src/core/context.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/core/context.rs)
- Evidence: [`engine-v2/src/core/backend.rs`](/C:/Users/Calvin/Desktop/VideoForge1/engine-v2/src/core/backend.rs)
