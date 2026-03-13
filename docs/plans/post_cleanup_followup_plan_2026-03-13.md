# Post-Cleanup Follow-up Plan

Status: active
Created: 2026-03-13
Scope: next logical cleanup/refactor tracks after completing the 8-step codebase cleanup plan
Use this for: tracking the remaining high-value refactor and hygiene work
Do not use this for: release notes or shipped-state claims without rerunning validation

## 1. Objectives

- Finish the remaining structural cleanup left after the completed PR 1-8 plan.
- Reduce the largest remaining multi-responsibility files on both the Rust and UI sides.
- Restore strict lint hygiene in addition to passing tests/typecheck.
- Keep each track reviewable and independently shippable.

## 2. Current Audit Summary

- `src-tauri/src/commands/native_engine.rs` still owns the direct native pipeline implementation, ffprobe/source profiling, FFmpeg streaming wrappers, encoder glue, and tests.
- `ui/src/components/InputOutputPanel.tsx` is still the largest UI file and mixes config schema/defaults, helper logic, icons, and component rendering.
- `src-tauri/src/commands/upscale.rs` and `src-tauri/src/video_pipeline.rs` still contain broad orchestration/config surfaces and `too_many_arguments` suppressions.
- `cargo clippy --workspace --all-targets -- -D warnings` currently fails on at least one needless-return issue, so lint hygiene is not fully green.

## 3. Execution Rules

- Keep behavior-preserving unless a task explicitly calls for behavior changes.
- Do not combine frontend and backend structural refactors in the same PR unless the boundary requires it.
- After each completed item, update this file by checking the box and briefly noting validation status/date.
- Validation is part of every step, not a final-only activity.

## 4. Priority-Ranked Plan

### Track 0: Restore Strict Lint Baseline

Status: complete (2026-03-13)

- [x] Fix current `clippy -D warnings` failures.
- [x] Re-run strict lint validation and confirm clean output.
- [x] Add the lint command to the working validation checklist in this plan once green.

Files in scope:
- `src-tauri/src/commands/native_engine.rs`
- any other Rust files surfaced by `cargo clippy --workspace --all-targets -- -D warnings`

Changes:
- Remove trivial style/lint regressions left after the previous cleanup plan.
- Prefer minimal edits; this track is for restoring the baseline, not broad refactors.

Acceptance criteria:
- `cargo clippy --workspace --all-targets -- -D warnings` passes.
- `cargo test --workspace` still passes.

Validation:
- `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings`
- `cd src-tauri && cargo test --workspace`

### Track 1: Extract Native Direct Pipeline Modules

Status: complete (2026-03-13)

- [x] Extract source probing/profile types from `native_engine.rs`.
- [x] Extract FFmpeg bitstream source and mux sink wrappers from `native_engine.rs`.
- [x] Extract direct engine-v2 pipeline runner into a focused module.
- [x] Keep command boundary and route-selection behavior unchanged.

Files in scope:
- `src-tauri/src/commands/native_engine.rs`
- new modules under `src-tauri/src/commands/` such as:
  - `native_direct_pipeline.rs`
  - `native_streaming_io.rs`
  - `native_probe.rs`

Changes:
- Move `NativeVideoSourceProfile`, `NativeDirectPlan`, `FfmpegBitstreamSource`, `StreamingMuxSink`, and related helpers into smaller modules.
- Leave `upscale_request_native` as the command boundary and orchestrator entrypoint.
- Keep tests updated with the new module boundaries.

Acceptance criteria:
- `native_engine.rs` becomes materially smaller and more command-focused.
- Direct native path logic is easier to review in isolation.
- Existing native tests still pass.

Validation:
- `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings`
- `cd src-tauri && cargo test --workspace`

### Track 2: Decompose InputOutputPanel

Status: complete (2026-03-13)

- [x] Extract research config types/defaults/presets from `InputOutputPanel.tsx`.
- [x] Extract model display/helpers and reusable small UI helpers.
- [x] Extract the research controls into a focused section component. Validation: `cd ui && npx tsc --noEmit` on 2026-03-13.
- [x] Extract the source input block into a focused section component. Validation: `cd ui && npx tsc --noEmit` on 2026-03-13.
- [x] Extract the crop/transform geometry controls into a focused section component. Validation: `cd ui && npx tsc --noEmit` on 2026-03-13.
- [x] Extract the color/fps/output post-processing controls into a focused section component. Validation: `cd ui && npx tsc --noEmit` on 2026-03-13.
- [x] Extract the footer action controls into a focused section component. Validation: `cd ui && npx tsc --noEmit` on 2026-03-13.
- [x] Split the panel into smaller component sections with clear ownership. Validation: `cd ui && npx tsc --noEmit` on 2026-03-13.
- [x] Keep current UI behavior and visual language unchanged. Validation: `cd ui && npx tsc --noEmit` on 2026-03-13 after removing the stale inline fallback block.

Files in scope:
- `ui/src/components/InputOutputPanel.tsx`
- new files under `ui/src/components/` and/or `ui/src/utils/`

Changes:
- Move research config schema/defaults into a shared module.
- Move helper functions and potentially icon-only helpers out of the main panel file where it improves readability.
- Break the panel into smaller focused sections such as input selection, output settings, edit controls, and research controls.

Acceptance criteria:
- `InputOutputPanel.tsx` becomes materially smaller.
- Repeated config/helper logic is moved into shared modules.
- No user-visible regression in the panel flow.

Validation:
- `cd ui && npx tsc --noEmit`
- manual spot-check of input/output selection, model selection, and research controls

### Track 3: Refactor Python Upscale Orchestration And Video Pipeline Config

Status: complete (2026-03-13)

- [x] Replace broad argument lists in `video_pipeline.rs` with typed config/input structs where practical. Validation: `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings` and `cd src-tauri && cargo test --workspace` on 2026-03-13.
- [x] Reduce `too_many_arguments` suppressions in `upscale.rs` and `video_pipeline.rs`. Validation: `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings` on 2026-03-13; only the intentional flat Tauri IPC command signature remains suppressed.
- [x] Separate pipeline config shaping from execution orchestration. Validation: `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings` and `cd src-tauri && cargo test --workspace` on 2026-03-13.
- [x] Keep Tauri command surface behavior unchanged. Validation: `cd src-tauri && cargo test --workspace` on 2026-03-13.

Files in scope:
- `src-tauri/src/commands/upscale.rs`
- `src-tauri/src/video_pipeline.rs`

Changes:
- Introduce focused config structs for decode/encode/pipeline setup.
- Reduce orchestration sprawl in `run_upscale_job`.
- Keep the current DTO and user-facing command behavior stable.

Acceptance criteria:
- Fewer broad helper signatures.
- Clearer ownership between Tauri command boundary, worker orchestration, and FFmpeg pipeline setup.
- No regression in Python-path tests.

Validation:
- `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings`
- `cd src-tauri && cargo test --workspace`

### Track 4: Validation And Contract Hardening

Status: in progress (2026-03-13)

- [x] Add/update tests around newly extracted native module boundaries. Added focused tests for native probe/output shaping and streaming mux helpers on 2026-03-13; native-feature execution remains blocked by an existing `--features native_engine` compile failure outside these new tests.
- [x] Add/update tests around smoke-mode parsing structure if needed. Added direct smoke argument parsing coverage on 2026-03-13. Validation: `cd src-tauri && cargo test --workspace` on 2026-03-13.
- [ ] Confirm the maintained validation matrix reflects actual expectations. Default matrix is green on 2026-03-13, but opt-in native-feature validation currently fails to compile before tests run.

Files in scope:
- extracted Rust modules from Tracks 1 and 3
- `src-tauri/src/bin/smoke.rs`
- relevant test modules/docs

Changes:
- Add targeted tests around extraction seams rather than only end-to-end flows.
- Keep this track focused on coverage and maintenance hardening, not major refactoring.

Acceptance criteria:
- Extracted modules have direct tests where the seam is easy to regress.
- Validation commands are documented and still accurate.

Validation:
- `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings`
- `cd src-tauri && cargo test --workspace`
- `cd ui && npx tsc --noEmit`
- Optional native-feature follow-up (currently blocked by existing compile errors outside Track 4 changes):
  - `cd src-tauri && cargo test --features native_engine <targeted test>`

## 5. Suggested Order

1. Track 0: Restore strict lint baseline
2. Track 1: Extract native direct pipeline modules
3. Track 2: Decompose `InputOutputPanel.tsx`
4. Track 3: Refactor Python upscale orchestration and video pipeline config
5. Track 4: Validation and contract hardening

## 6. Progress Log

- [x] Track 0 complete
- [x] Track 1 complete
- [x] Track 2 complete
- [x] Track 3 complete
- [ ] Track 4 complete

## 7. Working Validation Matrix

- Rust lint:
  - `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings`
- Rust tests:
  - `cd src-tauri && cargo test --workspace`
- UI typecheck:
  - `cd ui && npx tsc --noEmit`

## 8. Definition Of Done

- Strict Rust linting is green again.
- Remaining oversized orchestration files have been reduced into clearer module boundaries.
- The largest remaining UI hotspot has been decomposed into smaller maintainable pieces.
- Validation remains green after each track, and this file is kept up to date as work lands.
