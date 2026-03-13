# Cleanup Follow-up Handoff

Date: 2026-03-13
Status: active
Source tracker: `docs/plans/post_cleanup_followup_plan_2026-03-13.md`

## Purpose

This handoff captures the complete post-cleanup implementation plan and the current execution status so work can continue without reconstructing context from terminal history.

## Current Summary

- Track 0 is complete.
- Track 1 is complete.
- Track 2 is in progress.
- Track 3 has not started.
- Track 4 has not started.

## Validation Baseline

Latest confirmed green commands:

- `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings`
- `cd src-tauri && cargo test --workspace`
- `cd ui && npx tsc --noEmit`

## Complete Plan And Progress

### Track 0: Restore Strict Lint Baseline

Status: complete

Goals:

- Restore strict Rust lint hygiene.
- Reconfirm the Rust test baseline after lint fixes.

Planned items:

- [x] Fix current `clippy -D warnings` failures.
- [x] Re-run strict lint validation and confirm clean output.
- [x] Add the lint command to the working validation checklist in the plan.

Implemented:

- Fixed the strict `clippy` regression in `src-tauri/src/commands/native_engine.rs`.
- Moved the `smoke.rs` test module to the end of the file to satisfy `clippy::items-after-test-module`.

Validation:

- [x] `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cd src-tauri && cargo test --workspace`

### Track 1: Extract Native Direct Pipeline Modules

Status: complete

Goals:

- Reduce `src-tauri/src/commands/native_engine.rs` into a more command-focused module.
- Move probing, streaming IO, and direct pipeline implementation into focused modules.

Planned items:

- [x] Extract source probing/profile types from `native_engine.rs`.
- [x] Extract FFmpeg bitstream source and mux sink wrappers from `native_engine.rs`.
- [x] Extract direct engine-v2 pipeline runner into a focused module.
- [x] Keep command boundary and route-selection behavior unchanged.

Implemented:

- Added `src-tauri/src/commands/native_probe.rs`.
- Added `src-tauri/src/commands/native_streaming_io.rs`.
- Added `src-tauri/src/commands/native_direct_pipeline.rs`.
- Updated `src-tauri/src/commands/mod.rs` to register the new modules.
- Rewired `src-tauri/src/commands/native_routing.rs` to call the extracted direct pipeline module.
- Removed the duplicated direct pipeline and encoder-wrapper implementation from `src-tauri/src/commands/native_engine.rs`.
- Kept `upscale_request_native` and route selection behavior stable.

Validation:

- [x] `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cd src-tauri && cargo test --workspace`

### Track 2: Decompose InputOutputPanel

Status: in progress

Goals:

- Reduce `ui/src/components/InputOutputPanel.tsx`.
- Move static config and helpers out first, then split render sections into smaller components.
- Preserve current UI behavior and visual language.

Planned items:

- [x] Extract research config types/defaults/presets from `InputOutputPanel.tsx`.
- [x] Extract model display/helpers and reusable small UI helpers.
- [ ] Split the panel into smaller component sections with clear ownership.
- [ ] Keep current UI behavior and visual language unchanged.

Implemented so far:

- Added `ui/src/components/inputOutputPanel/researchConfig.ts`.
- Added `ui/src/components/inputOutputPanel/panelHelpers.ts`.
- Added `ui/src/components/inputOutputPanel/panelIcons.tsx`.
- Updated `ui/src/components/InputOutputPanel.tsx` to import those modules instead of owning the static definitions inline.
- Reduced `InputOutputPanel.tsx` from 2321 lines to 2214 lines in this first pass.

Validation:

- [x] `cd ui && npx tsc --noEmit`

Next concrete step:

- Extract the largest render sections out of `InputOutputPanel.tsx`, starting with research controls and then the input/output or edit-control sections.

### Track 3: Refactor Python Upscale Orchestration And Video Pipeline Config

Status: not started

Goals:

- Reduce orchestration sprawl in the Python path.
- Replace broad argument lists with focused config/input structs where practical.
- Keep the Tauri command surface stable.

Planned items:

- [ ] Replace broad argument lists in `video_pipeline.rs` with typed config/input structs where practical.
- [ ] Reduce `too_many_arguments` suppressions in `upscale.rs` and `video_pipeline.rs`.
- [ ] Separate pipeline config shaping from execution orchestration.
- [ ] Keep Tauri command surface behavior unchanged.

Target files:

- `src-tauri/src/commands/upscale.rs`
- `src-tauri/src/video_pipeline.rs`

Validation target:

- `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings`
- `cd src-tauri && cargo test --workspace`

### Track 4: Validation And Contract Hardening

Status: not started

Goals:

- Add targeted coverage around the new extraction seams.
- Confirm the validation matrix reflects the actual maintained contract.

Planned items:

- [ ] Add/update tests around newly extracted native module boundaries.
- [ ] Add/update tests around smoke-mode parsing structure if needed.
- [ ] Confirm the maintained validation matrix reflects actual expectations.

Target files:

- extracted Rust modules from Tracks 1 and 3
- `src-tauri/src/bin/smoke.rs`
- relevant docs/test modules

Validation target:

- `cd src-tauri && cargo clippy --workspace --all-targets -- -D warnings`
- `cd src-tauri && cargo test --workspace`
- `cd ui && npx tsc --noEmit`

## Recommended Resume Point

Resume with Track 2.

Recommended sequence:

1. Extract the research controls block from `ui/src/components/InputOutputPanel.tsx` into a focused subcomponent.
2. Re-run `cd ui && npx tsc --noEmit`.
3. Continue splitting the next largest render section while preserving props and behavior.
4. Mark Track 2 complete in `docs/plans/post_cleanup_followup_plan_2026-03-13.md` once the panel is materially decomposed.

## Files Most Recently Touched In This Follow-up Phase

- `docs/plans/post_cleanup_followup_plan_2026-03-13.md`
- `docs/handoff_cleanup_followup_status_2026-03-13.md`
- `src-tauri/src/commands/mod.rs`
- `src-tauri/src/commands/native_direct_pipeline.rs`
- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/commands/native_probe.rs`
- `src-tauri/src/commands/native_routing.rs`
- `src-tauri/src/commands/native_streaming_io.rs`
- `ui/src/components/InputOutputPanel.tsx`
- `ui/src/components/inputOutputPanel/panelHelpers.ts`
- `ui/src/components/inputOutputPanel/panelIcons.tsx`
- `ui/src/components/inputOutputPanel/researchConfig.ts`

## Source Of Truth

Keep updating:

- `docs/plans/post_cleanup_followup_plan_2026-03-13.md`

Use this handoff file as a snapshot for transfer, not as the long-term tracker.
