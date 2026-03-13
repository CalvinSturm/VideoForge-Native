# Codebase Cleanup Implementation Plan

Status: Planning snapshot
Created: 2026-03-13
Scope: UI contract cleanup, Rust module cleanup, metadata alignment, and type/utility consolidation
Use this for: Sequencing cleanup work into reviewable, low-risk steps
Do not use this for: Current shipped status without checking the latest code and validation results

## 1. Objectives
- Reduce duplicated logic and drift across the UI, Tauri boundary, and native runtime wrapper.
- Improve reviewability by shrinking oversized files and isolating responsibilities.
- Tighten type contracts so error handling, job updates, and Tauri DTOs stop relying on weak `string`/inline shapes.
- Keep each cleanup step independently shippable with clear validation.

## 2. Current Cleanup Findings
- `[App.tsx](/C:/Users/Calvin/Desktop/VideoForge1/ui/src/App.tsx)` was carrying hook-owned logic; that pattern likely still exists in smaller pockets across the UI.
- `[useUpscaleJob.ts](/C:/Users/Calvin/Desktop/VideoForge1/ui/src/hooks/useUpscaleJob.ts)` still owns general-purpose model/runtime helpers that should live in shared utilities.
- UI job completion, error normalization, log writes, and toast handling are repeated in slightly different forms across hooks and event handlers.
- `[native_engine.rs](/C:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/commands/native_engine.rs)` still combines runtime discovery, env setup, ffmpeg wrappers, execution routing, fallback policy, and tests in one large module.
- `[smoke.rs](/C:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/bin/smoke.rs)` still uses a broad flat args struct and mixes multiple execution modes in one file.
- Tauri request/response payloads are not fully centralized as a single typed contract on both the TS and Rust sides.
- Some UI callbacks still use broad `(msg: string, type: string)` signatures where the valid domain is already finite.

## 3. Cleanup Principles
- Keep refactors behavior-preserving unless a task explicitly changes behavior.
- Prefer extraction and unification over local patching.
- Land contract cleanup before deeper optimization or larger architecture changes.
- Keep PRs narrow enough that a reviewer can validate intent from one file group.

## 4. Priority-ranked Implementation Plan

### PR 1: Extract Shared UI Domain Helpers
**Goal:** Move model/runtime helper logic out of hook-local scope into shared utilities.

**Files:**
- `ui/src/hooks/useUpscaleJob.ts`
- `ui/src/hooks/useRaveIntegration.ts`
- `ui/src/types.ts`
- New utility module under `ui/src/lib/` or `ui/src/utils/`

**Changes:**
- Extract `getScaleFromModel`.
- Extract `inferNativePrecision`.
- Extract any pure payload-shaping helpers that do not require hook state.
- Keep hook files focused on orchestration rather than domain parsing.

**Acceptance criteria:**
- No model/runtime parsing helpers remain private to `useUpscaleJob` unless they truly require hook-local state.
- Helper tests or type-safe call sites exist for the extracted functions.
- Hook files shrink and become easier to review.

**Validation:**
- `cd ui && npx tsc --noEmit`

**Risk notes:**
- Low risk if extraction remains behavior-preserving.

### PR 2: Unify UI Job/Toast/Error Utilities
**Goal:** Stop repeating the same job-finalization and error-reporting shapes across the UI.

**Files:**
- `ui/src/hooks/useUpscaleJob.ts`
- `ui/src/hooks/useTauriEvents.ts`
- `ui/src/Store/useJobStore.tsx`
- `ui/src/types.ts`
- New utility module under `ui/src/lib/` or `ui/src/utils/`

**Changes:**
- Introduce shared helpers for:
  - marking a job as done
  - marking a job as failed
  - applying structured error metadata
  - appending standardized log lines
  - normalizing toast payloads
- Remove repeated inline object spreads that construct job success/error updates.

**Acceptance criteria:**
- Success/error job state shaping is centralized.
- Log/toast formatting rules are consistent between hook-driven actions and event-driven updates.
- UI state updates are easier to diff and audit.

**Validation:**
- `cd ui && npx tsc --noEmit`
- Manual spot-check of one upscale path, one export path, and one failure path

**Risk notes:**
- Medium risk if shared helpers over-generalize different job lifecycles.

### PR 3: Tighten UI Types And Callback Contracts
**Goal:** Replace weak UI callback signatures and reduce `as any` / ad hoc coercion.

**Files:**
- `ui/src/types.ts`
- `ui/src/hooks/useTauriEvents.ts`
- `ui/src/hooks/useUpscaleJob.ts`
- `ui/src/App.tsx`
- Affected components consuming toast/job/event callbacks

**Changes:**
- Introduce explicit reusable types for toast level, log event category, and job mutation helpers.
- Replace `(msg: string, type: string)` with a narrower shared type where possible.
- Remove `as any` and unnecessary `as const` where type definitions can carry the intent.
- Tighten event payload parsing around known Tauri event shapes.

**Acceptance criteria:**
- UI code no longer needs broad stringly-typed callback interfaces for finite domains.
- Type assertions decrease materially in the touched files.
- Event and toast call sites become self-documenting.

**Validation:**
- `cd ui && npx tsc --noEmit`

**Risk notes:**
- Low to medium risk depending on how many components currently accept loose callback types.

### PR 4: Centralize Tauri DTO Contracts
**Goal:** Give Tauri request/response payloads one clear source of truth on the UI side and a clearer matching shape on the Rust side.

**Files:**
- `ui/src/types.ts`
- `ui/src/hooks/useUpscaleJob.ts`
- `src-tauri/src/commands/upscale.rs`
- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/lib.rs`
- Potential new shared Rust DTO module under `src-tauri/src/`

**Changes:**
- Group UI-facing Tauri DTOs by command domain instead of leaving them scattered among general UI types.
- Introduce or improve Rust-side DTO structs used at the command boundary.
- Reduce inline request/result object shapes in the UI.
- Document command-boundary ownership for each DTO family.

**Acceptance criteria:**
- Each Tauri command used by the UI has a named TS type and a named Rust boundary struct.
- Fewer inline anonymous payload shapes exist in hooks/components.
- Native and Python command results are easier to compare and evolve safely.

**Validation:**
- `cd ui && npx tsc --noEmit`
- `cd src-tauri && cargo test --workspace`

**Risk notes:**
- Medium risk if there are hidden payload shape mismatches currently masked by loose typing.

### PR 5: Split Native Runtime Discovery And Env Management
**Goal:** Break `[native_engine.rs](/C:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/commands/native_engine.rs)` into smaller units around runtime path discovery and env setup.

**Files:**
- `src-tauri/src/commands/native_engine.rs`
- New modules such as:
  - `src-tauri/src/commands/native_runtime_paths.rs`
  - `src-tauri/src/commands/native_runtime_env.rs`
  - or a nested `native_engine/` module tree

**Changes:**
- Extract path discovery and PATH/DLL staging into focused modules.
- Keep command orchestration separate from runtime bootstrapping.
- Localize tests near the extracted runtime helpers.

**Acceptance criteria:**
- Native runtime discovery no longer lives inline beside command execution and mux/ffmpeg logic.
- Tests for runtime discovery/env mutation are colocated with the extracted logic.
- `native_engine.rs` becomes materially smaller and easier to review.

**Validation:**
- `cd src-tauri && cargo test --workspace`

**Risk notes:**
- Medium risk because runtime setup order matters.

### PR 6: Split Native Tool Execution And Fallback Routing
**Goal:** Isolate direct execution, CLI fallback routing, and ffmpeg boundary helpers from the rest of the native command file.

**Files:**
- `src-tauri/src/commands/native_engine.rs`
- New modules such as:
  - `native_engine/tool_runner.rs`
  - `native_engine/fallback.rs`
  - `native_engine/ffmpeg_bridge.rs`

**Changes:**
- Extract direct-run setup and tool invocation helpers.
- Extract fallback decision logic and reason-code mapping.
- Extract ffmpeg bridge types such as streaming source/sink helpers.

**Acceptance criteria:**
- Fallback policy is readable without scanning unrelated runtime setup code.
- ffmpeg helper types are grouped together.
- The main Tauri command surface becomes a thin orchestration layer.

**Validation:**
- `cd src-tauri && cargo test --workspace`

**Risk notes:**
- Medium risk because extracted modules still share the same execution contracts.

### PR 7: Refactor Smoke Binary Into Mode-oriented Structure
**Goal:** Make `[smoke.rs](/C:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/bin/smoke.rs)` easier to maintain by grouping arguments and mode-specific logic.

**Files:**
- `src-tauri/src/bin/smoke.rs`
- Optional new helper modules under `src-tauri/src/bin/smoke/` if the project prefers that layout

**Changes:**
- Group args by mode:
  - base prerequisites
  - Python IPC
  - SHM roundtrip
  - Python E2E
  - native E2E
- Replace the flat `Args` bag with mode-oriented config structs.
- Move repeated validation helpers closer to the mode that uses them.

**Acceptance criteria:**
- `smoke.rs` reads as a dispatcher over mode-specific checks rather than one long sequence.
- Feature-gated native arguments are localized and clearer.
- Clippy exceptions around long parameter lists are reduced where practical.

**Validation:**
- `cd src-tauri && cargo test --workspace`

**Risk notes:**
- Low to medium risk if refactor stays behavior-preserving.

### PR 8: Metadata And Package Contract Hygiene Sweep
**Goal:** Remove remaining obvious metadata drift and add a lightweight maintenance checklist.

**Files:**
- `package.json`
- `ui/package.json`
- `src-tauri/Cargo.toml`
- `src-tauri/tauri.conf.json`
- `README.md`

**Changes:**
- Verify version alignment strategy and document the intended source of truth.
- Check package naming/product casing for consistency.
- Check README commands and metadata references against the actual repo.
- Remove any remaining placeholder package metadata.

**Acceptance criteria:**
- Repo/package/app metadata no longer disagrees without an intentional reason.
- README package/build references match current reality.
- A small checklist exists for future release hygiene.

**Validation:**
- Manual metadata audit
- `cd ui && npx tsc --noEmit`
- `cd src-tauri && cargo test --workspace`

**Risk notes:**
- Low risk.

## 5. Suggested Execution Order
1. PR 1: Extract shared UI domain helpers
2. PR 2: Unify UI job/toast/error utilities
3. PR 3: Tighten UI types and callback contracts
4. PR 4: Centralize Tauri DTO contracts
5. PR 5: Split native runtime discovery and env management
6. PR 6: Split native tool execution and fallback routing
7. PR 7: Refactor smoke binary into mode-oriented structure
8. PR 8: Metadata and package contract hygiene sweep

## 6. Working Rules For Each Step
- Before starting a PR-sized task, record the exact files in scope.
- Keep behavioral changes out of cleanup PRs unless they are required to complete the refactor safely.
- Run validation after every step, not just at the end of the whole cleanup plan.
- If an extracted helper cannot be named clearly, do not extract it yet.
- If two candidate cleanups touch the same file heavily, sequence them instead of combining them.

## 7. Minimum Validation Matrix
- UI-only cleanup:
  - `cd ui && npx tsc --noEmit`
- Rust-only cleanup:
  - `cd src-tauri && cargo test --workspace`
- Tauri contract cleanup:
  - `cd ui && npx tsc --noEmit`
  - `cd src-tauri && cargo test --workspace`

## 8. Definition Of Done
- Major orchestration files are smaller and more single-purpose.
- Shared helpers/types exist in one place instead of being redefined per hook/component.
- Tauri boundary contracts are named, typed, and easier to evolve safely.
- Validation remains green after each cleanup step.
