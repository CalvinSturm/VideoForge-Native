# Native Engine Handoff

Date: 2026-03-07

## Current Status

The native-direct and native-cli family now share a materially larger control plane in `src-tauri/src/commands/native_engine.rs`.

What is now unified:

- one shared native job contract: `NativeJobSpec`
- one shared result contract: `NativeUpscaleResult`
- one shared perf contract: `NativePerfReport`
- one shared executor routing/fallback contract
- one shared native runtime override/env guard path
- one shared runtime path resolver for FFmpeg/ffprobe/native deps
- one shared tool request layer for benchmark and smoke
- one shared result/presentation layer for benchmark and smoke output

What still diverges:

- direct native execution still owns in-process decode/infer/encode behavior
- CLI-native still owns subprocess execution via `rave`
- lower-level executor behavior is still different by design

This work intentionally did not touch Python-path performance or architecture.

## Commits Landed In This Refactor Sequence

- `b8b01a7` `Unify native direct and CLI job contracts`
- `20b8ac9` `Add native route and fallback reporting`
- `512e62a` `Unify native runtime and CLI prep`
- `dade877` `Consolidate native execution planning`
- `6c8f37c` `Unify native direct launch profiles`
- `0954f19` `Share native tool execution path`
- `6bd9368` `Share native result formatting helpers`
- `66f7497` `Share native tool request builders`
- `9aa16f1` `Share native benchmark policy helpers`
- `e8e032a` `Share native smoke and bench helpers`

## Key Code Changes

### Shared job/result/perf contracts

- `src-tauri/src/commands/native_engine.rs`
  - added shared `NativeJobSpec`
  - centralized validation/policy above executor selection
  - direct native and CLI-native now both shape `NativeUpscaleResult`
  - `NativePerfReport` now carries shared route/fallback fields and shared perf fields

### Shared executor routing and fallback

- `src-tauri/src/commands/native_engine.rs`
  - added centralized native executor routing helpers
  - direct success, CLI-only success, and direct-fallback-to-CLI now all map through the same route/result layer
  - fallback behavior remained intact

### Shared runtime control plane

- `src-tauri/src/commands/native_engine.rs`
  - added `NativeRuntimeOverrides`
  - added `NativeRuntimeEnvGuard`
  - added shared runtime path discovery helpers
  - direct native now resolves and uses the discovered `ffprobe` path instead of assuming `"ffprobe"`

- `src-tauri/src/rave_cli.rs`
  - CLI-native PATH/runtime setup now reuses shared runtime path resolution

- `src-tauri/src/commands/rave.rs`
  - CLI preparation moved toward prepared-command execution instead of rebuilding policy inside the Tauri wrapper

### Shared direct launch-policy objects

- `src-tauri/src/commands/native_engine.rs`
  - added `NativeVideoSourceProfile`
  - added `NativeVideoOutputProfile`
  - direct-native launch policy now flows through source/output descriptors instead of loose width/height/fps shaping

### Shared benchmark and smoke tool contracts

- `src-tauri/src/commands/native_engine.rs`
  - added `NativeToolRunRequest`
  - added `run_native_tool_request(...)`
  - added shared tool request helpers for:
    - default TRT cache policy
    - warmup output naming
    - runtime filesystem prep
    - route labeling
    - result summary JSON
    - result summary lines
    - benchmark event payloads
    - smoke banner/success lines

- `src-tauri/src/bin/videoforge_bench.rs`
  - now delegates native request building and native result/event shaping to shared helpers

- `src-tauri/src/bin/smoke.rs`
  - now delegates native banner/success formatting to shared helpers

## Current Architecture Snapshot

For native-family video execution today:

- app/UI path:
  - request -> `upscale_request_native(...)` -> `NativeJobSpec::resolve(...)` -> executor selection -> direct native or CLI-native -> shared result/perf contract

- tooling path:
  - benchmark/smoke -> `NativeToolRunRequest` -> shared runtime/cache/output prep -> `run_native_tool_request(...)` -> `upscale_request_native(...)` -> shared result/perf/presentation helpers

This is much better than the previous state where app, benchmark, smoke, and CLI wrappers each carried their own partial native routing and shaping logic.

## Validation Performed

Repeatedly passed during this sequence:

- `cargo check --features native_engine`
- `cargo test --features native_engine progress_summary --lib`
- `cargo test --features native_engine native_runtime_env_guard_restores_previous_values --lib`
- `cargo test --features native_engine max_batch_allows_one_through_eight --lib`
- `cargo test --features native_engine prepared_upscale_command_injects_profile_once --lib`
- `cargo test --features native_engine runtime_path_resolver_keeps_cli_bin_dir --lib`
- `cargo test --features native_engine source_profile_derives_expected_output_profile --lib`
- `cargo test --features native_engine tool_request_runtime_overrides_follow_cache_presence --lib`
- `cargo test --features native_engine tool_request_builder_normalizes_optional_output_and_route_label --lib`
- `cargo test --features native_engine tool_request_prepare_runtime_filesystem_creates_cache_dir --lib`
- `cargo test --features native_engine native_benchmark_done_payload_uses_shared_request_fields --lib`
- `cargo test --features native_engine native_tool_helpers_shape_smoke_output --lib`

Notes:

- `ui` typecheck was intentionally not part of this refactor sequence.
- Existing warnings remain in generated/native code and some unused direct-path scaffolding, but no new compile blockers were introduced.

## Repository State At Handoff

Tracked working tree should be clean after `e8e032a`.

Untracked items expected at handoff time:

- cross-engine audit/planning markdown files
- `artifacts/`

Those audit/planning docs were intentionally excluded from the native refactor commits at the time of handoff.

## Strongest Current Architectural Conclusion

The direct-native vs native-cli split should remain an execution-detail split, not a control-plane split.

The work above moved the codebase closer to that model:

- policy is increasingly shared
- tooling is increasingly shared
- result/perf/route reporting is shared

What remains should be handled the same way: keep different executors, but keep one contract layer above them.

## Best Next Work

### 1. Unify remaining native tool output policy

Goal:

- remove any remaining benchmark/smoke-local naming or presentation drift that is still not covered by shared helpers

Likely files:

- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/bin/videoforge_bench.rs`
- `src-tauri/src/bin/smoke.rs`

Why next:

- low-risk continuation of the current direction
- keeps tooling aligned with app-native behavior

### 2. Introduce a shared native video job descriptor below `NativeJobSpec`

Goal:

- make direct-native launch assumptions explicit in a reusable descriptor, rather than leaving probe/output/runtime details spread across direct-path setup

Likely scope:

- move more direct launch setup behind prepared plan/descriptor objects
- keep executor-specific behavior separate, but share the launch contract

Likely files:

- `src-tauri/src/commands/native_engine.rs`

Why next:

- this is the remaining structural gap below the already-shared job/result layer
- it will make future benchmarking and future adapters less drift-prone

### 3. Normalize benchmarkability without expanding telemetry scope too broadly

Goal:

- keep using `NativePerfReport` as the single schema
- improve comparison quality between direct and CLI while preserving current routing/behavior

Likely scope:

- ensure direct and CLI both emit comparable route/perf/cache fields everywhere native tooling consumes them

Likely files:

- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/commands/rave.rs`
- `src-tauri/src/rave_cli.rs`
- `src-tauri/src/bin/videoforge_bench.rs`

Why next:

- the schema is now shared, so comparison improvements no longer require more contract churn

## Recommended Guardrails For Next Codex Instance

- Do not touch the Python engine unless the task explicitly changes.
- Keep using `apply_patch` for file edits.
- Keep commits scoped to the native family only.
- Do not include the root audit markdown files or `artifacts/` in commits.
- Preserve current direct-to-CLI fallback behavior unless the task explicitly changes routing semantics.
- Prefer adding shared helpers in `src-tauri/src/commands/native_engine.rs` over duplicating logic in `smoke` or `videoforge_bench`.

## Useful Files

- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/commands/rave.rs`
- `src-tauri/src/rave_cli.rs`
- `src-tauri/src/bin/videoforge_bench.rs`
- `src-tauri/src/bin/smoke.rs`
- `engine-v2/src/lib.rs`
- `engine-v2/src/engine/pipeline.rs`

## Short Resume Prompt For The Next Codex Instance

Resume the native-only direct-native/native-cli control-plane consolidation from commit `e8e032a`.

Assume:

- shared job/result/perf contracts already exist
- runtime override/path discovery is already shared
- benchmark and smoke already use shared native helper paths for most request/result formatting

Continue by removing the remaining native tool/output/setup drift without changing Python behavior or breaking current direct-to-CLI fallback semantics.
