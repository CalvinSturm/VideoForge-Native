# VideoForge Handoff (Canonical)

Updated: 2026-02-28
Branch: `main`
Status: Native RAVE engine path is working on current `origin/main` (`66ac16d`).

## What Is Done

Recent commits that established the current working state:

1. `9463f00` (2026-02-27): NVENC register/map parameter compatibility fixes.
2. `52603de` (2026-02-27): Restored native runtime launch and fixed Windows FFmpeg build wiring.
3. `8a5222f` (2026-02-28): Finalized native batch path and runtime diagnostics cleanup.
4. `66ac16d` (2026-02-28): Added native perf gate and hardened NVDEC decode copy/event handling.

## Regression Guardrails

Treat these as required checks before/after any native engine changes:

1. `cargo check --manifest-path src-tauri/Cargo.toml`
2. Native smoke path and perf gate in CI:
- `tools/ci/check_native_smoke_perf.ps1`
- See [SMOKE_TESTS.md](/C:/Users/Calvin/Desktop/VideoForge1/docs/SMOKE_TESTS.md)
3. Confirm no regressions in:
- `engine-v2/src/codecs/nvdec.rs`
- `engine-v2/src/codecs/nvenc.rs`
- `engine-v2/src/engine/pipeline.rs`
- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/bin/smoke.rs`

## Known Risk Areas

1. CUDA context ownership across threads in encode path.
2. NVDEC event ordering and decode-copy synchronization.
3. Runtime DLL/provider ordering on Windows (FFmpeg/TensorRT search paths).

## Working Agreement To Avoid Regressions

1. Keep this file as the single source of truth for handoff status.
2. Do not duplicate handoff status in multiple active files.
3. If a crash is fixed, update this file with:
- exact commit
- exact verification command
- exact residual risk (if any)

## Archived Notes

Older investigation notes were removed from the repository and archived externally.
