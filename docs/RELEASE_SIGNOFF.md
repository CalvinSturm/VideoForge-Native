# Release Signoff Record

Use this file as a per-release checklist + evidence log.

## Release

- Version/tag: pending
- Branch: `main`
- Commit SHA: `d0120d7710b8df9156907991651f3d7649e26557`
- Date (UTC): `2026-02-23T01:56:43Z`
- Owner: Calvin

## Required Checks

- [x] `cargo fmt --all -- --check` (from `src-tauri`)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` (from `src-tauri`)
- [x] `cargo test` (from `src-tauri`)
- [x] `./scripts/check_deps.sh` (if present in repo) - skipped, file not present
- [x] `./scripts/check_docs.sh` (if present in repo) - skipped, file not present

Windows helper:

```powershell
pwsh -NoProfile -File tools/ci/run_release_signoff_windows.ps1
```

## GPU CI Status

- [ ] `ENABLE_GPU_CI=1` is set.
- [ ] `gpu-rave` passed 3 consecutive runs.
- [ ] Benchmark baseline calibrated from recent artifacts.
- [ ] `ENABLE_GPU_THRESHOLD_GATE=1` is set and verified.

## Branch Protection

- [ ] Required status check: `Rust checks`
- [ ] Required status check: `GPU RAVE checks` (release branches)
- [ ] PR review required before merge
- [ ] Direct pushes to protected branches disabled

## RAVE Contract Validation

- [ ] `validate --json` includes `policy` + host-copy audit fields
- [ ] `benchmark --json` includes `policy` + stage timing fields
- [ ] `upscale --json` includes `policy`
- [ ] stdout-only JSON contract confirmed (stderr not parsed by consumers)
- [ ] `max_batch > 1` fail-fast behavior verified

## Defect Gate

- [ ] No open P0 defects
- [ ] No open P1 defects
- [ ] Any accepted waivers documented below

## Notes / Waivers

- Local signoff script run passed at `2026-02-23T01:56:43Z`:
  - `pwsh -NoProfile -File tools/ci/run_release_signoff_windows.ps1`
