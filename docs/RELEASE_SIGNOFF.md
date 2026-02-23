# Release Signoff Record

Use this file as a per-release checklist + evidence log.

## Release

- Version/tag: pending
- Branch: `main`
- Commit SHA: `e06486de40c9ac6cb10fc620a8aff2bbd3c83517`
- Date (UTC): `2026-02-23T02:25:09Z`
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

- [x] `validate --json` includes `policy` + host-copy audit fields
- [x] `benchmark --json` includes `policy` + stage timing fields
- [x] `upscale --json` includes `policy`
- [x] stdout-only JSON contract confirmed (stderr not parsed by consumers)
- [x] `max_batch > 1` fail-fast behavior verified

## Defect Gate

- [ ] No open P0 defects
- [ ] No open P1 defects
- [ ] Any accepted waivers documented below

## Notes / Waivers

- Local signoff script run passed at `2026-02-23T01:56:43Z`:
  - `pwsh -NoProfile -File tools/ci/run_release_signoff_windows.ps1`
- Local signoff script run passed at `2026-02-23T02:25:09Z`:
  - `pwsh -NoProfile -File tools/ci/run_release_signoff_windows.ps1`
- RAVE JSON contract checks are enforced in CI via:
  - `tools/ci/check_rave_json_contract.ps1`
  - `.github/workflows/ci.yml` (mock + GPU lanes)
