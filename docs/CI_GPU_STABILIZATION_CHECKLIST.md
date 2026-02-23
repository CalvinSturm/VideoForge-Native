# CI GPU Stabilization And Release Checklist

Last updated: 2026-02-23

This checklist is the remaining operational hardening for VideoForge + `rave-*` integration.

## 1) GPU Runner Stability

- [ ] Confirm self-hosted GPU runner labels match workflow requirements: `self-hosted`, `windows`, `gpu`.
- [ ] Validate runner has stable NVIDIA driver + CUDA/TensorRT runtime libraries.
- [ ] Set repository variable `ENABLE_GPU_CI=1`.
- [ ] Run `gpu-rave` job successfully at least 3 consecutive times.
- [ ] Confirm artifacts are uploaded on success/failure:
  - `rave_validate_gpu.stdout.json`
  - `rave_benchmark_gpu.stdout.json`
  - `rave_upscale_gpu.stdout.json`
  - `rave_benchmark_threshold_report.json`

## 2) Benchmark Baseline Calibration

- [ ] Collect multiple benchmark artifacts from the same runner class (recommended: 5+ runs).
- [ ] Recompute baseline using:

```powershell
pwsh -NoProfile -File tools/ci/calibrate_rave_gpu_baseline.ps1 `
  -InputGlob "artifacts/runs/*/rave_benchmark_gpu.stdout.json" `
  -OutputJson "tools/ci/rave_benchmark_baseline.windows-gpu.json" `
  -ExistingBaselineJson "tools/ci/rave_benchmark_baseline.windows-gpu.json" `
  -Target "windows-gpu"
```

- [ ] Optional compatibility alias: `tools/ci/update_rave_benchmark_baseline.ps1` (same parameters).

- [ ] Review baseline medians and threshold percentages for false-positive risk.
- [ ] Commit updated baseline JSON.

## 3) Threshold Gate Rollout

- [ ] Keep `ENABLE_GPU_THRESHOLD_GATE` unset or `0` while calibrating.
- [ ] Monitor threshold report output across multiple runs.
- [ ] Set repository variable `ENABLE_GPU_THRESHOLD_GATE=1` when stable.
- [ ] Verify expected behavior:
  - warnings only when gate disabled
  - hard CI failure on regression when gate enabled

## 4) Contract Gate Verification

- [ ] Confirm JSON contract checks pass for all commands:
  - validate (`policy`, host-copy audit fields)
  - benchmark (`fps`, stage timings, `policy`)
  - upscale (`input/output/model/codec/size/elapsed/policy`)
- [ ] Confirm `--json` output remains single final object on stdout.
- [ ] Confirm stderr is used only for logs/progress/diagnostics.

## 5) Branch Protection / Merge Policy

- [ ] Require `Rust checks` status check.
- [ ] Require `GPU RAVE checks` status check for branches targeting release.
- [ ] Require pull request review before merge.
- [ ] Disable direct pushes to protected release branches.

## 6) Release Signoff

- [ ] Run required validation set:
  - `cargo fmt --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test`
  - `./scripts/check_deps.sh`
  - `./scripts/check_docs.sh`
- [ ] Optional Windows helper: `pwsh -NoProfile -File tools/ci/run_release_signoff_windows.ps1`
- [ ] Fill release record: `docs/RELEASE_SIGNOFF.md`
- [ ] Confirm Windows runtime doc is current: `docs/WINDOWS_RAVE_RUNTIME.md`.
- [ ] Confirm roadmap doc reflects status: `docs/VIDEOFORGE_2_WEEK_EXECUTION_PLAN.md`.
- [ ] Confirm no open P0/P1 integration defects.
