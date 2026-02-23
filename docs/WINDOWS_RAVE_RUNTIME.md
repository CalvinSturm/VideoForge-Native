# Windows RAVE Runtime Guide

This guide is for running VideoForge + `rave-*` integration on Windows.

## Scope

- Host OS: Windows (PowerShell)
- VideoForge repo root: `C:\Users\Calvin\Desktop\VideoForge1`
- RAVE workspace path: `third_party\rave`

## 1) Build `rave` Binary (Recommended)

From repo root:

```powershell
cargo build --manifest-path third_party\rave\Cargo.toml -p rave-cli --release
```

Expected binary path:

- `third_party\rave\target\release\rave.exe`

VideoForge backend now prefers this workspace target path automatically.

## 2) Profile Policy Control

VideoForge backend profile defaults:

- Debug builds: `dev`
- Release builds: `production_strict`

Optional override (current shell only):

```powershell
$env:VIDEOFORGE_RAVE_PROFILE = "dev"
# or
$env:VIDEOFORGE_RAVE_PROFILE = "production_strict"
```

Persistent user override:

```powershell
setx VIDEOFORGE_RAVE_PROFILE "production_strict"
```

Note: `setx` applies to new shells only.

## 3) Strict Validate (Mock, Non-GPU Safe)

```powershell
$env:RAVE_MOCK_RUN = "1"
cargo run --manifest-path third_party\rave\Cargo.toml -p rave-cli --features audit-no-host-copies --bin rave -- validate --profile production_strict --json --best-effort
```

This path validates strict-policy mapping without requiring live CUDA initialization.

## 4) Direct CLI Commands

### Upscale

```powershell
cargo run --manifest-path third_party\rave\Cargo.toml -p rave-cli --bin rave -- upscale --json -i <input.mp4> -m <model.onnx> -o <output.mp4> --profile production_strict
```

### Benchmark

```powershell
cargo run --manifest-path third_party\rave\Cargo.toml -p rave-cli --bin rave -- benchmark --json -i <input.mp4> -m <model.onnx> --profile production_strict
```

### Validate with optional fixture

```powershell
cargo run --manifest-path third_party\rave\Cargo.toml -p rave-cli --bin rave -- validate --json --profile production_strict --fixture <fixture.json>
```

## 5) Tauri/UI Integration Notes

- Native ONNX video path in UI is wired through `rave_upscale`.
- Queue panel shows policy/audit metadata when present.
- Validate button in UI runs strict validate in mock mode.

## 6) Troubleshooting

### A) `rave` not found / spawn failures

- Build release binary first (Section 1).
- Confirm file exists:

```powershell
Test-Path third_party\rave\target\release\rave.exe
```

### B) Cargo workspace error (`failed to find a workspace root`)

Use the workspace manifest, not crate-local manifest:

```powershell
--manifest-path third_party\rave\Cargo.toml
```

### C) crates.io/network fetch failures

If network is blocked in your environment, Cargo cannot resolve dependencies.
Use a network-enabled shell/runner for first dependency resolution.

### D) Strict validate fails due to host-copy audit

`production_strict` expects audit capability when strict no-host-copies is enforced.
Use:

```powershell
--features audit-no-host-copies
```

### E) CUDA/driver/provider runtime errors

- Verify NVIDIA driver is installed and compatible.
- Verify required runtime libraries are available in process environment.
- Prefer actionable stderr diagnostics from `rave-cli`; parse JSON from stdout only.

### F) Error category meanings (UI/Logs)

VideoForge now tags many failures with a stable category.

- `policy_violation`: strict profile policy failed (for example strict no-host-copies audit requirement).
- `provider_loader_error`: runtime/provider loader or initialization failed (CUDA/ORT/TensorRT/driver path).
- `runtime_dependency_missing`: required dependency is missing/not discoverable.
- `input_contract_error`: unsupported/invalid input contract (for example `max_batch > 1`) or JSON contract mismatch.

Suggested operator response:

1. `policy_violation`: re-run with required strict features/capabilities enabled; verify profile intent.
2. `provider_loader_error`: verify driver/runtime installation and provider library resolution.
3. `runtime_dependency_missing`: install missing binary/library/tool and re-run.
4. `input_contract_error`: fix CLI/input arguments; keep `max_batch` at `1` until micro-batching is implemented.

## 7) Quick PowerShell Session Template

```powershell
cd C:\Users\Calvin\Desktop\VideoForge1
$env:VIDEOFORGE_RAVE_PROFILE = "production_strict"
$env:RAVE_MOCK_RUN = "1"

cargo build --manifest-path third_party\rave\Cargo.toml -p rave-cli --release
cargo run --manifest-path third_party\rave\Cargo.toml -p rave-cli --features audit-no-host-copies --bin rave -- validate --profile production_strict --json --best-effort
```

## 8) GPU Benchmark Threshold Gate (CI)

CI artifacts include `rave_benchmark_gpu.stdout.json` and a threshold comparison report.

Repo variables:

- `ENABLE_GPU_CI=1` enables the GPU job.
- `ENABLE_GPU_THRESHOLD_GATE=1` turns threshold violations into CI failures.

Baseline file:

- `tools/ci/rave_benchmark_baseline.windows-gpu.json`

Comparator script:

- `tools/ci/check_rave_benchmark_thresholds.ps1`
- `tools/ci/check_rave_json_contract.ps1` (schema/required-field contract checks for validate/benchmark/upscale artifacts)
- `tools/ci/calibrate_rave_gpu_baseline.ps1` (recompute baseline medians from collected benchmark artifacts)

Recommended rollout:

1. Run GPU CI with threshold gate disabled.
2. Inspect benchmark artifacts over several runs.
3. Tune baseline/threshold values for your runner class.
4. Enable `ENABLE_GPU_THRESHOLD_GATE=1`.

Reference checklist:

- `docs/CI_GPU_STABILIZATION_CHECKLIST.md`
