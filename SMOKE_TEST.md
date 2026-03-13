# VideoForge Smoke Test Runbook

## Prerequisites

- Windows machine with **NVIDIA GPU** (NVENC-capable — GTX 1050+ / Quadro P400+)
- CUDA Toolkit 12.x installed (matching your driver)
- FFmpeg in PATH (includes ffprobe)
- Python AI environment activated (conda/venv with torch, onnxruntime-gpu, etc.)
- A short test video (e.g. `test_720p.mp4`)

## Quick Launchers

For app-level manual checks from the repo root:

- `run.bat`
  - standard dev launcher
- `run_native_engine.bat`
  - native direct launcher
- `run_native_engine_debug.bat`
  - native direct launcher with startup-focused debug dumps enabled

The debug launcher writes artifacts under:

- `artifacts/nvdec_debug/run_native_engine_debug/`

## 1. Python IPC Handshake (No GPU Required)

```powershell
cargo run --manifest-path src-tauri/Cargo.toml --bin smoke -- \
  --model RCAN_x4 --precision fp32 --timeout 60
```

**Expected output:**

```
[PASS] FFmpeg in PATH
[PASS] FFprobe in PATH
[PASS] Python environment
[PASS] Python spawn
[PASS] Python Zenoh handshake
[PASS] Model load (RCAN_x4)
```

## 2. SHM Roundtrip (GPU Recommended)

```powershell
cargo run --manifest-path src-tauri/Cargo.toml --bin smoke -- \
  --model RCAN_x4 --shm-roundtrip --precision fp32 --timeout 90
```

**Expected output:** All checks above plus:

```
[PASS] SHM created
[PASS] SHM header validated
[PASS] Synthetic frame written → SLOT_READY_FOR_AI
[PASS] process_one_frame sent
[PASS] FRAME_DONE received
[PASS] Output validated (NNN bytes, non-zero)
```

## 3. Python E2E Pipeline (Full FFmpeg Path)

```powershell
cargo run --manifest-path src-tauri/Cargo.toml --bin smoke -- \
  --e2e-python --input test_720p.mp4 --e2e-model RCAN_x4 \
  --e2e-scale 1 --precision fp32 --timeout-ms 600000 --keep-temp
```

**Validates:** Input probing → job execution → output dimensions/duration → cleanup.

## 4. Native Engine E2E (GPU Required, feature-gated)

Native smoke requires:

- `native_engine` Cargo feature enabled for the smoke binary
- an ONNX model path passed via `--e2e-onnx`

Optional direct-route request:

- add `--native-direct` to force `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`

```powershell
cargo run --manifest-path src-tauri/Cargo.toml --bin smoke \
  --features native_engine -- \
  --e2e-native --input test_720p.mp4 \
  --e2e-onnx path/to/model.onnx --e2e-scale 4 \
  --precision fp32 --keep-temp
```

Direct-route variant:

```powershell
cargo run --manifest-path src-tauri/Cargo.toml --bin smoke \
  --features native_engine -- \
  --e2e-native --native-direct --input test_720p.mp4 \
  --e2e-onnx path/to/model.onnx --e2e-scale 4 \
  --precision fp32 --keep-temp
```

> **Note:** Requires the `native_engine` Cargo feature. If you see
> `NVENC error code 5 (NV_ENC_ERR_DEVICE_NOT_EXIST)`, verify:
>
> 1. CUDA context is bound to the correct GPU (`nvidia-smi -L`)
> 2. The GPU supports NVENC (check NVIDIA's support matrix)
> 3. Driver version matches CUDA Toolkit version

## 5. GPU CI Runner Activation

The `gpu-rave` CI job in `.github/workflows/ci.yml` is **already configured** but
gated behind the `ENABLE_GPU_CI` repository variable.

**To activate:**

1. Go to **Settings → Variables and secrets → Actions → Variables**
2. Add variable: `ENABLE_GPU_CI` = `1`
3. Register a self-hosted runner with labels: `self-hosted, windows, gpu`
4. Set runner environment variables:
   - `RAVE_GPU_INPUT` — absolute path to test video on the runner
   - `RAVE_GPU_MODEL` — absolute path to ONNX model on the runner
5. (Optional) Set `ENABLE_GPU_THRESHOLD_GATE` = `1` to enforce benchmark regression gates

**CI checks performed automatically:**

- `cargo check --features native_engine`
- RAVE validate (production_strict profile)
- RAVE benchmark (with baseline comparison)
- RAVE upscale E2E (output validation)

## Exit Codes

| Code | Meaning |
|------|---------|
| 0    | All checks passed |
| 1    | One or more checks failed |

## Notes

- This runbook is for current operational commands.
- Historical smoke notes were moved to `docs/archive/SMOKE_TESTS.md`.
- For system-truth docs, start with `docs/README.md` rather than archived handoffs or plans.
- If you also want the run to be ingestible by RunScope, enable `VIDEOFORGE_ENABLE_RUN_ARTIFACTS=1` before running the app or smoke command. VideoForge will then write a `.videoforge_runs/<job_id>/` bundle next to the output, including `videoforge_run.json`.
- After a run finishes, `run_latest_runscope_ingest.bat` will print the newest artifact bundle path plus a copy-paste `runscope ingest "<bundle-dir>"` command.
