# VideoForge Smoke Tests

Repeatable verification commands for both the Python pipeline (default) and the
native-engine MVP.  Run these after every major change and on a fresh clone.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Rust toolchain** | `rustup toolchain install stable` |
| **Node / npm** | ≥ 18 (for `npm run dev`) |
| **FFmpeg + FFprobe** | In `PATH`.  NVENC support recommended for GPU encode. |
| **Python venv** | torch ≥ 2.0, OpenCV, numpy, zenoh 1.0.x.  See setup below. |
| **NVIDIA GPU** | CUDA ≥ 11.8 recommended.  CPU fallback works but is slow. |
| **Model weights** | `.pth` files under `weights/` (see [Model Setup](#model-setup)). |

---

## Python Venv Setup (Development)

```powershell
# Create and activate a venv (Python 3.10 recommended)
python -m venv venv310
.\venv310\Scripts\Activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy zenoh==1.0.2

# Tell VideoForge where this venv is
$Env:VIDEOFORGE_DEV_PYTHON = "$(pwd)\venv310\Scripts\python.exe"
```

> The binary search order is:
> 1. `%LOCALAPPDATA%\VideoForge\python\python.exe` (installed runtime)
> 2. `VIDEOFORGE_DEV_PYTHON` env var
> 3. Hardcoded dev-venv fallback paths (see `src-tauri/src/python_env.rs`)

---

## Model Setup

Place model weight files under the `weights/` directory in the repo root.
The Python worker looks in `{repo_root}/weights/` and `{repo_root}/python/weights/`.

```
weights/
  RCAN_x4.pth
  RCAN_x2.pth
  RealESRGAN_x4plus.pth
  ...
```

Model identifiers sent from the UI must match keys recognised by `ModelLoader`
in `python/shm_worker.py` (see `Config.VALID_MODELS`).

---

## A) One-Command Smoke Harness (PowerShell)

Run from the repository root:

```powershell
# Quick check — prerequisites + Python IPC handshake only (no video needed):
.\tools\smoke.ps1

# Full check with a test clip and RCAN_x4:
.\tools\smoke.ps1 -InputFile C:\test\sample_720p.mp4 -Model RCAN_x4

# Expected exit code: 0 on success, 1 on failure
```

The script will print `[PASS]` / `[FAIL]` / `[SKIP]` for each step.

---

## A1) Native Perf Regression Gate (repeatable)

Use this to catch native-engine performance regressions early with a hard
median-FPS threshold.

```powershell
# From repo root
$in = (Resolve-Path .\test_input.mp4).Path
$onnx = 'C:\Users\Calvin\Desktop\rave\tests\assets\models\resize2x_rgb.onnx'

.\tools\ci\check_native_smoke_perf.ps1 `
  -Input $in `
  -Onnx $onnx `
  -Scale 2 `
  -Precision fp32 `
  -Runs 3 `
  -MinMedianFps 40 `
  -OutJson .\artifacts\native_perf_report.json
```

Behavior:
- Runs `smoke.exe --e2e-native` repeatedly.
- Computes FPS from input frame count and wall-clock time.
- Fails (non-zero exit) when `median_fps < MinMedianFps`.

---

## B) Python Pipeline (default path)

### Step 1 — Build and run the Rust smoke binary

```powershell
cd src-tauri

# Build the smoke binary (debug, fast)
cargo build --bin smoke

# Run prerequisite + IPC check only
.\target\debug\smoke.exe --timeout 60

# Run with model load check
.\target\debug\smoke.exe --model RCAN_x4 --precision fp32
```

**Expected output (all pass):**

```
=== VideoForge Smoke Test ===

── Prerequisites ───────────────────────────────────────────────
[PASS] FFmpeg in PATH
[PASS] FFprobe in PATH

── Python Environment ──────────────────────────────────────────
[PASS] Python environment
       python = C:\path\to\venv310\Scripts\python.exe
       script = C:\...\python\shm_worker.py

── Python IPC Handshake ────────────────────────────────────────
[PASS] Zenoh listener
[PASS] Python spawn
[PASS] Python Zenoh handshake
[PASS] Model load (RCAN_x4)

── Native Engine ───────────────────────────────────────────────
[SKIP] native_engine feature: BLOCKED ...

─────────────────────────────────────────────────────────────────
Result: ALL CHECKS PASSED
```

### Step 2 — Full end-to-end upscale via the app

```powershell
# In repo root — launches Tauri + Vite dev server
npm run dev
```

Then in the browser DevTools console (F12 → Console):

```javascript
// Python pipeline — full upscale
await window.__TAURI__.core.invoke("upscale_request", {
  inputPath:  "C:/test/sample_720p.mp4",   // change to your file
  outputPath: "",                           // auto-generated
  model:      "RCAN_x4",
  editConfig: {
    trim_start: 0, trim_end: 0, crop: null,
    rotation: 0, flip_h: false, flip_v: false, fps: 0,
    color: { brightness: 0, contrast: 0, saturation: 0, gamma: 1 }
  },
  scale: 4
})
```

**Expected return value:**

```
"C:/test/sample_720p_<timestamp>_upscaled.mp4"
```

**Expected log patterns** (visible in the DevTools console and Rust stdout):

| Log | Source | Meaning |
|-----|--------|---------|
| `[INFO] Upscale request started` | Rust | Command received |
| `[INFO] Python worker spawned` | Rust | Python process started |
| `[INFO] Python worker handshake received` | Rust | Zenoh IPC alive |
| `[Python] Precision: FP32 ...` | Python stdout | Model precision set |
| `[Python] Loaded: RCAN_x4 ...` | Python stdout | Weights loaded |
| `[INFO] Model loaded` | Rust | `load_model` IPC round-trip OK |
| `[INFO] SHM ring buffer opened and reset` | Rust | Shared memory ready |
| `[INFO] Python frame loop started` | Rust | GPU polling loop running |
| `[DEBUG] Frame written to SHM → READY_FOR_AI` | Rust | Decode in progress |
| `[DEBUG] Slot READY_FOR_ENCODE` | Rust | AI inference complete per frame |
| `[INFO] Encode complete` | Rust | FFmpeg wrote output |
| `[INFO] Upscale request complete` | Rust | Job done |

### What to check in the output file

1. Open the output `.mp4` — it should play without errors.
2. Resolution should be 4× the input (e.g. 720p → 2880p for `scale=4`).
3. Audio should be preserved (same track as input).
4. Duration should match the input (minus any trim).

### Common failure modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Python worker handshake timeout (60s)` | Python failed to start or Zenoh not connecting | Check `VIDEOFORGE_DEV_PYTHON`, ensure zenoh 1.0.x is installed in venv |
| `Failed to load model: Load Failed: ...` | Weight file not found | Place `.pth` file under `weights/`, use correct model identifier |
| `SHM init failed` | SHM file creation failed | Check temp dir permissions (`%TEMP%`) |
| `AI processing timeout (30s)` | GPU OOM or inference hung | Use smaller input, reduce tile size in `Config.TILE_SIZE` |
| `error while running tauri application` | Tauri setup error | Run `npm run ui-install` then retry |

---

## C) SHM Roundtrip Smoke (no FFmpeg, no UI, no model weights)

Exercises the full Rust→Python→Rust SHM frame path using a scale=1 passthrough
(no AI inference, no weights required).  Use this to rule out IPC/SHM issues
before chasing FFmpeg or model problems.

### Command

```powershell
cd src-tauri
cargo build --bin smoke
.\target\debug\smoke.exe --shm-roundtrip
```

### Expected output

```
── SHM Roundtrip ─────────────────────────────────────────────
[PASS] Zenoh listener
[PASS] Python spawn
[PASS] Python Zenoh handshake
[PASS] SHM created (path: C:\Temp\vf_buffer_XXXXX.bin)
[PASS] SHM header validated
[PASS] Synthetic frame written → SLOT_READY_FOR_AI
[PASS] process_one_frame sent
[PASS] FRAME_DONE received
[PASS] Output validated (192 bytes, non-zero)
─────────────────────────────────────────────────────────────
Result: ALL CHECKS PASSED
```

### Failure labels and fixes

| Label | Meaning | Fix |
|-------|---------|-----|
| `SHM_CREATE_TIMEOUT` | Python didn't reply to create_shm in 10 s | Check Python env, zenoh port conflict |
| `SHM_OPEN_FAILED` | Rust can't open temp file | Windows file lock? Check %TEMP% perms |
| `FRAME_DONE_TIMEOUT` | Python didn't process in 5 s | Python crash — run with stderr visible |
| `OUTPUT_ALL_ZEROS` | Frame processed but output is zeros | scale=1 passthrough broken — check mmap write |

---

## D) Native Engine Path

> **Status: BLOCKED** — see `docs/NATIVE_ENGINE_MVP.md` for full context.

The `native_engine` feature flag is off by default because `engine-v2` requires
`ort = "^2.0"` which has not been published as a stable release on crates.io
(only release candidates exist, e.g. `2.0.0-rc.11`).

**To unblock and enable:**

1. Edit `engine-v2/Cargo.toml`:
   ```toml
   ort = { version = "2.0.0-rc.11", features = ["cuda", "tensorrt"] }
   ```
2. Uncomment in `src-tauri/Cargo.toml`:
   ```toml
   videoforge-engine = { path = "../engine-v2", optional = true }
   ```
3. Change the feature definition:
   ```toml
   native_engine = ["dep:videoforge-engine"]
   ```
4. Build and run:
   ```powershell
   cd src-tauri
   cargo build --release --features native_engine
   ```
5. From DevTools console:
   ```javascript
   await window.__TAURI__.core.invoke("upscale_request_native", {
     inputPath:  "C:/test/sample_720p.mp4",
     outputPath: "",
     modelPath:  "C:/models/rcan_x4.onnx",
     scale: 4,
     precision: "fp32",
     audio: true
   })
   ```

**Expected when feature is disabled:**

```json
{"code":"FEATURE_DISABLED","message":"The native_engine feature is not compiled in..."}
```

---

## E) CI Local Equivalents

Run these commands locally to replicate what GitHub Actions checks:

```powershell
cd src-tauri

# Format check (must be clean before push)
cargo fmt --all -- --check

# Full unit test suite
cargo test

# Individual test modules
cargo test --test-output immediate shm
cargo test --test-output immediate ipc
cargo test --test-output immediate utils
```

Expected output:
```
running 18 tests
test ipc::protocol::tests::... ok  (×9)
test shm::tests::...           ok  (×6)
test utils::tests::...         ok  (×3)

test result: ok. 18 passed; 0 failed
```

---

## F) RUST_LOG levels

Set `RUST_LOG` to control log verbosity:

```powershell
# Development — see all VideoForge logs
$Env:RUST_LOG = "videoforge=debug"
npm run dev

# Production — info only (default)
$Env:RUST_LOG = "videoforge=info"

# Quiet — errors only
$Env:RUST_LOG = "error"
```

Key log targets:
- `videoforge::commands::upscale` — IPC round-trips, SHM events, encode progress
- `videoforge::shm` — SHM header validation (debug)
- `videoforge::python_env` — Python path resolution

Per-frame SHM slot transitions are logged at `debug` level and do not appear at
`info` — this avoids log spam during normal video processing.
