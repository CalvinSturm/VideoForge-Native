<p align="center">
  <img src="ui/src/VideoForge_icon.png" alt="VideoForge" width="120" />
</p>

<h1 align="center">VideoForge</h1>

<p align="center">
  <strong>Local-first, deterministic AI super-resolution for professional image &amp; video enhancement.</strong>
</p>

<p align="center">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square" />
  <img alt="Tauri" src="https://img.shields.io/badge/Tauri-2.0-orange?style=flat-square" />
  <img alt="React" src="https://img.shields.io/badge/React-19-61dafb?style=flat-square" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=flat-square" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-11.7+-76b900?style=flat-square" />
  <img alt="License" src="https://img.shields.io/badge/license-Proprietary-lightgrey?style=flat-square" />
</p>

---

## Current Docs

- [`docs/architecture_status_truth.md`](docs/architecture_status_truth.md) — canonical architecture and status source.
- [`docs/capability_matrix.md`](docs/capability_matrix.md) — canonical feature and support matrix.
- [`docs/runtime_path_contracts.md`](docs/runtime_path_contracts.md) — canonical route, gating, and fallback contracts.
- [`docs/state_and_persistence.md`](docs/state_and_persistence.md) — canonical operational state model.
- [`docs/metrics_trust.md`](docs/metrics_trust.md) — canonical metric provenance and comparison rules.
- [`docs/README.md`](docs/README.md) — entrypoint for current docs and status references.
- [`docs/release_hygiene_checklist.md`](docs/release_hygiene_checklist.md) — metadata/version alignment checklist for release and packaging changes.
- [`docs/handoff_native_direct_glitch_followup_2026-03-13.md`](docs/handoff_native_direct_glitch_followup_2026-03-13.md) — resolved native direct investigation record to resume from if the regression returns.
- [`SMOKE_TEST.md`](SMOKE_TEST.md) — smoke test runbook for current manual validation commands.

Historical audits, plans, and completed handoffs now live under [`docs/archive/`](docs/archive/).

---

## Overview

VideoForge is a local-first desktop application for AI-powered image and video upscaling. It combines a Rust orchestration layer, a Python inference engine, and an optional native Rust video engine behind a Tauri desktop app and React UI.

### Core Philosophy

- **Privacy First** — All processing happens locally on your GPU. No network calls, no telemetry.
- **Determinism** — Supported models (RCAN, EDSR) produce bit-identical output across runs. GAN models (RealESRGAN) are clearly labeled as non-deterministic.
- **User Authority** — Full control over trim, crop, color grading, model selection, and precision mode. Preview before you commit.
- **Engine Flexibility** — Video jobs can run through the Python worker path or the opt-in native `engine-v2` path, depending on model/runtime eligibility.

### Native Engine Reality Check

- The native family is for **video jobs only** and is only attempted for **ONNX** models.
- There are two native execution modes:
  - **Native direct**: in-process `engine-v2`
  - **Native CLI-backed**: native-family fallback through the `rave` adapter
- The direct path now uses packet-aware demux/mux boundaries in the host and a safer decode-to-preprocess lifetime contract aligned with the working `third_party/rave` reference path.
- The Python path remains the broadest-compatibility route and the default for non-ONNX or non-video jobs.

---

## Features

| Category | Details |
|----------|---------|
| **AI Upscaling** | Python path supports broad local model coverage; native family is currently limited to eligible ONNX video models |
| **Video Pipeline** | Python path: FFmpeg decode → SHM ring buffer → PyTorch inference → FFmpeg encode. Native direct path: packet-aware demux → `engine-v2` → packet-aware mux |
| **Editing** | Trim, crop, rotation, color grading (brightness, contrast, saturation, hue), FPS override |
| **Research Layer** | Python path only: multi-model blending, frequency band analysis, hallucination detection, spatial routing |
| **Auto Grading** | Python-oriented grading and analysis flow; not a native-family contract |
| **Precision Modes** | FP32, FP16, and deterministic (forces `cudnn.deterministic`, disables TF32) |
| **Job Queue** | Batch processing with per-job progress and ETA estimation |
| **Professional UI** | Tiled mosaic layout (react-mosaic), video preview with crop overlay, interactive timeline |

### Shipped Support Matrix

| Engine / route | Media | Model formats | Practical model support | Relative speed |
|---|---|---|---|---|
| **Python sidecar** | Image, video | PyTorch weights, broad local model support | RCAN, EDSR, RealESRGAN, SwinIR, HAT, Swin2SR, diffusion, lightweight models, research/blending path | Broadest compatibility, generally slower than native direct for eligible video jobs |
| **Native direct** | Video only | ONNX only | Compatible ONNX video models routed through `engine-v2` | Fastest expected path for supported video jobs |
| **Native CLI-backed** | Video only | ONNX only | Compatible ONNX video models routed through the `rave` adapter | Native-family compatibility route; typically less ideal than native direct |

Notes:
- The UI only attempts the native family for video jobs when native mode is enabled and the selected model is ONNX.
- Research/blending features are part of the Python path, not the native contract.
- Feature lists above should be read as repo capabilities, not a claim that every route supports every feature.
- Speed comparisons should always be read with route, cache state, batch, and fallback status in mind.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  React / TypeScript UI                                          │
│  Vite · BlueprintJS · Zustand · react-mosaic                    │
│  ├─ InputOutputPanel  ── Model selection, editing, export       │
│  ├─ PreviewPanel      ── Video/image preview with crop overlay  │
│  ├─ AIUpscaleNode     ── Upscale config & research controls     │
│  ├─ JobsPanel         ── Queue with progress & ETA              │
│  └─ Timeline          ── Trim, timeline scrubbing               │
└────────────────────┬────────────────────────────────────────────┘
                     │  Tauri IPC (invoke / listen)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Rust Backend (Tauri 2.0)                                       │
│  ├─ lib.rs             ── Commands, process mgmt, SHM orch.    │
│  ├─ video_pipeline.rs  ── FFmpeg decode/encode, NVENC probing   │
│  ├─ shm.rs             ── 3-slot ring buffer (memmap2)          │
│  ├─ control.rs         ── Zenoh pub/sub parameter sync          │
│  ├─ edit_config.rs     ── FFmpeg filter chain builder           │
│  └─ models.rs          ── Model file discovery & metadata       │
└───────┬─────────────────────────────┬───────────────────────────┘
        │  stdin/stdout pipes         │  SHM (mmap) + Zenoh IPC
        ▼                             ▼
   FFmpeg (decode/encode)        Python AI Sidecar
   NVDEC / NVENC                 ├─ shm_worker.py       ── Main inference loop
                                 ├─ model_manager.py    ── Model loading & VRAM mgmt
                                 ├─ arch_wrappers.py    ── Architecture adapters
                                 ├─ blender_engine.py   ── GPU blending & detail ops
                                 ├─ research_layer.py   ── Multi-model SR framework
                                 ├─ sr_settings_node.py ── Settings & feature gating
                                 └─ auto_grade_analysis.py ── Auto color grading
```

### Engine v2 (GPU-Native Video Path)

`engine-v2/` is the native video engine used by the opt-in native path:

- **NVDEC → CUDA Preprocessing → TensorRT/ONNX Inference → NVENC** — no CPU round-trips
- CUDA custom kernels for NV12↔RGB conversion, scaling, and format transforms
- RAII-based VRAM management with bucketed buffer pools
- Packet-aware FFmpeg demux/mux boundaries in the current direct-native host path

---

## Data Flow (Video Upscale)

### Python sidecar path

```
1. User selects input video, model, and edit settings in the UI
2. UI sends upscale_request via Tauri IPC to the Rust backend
3. Rust spawns the Python AI worker and performs a Zenoh handshake
4. Rust allocates a 3-slot SHM ring buffer (input + output regions per slot)
5. Processing loop:
   ┌──────────┐     ┌──────────────────┐     ┌──────────────┐
   │  FFmpeg  │────▶│  SHM Input Slot  │────▶│  Python GPU  │
   │  Decode  │     │  (raw RGB24)     │     │  Inference   │
   └──────────┘     └──────────────────┘     └──────┬───────┘
                                                    │
   ┌──────────┐     ┌──────────────────┐            │
   │  FFmpeg  │◀────│  SHM Output Slot │◀───────────┘
   │  Encode  │     │  (upscaled RGB)  │
   └──────────┘     └──────────────────┘
6. Rust streams encoded frames to FFmpeg → final MP4 (NVENC H.264/H.265)
7. Cleanup: SHM files removed, Python worker terminated
```

### Native direct path

```
1. User selects a video job with an ONNX model and native mode enabled
2. UI sends upscale_request_native via Tauri IPC to the Rust backend
3. Backend applies compile-time and runtime native gating
4. Host performs packet-aware FFmpeg demux on the input container
5. Direct processing loop:
   packet-aware demux
      -> NVDEC
      -> CUDA preprocess
      -> TensorRT / ONNX Runtime inference
      -> CUDA postprocess
      -> NVENC
      -> packet-aware FFmpeg mux
6. Final MP4 is written to the requested output path
```

---

## Supported AI Models

| Architecture | Type | Deterministic | Capabilities |
|-------------|------|:---:|--------------|
| **RCAN** | CNN | ✅ | Temporal, edge-aware, luma blend, sharpen |
| **EDSR** | CNN | ✅ | Temporal, edge-aware, luma blend, sharpen |
| **RealESRGAN** | GAN | ❌ | Full pipeline + secondary model blending |
| **SwinIR** | Transformer | ❌ | Full pipeline + secondary model blending |
| **HAT** | Transformer | ❌ | Full pipeline + secondary model blending |
| **Swin2SR** | Transformer | ❌ | Full pipeline + secondary model blending |
| **Diffusion** | Diffusion | ❌ | Sharpen |
| **Lightweight** | CNN | varies | Sharpen (fast preview) |

Model weights are loaded from the `weights/` directory and scanned automatically at startup.

---

## Prerequisites

- **OS:** Windows 10/11 (primary platform)
- **GPU:** NVIDIA GPU with CUDA 11.7+ and NVENC support
- **Software:**
  - [Node.js](https://nodejs.org/) ≥ 18
  - [Rust](https://rustup.rs/) (stable toolchain)
  - [FFmpeg & FFprobe](https://ffmpeg.org/) available either in `PATH` or in supported repo/runtime locations discovered by the native runtime helpers
  - Python 3.10+ (bundled or installed to `%LOCALAPPDATA%/VideoForge/python/`)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/VideoForgeRepo/VideoForgeV1.5.git
cd VideoForge
```

### 2. Install dependencies

```bash
# Install root Tauri CLI
npm install

# Install UI dependencies
npm run ui-install

# Install Python dependencies (into your venv or bundled runtime)
pip install -r requirements.txt
```

### 3. Run in development mode

```bash
npm run dev
```

This launches the Tauri app with Vite hot-reload for the UI. The Rust backend compiles and starts automatically.

Alternatively, use the convenience script:

```bat
run.bat
```

### 4. Run the native engine in development mode

Native development requires the `native_engine` Cargo feature and runtime opt-in.

CLI option:

```bash
npm run dev:native
```

One-click launchers:

```bat
run_native_engine.bat
run_native_engine_debug.bat
```

Launcher behavior:

- `run_native_engine.bat`
  - enables `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1`
  - enables `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`
  - starts `npm run dev:native`
- `run_native_engine_debug.bat`
  - enables the same native direct path
  - also enables a small startup-focused debug capture window for NVDEC, pipeline, postprocess-kernel, NVENC, and mux logging
  - writes dumps under `artifacts/nvdec_debug/run_native_engine_debug/`

Important routing note:

- The UI only attempts the native family for **video** jobs when native mode is enabled and the selected model is **ONNX**.
- If those conditions are not met, the job uses the Python path instead.

### 5. Production build

```bash
npm run build
```

---

## Development

### Build Commands

| Command | Description |
|---------|-------------|
| `npm run dev` | Launch Tauri + Vite dev server with hot-reload |
| `npm run dev:native` | Launch Tauri + Vite with the `native_engine` feature enabled |
| `npm run dev:native:cached` | Launch native dev mode with TensorRT engine cache enabled |
| `npm run build` | Production build (compiles Rust + bundles UI) |
| `npm run build:native` | Production build with the `native_engine` feature enabled |
| `npm run build:native:cached` | Production native build with TensorRT engine cache enabled |
| `npm run ui-install` | Install UI npm dependencies |
| `cd src-tauri && cargo test --workspace` | Run Rust workspace tests |
| `cd ui && npx tsc --noEmit` | Type-check the TypeScript UI |

### Native Runtime Flags

Common native runtime flags:

| Env var | Purpose |
|---|---|
| `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1` | Runtime opt-in for the native family |
| `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` | Prefer the in-process `engine-v2` direct path |
| `VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE=1` | Enable TensorRT engine caching |
| `VIDEOFORGE_TRT_CACHE_DIR=<path>` | Override TensorRT cache location |

Useful debug-only flags:

| Env var | Purpose |
|---|---|
| `VIDEOFORGE_NVDEC_DEBUG_DUMP=1` | Enable NVDEC dump support |
| `VIDEOFORGE_NVDEC_DEBUG_DUMP_FRAMES=<N>` | Dump a decoded frame window |
| `VIDEOFORGE_NVDEC_DEBUG_DUMP_START_FRAME=<F>` | Start frame for NVDEC windowed dumps |
| `VIDEOFORGE_PIPELINE_DEBUG_DUMP_FRAMES=<N>` | Dump preprocess and postprocess frame windows |
| `VIDEOFORGE_PIPELINE_DEBUG_DUMP_START_FRAME=<F>` | Start frame for pipeline-window dumps |
| `VIDEOFORGE_POSTPROCESS_KERNEL_DEBUG_DUMP_FRAMES=<N>` | Dump postprocess-kernel input frame windows |
| `VIDEOFORGE_POSTPROCESS_KERNEL_DEBUG_DUMP_START_FRAME=<F>` | Start frame for postprocess-kernel dumps |
| `VIDEOFORGE_NVENC_DEBUG_DUMP_FRAMES=<N>` | Dump NVENC handoff surfaces and encoded packets |
| `VIDEOFORGE_NATIVE_MUX_DEBUG=1` | Enable mux-side debug logging |

Debug artifacts are written under `artifacts/` and should not be committed.

### Run Artifacts And RunScope

Optional run artifacts are enabled with:

| Env var | Purpose |
|---|---|
| `VIDEOFORGE_ENABLE_RUN_ARTIFACTS=1` | Write per-run artifact bundles under the output-adjacent `.videoforge_runs/<job_id>/` directory |

When enabled, both Python and native command paths now write:

- `videoforge.run_manifest.v1.json`
- `videoforge.runtime_config_snapshot.v1.json`
- `videoforge.run_observed_metrics.v1.json`
- `videoforge_run.json`

`videoforge_run.json` is a RunScope-ingestible producer bundle intended for the `runscope` workstream.

Practical workflow:

1. Run a VideoForge job with `VIDEOFORGE_ENABLE_RUN_ARTIFACTS=1`.
2. Locate the output-adjacent `.videoforge_runs/<job_id>/` directory.
3. Point RunScope ingestion at that directory or directly at `videoforge_run.json`.

Concrete example:

```bash
runscope ingest "<output-path>/.videoforge_runs/<job_id>"
```

Helper:

```bat
run_latest_runscope_ingest.bat
```

That launcher prints the newest VideoForge artifact bundle it can find under the repo and a copy-paste `runscope ingest "<bundle-dir>"` command.

### Metadata Alignment

- Version source of truth is shared across `package.json`, `ui/package.json`, `src-tauri/Cargo.toml`, and `src-tauri/tauri.conf.json`.
- Package ids remain lowercase for tooling compatibility, while the user-facing product name is `VideoForge`.
- Use [`docs/release_hygiene_checklist.md`](docs/release_hygiene_checklist.md) before release or packaging changes.

### Project Structure

```
VideoForge/
├── src-tauri/              # Rust backend (Tauri 2.0)
│   ├── src/                #   Source files (lib.rs, video_pipeline.rs, shm.rs, ...)
│   ├── Cargo.toml          #   Rust dependencies
│   └── tauri.conf.json     #   Tauri window & build config
├── ui/                     # React/TypeScript frontend
│   ├── src/                #   Components, stores, hooks, utils
│   ├── package.json        #   UI dependencies (React 19, BlueprintJS, Zustand, ...)
│   └── vite.config.ts      #   Vite bundler config
├── python/                 # Python AI worker (sidecar process)
│   ├── shm_worker.py       #   Main inference loop & Zenoh subscriber
│   ├── model_manager.py    #   Model registry, weight loading, VRAM eviction
│   ├── arch_wrappers.py    #   Architecture-specific adapters
│   ├── blender_engine.py   #   GPU blending, EMA, edge detection
│   ├── research_layer.py   #   Multi-model SR blending framework
│   ├── sr_settings_node.py #   Settings management & dispatch
│   └── auto_grade_analysis.py  # Auto color grading analysis
├── engine-v2/              # Next-gen GPU-native upscale engine (Rust/CUDA)
│   ├── src/
│   │   ├── core/           #     GPU context, CUDA kernels, types
│   │   ├── codecs/         #     NVDEC, NVENC, FFI bindings
│   │   ├── backends/       #     TensorRT inference backend
│   │   └── engine/         #     Pipeline orchestration, inference loop
│   └── Cargo.toml
├── weights/                # AI model weights directory (not tracked in git)
├── docs/                   # Architecture docs & roadmap
├── requirements.txt        # Python dependencies
├── package.json            # Root workspace (Tauri CLI)
├── run.bat                 # One-click standard dev launcher
├── run_native_engine.bat   # One-click native direct dev launcher
├── run_latest_runscope_ingest.bat # Print latest RunScope ingest command for VideoForge artifacts
└── run_native_engine_debug.bat # One-click native direct startup-debug launcher
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Desktop Shell** | Tauri 2.0 | Window management, IPC, process lifecycle |
| **Backend** | Rust + Tokio | Pipeline orchestration, SHM, FFmpeg control |
| **IPC** | Zenoh + memmap2 | Low-latency signaling + zero-copy frame transfer |
| **AI Engine** | Python + PyTorch + CUDA 11.7 | Model inference (RealESRGAN, RCAN, SwinIR, ...) |
| **Frontend** | React 19 + TypeScript + Vite 6 | Interactive UI |
| **UI Framework** | BlueprintJS 6 | Professional component library |
| **State** | Zustand 5 | Lightweight state management |
| **Layout** | react-mosaic | Tiled, rearrangeable panel layout |
| **Video I/O** | FFmpeg (NVDEC/NVENC) | Hardware-accelerated decode/encode |
| **Next-Gen Engine** | Rust + cudarc + ort (TensorRT) | Fully GPU-resident pipeline (engine-v2) |

---

## Platform Notes

- **Windows-primary**: Python runtime resolves to `%LOCALAPPDATA%/VideoForge/python/` in distribution builds. Development uses local venvs.
- **NVIDIA GPU required**: CUDA 11.7+ with NVENC support for hardware encoding.
- **Native engine is opt-in**: build with `--features native_engine` and enable `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1` at runtime.
- **Model weights** are scanned from `weights/` directories relative to the installation path.

---

## Status

Current repo state, based on the canonical docs and checked-in code:

- Python remains the default engine path.
- Native direct and native-cli now share a larger control plane and result contract.
- The direct native path now uses packet-aware demux/mux boundaries instead of the earlier chunked streaming path.
- Native direct startup reliability has been improved by aligning decode-to-preprocess buffer lifetime handling with the working `third_party/rave` reference pattern.
- The canonical architecture, capability, routing, persistence, and metrics docs are now in place.
- The cleanup follow-up refactor tracks are complete and the maintained validation matrix is green.
- Optional run manifests now have parity across Python and native command paths through the shared artifact system.
- Optional run artifacts now include a RunScope-ingestible `videoforge_run.json` bundle plus raw runtime snapshot and observed-metrics JSON alongside the manifest.
- Remaining work is focused on selective native productization and refining user-facing status/support language, not on broad cleanup recovery.

For current planning detail, start with:

- [`docs/architecture_status_truth.md`](docs/architecture_status_truth.md)
- [`docs/capability_matrix.md`](docs/capability_matrix.md)
- [`docs/runtime_path_contracts.md`](docs/runtime_path_contracts.md)
- [`docs/README.md`](docs/README.md)

---

## Contributing

Contributions are welcome. When contributing models or pipeline changes, ensure that determinism and performance characteristics are preserved where applicable. Start with [`docs/README.md`](docs/README.md) for the current doc set.

---

## License

Proprietary — All rights reserved.
