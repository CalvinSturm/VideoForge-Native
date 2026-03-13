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
- [`implementation_plan.md`](implementation_plan.md) — native engine execution tracker and archived status summary.
- [`docs/native_engine_handoff_2026-03-07.md`](docs/native_engine_handoff_2026-03-07.md) — latest native-only handoff and recommended next steps.
- [`docs/audits/video_upscaler_audit_2026-03-07.md`](docs/audits/video_upscaler_audit_2026-03-07.md) — workspace audit and bottleneck review.
- [`docs/plans/video_upscaler_patch_plan_2026-03-07.md`](docs/plans/video_upscaler_patch_plan_2026-03-07.md) — PR-shaped cleanup and measurement plan.
- [`docs/plans/video_upscaler_benchmark_plan_2026-03-07.md`](docs/plans/video_upscaler_benchmark_plan_2026-03-07.md) — benchmark policy and fixture plan.
- [`docs/release_hygiene_checklist.md`](docs/release_hygiene_checklist.md) — metadata/version alignment checklist for release and packaging changes.

---

## Overview

VideoForge is a local-first desktop application for AI-powered image and video upscaling. It combines a Rust orchestration layer, a Python inference engine, and an optional native Rust video engine behind a Tauri desktop app and React UI.

### Core Philosophy

- **Privacy First** — All processing happens locally on your GPU. No network calls, no telemetry.
- **Determinism** — Supported models (RCAN, EDSR) produce bit-identical output across runs. GAN models (RealESRGAN) are clearly labeled as non-deterministic.
- **User Authority** — Full control over trim, crop, color grading, model selection, and precision mode. Preview before you commit.
- **Engine Flexibility** — Video jobs can run through the Python worker path or the opt-in native `engine-v2` path, depending on model/runtime eligibility.

---

## Features

| Category | Details |
|----------|---------|
| **AI Upscaling** | RealESRGAN, RCAN, EDSR, SwinIR, HAT, Swin2SR, diffusion, and lightweight models |
| **Video Pipeline** | FFmpeg decode → SHM ring buffer → PyTorch inference → FFmpeg encode (H.264/H.265 NVENC) |
| **Editing** | Trim, crop, rotation, color grading (brightness, contrast, saturation, hue), FPS override |
| **Research Layer** | Multi-model blending, frequency band analysis, hallucination detection, spatial routing |
| **Auto Grading** | Histogram analysis, white balance correction, noise estimation, skin tone detection |
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
- Streamed FFmpeg demux/mux boundaries in the current direct-native host path

---

## Data Flow (Video Upscale)

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
  - Python 3.10+ (bundled or installed to `%APPDATA%/VideoForge/python/`)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-org/VideoForge.git
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

### 4. Production build

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
| `npm run build` | Production build (compiles Rust + bundles UI) |
| `npm run build:native` | Production build with the `native_engine` feature enabled |
| `npm run ui-install` | Install UI npm dependencies |
| `cd src-tauri && cargo test --workspace` | Run Rust workspace tests |
| `cd ui && npx tsc --noEmit` | Type-check the TypeScript UI |

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
└── run.bat                 # One-click dev launcher
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

- **Windows-primary**: Python runtime resolves to `%APPDATA%/Local/VideoForge/python/` in distribution builds. Development uses local venvs.
- **NVIDIA GPU required**: CUDA 11.7+ with NVENC support for hardware encoding.
- **Native engine is opt-in**: build with `--features native_engine` and enable `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1` at runtime.
- **Model weights** are scanned from `weights/` directories relative to the installation path.

---

## Status

Current repo state, based on the canonical docs and checked-in code:

- Python remains the default engine path.
- Native direct and native-cli now share a larger control plane and result contract.
- The direct native path has removed the main temp-file boundaries in favor of streamed demux/mux.
- The canonical architecture, capability, routing, persistence, and metrics docs are now in place.
- The cleanup follow-up refactor tracks are complete and the maintained validation matrix is green.
- Optional run manifests now have parity across Python and native command paths through the shared artifact system.
- Remaining work is focused on selective native productization and refining user-facing status/support language, not on broad cleanup recovery.

For current planning detail, start with:

- [`docs/architecture_status_truth.md`](docs/architecture_status_truth.md)
- [`docs/capability_matrix.md`](docs/capability_matrix.md)
- [`docs/runtime_path_contracts.md`](docs/runtime_path_contracts.md)
- [`implementation_plan.md`](implementation_plan.md)
- [`docs/native_engine_handoff_2026-03-07.md`](docs/native_engine_handoff_2026-03-07.md)
- [`docs/README.md`](docs/README.md)

---

## Contributing

Contributions are welcome. When contributing models or pipeline changes, ensure that determinism and performance characteristics are preserved where applicable. Start with [`docs/README.md`](docs/README.md) for the current doc set.

---

## License

Proprietary — All rights reserved.
