# VideoForge Architecture & Operational Guide

This guide provides a technical deep-dive into the VideoForge codebase, intended for staff engineers and new contributors. It covers the system's architecture, operational flows, and performance characteristics.

## Coverage

The following components were audited for this guide:

- **Rust Backend**: `src-tauri/` (Tauri commands, SHM management, Zenoh orchestration, video pipeline).
- **AI Sidecar**: `python/` (PyTorch inference, model registry, SHM worker).
- **Native Engine**: `engine-v2/` (GPU-native TensorRT pipeline).
- **Frontend**: `ui/src/` (Zustand state management, Tauri integration).
- **IPC Protocol**: `ipc/` (JSON schemas for Zenoh envelopes).
- **Third-Party**: `third_party/rave` (Production-grade video processing crates).
- **Tooling**: `.github/workflows/` (CI/GPU lane), `tools/` (Benchmarks and smoke tests).

---

## 1. Executive Summary

VideoForge is a professional-grade video super-resolution workstation.

- **Goal**: Provide deterministic, flicker-free video upscaling using editor-grade CNNs (RCAN, EDSR).
- **Target Audience**: Video editors and archivists requiring bit-exact enhancement.
- **Non-Goals**: Generative synthesis (GANs are discouraged), real-time low-latency streaming (prioritizes quality and determinism), or general-purpose video editing.

---

## 2. Quickstart

### Prerequisites

- Windows 10/11 (CUDA-capable GPU required).
- NVIDIA Drivers + CUDA Toolkit 12.x.
- Python 3.10+ (with `venv`).
- Node.js + `pnpm`.

### Setup & Build

```powershell
# Install UI dependencies
cd ui
pnpm install

# Build backend and frontend
cd ..
npm run tauri build
```

### Dev Loop

```powershell
# Run in dev mode (hot reload)
npm run tauri dev

# Run Python worker independently for debugging
cd python
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
python shm_worker.py --port 5000
```

### Testing

- **Rust**: `cargo test` inside `src-tauri`.
- **Python**: `pytest` inside `python`.
- **Full System**: `powershell tools/smoke.ps1`.

---

## 3. System Diagram

```mermaid
graph TD
    Frontend[React UI] <--> |Tauri Invoke / Events| Backend[Tauri Rust Backend]
    Backend <--> |Zenoh IPC (JSON)| Sidecar[Python AI Worker]
    Backend <--> |Shared Memory Ring| Sidecar
    Backend <--> |Stdio Pipes| FFmpeg[FFmpeg Process]
    FFmpeg <--> |Raw Pixels| Backend
    Backend -.-> |Optional| Native[Native Engine-v2]
```

---

## 4. Module Map

### Rust Layer (`src-tauri/src/`)

- **`main.rs` / `lib.rs`**: System initialization. Wires Tauri commands (`upscale_request`, `rave_upscale`) and background tasks (`spawn_system_monitor`).
- **`shm.rs`**: **Critical Infrastructure**. Implements `VideoShm` ring buffer. Uses `AtomicU32` CAS for state transitions: `EMPTY` -> `RUST_WRITING` -> `READY_FOR_AI` -> `AI_PROCESSING` -> `READY_FOR_ENCODE`.
- **`upscale.rs`**: Orchsestrates `run_upscale_job`. Manages life-cycle of `VideoDecoder`, `VideoEncoder`, and the `Zenoh` IPC session.
- **`video_pipeline.rs`**: FFmpeg abstraction. `VideoDecoder` spawns `-hwaccel cuda` inputs; `VideoEncoder` handles `-c:v h264_nvenc` outputs.
- **`control.rs`**: Real-time parameter sync via `ResearchConfig` and `ControlChannel`.
- **`ipc/protocol.rs`**: Typed message definitions (`RequestEnvelope`, `ResponseEnvelope`).

### Python Layer (`python/`)

- **`shm_worker.py`**: The `AIWorker` class implements the main command loop. Dispatches to `handle_create_shm`, `load_model`, and `_process_slot`.
- **`inference_engine.py`**: Stateless core. `inference()` handles per-frame SR; `inference_batch()` processes multiple frames. Manages `PinnedStagingBuffers` for DMA.
- **`model_manager.py`**: `ModelLoader` handles logic for weight resolution (`weights/*.pth`) and VRAM eviction for "heavy" models (Transformers).
- **`research_layer.py`**: Multi-model blending pipeline. Implements `SpatialRouter` for edge/texture-aware routing.

### UI Layer (`ui/src/`)

- **`Store/useJobStore.tsx`**: Zustand store for app-wide state: `isProcessing`, `progressPercent`, `upscaleConfig`.
- **`hooks/useUpscaleJob.ts`**: High-level orchestrator. Toggles between Python (`upscale_request`) and Native (`rave_upscale`) engines.
- **`components/upscale/`**: UI modules for specific Model Parameters.

---

## 5. Runtime Model

### Processes & Lifecycle

1. **Tauri Host (Rust)**: Parent process. Manages window and orchestration.
2. **Python Worker**: Lazy-spawned sidecar. Managed by `python_env.rs::ProcessGuard` (RAII kill-on-drop).
3. **FFmpeg Pipeline**: Decoder Process -> Rust Pipe -> SHM -> Python GPU -> SHM -> Rust Pipe -> Encoder Process.

### Concurrency & Scheduling

- **Rust**: Tokio multi-threaded executor. Background `system-stats` emitter every 2s.
- **Python**: Single-threaded event loop to prevent GIL contention and CUDA context fragmentation. Background "Watchdog" thread (`watchdog.py`) monitors parent PID for safety.
- **IPC**: Zenoh provides async pub-sub. SHM provides a 6-slot circular buffer to hide I/O latency.

---

## 6. Critical Flows

### Flow 1: Video Upscale Pipeline (Python Path)

- **Trigger**: `ui/src/hooks/useUpscaleJob.ts::startUpscale`.
- **Backend**: `upscale.rs::run_upscale_job` initializes `VideoShm` and `Zenoh`.
- **Handshake**: Rust sends `CommandKind::CreateShm` to Python.
- **Loop**:
    1. FFmpeg writes to `VideoDecoder::stdout`.
    2. Rust `VideoShm::transition_slot_state` to `RUST_WRITING`.
    3. Rust signals `READY_FOR_AI`.
    4. Python `shm_worker.py::_process_slot` performs SR.
    5. Python signals `READY_FOR_ENCODE`.
    6. Rust writes to `VideoEncoder::stdin`.

### Flow 2: Research Parameter Hot-Reload

- **Trigger**: User moves slider in `ui/src/components/AIUpscaleNode.tsx`.
- **Path**: `tauri::command` -> `control.rs::set_research_config`.
- **Broadcast**: Rust publishes to Zenoh topic `vf/control/params`.
- **Python Side**: Background listener in `shm_worker.py` receives JSON -> Calls `research_layer.py::update_params`.

### Flow 3: VRAM-Safe Model Switching

- **Trigger**: `ui` changes `primaryModelId`.
- **Path**: `model_manager.py::_ensure_loaded`.
- **Logic**: If new model is "heavy" (`_is_heavy`), existing models are evicted via `unload_heavy_models()` -> `torch.cuda.empty_cache()`.

---

## 7. Data Model

- **SHM Header**: Defined in `shm_worker.py::Config` and `src-tauri/src/shm.rs`. Contains `magic_byte`, `version`, `width`, `height`, `scale`, and atomics for ring pointers.
- **Weights**: Safetensors or TorchScript/Pickle in `weights/` directory.
- **IPC Schemas**: JSON-based. See `ipc/protocol.schema.json`.

---

## 8. External Dependencies

- **FFmpeg 6.0+**: Essential for decoding/encoding. Must have `h264_nvenc` and `hevc_nvenc`.
- **Zenoh 1.0.x**: High-perf orchestration bus.
- **ONNX Runtime**: Used by `engine-v2` for native paths.

---

## 9. Configuration & Precedence

1. **CLI Flags**: Highest precedence (e.g., `--port`, `--shm-ring-size`).
2. **Env Vars**: `VIDEOFORGE_DEV_PYTHON`, `RUST_LOG`.
3. **Hardcoded Defaults**: `shm_worker.py::Config` (e.g., `TILE_SIZE=512`).

---

## 10. Observability

- **Logs**: Real-time log capture in `ui/src/components/LogsPanel.tsx`. Backend emits events for every frame completion or model load failure.
- **System Stats**: CPU/RAM/VRAM metrics emitted as Tauri events.
- **SHM Diagnostics**: Rust backend verifies `Global Header` version on mount to prevent stale buffer access.

---

## 11. Security & Trust Boundaries

- **Input Validation**: `upscale.rs` validates paths exist and dimensions are positive before spawning workers.
- **Sandboxing**: Python worker restricted to a loop; no network access (except local Zenoh bus).
- **Model Safety**: `model_manager.py::ALLOW_UNSAFE_LOAD` (default `True`) allows legacy pickles. Set to `False` for production strict mode to only allow Safetensors.

---

## 12. Performance

- **Hot Path**: `VideoShm::transition_slot_state` uses `compare_exchange` to avoid mutex locks.
- **Zero-Copy**: Pinned memory buffers in `inference_engine.py` allow `torch.from_numpy` to alias system-memory pixels without a copy.
- **Scaling Limit**: PCIe bandwidth is the primary bottleneck for raw 4K upscaling.

---

## 13. Invariants & Contracts

- **Thread Safety**: `VideoShm` is `unsafe impl Send + Sync`. Correctness is enforced by atomic state checks.
- **Deterministic Output**: Enforced by `enforce_deterministic_mode()` in `inference_engine.py` (disables `cudnn.benchmark`).
- **One-Active-Job**: UI/Rust state prevents concurrent upscale jobs to avoid GPU contention.

---

## 14. Risk Register

| Risk | Severity | Likelihood | Mitigation |
| :--- | :--- | :--- | :--- |
| SHM Version Mismatch | High | Medium | Version check in `shm.rs::VideoShm::open`. |
| VRAM Fragmentation | Medium | High | `model_manager.py` aggressive cache clearing on heavy model load. |
| Python Environment Drift | Medium | Low | `pyproject.toml` pins all dependencies. |

---

## 15. Action Plan: Top Improvements

1. **Unified Python Config**: Migrate all `requirements.txt` logic to `pyproject.toml` (Location: `python/`, Effort: Low).
2. **Dynamic Tiling**: Auto-detect VRAM and set `TILE_SIZE` (Location: `model_manager.py`, Impact: High).
3. **Rust-Native Decoding**: Move FFmpeg calls inside `engine-v2` crates for lower overhead (Location: `third_party/rave`, Effort: High).
4. **EMA Buffer Visibility**: Expose Blender Engine EMA stats to UI for better quality tuning.
5. **DirectML Backend**: Enable non-NVIDIA GPU support via ONNX Dynamic EP.
6. **Strict Pickle Lockdown**: Toggle `ALLOW_UNSAFE_LOAD` to `False` by default.
7. **Typed IPC Migration**: Finish migration of all Zenoh messages to `RequestEnvelope::typed`.
8. **Automated VRAM Bench**: Add a "Stress Test" command to auto-detect OOM limits.
9. **Linux Support**: Resolve Windows-native `EventSync` dependencies for cross-platform portability.
10. **CI Smoke Expansion**: Add a full "End-to-End" video roundtrip to the `gpu-rave` CI lane.

---

**Audit Date**: 2026-02-26  
**Status**: Stable  
**Verified by**: Architecture Audit Tool
