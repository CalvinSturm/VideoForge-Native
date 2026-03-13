# AGENTS.md

## TL;DR
- `ui/` is the frontend: a React 19 + Vite + Zustand desktop UI rendered inside Tauri.
- `src-tauri/` is the app shell and desktop backend/API: Tauri commands, process orchestration, FFmpeg pipeline control, model discovery, and engine management.
- `python/` is the Python super-resolution sidecar engine: PyTorch-based inference, model loading, shared-memory worker loop, and optional research/blending layers.
- `engine-v2/` is the native Rust engine crate: a GPU-resident NVDEC -> preprocess -> TensorRT/ORT -> NVENC pipeline.
- The app has two primary processing engine families: the Python sidecar engine and the native Rust engine. The native path itself has two execution modes: direct in-process `engine-v2` and a CLI-backed native fallback path.
- Engine routing is explicit: the UI only attempts the native path for video jobs when `useNativeEngine` is enabled and the selected model is ONNX; otherwise it uses the Python pipeline.
- Model files are discovered from local `weights/` directories and from `%LOCALAPPDATA%/VideoForge/python/weights`; ONNX models are the native-engine candidates.
- Persistence appears mostly local-file based: input/output media on disk, model weights on disk, optional per-run manifests next to outputs, and UI job state in memory.
- There is no obvious HTTP server or database layer in the inspected repo; the “API” is Tauri invoke commands plus local IPC between Rust and Python.
- Read `ui/src/App.tsx`, `ui/src/hooks/useUpscaleJob.ts`, `src-tauri/src/lib.rs`, `src-tauri/src/commands/upscale.rs`, `src-tauri/src/commands/native_engine.rs`, `python/shm_worker.py`, and `engine-v2/src/lib.rs` first.

## Docs Authority
- Treat the five canonical docs in `docs/` as the primary source of truth for current behavior:
  - `docs/architecture_status_truth.md`
  - `docs/capability_matrix.md`
  - `docs/runtime_path_contracts.md`
  - `docs/state_and_persistence.md`
  - `docs/metrics_trust.md`
- Treat `docs/README.md` as the docs entrypoint.
- Treat `docs/archive/README.md` and files under `docs/archive/` as historical context only.
- Treat handoff docs as working transfer context, not durable architecture authority, unless a canonical doc explicitly delegates to them.
- Treat `SMOKE_TEST.md` as the live operational smoke runbook.

## Repo Shape
- `ui/`: frontend app.
- `src-tauri/`: Tauri desktop host and Rust backend/API.
- `python/`: Python engine package and tests.
- `engine-v2/`: native Rust engine crate.
- `ipc/`: shared protocol/schema files for host/worker SHM and IPC contracts.
- `weights/`: local model/checkpoint storage.
- `third_party/`: bundled native runtimes/deps such as FFmpeg/TensorRT paths referenced by the native path.
- `scripts/`: utility scripts such as model conversion.
- `docs/`: architecture notes, audits, handoff docs, and plans.
- `artifacts/`: build/runtime artifacts, including TensorRT cache roots referenced by scripts/env vars.
- `tools/`: auxiliary tooling area; not inspected in detail.
- `src-tauri/src/bin/`: Rust utility binaries such as smoke tests.

## Frontend
- Lives in `ui/`.
- Framework/runtime: React 19 + Vite + TypeScript, with BlueprintJS, `react-mosaic-component`, Zustand, and Tauri API bindings.
- App entrypoint: `ui/src/index.tsx`.
- Main shell: `ui/src/App.tsx`.
- State/model types: `ui/src/types.ts` and `ui/src/Store/useJobStore.tsx`.
- Upload/queue/results flow at a high level:
  - User picks input/output/model in the React UI.
  - UI builds an upscale payload from current edit/upscale state.
  - UI routes to either `upscale_request_native` or `upscale_request`.
  - Progress/status comes back through Tauri events and updates the in-memory job list.
  - Completed jobs retain output paths for reveal/open actions.
- Frontend changes:
  - UI shell/layout: start in `ui/src/App.tsx`.
  - Job submission/routing: start in `ui/src/hooks/useUpscaleJob.ts`.
  - Shared UI state: start in `ui/src/Store/useJobStore.tsx`.
  - Shared DTOs/types: start in `ui/src/types.ts`.

## Backend / API
- Lives in `src-tauri/`.
- Framework/runtime: Tauri 2 + Rust.
- There is no separate network API visible in lightweight inspection; the app-facing API is Tauri commands registered in `src-tauri/src/lib.rs`.
- Main entrypoints:
  - Desktop entry: `src-tauri/src/main.rs`.
  - Tauri wiring and command registration: `src-tauri/src/lib.rs`.
- Key backend module boundaries:
  - `src-tauri/src/commands/upscale.rs`: Python-engine orchestration path.
  - `src-tauri/src/commands/native_engine.rs`: native-engine command, runtime gating, direct-vs-CLI routing.
  - `src-tauri/src/commands/engine.rs`: engine install/status/reset and model listing.
  - `src-tauri/src/video_pipeline.rs`: FFmpeg decode/encode boundary.
  - `src-tauri/src/python_env.rs`: Python runtime and `shm_worker.py` resolution.
  - `src-tauri/src/models.rs`: local model discovery and weight metadata.
  - `src-tauri/src/run_manifest.rs`: optional per-run manifest persistence.
  - `src-tauri/src/edit_config.rs`: edit/filter config translation for FFmpeg.
- Backend/API changes:
  - New Tauri command or app orchestration: `src-tauri/src/lib.rs` and `src-tauri/src/commands/`.
  - Python worker launch or environment issues: `src-tauri/src/python_env.rs`.
  - Model discovery/config metadata: `src-tauri/src/models.rs`.
  - Local run artifacts: `src-tauri/src/run_manifest.rs`.

## Video Upscaling Engines
- Use this section as a quick orientation map.
- For durable route/gating/support truth, prefer the canonical docs listed above.

- Primary engine families found:
  - Python sidecar engine.
  - Native Rust engine.
- Execution modes found under the native family:
  - Direct in-process native path using `engine-v2`.
  - CLI-backed native fallback path invoked through the Tauri native command.

- Engine 1: Python sidecar engine
  - Lives in `python/`.
  - Main entrypoint appears to be `python/shm_worker.py`.
  - Core responsibilities:
    - shared-memory worker loop
    - Zenoh-based IPC participation
    - PyTorch precision/determinism configuration
    - model loading and frame inference via `python/model_manager.py`
    - optional research/blending/postprocessing via `research_layer.py` and `blender_engine.py`
  - Inference runtime appears to be PyTorch, with model ecosystems including `basicsr`, `realesrgan`, and `spandrel`.
  - Model formats include PyTorch weights and other local checkpoint formats; this is the general/default path.

- Engine 2: Native Rust engine
  - Lives in `engine-v2/`.
  - Exposed to the app through the optional `videoforge-engine` dependency in `src-tauri/Cargo.toml`.
  - Main boundary files:
    - `engine-v2/src/lib.rs`
    - `engine-v2/src/engine/pipeline.rs`
    - `engine-v2/src/backends/tensorrt.rs`
    - `engine-v2/src/codecs/mod.rs`
  - Core responsibilities:
    - GPU-native pipeline orchestration
    - hardware decode/encode
    - device preprocess/postprocess stages
    - TensorRT-backed inference through ONNX Runtime
  - Inference runtime appears to be ONNX Runtime with TensorRT EP, with CUDA EP allowed as fallback inside the engine backend and CPU EP excluded.
  - This engine is clearly ONNX-oriented and optimized for GPU-resident video paths.

- Native execution modes
  - Direct mode:
    - Enabled only when the app is compiled with `native_engine` and `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1`.
    - Further gated by `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`.
    - Routes into `engine-v2` directly.
  - CLI-backed mode:
    - If direct mode is off, or if direct mode fails in some encoder/pipeline cases, the native command can route through a CLI-backed path and returns engine identifiers like `native_via_rave_cli`.
    - This is best treated as a delivery mode for the native engine family, not a separate frontend-facing engine family.

- How the app chooses between engines
  - Frontend routing:
    - Native path is attempted only when `useNativeEngine` is enabled and the job is `video`.
    - The selected model must be ONNX; otherwise the UI warns and falls back to the Python path.
  - Backend/runtime gating:
    - The native command exists even when not compiled, but returns a structured `FEATURE_DISABLED` error if the `native_engine` feature is off.
    - Even when compiled, runtime opt-in is required through `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1`.
    - Direct native execution additionally requires `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`.

- Shared vs engine-specific responsibilities
  - Shared/app-level:
    - frontend job submission
    - Tauri command boundary
    - model discovery metadata
    - edit config / output path selection
    - optional run manifest writing
  - Python-specific:
    - SHM worker lifecycle
    - PyTorch model loading/inference
    - research/blending features
  - Native-specific:
    - NVDEC/NVENC codec pipeline
    - TensorRT/ORT inference
    - GPU-resident bounded pipeline stages

- Important files/directories to read first
  - `src-tauri/src/commands/upscale.rs`
  - `src-tauri/src/commands/native_engine.rs`
  - `src-tauri/src/python_env.rs`
  - `python/shm_worker.py`
  - `python/model_manager.py`
  - `engine-v2/src/lib.rs`
  - `engine-v2/src/engine/pipeline.rs`
  - `engine-v2/src/backends/tensorrt.rs`

## Shared Code
- `ipc/`: shared protocol/schema files for SHM and IPC contracts.
- `src-tauri/src/models.rs`: model metadata discovery shared by UI/backend decisions.
- `src-tauri/src/edit_config.rs`: edit config translation shared across export/upscale flows.
- `src-tauri/src/run_manifest.rs`: shared run metadata persistence shape.
- `ui/src/types.ts`: frontend job/result/request-related DTOs.
- `ui/src/Store/useJobStore.tsx`: centralized frontend upscale config and job/system state.

## Data Model
- Input video / media path
  - Defined in frontend state and passed into Tauri commands.
  - Related to edit config, output path, and job execution.
  - Persisted as normal filesystem paths chosen by the user.

- Job
  - Defined in `ui/src/types.ts` as `Job`.
  - Tracks id, status, progress, ETA, output path, error data, and native-engine result metadata.
  - Appears persisted only in frontend memory during app runtime.

- Upscale config / job config
  - Frontend config lives in `ui/src/Store/useJobStore.tsx` as `UpscaleConfig`.
  - Backend Python-job config lives in `src-tauri/src/commands/upscale.rs` as `UpscaleJobConfig`.
  - Relates model choice, scale, resolution mode, and engine choice to a concrete run.
  - Appears persisted in memory, with optional per-run snapshotting via run manifests.

- Edit config
  - Frontend shape is in `ui/src/types.ts`.
  - Rust-side translation lives in `src-tauri/src/edit_config.rs`.
  - Relates trim/crop/rotation/color/fps settings to export/upscale execution.
  - Appears transient, per-request.

- Model metadata / registry
  - Defined in `src-tauri/src/models.rs` as `ModelInfo`.
  - Includes `id`, `scale`, `filename`, `format`, and absolute `path`.
  - Drives frontend model lists and engine routing decisions.
  - Persisted as files on disk under `weights/` or `%LOCALAPPDATA%/VideoForge/python/weights`.

- Run manifest
  - Defined in `src-tauri/src/run_manifest.rs` as `RunManifestV1`.
  - Captures job id, paths, scale, precision, model key, worker caps, protocol versions, and app version.
  - Persisted only when enabled, under an output-adjacent `.videoforge_runs/<job_id>/` directory.

- Worker capability metadata
  - Defined in `src-tauri/src/python_env.rs` as `WorkerCaps`.
  - Snapshotted in `src-tauri/src/run_manifest.rs`.
  - Describes IPC/protocol/determinism toggles for the Python worker path.

- Native runtime capability/config flags
  - Defined operationally in `src-tauri/src/commands/native_engine.rs`.
  - Includes compile-time feature gating plus env vars for native enablement, direct mode, and TensorRT cache behavior.
  - Appears process-local rather than database-persisted.

- Users/accounts/history
  - No user/account model or database schema was found in lightweight inspection.
  - History beyond the current session appears limited to output files and optional run manifests.

## Request / Video Flow
- Python path:
  - video input -> frontend config/edit state -> Tauri `upscale_request` -> Python env resolution -> Zenoh handshake -> SHM ring allocation -> FFmpeg decode -> Python frame inference -> FFmpeg encode -> output file -> optional run manifest -> frontend job completion
- Native path:
  - video input -> frontend decides native eligibility -> Tauri `upscale_request_native` -> runtime gating -> direct `engine-v2` or CLI-backed native mode -> hardware decode -> GPU preprocess -> TensorRT/ORT inference -> hardware encode / mux -> output file -> frontend job completion
- Shared routing shape:
  - user choice (`useNativeEngine`) + media type (`video`) + model format (`onnx` vs `pytorch`) + runtime capability flags determine which engine family executes

## Build / Run / Test
- Root commands from `package.json`:
  - `npm run dev`
  - `npm run dev:native`
  - `npm run build`
  - `npm run build:native`
  - `npm run ui-install`
- Tauri build wiring:
  - `src-tauri/tauri.conf.json` runs the UI dev server from `ui/` and builds frontend assets from `ui/dist`.
- Python setup from repo evidence:
  - `pip install -r requirements.txt`
  - `python/pyproject.toml` also defines the sidecar package and optional `dev` extras.
- Test/typecheck commands called out in `README.md`:
  - `cd src-tauri && cargo test`
  - `cd ui && npx tsc --noEmit`
- Test locations found:
  - `python/tests/`
  - Rust inline tests in files such as `src-tauri/src/run_manifest.rs`
  - smoke-style binaries under `src-tauri/src/bin/`

## Read This First
1. `README.md`
2. `package.json`
3. `src-tauri/src/lib.rs`
4. `ui/src/App.tsx`
5. `ui/src/hooks/useUpscaleJob.ts`
6. `src-tauri/src/commands/upscale.rs`
7. `src-tauri/src/commands/native_engine.rs`
8. `src-tauri/src/python_env.rs`
9. `src-tauri/src/models.rs`
10. `python/shm_worker.py`
11. `python/model_manager.py`
12. `engine-v2/src/lib.rs`
13. `engine-v2/src/engine/pipeline.rs`
14. `engine-v2/src/backends/tensorrt.rs`

## Unknowns / Ambiguities
- The repo references `rave-cli`, but the inspected workspace only exposed build artifacts under `rave-cli/target`, not live source. The CLI-backed native path is confirmed, but its full source boundary was not.
- `engine-v2` is clearly present as a native crate, but lightweight inspection did not confirm whether additional native backend implementations exist beyond the TensorRT backend.
- A separate persistent job database or settings store was not found in lightweight inspection; this appears to be a local-file + in-memory desktop app, but absence was not exhaustively proven.
- `tools/` and some docs may contain additional operational workflows that were intentionally not expanded because they were not needed for the onboarding map.

## Evidence
- Evidence: `package.json`
- Evidence: `README.md`
- Evidence: `ui/package.json`
- Evidence: `ui/src/index.tsx`
- Evidence: `ui/src/App.tsx`
- Evidence: `ui/src/hooks/useUpscaleJob.ts`
- Evidence: `ui/src/types.ts`
- Evidence: `ui/src/Store/useJobStore.tsx`
- Evidence: `src-tauri/Cargo.toml`
- Evidence: `src-tauri/tauri.conf.json`
- Evidence: `src-tauri/src/main.rs`
- Evidence: `src-tauri/src/lib.rs`
- Evidence: `src-tauri/src/commands/upscale.rs`
- Evidence: `src-tauri/src/commands/native_engine.rs`
- Evidence: `src-tauri/src/commands/engine.rs`
- Evidence: `src-tauri/src/commands/rave.rs`
- Evidence: `src-tauri/src/python_env.rs`
- Evidence: `src-tauri/src/models.rs`
- Evidence: `src-tauri/src/run_manifest.rs`
- Evidence: `engine-v2/Cargo.toml`
- Evidence: `engine-v2/src/lib.rs`
- Evidence: `engine-v2/src/engine/mod.rs`
- Evidence: `engine-v2/src/engine/pipeline.rs`
- Evidence: `engine-v2/src/backends/mod.rs`
- Evidence: `engine-v2/src/backends/tensorrt.rs`
- Evidence: `engine-v2/src/codecs/mod.rs`
- Evidence: `python/pyproject.toml`
- Evidence: `python/shm_worker.py`
- Evidence: `python/model_manager.py`
- Evidence: `requirements.txt`
- Evidence: `ipc/protocol.schema.json`
- Evidence: `ipc/shm_protocol.json`
