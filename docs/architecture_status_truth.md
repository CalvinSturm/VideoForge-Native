# Architecture and Status Truth

Purpose: short, durable description of the system as it exists now.

Use this doc for:
- engine family definitions
- route selection rules
- compile-time vs runtime gating
- shipped vs experimental status

Do not use this doc for:
- resume-point tracking
- implementation task lists
- speculative design

Status language:
- `Current truth`: supported directly by checked-in code
- `Inference`: likely true, but not fully provable from this workspace
- `Needs verification`: requires runtime confirmation
- `Planned`: appears in plans or handoffs, not authoritative by itself

## Canonical Summary

`Current truth`:
- VideoForge has two engine families:
  - Python sidecar path via `upscale_request`
  - Native video path via `upscale_request_native`
- The native family has two execution routes:
  - direct in-process route using `engine-v2`
  - CLI-backed route using the `rave` binary adapter
- The UI only attempts the native family for video jobs when `useNativeEngine` is enabled and the selected model format is ONNX.
- The Python path remains the default and broadest-compatibility path.

`Current truth` source files:
- `ui/src/hooks/useUpscaleJob.ts`
- `src-tauri/src/commands/upscale.rs`
- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/models.rs`
- `src-tauri/src/lib.rs`

## Engine Families

### Python sidecar family

`Current truth`:
- App entrypoint: `upscale_request`
- Primary scope: image jobs, video jobs, PyTorch model support, broader model compatibility
- Runtime shape: Rust host + Python worker + FFmpeg decode/encode + SHM/IPC
- Progress is emitted through Tauri events from `src-tauri/src/commands/upscale.rs`

Key files:
- `src-tauri/src/commands/upscale.rs`
- `src-tauri/src/video_pipeline.rs`
- `src-tauri/src/python_env.rs`
- `python/shm_worker.py`
- `python/model_manager.py`

Status:
- `Shipped/default`

### Native video family

`Current truth`:
- App entrypoint: `upscale_request_native`
- Scope: ONNX-backed video jobs only
- App-facing command exists even when the feature is not compiled in; in that case it returns `FEATURE_DISABLED`
- Runtime opt-in is required even when compiled

Key files:
- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/rave_cli.rs`
- `engine-v2/src/lib.rs`

Status:
- `Opt-in`

## Native Family Routes

### Route A: direct native

`Current truth`:
- Selected only when `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1` and `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`
- Uses the in-process `engine-v2` path
- Current direct path uses packet-aware FFmpeg demux into the engine and packet-aware FFmpeg muxing out of the engine
- Some direct failures fall back to CLI-native

Key files:
- `src-tauri/src/commands/native_engine.rs`
- `engine-v2/src/engine/pipeline.rs`
- `engine-v2/src/backends/tensorrt.rs`

Status:
- `Opt-in experimental/productizing`

### Route B: CLI-backed native

`Current truth`:
- Used when native is enabled but direct mode is not requested
- Also used as selected fallback from some direct-native failures
- App contract is known in this repo; downstream `rave` internal execution is not fully inspectable here

Key files:
- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/rave_cli.rs`
- `src-tauri/src/commands/rave.rs`

Status:
- `Opt-in compatibility route`

## Routing Rules

### UI routing

`Current truth`:
- UI attempts native only when:
  - `upscaleConfig.useNativeEngine === true`
  - media mode is `video`
  - selected model format is `onnx`
- If the selected model is not ONNX, the UI warns and falls back to the Python path

Primary file:
- `ui/src/hooks/useUpscaleJob.ts`

### Backend routing

`Current truth`:
- Native command validates inputs first
- If `native_engine` feature is missing, returns `FEATURE_DISABLED`
- If runtime opt-in is off, returns `NATIVE_ENGINE_DISABLED`
- If direct mode is requested, backend attempts direct-native and may fall back to CLI-native for selected error classes
- If direct mode is not requested, backend uses CLI-native

Primary file:
- `src-tauri/src/commands/native_engine.rs`

## Compile-Time vs Runtime Gating

| Gate | Meaning | Source |
|---|---|---|
| `native_engine` Cargo feature | Compiles the direct native implementation into the app | `src-tauri/src/commands/native_engine.rs` |
| `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1` | Enables use of the native family at runtime | `src-tauri/src/commands/native_engine.rs` |
| `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` | Requests the direct in-process native route | `src-tauri/src/commands/native_engine.rs` |
| ONNX model format | Native-family eligibility at UI routing layer | `ui/src/hooks/useUpscaleJob.ts`, `src-tauri/src/models.rs` |

## Shipped vs Experimental

`Current truth`:
- Shipped/default:
  - Python engine family
  - filesystem-discovered models
  - in-memory UI queue
- Shipped but gated:
  - app-facing native command surface
- Experimental/productizing:
  - direct-native route as default user path
  - route-comparable benchmarking across all paths
  - runtime capability claims beyond what the capability matrix explicitly states

## Known Non-Contracts

`Current truth`:
- Queue pause/resume is not part of the supported product surface or backend command contract
- Planning docs and handoff docs are not canonical authority
- CLI adapter contract does not prove internal `rave` runtime behavior

## Update Trigger

Update this doc when any of the following changes:
- route selection rules
- feature gating
- fallback rules
- shipped vs experimental status
- engine family definitions
