# State and Persistence

Purpose: describe what state exists, who owns it, where it lives, and whether it persists.

## Summary

`Current truth`:
- The repo does not present an obvious database-backed application state model.
- Most state is either:
  - filesystem-backed
  - process-local env/runtime state
  - in-memory UI state

## State Inventory

| State class | Owner | Storage | Lifetime | Durability | Source |
|---|---|---|---|---|---|
| Input media path | UI request + backend command | Filesystem path chosen by user | Per job | Durable outside app | `ui/src/hooks/useUpscaleJob.ts` |
| Output media path | UI request + backend command | Filesystem path chosen or generated | Per job | Durable outside app | `ui/src/hooks/useUpscaleJob.ts`, `src-tauri/src/commands/upscale.rs`, `src-tauri/src/commands/native_engine.rs` |
| UI jobs list | React app state | In memory | App session | Transient | `ui/src/types.ts`, `ui/src/App.tsx` |
| UI processing flags and upscale config | Zustand store | In memory | App session | Transient | `ui/src/Store/useJobStore.tsx` |
| Last output path | Zustand store | In memory | App session | Transient | `ui/src/Store/useJobStore.tsx` |
| Model registry | Backend model discovery | Filesystem scan | Recomputed on demand | Durable because files are durable | `src-tauri/src/models.rs` |
| Installed Python runtime | Backend runtime discovery | `%LOCALAPPDATA%/VideoForge/python/` | Across sessions | Durable | `src-tauri/src/python_env.rs` |
| Weights in dev paths | Backend model discovery | `weights/`, `python/weights/`, parent-relative paths | Across sessions | Durable | `src-tauri/src/models.rs` |
| Python worker capabilities | Backend launch config | Process memory, optional manifest snapshot | Per run | Mostly transient | `src-tauri/src/python_env.rs`, `src-tauri/src/run_manifest.rs` |
| Run manifest | Backend optional artifact | Output-adjacent `.videoforge_runs/<job_id>/` | Per run | Durable when enabled | `src-tauri/src/run_manifest.rs`, `src-tauri/src/commands/upscale.rs`, `src-tauri/src/commands/native_routing.rs` |
| Native runtime flags | Process env vars | Process environment | Process lifetime | Transient unless set externally | `src-tauri/src/commands/native_engine.rs` |
| TensorRT cache | Native runtime | Temp or configured cache dir | Across matching runs | Durable on disk | `src-tauri/src/commands/native_engine.rs` |
| Python SHM backing store | Host/worker runtime | Temp/file-backed SHM | Per run | Transient | `src-tauri/src/commands/upscale.rs`, `python/shm_worker.py` |

## Important Non-Persistence Facts

`Current truth`:
- UI job history is not persisted across app restarts
- Pause/resume state is not durable because it is not a real backend execution contract
- Native runtime enablement is env-driven, not stored as an application setting in the inspected code
- Run-artifact persistence is opt-in through `VIDEOFORGE_ENABLE_RUN_ARTIFACTS=1`

## Run Manifest Scope

`Current truth`:
- Run manifest support exists for both Python and native command paths through `maybe_write_run_manifest`
- App command paths keep manifests off by default unless `VIDEOFORGE_ENABLE_RUN_ARTIFACTS=1` is set
- Native manifests now record engine/route/runtime fields through the shared artifact schema

## Operational Guidance

When documenting or changing state, always answer:
- who owns it
- where it lives
- whether it survives restart
- whether it is user-visible
- whether it is route-specific

## Update Trigger

Update this doc when adding:
- persisted settings
- queue persistence
- new manifests
- caches
- temp artifact rules
- runtime flags that affect execution
