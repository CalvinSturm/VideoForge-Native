# Runtime Path Contracts

Purpose: define each execution path as a contract, not a loose description.

Status language:
- `Current truth`
- `Inference`
- `Needs verification`

## Contract Shape

For each route, keep four facts separate:
- requested route
- resolved route
- executed route
- fallback behavior

UI intent is not execution truth.

## Route 1: Python path

### Contract

- Requested by: UI calling `upscale_request`
- Resolved by: Tauri command + Python environment resolution
- Executed by: Rust host path with Python worker sidecar
- Route ID to use in docs: `python_sidecar`

### Current truth

- Entry command: `src-tauri/src/commands/upscale.rs::upscale_request`
- Python environment is resolved from installed runtime or dev fallback locations
- Worker launch arguments are built in `src-tauri/src/python_env.rs`
- Progress events are emitted through `upscale-progress`
- Run manifest writing exists here and is gated off by default in the current app command path
- Run-start snapshot logging now exists with schema `videoforge.runtime_config_snapshot.python.v1`

### Needs verification

- Backend-owned cancel semantics
- Runtime-effective worker capability flags beyond the subset actually consumed by the worker

## Route 2: Native direct

### Contract

- Requested by: `upscale_request_native` when direct mode is enabled
- Resolved by: env-gated executor selection in native command layer
- Executed by: in-process `engine-v2`
- Route ID to use in docs: `native_direct`

### Current truth

- Compile-time gate: `native_engine` Cargo feature
- Runtime gate: `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1`
- Direct gate: `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`
- FFmpeg runtime path resolution is handled in `src-tauri/src/commands/native_engine.rs`
- Direct pipeline uses packet-aware FFmpeg demux, in-process `engine-v2`, and packet-aware FFmpeg muxing in the host path
- Result includes requested/executed executor, direct attempted flag, fallback flags, and perf fields
- Run-start snapshot logging now exists with schema `videoforge.runtime_config_snapshot.native.v1`

### Fallback contract

`Current truth`:
- Selected direct failures with `ENCODER_INIT` or `PIPELINE` plus NVENC/software-fallback signatures can fall back to CLI-native

Primary file:
- `src-tauri/src/commands/native_engine.rs`

### Needs verification

- Exact downstream runtime parity between app-native and tool-native bootstraps

## Route 3: Native CLI

### Contract

- Requested by: `upscale_request_native` when native is enabled without direct mode
- Also reached by: direct-native fallback
- Executed by: `rave` subprocess adapter
- Route ID to use in docs: `native_via_rave_cli`

### Current truth

- Adapter code lives in `src-tauri/src/rave_cli.rs`
- Output contract is final JSON on stdout plus progress summaries parsed from stderr
- App-facing result contract is normalized in `src-tauri/src/commands/native_engine.rs`

### Inference

- Internal execution behavior of the `rave` binary is not fully inspectable in this workspace

### Contract limit

Do not infer any of the following from adapter code alone:
- internal frame movement
- exact downstream metric collection method
- exact downstream runtime fallback behavior

## Evidence Sources

Preferred evidence when diagnosing runtime truth:
1. route-identifying run logs and result objects
2. command-level code
3. smoke/bench output
4. README or handoff docs

## Runtime Snapshot Contract

`Current truth`:
- Python runs log `videoforge.runtime_config_snapshot.python.v1` at run start
- Native runs log `videoforge.runtime_config_snapshot.native.v1` at run start
- Native direct-to-CLI fallback also logs a `route_fallback` native snapshot

The snapshots are runtime-owned logs intended to capture resolved execution intent at launch time, not UI request state.

## Update Trigger

Update when any route selection, gating, result field, or fallback rule changes.
