# Native Engine Handoff

Date: 2026-03-06

## Current Status

Direct in-process native engine validation is working end-to-end for both:

- `weights/2x_SPAN_soft.onnx` with `scale=2`
- `weights/rcan_4x.onnx` with `scale=4`

The earlier blockers in PR 2 are resolved:

- decode/preprocess event handoff no longer crashes with `0xC0000005`
- TensorRT metadata no longer mis-infers dynamic-shape models as `scale=2`
- RCAN 4x direct-path smoke now muxes successfully

Default app behavior is still unchanged:

- `upscale_request_native()` still routes to `run_native_via_rave_cli()` by default
- true in-process path is opt-in behind `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`

## Key Code Changes

### Stream ordering / direct-path hardening

- `engine-v2/src/core/context.rs`
  - `GpuContext::sync_stream()` now calls `cuStreamSynchronize`

- `engine-v2/src/codecs/sys.rs`
  - added `CUDA_ERROR_NOT_READY`
  - added FFI bindings for `cuEventQuery`, `cuEventSynchronize`, `cuStreamSynchronize`

- `engine-v2/src/engine/pipeline.rs`
  - introduced `StreamReadyEvent`
  - decode -> preprocess now uses per-frame event handoff
  - preprocess -> inference now uses per-frame event handoff
  - postprocess -> encode now uses per-frame event handoff
  - added temporary boundary logs (`PIPELINE-BND: ...`) for debugging
  - added CUDA context rebinding after async suspension points

- `engine-v2/src/backends/tensorrt.rs`
  - `process()` rebinds the CUDA context after the async mutex lock

- `engine-v2/src/codecs/nvdec.rs`
  - fixed `get_raw_stream()` to use `cudarc::driver::CudaStream.stream.cast()` instead of UB pointer punning
  - current decode copy path is still using synchronous `cuMemcpy2D_v2`
  - decoder currently records an already-satisfied event after synchronous copy/unmap
  - there is still extra NVDEC debug logging in this file

### Direct-path routing

- `src-tauri/src/commands/native_engine.rs`
  - added `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` switch
  - when enabled, `upscale_request_native()` calls `run_native_pipeline(...)`
  - otherwise existing CLI-backed behavior remains unchanged

### Dynamic-scale inference fix

- `engine-v2/src/backends/tensorrt.rs`
  - `extract_metadata()` now:
    - computes scale from static tensor ratios when possible
    - otherwise infers scale from model filename patterns
    - errors if dynamic axes exist and scale cannot be inferred
  - added tests for filename-based scale inference

### Mux-path hardening

- `src-tauri/src/commands/native_engine.rs`
  - FFmpeg mux now captures stdout/stderr instead of discarding it
  - temp files are preserved on mux failure
  - temp files are only deleted after successful mux

## Validation Performed

### Unit test

Passed:

- `cargo test --manifest-path engine-v2/Cargo.toml infer_scale_from_model_path`

### Direct native smoke

Commands used:

```powershell
$env:VIDEOFORGE_ENABLE_NATIVE_ENGINE='1'
$env:VIDEOFORGE_NATIVE_ENGINE_DIRECT='1'
$env:RUST_LOG='info'
.\src-tauri\target\debug\smoke.exe --e2e-native --input .\test_input.mp4 --e2e-onnx .\weights\2x_SPAN_soft.onnx --e2e-scale 2 --precision fp32 --keep-temp
.\src-tauri\target\debug\smoke.exe --e2e-native --input .\test_input.mp4 --e2e-onnx .\weights\rcan_4x.onnx --e2e-scale 4 --precision fp32 --keep-temp
```

Observed:

- `2x_SPAN_soft.onnx` passed with output `256x256`
- `rcan_4x.onnx` passed repeatedly with output `512x512`

## Important Notes

### Still not production default

The in-process path is still not the default runtime. If the next task is to make the Tauri-native path the real shipped path, routing still needs to change.

### Debug instrumentation remains

There are still many temporary `info!` logs in:

- `engine-v2/src/codecs/nvdec.rs`
- `engine-v2/src/engine/pipeline.rs`

These were useful for narrowing the crash boundary and should probably be reduced once the next work item is chosen.

### Synchronous NVDEC copy remains

`nvdec.rs` is still using synchronous `cuMemcpy2D_v2` instead of the earlier async design. This was an intentional stabilization step during root-cause work. If the next Codex instance continues PR 2 performance work, this area should be revisited carefully.

## Recommended Next Work

Most sensible next target:

1. Clean up temporary debug logging in `nvdec.rs` and `pipeline.rs`
2. Re-audit whether decode copy can safely return to async submission without reintroducing the direct-path crash
3. If correctness is stable, continue PR 2 by reducing remaining coarse sync costs and then move to batching work

Alternative next target:

1. Make the direct in-process native path the default runtime path instead of the CLI-backed path
2. Update docs to reflect that runtime change

## Known Environment Quirk

There was one transient Windows rebuild issue:

- `cargo build --manifest-path src-tauri/Cargo.toml --bin smoke --features native_engine`
- failed once because `src-tauri/target/debug/smoke.exe` was locked

This was not a code bug. Re-running after the process exited worked.
