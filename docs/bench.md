# Headless Benchmark Harness (`videoforge_bench`)

This harness runs the same Rust/Python upscale pipeline used by the Tauri app, but without the UI.

It is intended for repeatable local benchmarking and later integration tests. It does not change default app behavior.

## Prereqs

- `ffmpeg` and `ffprobe` available in `PATH`
- VideoForge Python engine installed (or dev Python environment resolvable by `VIDEOFORGE_DEV_PYTHON`)
- Model weights installed/discoverable by the backend
- GPU driver/runtime installed (Windows-first workflow; NVIDIA path is the current primary path)

## CLI

Run from `src-tauri/`:

```powershell
cargo run --bin videoforge_bench -- `
  --input .\..\samples\input.mp4 `
  --output .\..\samples\output_bench.mp4 `
  --model RCAN_x4 `
  --scale 2 `
  --precision fp16
```

Dry-run (dependency validation only):

```powershell
cargo run --bin videoforge_bench -- `
  --dry-run `
  --input .\dummy.mp4 `
  --output .\dummy_out.mp4 `
  --model RCAN_x4 `
  --scale 2 `
  --precision fp16
```

Optional flags:

- `--deterministic` (uses backend deterministic mode if available; current pipeline accepts it)
- `--edit-config <json>` to apply crop/trim/rotation/color settings using the same `EditConfig` schema as the app

## PowerShell Runner

From repo root:

```powershell
.\tools\bench\run_bench.ps1 `
  -Input .\samples\input.mp4 `
  -Output .\samples\output_bench.mp4 `
  -Model RCAN_x4 `
  -Scale 2 `
  -Precision fp16
```

The script writes stdout JSONL to `tools/bench/out/bench_<timestamp>.jsonl`.

## JSONL Events

Stdout is JSON lines (one JSON object per line), so scripts can parse progress deterministically.

Examples:

```json
{"event":"start","input":"...","output":"...","model":"RCAN_x4","scale":2,"precision":"fp16"}
{"event":"progress","frame":37,"total_frames":240,"fps":11.8,"pct":15,"message":"Processing Frame 37/240","eta_secs":17,"output":null}
{"event":"done","output":"...","elapsed_ms":20345,"frames_encoded":240}
```

Error example:

```json
{"event":"error","message":"ffmpeg not found"}
```

Dry-run success:

```json
{"event":"dry_run_ok"}
```

## Model Keys

`--model` must match a backend-discovered model key (same backend discovery used by the app's model list). If unsure, use a known installed weight filename stem (for example `RCAN_x4` if the weight file is `RCAN_x4.pth`).
