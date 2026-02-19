# Native Engine MVP

`upscale_request_native` — GPU-resident video upscaling via engine-v2.

---

## Overview

The native engine path bypasses the Python sidecar entirely.  It runs the
full upscaling pipeline inside a single Rust async task using:

| Stage      | Implementation                             |
|------------|--------------------------------------------|
| Decode     | NVIDIA NVDEC (CUDA Video Codec SDK)        |
| Preprocess | CUDA kernels (NV12 → RGB planar float)    |
| Inference  | TensorRT via ONNX Runtime (ORT)           |
| Postprocess| CUDA kernels (RGB → NV12)                 |
| Encode     | NVIDIA NVENC                               |
| Container  | FFmpeg (demux input, mux output + audio)  |

The Python pipeline (`upscale_request`) remains the **default** and is
unaffected by this feature.

---

## How to Enable

### 1. Build requirements

- **CUDA Toolkit** ≥ 12.0 installed and in `PATH`
- **TensorRT** ≥ 8.6 installed
- **ONNX Runtime** with CUDA + TensorRT Execution Provider
  (set `ORT_LIB_LOCATION` if not on system path)
- **FFmpeg** in `PATH` (required for demux/mux, same as base requirement)

### 2. Build with the feature flag

```bash
cd src-tauri
cargo build --release --features native_engine
```

Or from the workspace root:

```bash
npm run build -- -- --features native_engine
```

### 3. Verify the feature is active

```bash
cargo test --features native_engine -p videoforge -- shm
```

---

## Calling the Command

### From the Tauri frontend (TypeScript)

```typescript
import { invoke } from "@tauri-apps/api/core";

const result = await invoke<{
  output_path: string;
  engine: string;
  frames_processed: number;
  audio_preserved: boolean;
}>("upscale_request_native", {
  inputPath: "/path/to/input.mp4",
  outputPath: "",           // auto-generated if empty
  modelPath: "/path/to/model.onnx",
  scale: 4,
  precision: "fp32",        // "fp32" | "fp16"
  audio: true,              // preserve audio from input
});

console.log("Output:", result.output_path);
console.log("Frames:", result.frames_processed);
```

### From the CLI (smoke test)

The command is a Tauri IPC command and cannot be called directly from the
shell.  Use the JavaScript snippet above in the DevTools console when the app
is running, or write a test harness using `tauri-driver`.

---

## Pipeline Details

### Container strategy (MVP)

```
Input .mp4 / .mkv / …
       │
       ▼ ffmpeg -i input -vcodec copy -an -bsf:v h264_mp4toannexb
       │
  temp/vf_native_input_<ts>.h264   (Annex B elementary stream, host file)
       │
       ▼ FileBitstreamSource → NvDecDecoder (NVDEC)
       │
  GPU frames (NV12, device resident)
       │
       ▼ PreprocessPipeline (NV12 → RGB planar float)
       │
       ▼ TensorRtBackend (inference)
       │
       ▼ PostprocessPipeline (RGB → NV12)
       │
  GPU frames (NV12, device resident)
       │
       ▼ NvEncoder (NVENC) → FileBitstreamSink
       │
  temp/vf_native_output_<ts>.h264
       │
       ▼ ffmpeg -i video_out -i original_input -c:v copy -c:a copy -map 0:v -map 1:a?
       │
  final output.mp4
```

### HEVC fallback

If the input cannot be demuxed to H.264 Annex B, the pipeline automatically
retries with the HEVC bitstream filter (`hevc_mp4toannexb`).  If both fail,
a structured error with code `DEMUX_FAILED` is returned.

---

## Known Limitations (MVP)

| Limitation | Detail |
|---|---|
| **Containers** | Only H.264 and HEVC elementary streams are supported by the native decoder.  ProRes, AV1, VP9, etc. are not. |
| **Audio** | Audio is copied unchanged from the original input by FFmpeg.  No transcoding.  If FFmpeg cannot copy the audio track, `audio_preserved = false` is returned. |
| **Duration** | Frame count is not validated against container duration in this MVP.  Short clips (< 10 frames) may produce incorrect output. |
| **Scale factor** | The TensorRT model must match the `scale` parameter.  Mismatches produce garbage output — no runtime check yet. |
| **Codec detection** | The pipeline tries H.264 first, then HEVC.  If the video codec is neither, use the Python pipeline (`upscale_request`). |
| **Variable frame rate** | VFR input is not handled.  Pre-convert to CFR with FFmpeg before passing to native engine. |
| **No progress events** | Unlike `upscale_request`, this command does not emit `upscale-progress` Tauri events.  Progress tracking is planned for a future milestone. |

---

## Fallback to Python Engine

If the native engine fails (build error, missing GPU capability, unsupported
codec), call `upscale_request` instead — it uses the Python sidecar and is
always available regardless of the `native_engine` feature flag.

```typescript
// Python engine fallback
const result = await invoke<string>("upscale_request", {
  inputPath: "/path/to/input.mp4",
  outputPath: "",
  model: "RCAN_x4",
  editConfig: { /* ... */ },
  scale: 4,
  precision: "fp32",
});
```

---

## Error Codes

| Code               | Meaning                                                        |
|--------------------|----------------------------------------------------------------|
| `FEATURE_DISABLED` | Binary was built without `--features native_engine`.           |
| `INPUT_NOT_FOUND`  | `input_path` does not exist on disk.                           |
| `MODEL_NOT_FOUND`  | `model_path` does not exist on disk.                           |
| `GPU_INIT`         | CUDA context creation failed (no GPU, wrong driver, etc.).     |
| `BACKEND_INIT`     | TensorRT backend failed to load model (bad ONNX, no TRT EP).  |
| `KERNEL_COMPILE`   | CUDA kernel compilation failed (check CUDA toolkit version).   |
| `DEMUX_FAILED`     | FFmpeg could not extract H.264 or HEVC elementary stream.      |
| `SOURCE_OPEN`      | Elementary stream file could not be opened.                    |
| `DECODER_INIT`     | NVDEC decoder init failed.                                     |
| `ENCODER_INIT`     | NVENC encoder init failed.                                     |
| `PIPELINE`         | Runtime error during decode/infer/encode.                      |
| `MUX_FAILED`       | FFmpeg mux (video + audio) failed.                             |

---

## Smoke Test (manual verification steps)

```bash
# 1. Build with the native engine feature
cd src-tauri
cargo build --features native_engine 2>&1 | tail -20

# 2. Verify the feature compiled in (look for NvDecDecoder in the binary)
nm -C target/debug/app_lib.dll 2>/dev/null | grep NvDecDecoder | head -5

# 3. Run the app in dev mode with the feature
cd ..
RUST_LOG=videoforge=debug npm run dev -- -- --features native_engine

# 4. In the browser DevTools console, call the command:
#    (replace paths with real paths on your system)
await window.__TAURI__.core.invoke("upscale_request_native", {
  inputPath: "C:/test/sample_720p.mp4",
  outputPath: "",
  modelPath: "C:/models/rcan_x4.onnx",
  scale: 4,
  precision: "fp32",
  audio: true
})

# Expected result:
# { output_path: "C:/test/sample_720p_<ts>_native_upscaled.mp4",
#   engine: "native_v2",
#   frames_processed: <N>,
#   audio_preserved: true }
```

### Expected failure (feature not compiled):

```json
{"code":"FEATURE_DISABLED","message":"The native_engine feature is not compiled in..."}
```
