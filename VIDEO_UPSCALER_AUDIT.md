# Video Upscaler Audit

## 1. Executive summary
- Current architecture summary:
  - The application has two real engine families: a Python sidecar pipeline (`src-tauri/src/commands/upscale.rs`, `python/shm_worker.py`, `src-tauri/src/video_pipeline.rs`) and a native GPU pipeline exposed through `upscale_request_native` (`src-tauri/src/commands/native_engine.rs`, `engine-v2/`).
  - The native command has two delivery pathways, not two independent engine cores:
    - `native`: direct in-process `engine-v2` execution.
    - `native-cli`: a subprocess adapter that shells out to a prebuilt `rave` binary through `src-tauri/src/commands/rave.rs` and `src-tauri/src/rave_cli.rs`.
  - The Python path is fundamentally CPU-host-frame based: FFmpeg emits `rgb24` rawvideo into Rust, Rust copies frames into SHM, Python copies SHM into NumPy/Torch tensors, inference runs, then results come back as CPU RGB and are piped back into FFmpeg for encode.
  - The direct native path is fundamentally GPU-resident in the core: FFmpeg demuxes compressed elementary video to stdout, `engine-v2` does NVDEC -> preprocess -> TensorRT/ORT -> postprocess -> NVENC, then FFmpeg muxes the encoded bitstream.
- Top 5 bottlenecks:
  - Python video path forces decode-to-host and encode-from-host RGB24 boundaries, eliminating GPU-resident flow (`src-tauri/src/video_pipeline.rs:64-183`, `src-tauri/src/video_pipeline.rs:341-489`).
  - Python worker still performs multiple per-frame host copies and format churn even after SHM (`python/shm_worker.py:1488-1697`, `python/shm_worker.py:1782-1936`).
  - Python cross-process scheduling relies on SHM state polling and spin sleeps in both Rust and Python (`src-tauri/src/commands/upscale.rs:747-799`, `python/shm_worker.py:1938-2019`).
  - Direct native demux/mux still depends on FFmpeg subprocesses and a coarse chunk-based pseudo-packet source, including synthetic PTS assignment (`src-tauri/src/commands/native_engine.rs:1075-1188`).
  - Benchmarking and observability are asymmetric: native has internal metrics/profiler hooks, but command surfaces and the benchmark binary mostly expose wall time only; Python stage timing fields exist but are not populated (`engine-v2/src/engine/pipeline.rs`, `engine-v2/src/core/context.rs`, `src-tauri/src/commands/upscale.rs:62-90`, `src-tauri/src/bin/videoforge_bench.rs`).
- Top 5 highest ROI optimizations:
  - Stop treating `native-cli` as a parallel architecture; keep a shared native job spec/core and reduce CLI to a thin adapter.
  - Add real stage metrics to both engines and expose them through command results and the benchmark binary.
  - Replace Python path polling with event-driven completion everywhere possible and remove 200us/100us spin loops from the steady state.
  - Remove avoidable Python per-frame copies and redundant postprocess re-uploads by reusing tensors and operating on GPU tensors end-to-end inside the worker.
  - Harden the direct native wrapper around packetization/timestamps and make the FFmpeg demux/mux boundary explicit, measured, and more faithful to packet timing.
- Overall verdict on the current engine split:
  - The Python/native split is justified by capability differences: Python handles PyTorch models, image flows, research/blending layers, and broader model compatibility; native is the performance path for ONNX video.
  - The direct-vs-CLI split is hurting more than helping. It duplicates routing, error policy, config policy, and benchmarking surfaces while only one side (`native`) exposes the actual engine core in this repo.
- Overall verdict on `native` vs `native-cli`:
  - `native` should be the preferred future path.
  - `native-cli` should be reduced to a thin compatibility adapter over a shared native job contract, then deprecated once the direct path is operationally complete.

## 2. Repository architecture inventory
- Engine inventory:
  - Python sidecar engine:
    - Host command: `src-tauri/src/commands/upscale.rs`
    - Video decode/encode helpers: `src-tauri/src/video_pipeline.rs`
    - Worker: `python/shm_worker.py`
    - Model/runtime loader: `python/model_manager.py`
  - Native direct engine:
    - Host command: `src-tauri/src/commands/native_engine.rs`
    - Core pipeline: `engine-v2/src/engine/pipeline.rs`
    - Inference backend: `engine-v2/src/backends/tensorrt.rs`
    - Decode/encode: `engine-v2/src/codecs/nvdec.rs`, `engine-v2/src/codecs/nvenc.rs`
  - Native CLI adapter:
    - App wrapper: `src-tauri/src/commands/native_engine.rs::run_native_via_rave_cli`
    - CLI process runner: `src-tauri/src/commands/rave.rs`, `src-tauri/src/rave_cli.rs`
    - Actual `rave` binary source is not present in this workspace; only the adapter layer is inspectable.
- Pathway inventory:
  - UI dispatch:
    - `ui/src/hooks/useUpscaleJob.ts`
    - `ui/src/App.tsx`
  - Tauri commands:
    - `upscale_request` -> Python engine
    - `upscale_request_native` -> direct native or CLI native depending on env flags
    - `rave_upscale` / `rave_benchmark` -> explicit CLI-native subprocess calls
- Core traits / structs / interfaces:
  - Python path:
    - `UpscaleJobConfig`, `run_upscale_job`, `VideoDecoder`, `VideoEncoder`
    - SHM ring via `shm::VideoShm`
  - Native path:
    - `UpscaleBackend`
    - `UpscalePipeline`
    - `FrameDecoder`, `FrameEncoder`
    - `TensorRtBackend`
    - `NvDecoder`, `NvEncoder`
    - `FfmpegBitstreamSource`, `StreamingMuxSink`
- Dispatch points:
  - Frontend native eligibility: `ui/src/hooks/useUpscaleJob.ts:100-160`
  - Backend native gating: `src-tauri/src/commands/native_engine.rs:564-689`
  - Model format selection: `src-tauri/src/models.rs`
- Capability detection points:
  - Native runtime opt-in: `native_engine_runtime_enabled`, `native_engine_direct_enabled`
  - Python decode/encode feature probes: `probe_nvdec`, `probe_nvenc`
  - ONNX native batch policy: `src-tauri/src/models.rs:124-145`
  - Python ONNX provider selection: `python/model_manager.py:619-669`
- Fallback points:
  - UI native -> Python fallback for non-ONNX models: `ui/src/hooks/useUpscaleJob.ts:145-161`
  - Direct native -> CLI-native fallback on selected encoder/pipeline failures: `src-tauri/src/commands/native_engine.rs:629-679`
  - Python FFmpeg NVDEC/NVENC spawn fallback to software: `src-tauri/src/video_pipeline.rs:142-175`, `src-tauri/src/video_pipeline.rs:521-591`
  - Python ONNX CUDA EP -> CPU EP fallback: `python/model_manager.py:649-669`
- Subprocess boundaries:
  - Python path:
    - Python worker subprocess
    - FFmpeg decode subprocess
    - FFmpeg encode subprocess
  - Native direct:
    - FFmpeg demux subprocess
    - FFmpeg mux subprocess
  - Native CLI:
    - `rave` subprocess
    - Internals of that subprocess are not visible here
- Model/runtime ownership boundaries:
  - Python model ownership lives in the worker process and uses Torch / ORT sessions there.
  - Native direct model ownership lives in-process in `TensorRtBackend` guarded by `Mutex<Option<InferenceState>>`.
  - CLI-native owns model/runtime state out-of-process behind a JSON stdout contract.

## 3. Current pipeline maps
### Engine A: Python sidecar engine
- Short explanation:
  - The host probes input dimensions, creates a local SHM ring, spawns a Python worker, starts a frame loop in that worker, decodes frames with FFmpeg to raw RGB24, writes them into SHM, waits for the worker to mark slots ready, then pushes worker-produced RGB24 frames into an FFmpeg encoder.
- Pipeline diagram:
  - `input -> ffprobe -> ffmpeg decode -> raw rgb24 pipe -> Rust SHM write -> Python NumPy/Torch ingest -> preprocess/tile/model/blender -> RGB output slot -> Rust SHM read -> ffmpeg encode/mux -> output`
- Shared code:
  - Uses shared Tauri/UI routing and shared model discovery.
- Drift:
  - Separate model loading, precision policy, timing, fallback, and error surfaces from native.
- Temporary files or subprocesses:
  - Python worker subprocess
  - FFmpeg decoder subprocess
  - FFmpeg encoder subprocess
  - SHM is file-backed via `tempfile.mkstemp` in Python
- Copies / conversions:
  - FFmpeg decode outputs host `rgb24`
  - Rust copies into SHM slot
  - Python copies SHM view to NumPy arrays
  - RGB/BGR copies for model compatibility
  - NumPy float conversion and CPU->GPU upload
  - GPU->CPU output download
  - Possible re-upload for blender/research postprocess

### Engine B: Native engine family
- Short explanation:
  - The app-native command is the public native engine surface. It either runs the in-process `engine-v2` path or delegates to the external `rave` CLI path.
- Pipeline diagram:
  - `input -> native command routing -> direct engine-v2 OR rave subprocess -> output`
- Shared code:
  - Same app command, model path selection, batch defaults, output path handling, and UI result surface.
- Drift:
  - Different execution boundaries, error contracts, timing visibility, and compatibility behavior between direct and CLI.
- Temporary files or subprocesses:
  - Direct path uses FFmpeg demux and mux subprocesses.
  - CLI path uses a whole external subprocess adapter.
- Copies / conversions:
  - Direct core stays mostly GPU-resident after compressed bitstream demux.
  - CLI internals cannot be proven from this repo.

### Native
- Short explanation:
  - `upscale_request_native` with `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` uses FFmpeg only for compressed stream demux and output mux. The engine core uses NVDEC, CUDA preprocess kernels, ORT/TensorRT IO binding, RGB->NV12 postprocess, and NVENC.
- Pipeline diagram:
  - `input -> ffprobe -> ffmpeg elementary demux stdout -> NVDEC -> D2D copy -> preprocess -> TensorRT/ORT -> postprocess RGB->NV12 -> NVENC -> ffmpeg mux stdin (+ optional copied audio) -> output`
- Shared code:
  - App gating and batch policy with CLI-native.
- Drift:
  - Different error behavior and observability from CLI-native.
- Temporary files or subprocesses:
  - FFmpeg demux subprocess
  - FFmpeg mux subprocess
  - TensorRT cache directory under temp/artifacts when enabled
- Copies / conversions:
  - Compressed bitstream stays on host until NVDEC submission
  - NVDEC surfaces are copied D2D into owned buffers in `NvDecoder`
  - Model tensors remain device resident
  - Postprocess converts RGB planar tensor output to NV12
  - NVENC may fall back to legacy CUDA staging for registration compatibility

### Native-cli
- Short explanation:
  - `upscale_request_native` without the direct flag, or after selected direct failures, builds CLI args and invokes `crate::commands::rave::rave_upscale`, which shells out to a prebuilt `rave` binary and parses a JSON stdout contract.
- Pipeline diagram:
  - `input -> upscale_request_native -> run_native_via_rave_cli -> rave_upscale -> rave subprocess --json -> parsed JSON result -> output`
- Shared code:
  - Uses the same app-facing native command and many of the same policy inputs.
- Drift:
  - Separate process lifecycle, stderr/stdout contract, error classification heuristics, and unknown internal pipeline semantics because source is missing.
- Temporary files or subprocesses:
  - Prebuilt `rave` subprocess
  - Any internal temps are unknown from this repo
- Copies / conversions:
  - Proven extra boundary is process-level JSON/stdout/stderr marshaling and duplicated command-line adaptation.
  - Internal frame movement is inference, not proven.

### Where pipelines share code
- UI job selection and model discovery
- Native batch policy defaults via `src-tauri/src/models.rs`
- Some runtime gating and result DTOs in `src-tauri/src/commands/native_engine.rs`

### Where pipelines drift
- Python vs native model/session ownership
- Python vs native timing and metrics surfaces
- Direct native vs CLI-native error mapping and fallback policy
- Python postprocessing/research features are not represented in direct native

### Where temporary files or subprocesses appear
- Python worker subprocess
- FFmpeg decode/encode subprocesses in Python path
- FFmpeg demux/mux subprocesses in direct native
- `rave` subprocess in CLI-native
- File-backed SHM backing store in Python worker
- Optional TensorRT cache directories

### Where copies or conversions appear
- Python path:
  - NVDEC/FFmpeg to host RGB24
  - SHM slot copies
  - RGB/BGR layout churn
  - NumPy/Torch conversion churn
  - GPU->CPU result download
  - Optional cv2 resize
- Native path:
  - D2D NVDEC surface extraction copy
  - RGB tensor to NV12 postprocess
  - NVENC legacy staging fallback D2D copy
  - Software fallback path contains DtoH plus rawvideo software encode path, but current wrapper intentionally refuses to use it in-process

## 4. Findings
### Finding 1: Python video path is architecturally host-bound at both decode and encode
**Scope:** Python engine  
**Files:** `src-tauri/src/video_pipeline.rs`, `src-tauri/src/commands/upscale.rs`  
**Symbols:** `VideoDecoder::build_args`, `VideoEncoder::build_encoder_args`, `run_upscale_job`  
**Call path:** `upscale_request -> run_upscale_job -> VideoDecoder::new / VideoEncoder::new_with_audio`  
**Problem:** The Python video path decodes to host `rgb24` and encodes from host `rgb24`, with FFmpeg doing format conversion around the model.  
**Why it matters:** This dominates data movement cost, prevents zero-copy/GPU-resident evolution, and makes GPU decode less meaningful because frames still cross back to host memory immediately.  
**Evidence:** `VideoDecoder::build_args` emits `-f rawvideo -pix_fmt rgb24 -` and explicitly documents that FFmpeg transfers NVDEC frames to system memory when piping rawvideo. `VideoEncoder::build_encoder_args` accepts raw `rgb24` stdin, then applies `format=yuv420p` inside FFmpeg before encode.  
**Expected upside:** High  
**Implementation cost:** High  
**Risk:** Medium  
**Recommendation:** Treat the current Python video path as a compatibility path, not a performance path. Move all serious video-throughput investment to native. If Python video must remain, redesign around GPU-native decode/output surfaces instead of RGB24 rawvideo pipes.  
**Confidence:** High

### Finding 2: Python worker does multiple per-frame host copies and color-layout copies after SHM
**Scope:** Python engine  
**Files:** `python/shm_worker.py`  
**Symbols:** `_process_slot`, `_process_batch`, `inference`, `_inference_prealloc`  
**Call path:** `run_upscale_job -> create_shm/start_frame_loop -> AIWorker._frame_loop -> _process_slot/_process_batch`  
**Problem:** The worker copies SHM input into NumPy arrays, may copy RGB->BGR, builds float tensors, downloads outputs to CPU, may re-upload outputs for blender logic, and may resize again in OpenCV.  
**Why it matters:** This is the core throughput and memory-efficiency tax in the Python path. It also adds latency variance and CPU pressure.  
**Evidence:** `_process_slot` does `img_input = in_view.copy()`, optionally `img_input[:, :, ::-1].copy()`, then `inference()` converts to float32 and uploads to CUDA, then returns output via `.cpu().numpy()`, then optional blender logic converts back to Torch CUDA tensors, then final `out_view[:] = out_for_rust`. `_process_batch` repeats similar per-frame copies and postprocess churn.  
**Expected upside:** High  
**Implementation cost:** Medium  
**Risk:** Medium  
**Recommendation:** Make the Python worker reuse both input and output tensors by default, keep postprocess on GPU tensors when possible, and eliminate RGB/BGR `.copy()` churn by normalizing model color expectations in adapters.  
**Confidence:** High

### Finding 3: Python path still uses polling and spin sleeps across process boundaries
**Scope:** Python engine  
**Files:** `src-tauri/src/commands/upscale.rs`, `python/shm_worker.py`  
**Symbols:** `poll_task`, `_frame_loop`, `_wait_for_input_event`  
**Call path:** `run_upscale_job -> poll_task` and `AIWorker.start_frame_loop -> _frame_loop`  
**Problem:** Rust polls each pending slot until `READY_FOR_ENCODE`, and Python uses adaptive sleep/backoff when no work is present.  
**Why it matters:** This creates avoidable CPU overhead, jitter, weaker benchmark repeatability, and complexity around timing.  
**Evidence:** Rust loops with `tokio::time::sleep(Duration::from_micros(200))` until slot state changes. Python loops with `time.sleep(0.0001)`, then `0.001`, then `0.005` in idle mode. Event-based sync exists but is Windows-only and optional.  
**Expected upside:** Medium  
**Implementation cost:** Medium  
**Risk:** Low  
**Recommendation:** Promote event-driven SHM synchronization to the default supported path or at minimum reduce polling frequency and expose queue wait metrics.  
**Confidence:** High

### Finding 4: Python timing surface exists but does not measure real stage timing
**Scope:** Python engine, benchmarkability  
**Files:** `src-tauri/src/commands/upscale.rs`, `src-tauri/src/bin/videoforge_bench.rs`  
**Symbols:** `StageTimingsMs`, `JobProgress`, `run_upscale_job`  
**Call path:** `upscale_request -> progress events`, `videoforge_bench -> run_upscale_job`  
**Problem:** The Python path has fields for decode/ai/encode timing but only populates total wall time in progress updates.  
**Why it matters:** You cannot benchmark bottlenecks accurately or compare engines apples-to-apples without decode/infer/encode split.  
**Evidence:** `StageTimingsMs` has `decode`, `ai`, `encode`, `total`, but the encoder progress path only sets `total`, leaving the others `None`. The benchmark binary prints high-level events, not stage timings.  
**Expected upside:** High  
**Implementation cost:** Low  
**Risk:** Low  
**Recommendation:** Instrument decoder read, worker wait, worker inference, and encoder write/finish timings and return them in both Tauri progress and benchmark JSON.  
**Confidence:** High

### Finding 5: Python ONNX loading can silently degrade to CPU execution
**Scope:** Python engine, determinism/reliability  
**Files:** `python/model_manager.py`, `python/loaders/onnx_loader.py`  
**Symbols:** `_load_onnx_model`, `_probe_onnx_session`  
**Call path:** `ModelLoader.load -> _load_module -> _load_onnx_model`  
**Problem:** ONNX models in the Python path probe CUDA EP and fall back to CPU EP when the probe times out.  
**Why it matters:** This creates hidden performance cliffs and benchmark invalidation, especially when a run is expected to be GPU-backed.  
**Evidence:** `_load_onnx_model` creates a CUDA session with `["CUDAExecutionProvider", "CPUExecutionProvider"]`; if `_probe_onnx_session` fails, it logs a warning and rebuilds with `CPUExecutionProvider` only.  
**Expected upside:** Medium  
**Implementation cost:** Low  
**Risk:** Low  
**Recommendation:** Make provider fallback explicit in job results and benchmark output. For benchmark mode, fail instead of silently routing to CPU unless explicitly requested.  
**Confidence:** High

### Finding 6: Direct native still depends on FFmpeg subprocesses for demux and mux
**Scope:** Native direct  
**Files:** `src-tauri/src/commands/native_engine.rs`  
**Symbols:** `FfmpegBitstreamSource`, `StreamingMuxSink`, `run_engine_pipeline`  
**Call path:** `upscale_request_native -> run_native_pipeline -> run_engine_pipeline`  
**Problem:** The direct path is not fully self-contained; compressed stream ingress and mux/finalize are delegated to FFmpeg subprocesses.  
**Why it matters:** This is acceptable for an MVP, but it preserves startup overhead, process-management complexity, and some observability gaps around demux/mux time and failures.  
**Evidence:** `FfmpegBitstreamSource::spawn` launches FFmpeg to emit Annex B elementary video to stdout. `StreamingMuxSink::ensure_started` launches FFmpeg to mux encoded packets from stdin and optionally copy audio from the original input.  
**Expected upside:** Medium  
**Implementation cost:** Medium  
**Risk:** Medium  
**Recommendation:** Keep the FFmpeg boundary short-term, but treat it as an explicit adapter layer with measured startup and finalize timings. Long-term, move toward a shared in-process demux/mux layer if direct native remains the strategic path.  
**Confidence:** High

### Finding 7: Direct native packetization and timestamps are lossy at the wrapper boundary
**Scope:** Native direct  
**Files:** `src-tauri/src/commands/native_engine.rs`  
**Symbols:** `FfmpegBitstreamSource::read_packet`  
**Call path:** `run_engine_pipeline -> FfmpegBitstreamSource -> NvDecoder`  
**Problem:** The wrapper reads 1MB chunks from FFmpeg stdout, not true packet units, then invents monotonically increasing PTS values and heuristically detects keyframes from Annex B bytes.  
**Why it matters:** This weakens determinism and timing fidelity and makes future audio/video sync, B-frame, and timestamp-sensitive benchmarking harder.  
**Evidence:** `read_packet` allocates a 1MB buffer, does `stdout.read`, truncates to bytes read, infers keyframe by scanning byte windows, and sets `pts` from `pts_counter += 1`.  
**Expected upside:** High  
**Implementation cost:** Medium  
**Risk:** Medium  
**Recommendation:** Preserve real packet boundaries and timestamps from demux, or explicitly document/measure the current approximation and constrain supported codecs/container behaviors accordingly.  
**Confidence:** High

### Finding 8: Direct native defaults to synchronous NVDEC surface copy unless env opt-in is set
**Scope:** Native direct  
**Files:** `engine-v2/src/codecs/nvdec.rs`  
**Symbols:** `NvDecoder::async_copy_enabled`, `NvDecoder::map_and_copy`  
**Call path:** `run_engine_pipeline -> NvDecoder::decode_next -> map_and_copy`  
**Problem:** The decoder supports async D2D copy plus cross-stream events, but the default path performs synchronous `cuMemcpy2D_v2` and unmaps immediately unless `VIDEOFORGE_NVDEC_ASYNC_COPY=1` is set.  
**Why it matters:** Synchronous copy reduces overlap between decode and downstream stages and leaves throughput on the table.  
**Evidence:** `async_copy_enabled()` checks only an env var. In `map_and_copy`, the async branch uses `cuMemcpy2DAsync_v2` and queued unmaps; the default branch uses blocking `cuMemcpy2D_v2`.  
**Expected upside:** Medium  
**Implementation cost:** Low  
**Risk:** Medium  
**Recommendation:** Benchmark async-copy mode and, if stable, make it the default for supported GPUs/drivers.  
**Confidence:** High

### Finding 9: Native wrapper carries dead or drifting fallback complexity around software encode
**Scope:** Native direct, maintainability  
**Files:** `src-tauri/src/commands/native_engine.rs`  
**Symbols:** `SoftwareBitstreamEncoder`, `NativeVideoEncoder`, `NativeVideoEncoderWrapper`  
**Call path:** `run_engine_pipeline -> NativeVideoEncoderWrapper::new/encode`  
**Problem:** The wrapper defines a software bitstream encoder and an enum variant for it, but current logic refuses in-process software fallback on init failure or first-frame NVENC failure and instead bubbles errors to the outer CLI fallback logic.  
**Why it matters:** This is architecture drift: complex fallback code exists without being the chosen failure mode, increasing maintenance surface and reviewer confusion.  
**Evidence:** `NativeVideoEncoderWrapper::new` returns `Err` on NVENC init failure with “refusing in-process software fallback”. `encode` also refuses mid-stream and first-frame fallback. Yet `SoftwareBitstreamEncoder` and `NativeVideoEncoder::Software` remain implemented.  
**Expected upside:** Medium  
**Implementation cost:** Low  
**Risk:** Low  
**Recommendation:** Either wire the in-process software fallback fully and benchmark it, or remove/disable the dead path and keep CLI fallback as the only wrapper-level contingency.  
**Confidence:** High

### Finding 10: Native direct has better core telemetry than Python, but the app does not surface it well
**Scope:** Native direct, benchmarkability  
**Files:** `engine-v2/src/engine/pipeline.rs`, `engine-v2/src/core/context.rs`, `src-tauri/src/commands/native_engine.rs`, `src-tauri/src/bin/videoforge_bench.rs`  
**Symbols:** `PipelineMetrics`, `PerfProfiler`, `run_engine_pipeline`, `run_native_bench`  
**Call path:** `upscale_request_native -> run_engine_pipeline -> UpscalePipeline::run`  
**Problem:** The native core tracks per-stage counters, per-stage latency, queue depth, overlap, ring contention, and VRAM, but the Tauri command returns only a small subset and the benchmark binary mostly reports elapsed time and frames.  
**Why it matters:** The instrumentation investment is not reaching the user-facing benchmark surface, so optimization loops remain guessy.  
**Evidence:** `PipelineMetrics` tracks preprocess/inference/postprocess/encode timing, `PerfProfiler` tracks stage stats, `TensorRtBackend` reports ring and inference metrics, but `NativeUpscaleResult` only returns engine, encoder mode/detail, frames, and TRT cache fields.  
**Expected upside:** High  
**Implementation cost:** Low  
**Risk:** Low  
**Recommendation:** Add a `NativePerfReport` to the command result and bench JSON with stage timing, queue depth peaks, VRAM peak, ring contention, and mux/demux startup/finalize timing.  
**Confidence:** High

### Finding 11: Native-cli is an expensive compatibility layer, not a true peer engine
**Scope:** Native direct vs native-cli  
**Files:** `src-tauri/src/commands/native_engine.rs`, `src-tauri/src/commands/rave.rs`, `src-tauri/src/rave_cli.rs`  
**Symbols:** `run_native_via_rave_cli`, `rave_upscale`, `run_upscale`  
**Call path:** `upscale_request_native -> run_native_via_rave_cli -> rave_upscale -> run_upscale -> rave subprocess`  
**Problem:** `native-cli` duplicates job shaping, max-batch validation, runtime gating, error mapping, output-path semantics, and benchmark entrypoints while hiding the core pipeline behind a subprocess JSON contract.  
**Why it matters:** This increases drift, obscures performance truth, and slows direct-native maturation because fallback becomes another product surface instead of a thin adapter.  
**Evidence:** `upscale_request_native` applies batch policy and runtime gating before dispatching either path. `run_native_via_rave_cli` rebuilds args, invokes `rave_upscale`, and returns `engine: native_via_rave_cli`. `rave.rs` separately validates batch args, profiles, opt-in rules, and error categories.  
**Expected upside:** High  
**Implementation cost:** Medium  
**Risk:** Medium  
**Recommendation:** Consolidate around a shared `NativeJobSpec` plus `NativeExecutor` abstraction. Keep CLI as a thin transport/execution adapter with the same perf/result schema as direct.  
**Confidence:** High

### Finding 12: Subtitle and metadata handling are absent or minimal across both video paths
**Scope:** Python engine, native direct  
**Files:** `src-tauri/src/video_pipeline.rs`, `src-tauri/src/commands/native_engine.rs`  
**Symbols:** `VideoEncoder::build_encoder_args`, `StreamingMuxSink::ensure_started`  
**Call path:** video encode/mux phases in both engines  
**Problem:** Both pipelines preserve video and optionally audio, but do not preserve subtitles or full container metadata.  
**Why it matters:** This is not the top performance issue, but it is pipeline incompleteness and can complicate “production-grade” parity and benchmark comparability.  
**Evidence:** Python encoder maps `0:v` and `1:a?` only. Native mux copies `-c:v copy` and optional `-c:a copy`, but no subtitle streams or metadata mappings are present.  
**Expected upside:** Low  
**Implementation cost:** Low  
**Risk:** Low  
**Recommendation:** Decide explicitly whether non-video streams are in scope. If yes, encode/mux adapters should preserve them or report that they are dropped.  
**Confidence:** High

### Finding 13: Python and native determinism contracts differ materially
**Scope:** Python engine, native direct, native-cli  
**Files:** `python/shm_worker.py`, `src-tauri/src/video_pipeline.rs`, `src-tauri/src/commands/native_engine.rs`, `engine-v2/src/backends/tensorrt.rs`  
**Symbols:** `configure_precision`, `enforce_deterministic_mode`, `VideoEncoder::build_encoder_args`, `PrecisionPolicy`  
**Call path:** model/runtime init and encode config in each engine  
**Problem:** Python has an explicit deterministic mode affecting Torch flags and batch behavior; native exposes only fp16/fp32 precision policy and encoder defaults, not a full determinism mode. CLI-native determinism depends on external binary behavior not visible here.  
**Why it matters:** Output comparability across engines is weaker than the UI suggests, and benchmark results may mix determinism and performance settings.  
**Evidence:** Python supports `"deterministic"` precision plus seeded execution and stricter Torch flags. Direct native only maps `"fp16"` to `PrecisionPolicy::Fp16`; everything else becomes fp32. No determinism-specific pipeline mode is wired into native command results.  
**Expected upside:** Medium  
**Implementation cost:** Medium  
**Risk:** Low  
**Recommendation:** Define one cross-engine determinism contract and make every benchmark report whether it was honored, partially honored, or unsupported.  
**Confidence:** High

### Finding 14: Benchmark tooling exists but is not yet a clean regression harness
**Scope:** Both engines, benchmarkability  
**Files:** `src-tauri/src/bin/videoforge_bench.rs`, `src-tauri/src/models.rs`, `src-tauri/src/commands/native_engine.rs`, `src-tauri/src/commands/upscale.rs`  
**Symbols:** `videoforge_bench`, `NativeUpscaleResult`, `run_upscale_job`  
**Call path:** standalone benchmark binary for Python or native  
**Problem:** The repo has a useful benchmark binary, but it does not emit a normalized stage breakdown, provider/fallback identity, memory metrics, or direct-vs-CLI parity metrics.  
**Why it matters:** Without this, performance regressions and pathway drift will continue to be diagnosed ad hoc from logs.  
**Evidence:** The bench tool reports start/progress/done, warmup, and some native metadata, but not decode/infer/encode split, queue depth, temp I/O, or provider/fallback incidence.  
**Expected upside:** High  
**Implementation cost:** Low  
**Risk:** Low  
**Recommendation:** Make `videoforge_bench` the canonical regression harness and require machine-readable perf reports from both engines.  
**Confidence:** High

## 5. Native vs native-cli verdict
- Recommendation:
  - Shared core plus thin adapters.
  - Prefer `native` as the strategic path.
  - Reduce `native-cli` to a compatibility adapter, then deprecate it when direct native covers the required rollout surface.
- Why both exist today:
  - `native` exists to run `engine-v2` in-process with direct access to the GPU pipeline.
  - `native-cli` exists as a safer compatibility path and operational fallback when the direct path is not enabled or selected direct failures occur.
- Duplicated responsibilities:
  - Runtime gating (`VIDEOFORGE_ENABLE_NATIVE_ENGINE`)
  - Batch validation and batch policy defaults
  - Precision/input/output argument shaping
  - Output path synthesis
  - Audio-preserve flag propagation
  - Benchmark surface exposure
  - Error adaptation into UI-facing JSON
- Drift areas:
  - Process lifecycle and logging
  - Error taxonomy and stderr heuristics
  - Performance observability
  - Determinism reporting
  - Unknown internal pipeline behavior in the external binary
- Current strengths of `native`:
  - Actual code is inspectable and optimizable.
  - GPU-resident core is already structured as a bounded streaming pipeline.
  - Internal telemetry already exists.
- Current weaknesses of `native`:
  - FFmpeg demux/mux wrappers still leak packet/timestamp fidelity.
  - Wrapper fallback/error logic has drift.
  - Direct/native result surface is too narrow for performance work.
- Current strengths of `native-cli`:
  - Operational escape hatch.
  - Useful as a rollout fallback while direct native matures.
- Current weaknesses of `native-cli`:
  - Extra process boundary and JSON contract
  - Separate error and benchmark semantics
  - Source absent in this repo, so optimization and drift control are weaker
- Recommended future architecture:
  - Introduce a shared `NativeJobSpec` and `NativePerfReport`.
  - Implement two executors:
    - `DirectNativeExecutor`
    - `CliNativeExecutor`
  - Keep all policy, validation, result schema, and benchmark hooks above the executor boundary.
  - Make CLI executor return the same perf/result structure as direct, or fail contract validation.

## 6. Ideal target architecture
- Ideal engine layering:
  - UI/job routing layer
  - Shared engine selection and validation layer
  - Shared job spec and result/perf schema
  - Engine executors:
    - Python compatibility executor
    - Direct native executor
    - Optional CLI-native adapter executor
- Ideal decode/infer/encode boundary design:
  - Direct native:
    - compressed packet source -> hardware decode -> GPU preprocess -> GPU inference -> GPU postprocess -> hardware encode -> mux
  - Python:
    - keep for non-native models and research features, but stop treating it as the long-term performance path
- Ideal streaming vs temp-file behavior:
  - No decoded-frame temp files.
  - No SHM file-backed RGB24 video path for the main performance engine.
  - Limited adapter subprocesses only where necessary, with explicit measured cost.
- Ideal shared core vs adapters split:
  - Shared:
    - job spec
    - engine selection
    - validation
    - batch policy
    - error/result schema
    - benchmark hooks
  - Adapter-specific:
    - direct in-process demux/decode/mux execution
    - CLI process spawn and contract parsing
- Ideal model/session ownership:
  - Long-lived model/session ownership per process
  - explicit warmup path
  - cache identity included in benchmark output
  - no silent provider fallback in benchmark mode
- Ideal benchmark/instrumentation hooks:
  - stage timings for demux, decode, preprocess, infer, postprocess, encode, mux/finalize
  - queue depths
  - VRAM/current/peak
  - temp disk bytes
  - fallback/provider identity
  - engine/pathway version tags
- Ideal determinism and error surfaces:
  - explicit determinism mode support matrix across engines
  - no silent engine/provider switching
  - stable structured errors with root cause and fallback decisions captured

## 7. Optimization roadmap
### Phase 1: Quick wins
- Exact tasks:
  - Surface native internal metrics in `NativeUpscaleResult`
  - Populate Python `StageTimingsMs.decode/ai/encode`
  - Add provider/fallback identity to Python and native benchmark output
  - Remove or clearly disable dead in-process software fallback code in direct native wrapper
  - Make direct native packet/timestamp approximation visible in logs and benchmark reports
- Affected files/modules:
  - `src-tauri/src/commands/native_engine.rs`
  - `src-tauri/src/commands/upscale.rs`
  - `src-tauri/src/bin/videoforge_bench.rs`
  - `python/model_manager.py`
- Dependencies:
  - None
- Expected payoff:
  - Immediate benchmarkability and lower architecture drift
- Risk notes:
  - Mostly additive; low regression risk

### Phase 2: Structural improvements
- Exact tasks:
  - Introduce a shared native job/result schema for direct and CLI
  - Replace Python polling-heavy completion with event-driven signaling where possible
  - Make native async NVDEC copy a supported default after validation
  - Reduce Python frame-loop copy churn by defaulting to tensor preallocation and GPU-side postprocess
- Affected files/modules:
  - `src-tauri/src/commands/native_engine.rs`
  - `src-tauri/src/commands/rave.rs`
  - `src-tauri/src/rave_cli.rs`
  - `python/shm_worker.py`
  - `engine-v2/src/codecs/nvdec.rs`
- Dependencies:
  - Phase 1 metrics should land first to validate gains
- Expected payoff:
  - Lower CPU overhead, better overlap, cleaner native architecture
- Risk notes:
  - Medium; touches process coordination and GPU synchronization

### Phase 3: Architecture upgrades
- Exact tasks:
  - Move further toward a shared in-process native media boundary instead of FFmpeg stdout/stdin chunk shims
  - Re-scope Python engine to compatibility/research/image workloads
  - Deprecate or severely narrow CLI-native once direct native is stable
- Affected files/modules:
  - `src-tauri/src/commands/native_engine.rs`
  - `engine-v2/`
  - `src-tauri/src/video_pipeline.rs`
  - `ui/src/hooks/useUpscaleJob.ts`
- Dependencies:
  - Phase 2 shared job spec and metrics
- Expected payoff:
  - Best long-term throughput, determinism, maintainability, and benchmarking
- Risk notes:
  - Higher; this is real architecture work, not hygiene

## 8. Top implementation priorities
1. Unify direct native and CLI-native behind a shared job spec/result schema.
   - Why now: It reduces immediate drift and makes every other native optimization easier to compare.
   - Affected engines/pathways: `native`, `native-cli`
   - Expected payoff: High
   - Implementation complexity: Medium
2. Expose native stage metrics, queue depth, and VRAM in command/bench results.
   - Why now: Existing telemetry is stranded in logs.
   - Affected engines/pathways: `native`
   - Expected payoff: High
   - Implementation complexity: Low
3. Expose real Python decode/AI/encode timings.
   - Why now: Current Python benchmark data is too coarse to guide ROI.
   - Affected engines/pathways: Python engine
   - Expected payoff: High
   - Implementation complexity: Low
4. Default or at least validate async NVDEC copy in direct native.
   - Why now: It directly improves overlap in the fastest path.
   - Affected engines/pathways: `native`
   - Expected payoff: Medium
   - Implementation complexity: Low
5. Remove or finalize the dead software fallback branch in direct native wrapper.
   - Why now: It is pure drift today.
   - Affected engines/pathways: `native`
   - Expected payoff: Medium
   - Implementation complexity: Low
6. Replace Python steady-state polling with stronger event-driven signaling.
   - Why now: It improves CPU efficiency and benchmark stability immediately.
   - Affected engines/pathways: Python engine
   - Expected payoff: Medium
   - Implementation complexity: Medium
7. Reduce Python postprocess re-upload/download churn.
   - Why now: It is one of the largest avoidable costs in the compatibility path.
   - Affected engines/pathways: Python engine
   - Expected payoff: Medium
   - Implementation complexity: Medium
8. Preserve real packet timing and packet boundaries in direct native demux.
   - Why now: It improves correctness and future muxing sophistication.
   - Affected engines/pathways: `native`
   - Expected payoff: Medium
   - Implementation complexity: Medium
9. Make provider fallback explicit and benchmark-fail on silent CPU fallback.
   - Why now: Hidden fallback invalidates comparisons.
   - Affected engines/pathways: Python engine, `native-cli` where applicable
   - Expected payoff: Medium
   - Implementation complexity: Low
10. Define one cross-engine determinism contract and report compliance.
   - Why now: Current “precision” semantics are not aligned across engines.
   - Affected engines/pathways: both engines, both native pathways
   - Expected payoff: Medium
   - Implementation complexity: Medium

## 9. Open questions / unknowns
- The internal source for the `rave` CLI binary is not present in this workspace, so `native-cli` internal frame movement, decode/mux strategy, and telemetry fidelity cannot be proven here.
- It is unclear whether direct native intends to support codecs beyond the currently probed H.264/HEVC set in the wrapper.
- It is unclear whether Windows-only event sync is expected to become cross-platform or remain optional.

## 10. Appendix
- Terminology clarifications:
  - “Python engine” means `upscale_request` plus `python/shm_worker.py`.
  - “Native” means direct in-process `engine-v2`.
  - “Native-cli” means the `rave` subprocess adapter invoked by `run_native_via_rave_cli`.
- Assumptions made:
  - Where `rave` internals are discussed, they are clearly labeled as inference from the adapter surface, because the binary source is absent.
- Useful file index:
  - `ui/src/hooks/useUpscaleJob.ts`
  - `src-tauri/src/commands/upscale.rs`
  - `src-tauri/src/video_pipeline.rs`
  - `python/shm_worker.py`
  - `python/model_manager.py`
  - `src-tauri/src/commands/native_engine.rs`
  - `src-tauri/src/commands/rave.rs`
  - `src-tauri/src/rave_cli.rs`
  - `engine-v2/src/engine/pipeline.rs`
  - `engine-v2/src/backends/tensorrt.rs`
  - `engine-v2/src/codecs/nvdec.rs`
  - `engine-v2/src/codecs/nvenc.rs`
- Optional line references where practical:
  - Python decode raw RGB24: `src-tauri/src/video_pipeline.rs:64-183`
  - Python encode raw RGB24 + format conversion: `src-tauri/src/video_pipeline.rs:341-489`
  - Python SHM orchestration and polling: `src-tauri/src/commands/upscale.rs:495-925`
  - Worker frame loop and batch processing: `python/shm_worker.py:1488-2019`
  - Native command routing: `src-tauri/src/commands/native_engine.rs:564-689`
  - Native core pipeline setup: `src-tauri/src/commands/native_engine.rs:780-996`
  - FFmpeg demux/mux adapters: `src-tauri/src/commands/native_engine.rs:1075-1385`
  - Native pipeline metrics: `engine-v2/src/engine/pipeline.rs`
  - TensorRT backend lifecycle and metrics: `engine-v2/src/backends/tensorrt.rs`
