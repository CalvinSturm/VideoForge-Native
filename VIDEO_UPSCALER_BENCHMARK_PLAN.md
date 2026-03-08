# Video Upscaler Benchmark Plan

## 1. Benchmark goals
- Quantify the real cost split between decode, preprocess, inference, postprocess, encode, and mux/finalize.
- Compare Python engine vs native engine fairly on the same media and model scale factor.
- Compare `native` vs `native-cli` as execution pathways, not just as user-visible labels.
- Detect hidden provider/engine fallback, especially Python ONNX CUDA -> CPU fallback and direct-native -> CLI fallback.
- Create a repeatable regression harness for throughput, latency, memory, and determinism work.

## 2. Comparison matrix
- Compare both engines:
  - Python engine vs native direct for video workloads where both are eligible.
  - Python engine vs native-cli where direct native is disabled or intentionally bypassed.
- Compare native vs native-cli:
  - Same input, same ONNX model, same scale, same audio-preserve setting, same output target.
  - Run direct native with and without async NVDEC copy if the feature remains configurable.
- Multiple input resolutions:
  - 720p
  - 1080p
  - 4K
- Short clip vs long clip:
  - 5-10 second clip for startup-sensitive latency
  - 60-120 second clip for steady-state throughput
- GPU path vs CPU path where relevant:
  - Python ONNX CUDA EP
  - Python ONNX CPU EP fallback
  - Native direct GPU path
- Throughput-sensitive workloads:
  - long 1080p and 4K clips with modest motion
- Latency-sensitive workloads:
  - short 720p or 1080p clip where startup and finalize dominate
- Quality-sensitive workloads:
  - same source/model pair across engines with deterministic or as-close-as-possible encode settings

## 3. Metrics
- Total wall clock time
- Frames per second
- Per-stage timing
- Decode time
- Preprocess time
- Inference time
- Postprocess time
- Encode time
- Mux/finalize time
- Peak memory
- Average memory
- Temp disk I/O
- GPU utilization, if available
- Model/session initialization cost
- Startup overhead
- Error/fallback incidence

Exact per-engine instrumentation targets:
- Python engine:
  - `src-tauri/src/commands/upscale.rs`
  - `src-tauri/src/video_pipeline.rs`
  - `python/shm_worker.py`
  - `python/model_manager.py`
- Native direct:
  - `src-tauri/src/commands/native_engine.rs`
  - `engine-v2/src/engine/pipeline.rs`
  - `engine-v2/src/backends/tensorrt.rs`
  - `engine-v2/src/core/context.rs`
- Native-cli:
  - `src-tauri/src/commands/native_engine.rs`
  - `src-tauri/src/commands/rave.rs`
  - `src-tauri/src/rave_cli.rs`
  - plus whatever perf fields the external `rave` subprocess can be made to return

## 4. Test media matrix
- Resolution:
  - 1280x720
  - 1920x1080
  - 3840x2160
- Duration:
  - 5s short clip
  - 30s medium clip
  - 120s long clip
- Motion complexity:
  - static/low-motion interview shot
  - medium-motion handheld/live-action
  - high-motion sports/action
- Compression characteristics:
  - clean mezzanine-like H.264
  - consumer highly compressed H.264
  - HEVC sample
- Scene complexity:
  - flat/cartoon/anime
  - natural live-action
  - fine-detail texture-heavy footage

Recommended minimal fixture set:
- `720p_short_lowmotion_h264`
- `1080p_short_texture_h264`
- `1080p_long_mixedmotion_h264`
- `4k_short_highdetail_hevc`
- `4k_long_mixedmotion_h264`

## 5. Execution rules
- Same inputs:
  - identical source clip for every compared run
- Same scale factor:
  - derived from the chosen model when necessary, but fixed across compared runs
- Same output settings where possible:
  - preserve audio consistently
  - record codec/encoder mode explicitly
- Warm vs cold run policy:
  - cold run: first run after process start and empty caches
  - warm run: after one warmup pass in the same process
- Number of repetitions:
  - minimum 3 cold runs
  - minimum 5 warm runs for steady-state comparisons
- Cache policy:
  - record TensorRT cache enabled/disabled
  - record warmup count
  - record Python model already loaded vs fresh worker spawn
- Deterministic run notes:
  - report whether deterministic mode is supported, requested, and actually applied
  - fail benchmark comparisons that silently fall back to a different provider/engine unless explicitly allowed
- Logging/tracing requirements:
  - capture machine-readable JSON benchmark output
  - preserve stderr tails for failed native direct demux/mux and CLI subprocess runs

## 6. Instrumentation additions
- Missing metrics or traces in current codebase:
  - Python decode/AI/encode split is missing in `src-tauri/src/commands/upscale.rs`.
  - Python provider identity/fallback reporting is missing in `python/model_manager.py`.
  - Native direct wrapper demux startup and mux finalize timing are missing in `src-tauri/src/commands/native_engine.rs`.
  - Native direct perf report is not surfaced even though `engine-v2` already tracks stage metrics in `engine-v2/src/engine/pipeline.rs` and `engine-v2/src/core/context.rs`.
  - Queue wait time and slot wait time are missing for the Python SHM path in both Rust and Python.
  - Temp disk I/O and SHM backing-file size are not reported anywhere.
  - Direct-native packet/timestamp approximation is not reported.
  - CLI-native subprocess startup cost and contract parse overhead are not reported separately.

Exact additions:
- Add `NativePerfReport` to `NativeUpscaleResult` in `src-tauri/src/commands/native_engine.rs`.
- Add `PythonPerfReport` to progress and final result surfaces in `src-tauri/src/commands/upscale.rs`.
- Emit provider name and fallback reason from `python/model_manager.py::_load_onnx_model`.
- Extend `src-tauri/src/bin/videoforge_bench.rs` to print:
  - engine/pathway
  - provider identity
  - stage timings
  - peak VRAM
  - queue depth peaks
  - warmup count
  - cache status
  - fallback incidence

## 7. Regression benchmark plan
- Make `src-tauri/src/bin/videoforge_bench.rs` the canonical harness.
- Store benchmark fixtures and expected configuration in versioned JSON alongside the repo or under `artifacts/benchmarks/`.
- For every optimization PR:
  - run the fixed fixture matrix
  - compare against the last accepted baseline
  - block merges on unexplained regressions above threshold
- Persist benchmark outputs as JSON artifacts so regressions can be diffed mechanically.
- Track at minimum:
  - median FPS
  - p95 wall time
  - stage timing deltas
  - peak VRAM
  - fallback incidence

## 8. Success criteria
- Highest-priority refactors are justified if they achieve one or more of:
  - `native` clearly outperforms `native-cli` on wall time and startup overhead for the same ONNX workload.
  - Python path shows reduced CPU utilization and at least modest FPS gains after copy/polling reductions.
  - Native direct exposes stable stage metrics with low run-to-run variance.
  - Hidden provider/engine fallback incidence drops to zero in benchmark mode.
  - The benchmark suite can explain where time moved after each PR, instead of only reporting total elapsed time.

## Required investigation details
- Concrete examples of bottlenecks:
  - Python decode/encode raw RGB24 boundary in `src-tauri/src/video_pipeline.rs`
  - Python worker copies in `python/shm_worker.py::_process_slot` and `_process_batch`
  - Direct-native chunk packetization in `src-tauri/src/commands/native_engine.rs::FfmpegBitstreamSource::read_packet`
- Concrete examples of duplicated logic:
  - Native job validation and batch policy in `src-tauri/src/commands/native_engine.rs` and `src-tauri/src/commands/rave.rs`
  - CLI process wrapping in `src-tauri/src/rave_cli.rs`
- Concrete examples of format conversion churn:
  - Python path `rgb24` decode and `format=yuv420p` encode in `src-tauri/src/video_pipeline.rs`
  - Python worker RGB/BGR copies and Torch/NumPy conversions in `python/shm_worker.py`
  - Native direct RGB tensor -> NV12 postprocess in `engine-v2/src/engine/pipeline.rs`
- Concrete examples of temp-file churn:
  - Python SHM backing file created via `tempfile.mkstemp` in `python/shm_worker.py::create_shm`
  - TensorRT cache directories in `src-tauri/src/commands/native_engine.rs` and `engine-v2/src/backends/tensorrt.rs`
- Concrete examples of subprocess overhead:
  - Python worker spawn in `src-tauri/src/commands/upscale.rs`
  - FFmpeg decode/encode in `src-tauri/src/video_pipeline.rs`
  - FFmpeg demux/mux in `src-tauri/src/commands/native_engine.rs`
  - `rave` subprocess in `src-tauri/src/rave_cli.rs`
- Concrete examples of initialization overhead:
  - Python worker handshake and model load in `src-tauri/src/commands/upscale.rs`
  - TensorRT backend initialization in `engine-v2/src/backends/tensorrt.rs::initialize`
- Concrete examples of missing instrumentation:
  - Native metrics stranded in logs in `engine-v2`
  - Python stage timing fields present but not filled in `src-tauri/src/commands/upscale.rs`

## Important reasoning standard
- Proven:
  - Python engine is host-frame based.
  - Direct native engine core is GPU-resident after compressed bitstream ingress.
  - `native-cli` is a subprocess adapter in this repo, not a visible peer engine core.
- Inference:
  - `native-cli` likely has extra startup and contract overhead beyond direct native because of the subprocess boundary.
  - Internal CLI-native frame movement cannot be proven here because the binary source is absent.
- Weak evidence:
  - Any claim about `rave` internal demux/decode/infer/encode stages beyond what its adapter contract exposes.
