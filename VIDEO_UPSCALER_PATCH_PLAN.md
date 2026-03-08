# Video Upscaler Patch Plan

## 1. Priority-ranked PR plan
### PR 1: Unify native job/result contracts across direct and CLI
**Goal:** Create one shared native execution contract so `native` and `native-cli` stop drifting at the app boundary.  
**Why this matters:** Ties directly to Findings 9, 10, and 11.  
**Files:** `src-tauri/src/commands/native_engine.rs`, `src-tauri/src/commands/rave.rs`, `src-tauri/src/rave_cli.rs`, `src-tauri/src/models.rs`, `ui/src/types.ts`  
**Primary symbols:** `upscale_request_native`, `run_native_via_rave_cli`, `NativeUpscaleResult`, `rave_upscale`  
**Changes:**  
- Introduce a shared `NativeJobSpec` with normalized fields for model path, scale, precision, audio, batch, and benchmark flags.
- Introduce a shared `NativePerfReport` and extend `NativeUpscaleResult` to carry it.
- Refactor direct and CLI pathways to consume the same job spec and return the same result contract.

**Acceptance criteria:**  
- Direct and CLI-native return the same JSON schema for success and perf metadata.
- Native batch policy and argument validation live in one place.
- UI no longer needs pathway-specific parsing beyond the `engine` field.

**Benchmark expectation:**  
- No direct throughput gain by itself.
- Validation: identical benchmark result schema for `native` and `native-cli`.

**Risk notes:** Contract changes affect UI parsing and existing logs.  
**Dependencies:** None

### PR 2: Surface real stage metrics from the native direct path
**Goal:** Expose existing `engine-v2` timing, queue, and VRAM telemetry to the app and benchmark tool.  
**Why this matters:** Ties to Findings 6, 7, and 10.  
**Files:** `src-tauri/src/commands/native_engine.rs`, `engine-v2/src/engine/pipeline.rs`, `engine-v2/src/backends/tensorrt.rs`, `engine-v2/src/core/context.rs`, `src-tauri/src/bin/videoforge_bench.rs`  
**Primary symbols:** `run_engine_pipeline`, `PipelineMetrics`, `PerfProfiler`, `TensorRtBackend`  
**Changes:**  
- Capture pipeline metrics snapshots after run completion.
- Add demux startup, backend init, encode flush, and mux finalize timing in the wrapper.
- Emit these metrics in `NativeUpscaleResult` and `videoforge_bench` JSON.

**Acceptance criteria:**  
- Native benchmark output includes decode/preprocess/infer/postprocess/encode/mux timings.
- Native benchmark output includes peak VRAM and queue depth peaks.
- Result schema remains stable across warmup and non-warmup runs.

**Benchmark expectation:**  
- No throughput change required.
- Validation: benchmark JSON contains non-null stage timing fields on successful direct native runs.

**Risk notes:** Mostly additive, low regression risk.  
**Dependencies:** PR 1 recommended but not strictly required

### PR 3: Add real Python stage timing and fallback identity
**Goal:** Make the Python path benchmarkable enough to compare against native honestly.  
**Why this matters:** Ties to Findings 3, 4, and 5.  
**Files:** `src-tauri/src/commands/upscale.rs`, `src-tauri/src/video_pipeline.rs`, `python/model_manager.py`, `src-tauri/src/bin/videoforge_bench.rs`  
**Primary symbols:** `run_upscale_job`, `StageTimingsMs`, `_load_onnx_model`  
**Changes:**  
- Measure decoder read, worker wait/AI time, encoder write, and finalization time.
- Record whether Python ONNX ran on CUDA EP or CPU EP.
- Emit normalized progress and benchmark JSON with timing and provider identity.

**Acceptance criteria:**  
- Python benchmark output includes non-null decode/ai/encode timings.
- Python runs explicitly report provider identity for ONNX models.
- CPU fallback is visible in command results and benchmark JSON.

**Benchmark expectation:**  
- No throughput change required.
- Validation: benchmark JSON must distinguish Torch vs ONNX, CUDA vs CPU EP, and stage timings.

**Risk notes:** Low; instrumentation only.  
**Dependencies:** None

### PR 4: Remove native wrapper fallback drift
**Goal:** Eliminate or finalize the dead in-process software fallback branch in the direct native wrapper.  
**Why this matters:** Ties to Finding 9.  
**Files:** `src-tauri/src/commands/native_engine.rs`  
**Primary symbols:** `SoftwareBitstreamEncoder`, `NativeVideoEncoder`, `NativeVideoEncoderWrapper`, `should_fallback_to_rave_cli`  
**Changes:**  
- Choose one policy: either fully support in-process software fallback or delete/feature-gate the unused path.
- Simplify error routing so direct native failures have one clear ownership boundary.
- Ensure encoder detail fields reflect the chosen policy accurately.

**Acceptance criteria:**  
- No unreachable or non-strategic fallback code remains in the direct wrapper.
- Error messages make it clear whether fallback is internal, external, or disabled.
- Tests or smoke runs confirm the chosen fallback policy.

**Benchmark expectation:**  
- Slight reduction in startup overhead and less ambiguity in failure diagnosis.
- Validation: failure cases produce one deterministic fallback path.

**Risk notes:** Medium if software fallback is still relied on externally.  
**Dependencies:** None

### PR 5: Promote async NVDEC copy to a supported fast path
**Goal:** Improve decode/preprocess overlap in direct native.  
**Why this matters:** Ties to Finding 8.  
**Files:** `engine-v2/src/codecs/nvdec.rs`, `engine-v2/src/engine/pipeline.rs`, `src-tauri/src/commands/native_engine.rs`  
**Primary symbols:** `NvDecoder::async_copy_enabled`, `map_and_copy`, `StreamReadyEvent`  
**Changes:**  
- Benchmark async-copy mode on supported hardware.
- Replace env-only opt-in with a validated runtime default or explicit config flag.
- Add metrics for pending unmaps and decode-copy latency.

**Acceptance criteria:**  
- Async copy path is covered by a documented runtime policy.
- No regression in output correctness or decoder stability.
- Native benchmark output includes evidence of improved decode/preprocess overlap.

**Benchmark expectation:**  
- Better throughput on decode-bound content.
- Validation: lower decode-to-preprocess idle time and higher FPS on native direct runs.

**Risk notes:** Medium; touches synchronization correctness.  
**Dependencies:** PR 2 recommended for measurement

### PR 6: Reduce Python worker copy churn and GPU re-upload churn
**Goal:** Cut avoidable host and device allocations in the Python video worker.  
**Why this matters:** Ties to Findings 2 and 3.  
**Files:** `python/shm_worker.py`, `python/model_manager.py`, `python/inference_engine.py` if used  
**Primary symbols:** `_process_slot`, `_process_batch`, `inference`, `_inference_prealloc`  
**Changes:**  
- Default to tensor preallocation when compatible.
- Keep blender/research postprocess on GPU tensors when possible instead of CPU roundtrips.
- Collapse redundant RGB/BGR copies behind adapter/model metadata.

**Acceptance criteria:**  
- Fewer per-frame allocations in logs or counters.
- No extra GPU upload after model inference for common no-research and research-light paths.
- Output parity remains within accepted tolerance for supported models.

**Benchmark expectation:**  
- Better Python-path FPS and lower CPU usage.
- Validation: reduced wall time and lower host allocation churn on repeated runs.

**Risk notes:** Medium; touches compatibility behavior in the worker.  
**Dependencies:** PR 3 recommended for measurement

### PR 7: Replace Python steady-state polling with stronger signaling
**Goal:** Reduce CPU spin overhead and timing jitter in the Python video path.  
**Why this matters:** Ties to Finding 3.  
**Files:** `src-tauri/src/commands/upscale.rs`, `python/shm_worker.py`, `src-tauri/src/win_events.rs` or cross-platform equivalent if added  
**Primary symbols:** `poll_task`, `_frame_loop`, `_signal_output_event`  
**Changes:**  
- Make event signaling the preferred completion path where supported.
- Reduce or remove 200us Rust polling and Python adaptive busy sleeps from the hot path.
- Add queue wait metrics so backpressure is measurable.

**Acceptance criteria:**  
- CPU idle spin behavior is materially reduced during Python video runs.
- Completion signaling remains reliable under load.
- Bench output includes worker wait/queue timing.

**Benchmark expectation:**  
- Lower CPU overhead and less run-to-run variance.
- Validation: reduced CPU utilization and tighter wall-time variance across repeated runs.

**Risk notes:** Medium; cross-process signaling changes can deadlock if implemented poorly.  
**Dependencies:** PR 3 recommended

### PR 8: Harden direct-native demux/mux fidelity and measurement
**Goal:** Make the FFmpeg adapter boundary explicit, correct, and measurable instead of heuristic.  
**Why this matters:** Ties to Findings 6 and 7.  
**Files:** `src-tauri/src/commands/native_engine.rs`  
**Primary symbols:** `FfmpegBitstreamSource`, `StreamingMuxSink`, `read_packet`, `ensure_started`  
**Changes:**  
- Preserve real packet timestamps where possible instead of synthetic counters.
- Reduce chunk-heurstic keyframe and packet handling.
- Add startup/finalize timing and stderr snapshots to the perf/error report.

**Acceptance criteria:**  
- Direct native result includes demux and mux timing.
- Packet timing behavior is documented and less heuristic.
- Error reports include actionable demux/mux context without scraping logs.

**Benchmark expectation:**  
- May improve correctness more than raw throughput.
- Validation: stable output timing and clearer failure attribution on problematic inputs.

**Risk notes:** Medium; touches media-boundary behavior.  
**Dependencies:** PR 2 recommended

## 2. Sequencing notes
- Start with PR 1 through PR 3 because they reduce drift and create measurement infrastructure.
- Land PR 4 next because it removes ambiguity in native failure behavior before more native optimization.
- Use PR 2 metrics to validate PR 5 and PR 8 on the direct native path.
- Use PR 3 metrics to validate PR 6 and PR 7 on the Python path.
- Defer broader architecture upgrades until these eight PRs have created a reliable measurement baseline.

## 3. Validation strategy
- After each PR, run `src-tauri/src/bin/videoforge_bench.rs` in both Python and native modes with the same short deterministic clip.
- Validate correctness by checking:
  - output file exists and decodes
  - reported engine/pathway matches expectation
  - frame counts are stable across repetitions
  - provider/fallback identity is explicit
- Validate determinism by repeating the same run under the same settings and comparing:
  - stage timings variance
  - engine/pathway identity
  - output checksum where codec settings allow meaningful comparison
- Validate performance by comparing:
  - wall time
  - FPS
  - per-stage timing
  - peak VRAM
  - CPU utilization where available

## 4. Suggested benchmark gates
- PR 1 gate:
  - Direct and CLI-native emit identical top-level result schema.
- PR 2 gate:
  - Native direct benchmark output must include non-null stage timings and peak VRAM.
- PR 3 gate:
  - Python benchmark output must include non-null decode/ai/encode timings and provider identity.
- PR 5 gate:
  - Async NVDEC copy must not regress native direct FPS on the baseline clip; target at least no regression and ideally a measurable improvement on 4K content.
- PR 6 gate:
  - Python worker changes should reduce median wall time by at least 5% on the standard 1080p clip or reduce CPU utilization materially.
- PR 7 gate:
  - Python path run-to-run wall-time variance should tighten, and idle CPU utilization should drop.
- PR 8 gate:
  - Demux/mux timing must be visible and output timing correctness must not regress on H.264 and HEVC fixtures.
