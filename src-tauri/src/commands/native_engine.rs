//! App-facing native video command surface.
//!
//! This module owns the native-family control plane exposed through
//! `upscale_request_native(...)`:
//! - compile-time and runtime gating
//! - direct-vs-CLI executor selection
//! - shared native result and perf contracts
//! - selected direct-to-CLI fallback policy
//!
//! Compile-time/runtime gates:
//! - when the `native_engine` Cargo feature is disabled, the command returns
//!   a structured `FEATURE_DISABLED` error
//! - `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1` is required before the native family
//!   will execute
//! - `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` requests the in-process direct route
//! - without the direct flag, the native command uses the CLI-backed route
//!
//! Native-family routes:
//! ```text
//! direct route:
//!   FFmpeg demux stdout -> Annex B elementary stream -> engine-v2
//!   NVDEC -> preprocess -> TensorRT/ORT -> postprocess -> NVENC
//!   -> FFmpeg mux stdin -> final .mp4
//!
//! CLI-backed route:
//!   upscale_request_native -> rave subprocess adapter -> final output
//! ```
//!
//! The direct route is streamed at the FFmpeg boundaries in the current
//! implementation; this module no longer relies on the older temp elementary
//! stream handoff described in prior docs.

use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::runtime_truth::{RunObservedMetrics, RuntimeConfigSnapshot};
use crate::tauri_contracts::NativeUpscaleRequest;
#[cfg(feature = "native_engine")]
pub(crate) use crate::commands::native_probe::{NativeVideoOutputProfile, NativeVideoSourceProfile};
#[cfg(feature = "native_engine")]
use crate::commands::native_routing::{run_native_job, NativeJobSpec};
#[cfg(feature = "native_engine")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "native_engine")]
static NATIVE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "native_engine")]
fn native_temp_token() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let pid = std::process::id();
    let seq = NATIVE_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{pid}_{ts}_{seq}")
}

/// Structured response returned by `upscale_request_native`.
#[derive(Debug, Serialize, Deserialize)]
pub struct NativePerfReport {
    pub frames_processed: u64,
    pub effective_max_batch: u32,
    pub trt_cache_enabled: bool,
    pub trt_cache_dir: Option<String>,
    pub requested_executor: Option<String>,
    pub executed_executor: Option<String>,
    pub direct_attempted: bool,
    pub fallback_used: bool,
    pub fallback_reason_code: Option<String>,
    pub fallback_reason_message: Option<String>,
    pub total_elapsed_ms: Option<u64>,
    pub frames_decoded: Option<u64>,
    pub frames_preprocessed: Option<u64>,
    pub frames_inferred: Option<u64>,
    pub frames_encoded: Option<u64>,
    pub preprocess_avg_us: Option<u64>,
    pub inference_frame_avg_us: Option<u64>,
    pub inference_dispatch_avg_us: Option<u64>,
    pub postprocess_frame_avg_us: Option<u64>,
    pub postprocess_dispatch_avg_us: Option<u64>,
    pub encode_avg_us: Option<u64>,
    pub vram_current_mb: Option<u64>,
    pub vram_peak_mb: Option<u64>,
}

/// Structured response returned by `upscale_request_native`.
#[derive(Debug, Serialize, Deserialize)]
pub struct NativeUpscaleResult {
    pub output_path: String,
    pub engine: String,
    pub encoder_mode: String,
    pub encoder_detail: Option<String>,
    pub audio_preserved: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_snapshot: Option<RuntimeConfigSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_metrics: Option<RunObservedMetrics>,
    #[serde(flatten)]
    pub perf: NativePerfReport,
}

/// Structured error returned by `upscale_request_native`.
#[derive(Debug, Serialize, Deserialize)]
pub struct NativeUpscaleError {
    pub code: String,
    pub message: String,
}

impl NativeUpscaleError {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
        }
    }
}

#[cfg(feature = "native_engine")]
pub use crate::commands::native_tooling::{
    default_native_tool_trt_cache_dir, native_benchmark_done_json,
    native_benchmark_result_json, native_benchmark_warmup_start_json,
    native_result_summary_json, native_smoke_success_lines, native_tool_run_banner,
    run_native_tool_request, NativeRuntimeOverrides, NativeToolRunRequest,
};

#[cfg(test)]
mod tests {
    #[cfg(feature = "native_engine")]
    use super::{
        default_native_tool_trt_cache_dir,
        native_benchmark_done_json, native_result_summary_json, native_smoke_success_lines,
        native_tool_run_banner,
        NativeJobSpec, NativePerfReport, NativeRuntimeOverrides, NativeToolRunRequest,
        NativeUpscaleError, NativeUpscaleResult, NativeVideoOutputProfile, NativeVideoSourceProfile,
    };
    #[cfg(feature = "native_engine")]
    use crate::commands::native_routing::{
        build_native_observed_metrics, build_native_runtime_snapshot, NativeExecutionRoute,
    };
    #[cfg(feature = "native_engine")]
    use crate::runtime_truth::{
        RunObservedMetrics, RunStatus, RuntimeConfigSnapshot, RuntimeEngineFamily,
        RUNTIME_CONFIG_SNAPSHOT_SCHEMA_V1, RUN_OBSERVED_METRICS_SCHEMA_V1,
    };
    use std::sync::{Mutex, OnceLock};

    fn env_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn native_runtime_env_guard_restores_previous_values() {
        let _guard = env_test_lock().lock().expect("env test lock");
        let direct_key = "VIDEOFORGE_NATIVE_ENGINE_DIRECT";
        let cache_dir_key = "VIDEOFORGE_TRT_CACHE_DIR";

        unsafe {
            std::env::set_var("VIDEOFORGE_ENABLE_NATIVE_ENGINE", "0");
            std::env::set_var(direct_key, "0");
            std::env::set_var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE", "0");
            std::env::set_var(cache_dir_key, "before");
        }

        {
            #[cfg(feature = "native_engine")]
            let _runtime_guard = NativeRuntimeOverrides::native_command(true)
                .with_trt_cache(true, Some(std::path::PathBuf::from("cache-dir")))
                .apply();

            #[cfg(feature = "native_engine")]
            {
                assert_eq!(
                    std::env::var("VIDEOFORGE_ENABLE_NATIVE_ENGINE").as_deref(),
                    Ok("1")
                );
                assert_eq!(std::env::var(direct_key).as_deref(), Ok("1"));
                assert_eq!(
                    std::env::var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE").as_deref(),
                    Ok("1")
                );
                assert_eq!(std::env::var(cache_dir_key).as_deref(), Ok("cache-dir"));
            }

            #[cfg(not(feature = "native_engine"))]
            {
                assert_eq!(
                    std::env::var("VIDEOFORGE_ENABLE_NATIVE_ENGINE").as_deref(),
                    Ok("0")
                );
                assert_eq!(std::env::var(direct_key).as_deref(), Ok("0"));
                assert_eq!(
                    std::env::var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE").as_deref(),
                    Ok("0")
                );
                assert_eq!(std::env::var(cache_dir_key).as_deref(), Ok("before"));
            }
        }

        assert_eq!(
            std::env::var("VIDEOFORGE_ENABLE_NATIVE_ENGINE").as_deref(),
            Ok("0")
        );
        assert_eq!(std::env::var(direct_key).as_deref(), Ok("0"));
        assert_eq!(
            std::env::var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE").as_deref(),
            Ok("0")
        );
        assert_eq!(std::env::var(cache_dir_key).as_deref(), Ok("before"));
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn source_profile_derives_expected_output_profile() {
        let source = NativeVideoSourceProfile {
            coded_width: 1920,
            coded_height: 1080,
            fps: 23.976,
            total_frames: 240,
            codec: videoforge_engine::codecs::sys::cudaVideoCodec::H264,
        };

        let output: NativeVideoOutputProfile = source.scaled_output(2);
        assert_eq!(output.width, 3840);
        assert_eq!(output.height, 2160);
        assert_eq!(output.nv12_pitch, 3840);
        assert_eq!(output.fps_num, 23_976);
        assert_eq!(output.fps_den, 1000);
        assert_eq!(output.frame_rate_arg(), "23976/1000");
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn tool_request_runtime_overrides_follow_cache_presence() {
        let _guard = env_test_lock().lock().expect("env test lock");
        let req = NativeToolRunRequest::new("in.mp4", "model.onnx", 2, "fp16")
            .with_output_path("out.mp4")
            .with_max_batch(Some(4))
            .with_native_direct(true)
            .with_trt_cache_dir(Some(std::path::PathBuf::from("cache-dir")));

        {
            let _runtime = req.runtime_overrides().apply();
            assert_eq!(
                std::env::var("VIDEOFORGE_ENABLE_NATIVE_ENGINE").as_deref(),
                Ok("1")
            );
            assert_eq!(
                std::env::var("VIDEOFORGE_NATIVE_ENGINE_DIRECT").as_deref(),
                Ok("1")
            );
            assert_eq!(
                std::env::var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE").as_deref(),
                Ok("1")
            );
            assert_eq!(std::env::var("VIDEOFORGE_TRT_CACHE_DIR").as_deref(), Ok("cache-dir"));
        }
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn tool_request_builder_normalizes_optional_output_and_route_label() {
        let direct = NativeToolRunRequest::new("in.mp4", "model.onnx", 2, "fp16")
            .with_optional_output_path(None)
            .with_native_direct(true);
        assert_eq!(direct.output_path, "");
        assert_eq!(direct.route_label(), "direct engine-v2 path");

        let cli = NativeToolRunRequest::new("in.mp4", "model.onnx", 2, "fp16")
            .with_optional_output_path(Some("out.mp4".to_string()));
        assert_eq!(cli.output_path, "out.mp4");
        assert_eq!(cli.route_label(), "default native command path");
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn tool_request_builder_derives_benchmark_cache_and_warmup_output() {
        let req = NativeToolRunRequest::new("in.mp4", "model.onnx", 2, "fp16")
            .with_output_path("results/final.mp4")
            .with_default_benchmark_trt_cache(true);
        assert_eq!(req.warmup_output_path(2), "results\\final.warmup2.mp4");
        assert_eq!(
            req.trt_cache_dir,
            Some(default_native_tool_trt_cache_dir())
        );
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn tool_request_prepare_runtime_filesystem_creates_cache_dir() {
        let cache_dir = std::env::temp_dir().join(format!(
            "videoforge-native-tool-cache-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&cache_dir);
        let req = NativeToolRunRequest::new("in.mp4", "model.onnx", 2, "fp16")
            .with_trt_cache_dir(Some(cache_dir.clone()));
        req.prepare_runtime_filesystem().expect("create cache dir");
        assert!(cache_dir.exists());
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn native_benchmark_done_payload_uses_shared_request_fields() {
        let request = NativeToolRunRequest::new("in.mp4", "model.onnx", 2, "fp16")
            .with_native_direct(true)
            .with_max_batch(Some(4));
        let report = NativeUpscaleResult {
            output_path: "out.mp4".to_string(),
            engine: "native_direct".to_string(),
            encoder_mode: "nvenc".to_string(),
            encoder_detail: None,
            audio_preserved: true,
            runtime_snapshot: None,
            observed_metrics: None,
            perf: NativePerfReport {
                requested_executor: Some("direct".to_string()),
                executed_executor: Some("direct".to_string()),
                direct_attempted: true,
                fallback_used: false,
                fallback_reason_code: None,
                fallback_reason_message: None,
                frames_processed: 120,
                effective_max_batch: 4,
                trt_cache_enabled: false,
                trt_cache_dir: None,
                total_elapsed_ms: Some(321),
                frames_decoded: Some(120),
                frames_preprocessed: None,
                frames_inferred: Some(120),
                frames_encoded: Some(120),
                preprocess_avg_us: None,
                inference_frame_avg_us: None,
                inference_dispatch_avg_us: None,
                postprocess_frame_avg_us: None,
                postprocess_dispatch_avg_us: None,
                encode_avg_us: None,
                vram_current_mb: None,
                vram_peak_mb: None,
            },
        };
        let payload = native_benchmark_done_json(&report, 999, &request, 2);
        assert_eq!(payload["event"], "done");
        assert_eq!(payload["mode"], "native");
        assert_eq!(payload["native_direct"], true);
        assert_eq!(payload["requested_max_batch"], 4);
        assert_eq!(payload["warmup_runs"], 2);
        assert_eq!(payload["elapsed_ms"], 999);
        assert_eq!(payload["output"], "out.mp4");
        assert!(payload["runtime_snapshot"].is_null());
        assert!(payload["observed_metrics"].is_null());
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn native_result_summary_json_embeds_runtime_truth_objects() {
        let runtime_snapshot = RuntimeConfigSnapshot::new(
            "run-1",
            "native_direct",
            RuntimeEngineFamily::Native,
            "in.mp4",
            "out.mp4",
        );
        let observed_metrics =
            RunObservedMetrics::new("run-1", "native_direct", RunStatus::Succeeded);
        let report = NativeUpscaleResult {
            output_path: "out.mp4".to_string(),
            engine: "native_direct".to_string(),
            encoder_mode: "nvenc".to_string(),
            encoder_detail: None,
            audio_preserved: true,
            runtime_snapshot: Some(runtime_snapshot),
            observed_metrics: Some(observed_metrics),
            perf: NativePerfReport {
                requested_executor: Some("direct".to_string()),
                executed_executor: Some("direct".to_string()),
                direct_attempted: true,
                fallback_used: false,
                fallback_reason_code: None,
                fallback_reason_message: None,
                frames_processed: 10,
                effective_max_batch: 1,
                trt_cache_enabled: false,
                trt_cache_dir: None,
                total_elapsed_ms: Some(42),
                frames_decoded: Some(10),
                frames_preprocessed: None,
                frames_inferred: Some(10),
                frames_encoded: Some(10),
                preprocess_avg_us: None,
                inference_frame_avg_us: None,
                inference_dispatch_avg_us: None,
                postprocess_frame_avg_us: None,
                postprocess_dispatch_avg_us: None,
                encode_avg_us: None,
                vram_current_mb: None,
                vram_peak_mb: None,
            },
        };

        let payload = native_result_summary_json(&report);
        assert_eq!(payload["runtime_snapshot"]["run_id"], "run-1");
        assert_eq!(payload["observed_metrics"]["route_id"], "native_direct");
        assert_eq!(payload["observed_metrics"]["status"], "succeeded");
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn native_tool_helpers_shape_smoke_output() {
        let request = NativeToolRunRequest::new("in.mp4", "model.onnx", 2, "fp16")
            .with_native_direct(true);
        assert_eq!(
            native_tool_run_banner(&request),
            "  Running native pipeline via direct engine-v2 path (this may take time)..."
        );

        let report = NativeUpscaleResult {
            output_path: "out.mp4".to_string(),
            engine: "native_direct".to_string(),
            encoder_mode: "nvenc".to_string(),
            encoder_detail: Some("h264".to_string()),
            audio_preserved: true,
            runtime_snapshot: None,
            observed_metrics: None,
            perf: NativePerfReport {
                frames_processed: 10,
                effective_max_batch: 1,
                trt_cache_enabled: false,
                trt_cache_dir: None,
                requested_executor: Some("direct".to_string()),
                executed_executor: Some("direct".to_string()),
                direct_attempted: true,
                fallback_used: false,
                fallback_reason_code: None,
                fallback_reason_message: None,
                total_elapsed_ms: None,
                frames_decoded: None,
                frames_preprocessed: None,
                frames_inferred: None,
                frames_encoded: None,
                preprocess_avg_us: None,
                inference_frame_avg_us: None,
                inference_dispatch_avg_us: None,
                postprocess_frame_avg_us: None,
                postprocess_dispatch_avg_us: None,
                encode_avg_us: None,
                vram_current_mb: None,
                vram_peak_mb: None,
            },
        };
        let lines = native_smoke_success_lines(&report);
        assert_eq!(lines[0], "frames=10 encoder_mode=nvenc encoder_detail=h264");
        assert!(lines.iter().any(|line| line == "encoder_mode=nvenc"));
        assert!(lines.iter().any(|line| line == "encoder_detail=h264"));
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn native_runtime_snapshot_serializes_direct_route_fields() {
        let job = NativeJobSpec {
            run_id: "run-321".to_string(),
            input_path: "in.mp4".to_string(),
            requested_output_path: "explicit.mp4".to_string(),
            model_path: "model.onnx".to_string(),
            scale: 4,
            precision: "fp16".to_string(),
            preserve_audio: false,
            max_batch: 2,
            trt_cache_enabled: true,
            trt_cache_dir: Some("cache-dir".to_string()),
        };
        let snapshot = build_native_runtime_snapshot(&job, &NativeExecutionRoute::direct());
        let value = serde_json::to_value(&snapshot).expect("serialize direct native snapshot");

        assert_eq!(value["schema_version"], RUNTIME_CONFIG_SNAPSHOT_SCHEMA_V1);
        assert_eq!(value["snapshot_kind"], "run_start");
        assert_eq!(value["run_id"], "run-321");
        assert_eq!(value["route_id"], "native_direct");
        assert_eq!(value["engine_family"], "native");
        assert_eq!(value["requested_executor"], "direct");
        assert_eq!(value["executed_executor"], "direct");
        assert_eq!(value["output_path"], "explicit.mp4");
        assert_eq!(value["model_path"], "model.onnx");
        assert_eq!(value["model_format"], "onnx");
        assert_eq!(value["extensions"]["native"]["requested_output_path"], "explicit.mp4");
        assert_eq!(value["extensions"]["native"]["preserve_audio"], false);
        assert!(value.get("fallback").is_none());
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn native_runtime_snapshot_serializes_route_and_fallback_fields() {
        let job = NativeJobSpec {
            run_id: "run-456".to_string(),
            input_path: "in.mp4".to_string(),
            requested_output_path: "".to_string(),
            model_path: "model.onnx".to_string(),
            scale: 2,
            precision: "fp16".to_string(),
            preserve_audio: true,
            max_batch: 4,
            trt_cache_enabled: true,
            trt_cache_dir: Some("cache-dir".to_string()),
        };
        let route = NativeExecutionRoute::cli_fallback(&NativeUpscaleError::new(
            "PIPELINE",
            "NVENC init failed",
        ));
        let snapshot = build_native_runtime_snapshot(&job, &route);
        let value = serde_json::to_value(&snapshot).expect("serialize native snapshot");

        assert_eq!(value["schema_version"], RUNTIME_CONFIG_SNAPSHOT_SCHEMA_V1);
        assert_eq!(value["run_id"], "run-456");
        assert_eq!(value["snapshot_kind"], "route_fallback");
        assert_eq!(value["engine_family"], "native");
        assert_eq!(value["route_id"], "native_via_rave_cli");
        assert_eq!(value["requested_executor"], "direct");
        assert_eq!(value["executed_executor"], "cli");
        assert_eq!(value["fallback"]["from_route_id"], "native_direct");
        assert_eq!(value["fallback"]["to_route_id"], "native_via_rave_cli");
        assert_eq!(value["fallback"]["reason_code"], "PIPELINE");
        assert_eq!(
            value["extensions"]["native"]["trt_cache_enabled"],
            true
        );
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn native_observed_metrics_maps_perf_report_without_inventing_missing_fields() {
        let perf = NativePerfReport {
            frames_processed: 120,
            effective_max_batch: 4,
            trt_cache_enabled: true,
            trt_cache_dir: Some("cache-dir".to_string()),
            requested_executor: Some("direct".to_string()),
            executed_executor: Some("direct".to_string()),
            direct_attempted: true,
            fallback_used: false,
            fallback_reason_code: None,
            fallback_reason_message: None,
            total_elapsed_ms: Some(3210),
            frames_decoded: Some(120),
            frames_preprocessed: Some(120),
            frames_inferred: Some(120),
            frames_encoded: Some(120),
            preprocess_avg_us: None,
            inference_frame_avg_us: Some(5100),
            inference_dispatch_avg_us: None,
            postprocess_frame_avg_us: None,
            postprocess_dispatch_avg_us: None,
            encode_avg_us: Some(1400),
            vram_current_mb: None,
            vram_peak_mb: Some(4096),
        };
        let metrics =
            build_native_observed_metrics("run-456", "native_direct", RunStatus::Succeeded, Some(&perf), None);
        let value = serde_json::to_value(&metrics).expect("serialize native observed metrics");

        assert_eq!(value["schema_version"], RUN_OBSERVED_METRICS_SCHEMA_V1);
        assert_eq!(value["run_id"], "run-456");
        assert_eq!(value["route_id"], "native_direct");
        assert_eq!(value["status"], "succeeded");
        assert_eq!(value["total_elapsed_ms"], 3210);
        assert_eq!(value["work_units_processed"], 120);
        assert_eq!(value["extensions"]["native"]["frames_preprocessed"], 120);
        assert_eq!(value["extensions"]["native"]["inference_frame_avg_us"], 5100);
        assert_eq!(value["extensions"]["native"]["encode_avg_us"], 1400);
        assert!(value["extensions"]["native"]
            .get("preprocess_avg_us")
            .is_none());
        assert!(value["extensions"]["native"]
            .get("vram_current_mb")
            .is_none());
        assert!(value.get("requested_executor").is_none());
    }
}

#[cfg(feature = "native_engine")]
fn classify_backend_init_error(model_path: &str, err: &str) -> String {
    if err.contains("Load model from")
        && err.contains("Type Error:")
        && (err.contains("tensor(float16)") || err.contains("tensor(float)"))
    {
        return format!(
            "Invalid ONNX artifact: {model_path} failed ORT model validation due to inconsistent tensor types. \
This is usually a broken or incompatible export, not a native runtime failure. Original error: {err}"
        );
    }

    format!("TensorRT backend init failed: {err}")
}

pub(crate) fn native_engine_runtime_enabled() -> bool {
    match std::env::var("VIDEOFORGE_ENABLE_NATIVE_ENGINE") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

#[cfg(feature = "native_engine")]
pub(crate) fn native_engine_direct_enabled() -> bool {
    match std::env::var("VIDEOFORGE_NATIVE_ENGINE_DIRECT") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

// =============================================================================
// Command (always present — returns FEATURE_DISABLED when feature is off)
// =============================================================================

/// Run the native engine-v2 GPU upscaling pipeline.
///
/// # Parameters
/// - `input_path`:   Source video file (MP4, MKV, …).
/// - `output_path`:  Destination MP4.  Auto-generated if empty.
/// - `model_path`:   Path to a TensorRT-compatible ONNX model file.
/// - `scale`:        Upscale factor (e.g. 4 for 4×).
/// - `precision`:    `"fp32"` | `"fp16"` (default `"fp32"`).
/// - `audio`:        Whether to preserve audio from input (default `true`).
///
/// # Returns
/// `Ok(NativeUpscaleResult)` on success, `Err(NativeUpscaleError)` on failure.
#[tauri::command]
pub async fn upscale_request_native(
    input_path: String,
    output_path: String,
    model_path: String,
    scale: u32,
    precision: Option<String>,
    audio: Option<bool>,
    max_batch: Option<u32>,
) -> Result<NativeUpscaleResult, String> {
    let request = NativeUpscaleRequest {
        input_path,
        output_path,
        model_path,
        scale,
        precision,
        audio,
        max_batch,
    };

    // Validate inputs regardless of feature flag.
    if !Path::new(&request.input_path).exists() {
        return Err(serde_json::to_string(&NativeUpscaleError::new(
            "INPUT_NOT_FOUND",
            format!("Input file not found: {}", request.input_path),
        ))
        .unwrap());
    }

    #[cfg(not(feature = "native_engine"))]
    {
        let _ = (
            &request.output_path,
            &request.model_path,
            request.scale,
            &request.precision,
            &request.audio,
            &request.max_batch,
        );
        Err(serde_json::to_string(&NativeUpscaleError::new(
            "FEATURE_DISABLED",
            "The native_engine feature is not compiled in. \
             Rebuild with `cargo build --features native_engine` or use the \
             Python pipeline (upscale_request).",
        ))
        .unwrap())
    }

    #[cfg(feature = "native_engine")]
    {
        if !native_engine_runtime_enabled() {
            return Err(serde_json::to_string(&NativeUpscaleError::new(
                "NATIVE_ENGINE_DISABLED",
                "Native engine is disabled by default for stability. \
                 Set VIDEOFORGE_ENABLE_NATIVE_ENGINE=1 to opt in explicitly.",
            ))
            .unwrap());
        }

        let job = NativeJobSpec::resolve(
            request.input_path,
            request.output_path,
            request.model_path,
            request.scale,
            request.precision,
            request.audio,
            request.max_batch,
        )?;

        run_native_job(job).await
    }
}

