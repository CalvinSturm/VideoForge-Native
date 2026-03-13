#[cfg(feature = "native_engine")]
use std::ffi::OsString;
#[cfg(feature = "native_engine")]
use std::path::{Path, PathBuf};

#[cfg(feature = "native_engine")]
use crate::commands::native_engine::{upscale_request_native, NativeUpscaleResult};

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone, Default)]
pub struct NativeRuntimeOverrides {
    pub enable_native: bool,
    pub direct: Option<bool>,
    pub trt_cache_enabled: Option<bool>,
    pub trt_cache_dir: Option<PathBuf>,
}

#[cfg(feature = "native_engine")]
pub struct NativeRuntimeEnvGuard {
    saved: Vec<(&'static str, Option<OsString>)>,
}

#[cfg(feature = "native_engine")]
impl NativeRuntimeOverrides {
    pub fn native_command(direct: bool) -> Self {
        Self {
            enable_native: true,
            direct: Some(direct),
            trt_cache_enabled: None,
            trt_cache_dir: None,
        }
    }

    pub fn with_trt_cache(mut self, enabled: bool, dir: Option<PathBuf>) -> Self {
        self.trt_cache_enabled = Some(enabled);
        self.trt_cache_dir = dir;
        self
    }

    pub fn apply(&self) -> NativeRuntimeEnvGuard {
        let mut vars: Vec<(&'static str, Option<OsString>)> = Vec::with_capacity(4);
        vars.push((
            "VIDEOFORGE_ENABLE_NATIVE_ENGINE",
            Some(OsString::from(if self.enable_native { "1" } else { "0" })),
        ));

        if let Some(direct) = self.direct {
            vars.push((
                "VIDEOFORGE_NATIVE_ENGINE_DIRECT",
                Some(OsString::from(if direct { "1" } else { "0" })),
            ));
        }

        if let Some(enabled) = self.trt_cache_enabled {
            vars.push((
                "VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE",
                Some(OsString::from(if enabled { "1" } else { "0" })),
            ));
        }

        if self.trt_cache_enabled.unwrap_or(false) {
            vars.push((
                "VIDEOFORGE_TRT_CACHE_DIR",
                self.trt_cache_dir.clone().map(OsString::from),
            ));
        }

        let mut saved = Vec::with_capacity(vars.len());
        for (key, value) in vars {
            saved.push((key, std::env::var_os(key)));
            match value {
                Some(v) => unsafe { std::env::set_var(key, v) },
                None => unsafe { std::env::remove_var(key) },
            }
        }

        NativeRuntimeEnvGuard { saved }
    }
}

#[cfg(feature = "native_engine")]
impl Drop for NativeRuntimeEnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.saved.drain(..).rev() {
            match value {
                Some(v) => unsafe { std::env::set_var(key, v) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
pub struct NativeToolRunRequest {
    pub input_path: String,
    pub output_path: String,
    pub model_path: String,
    pub scale: u32,
    pub precision: String,
    pub preserve_audio: bool,
    pub max_batch: Option<u32>,
    pub native_direct: bool,
    pub trt_cache_dir: Option<PathBuf>,
}

#[cfg(feature = "native_engine")]
impl NativeToolRunRequest {
    pub fn new(
        input_path: impl Into<String>,
        model_path: impl Into<String>,
        scale: u32,
        precision: impl Into<String>,
    ) -> Self {
        Self {
            input_path: input_path.into(),
            output_path: String::new(),
            model_path: model_path.into(),
            scale,
            precision: precision.into(),
            preserve_audio: true,
            max_batch: None,
            native_direct: false,
            trt_cache_dir: None,
        }
    }

    pub fn with_output_path(mut self, output_path: impl Into<String>) -> Self {
        self.output_path = output_path.into();
        self
    }

    pub fn with_optional_output_path(mut self, output_path: Option<String>) -> Self {
        self.output_path = output_path.unwrap_or_default();
        self
    }

    pub fn with_preserve_audio(mut self, preserve_audio: bool) -> Self {
        self.preserve_audio = preserve_audio;
        self
    }

    pub fn with_max_batch(mut self, max_batch: Option<u32>) -> Self {
        self.max_batch = max_batch;
        self
    }

    pub fn with_native_direct(mut self, native_direct: bool) -> Self {
        self.native_direct = native_direct;
        self
    }

    pub fn with_trt_cache_dir(mut self, trt_cache_dir: Option<PathBuf>) -> Self {
        self.trt_cache_dir = trt_cache_dir;
        self
    }

    pub fn with_default_benchmark_trt_cache(mut self, enabled: bool) -> Self {
        self.trt_cache_dir = enabled.then(default_native_tool_trt_cache_dir);
        self
    }

    pub fn route_label(&self) -> &'static str {
        if self.native_direct {
            "direct engine-v2 path"
        } else {
            "default native command path"
        }
    }

    pub fn warmup_output_path(&self, run_idx: u32) -> String {
        native_tool_warmup_output_path(&self.output_path, run_idx)
    }

    pub fn prepare_runtime_filesystem(&self) -> Result<(), String> {
        if let Some(cache_dir) = &self.trt_cache_dir {
            std::fs::create_dir_all(cache_dir).map_err(|err| {
                format!(
                    "Failed to create TensorRT cache directory {}: {err}",
                    cache_dir.display()
                )
            })?;
        }
        Ok(())
    }

    pub fn runtime_overrides(&self) -> NativeRuntimeOverrides {
        NativeRuntimeOverrides::native_command(self.native_direct)
            .with_trt_cache(self.trt_cache_dir.is_some(), self.trt_cache_dir.clone())
    }
}

#[cfg(feature = "native_engine")]
pub async fn run_native_tool_request(
    request: NativeToolRunRequest,
) -> Result<NativeUpscaleResult, String> {
    let _runtime_env = request.runtime_overrides().apply();
    upscale_request_native(
        request.input_path,
        request.output_path,
        request.model_path,
        request.scale,
        Some(request.precision),
        Some(request.preserve_audio),
        request.max_batch,
    )
    .await
}

#[cfg(feature = "native_engine")]
pub fn default_native_tool_trt_cache_dir() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("artifacts")
        .join("benchmarks")
        .join("trt_cache")
}

#[cfg(feature = "native_engine")]
pub fn native_tool_warmup_output_path(output: &str, run_idx: u32) -> String {
    let path = Path::new(output);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("mp4");
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    parent
        .join(format!("{stem}.warmup{run_idx}.{ext}"))
        .to_string_lossy()
        .to_string()
}

#[cfg(feature = "native_engine")]
pub fn native_result_summary_json(
    report: &NativeUpscaleResult,
) -> serde_json::Map<String, serde_json::Value> {
    use serde_json::{Map, Value};

    let mut map = Map::new();
    map.insert(
        "output".to_string(),
        Value::String(report.output_path.clone()),
    );
    map.insert("engine".to_string(), Value::String(report.engine.clone()));
    map.insert(
        "encoder_mode".to_string(),
        Value::String(report.encoder_mode.clone()),
    );
    map.insert(
        "encoder_detail".to_string(),
        report
            .encoder_detail
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "audio_preserved".to_string(),
        Value::Bool(report.audio_preserved),
    );
    map.insert(
        "requested_executor".to_string(),
        report
            .perf
            .requested_executor
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "executed_executor".to_string(),
        report
            .perf
            .executed_executor
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "direct_attempted".to_string(),
        Value::Bool(report.perf.direct_attempted),
    );
    map.insert(
        "fallback_used".to_string(),
        Value::Bool(report.perf.fallback_used),
    );
    map.insert(
        "fallback_reason_code".to_string(),
        report
            .perf
            .fallback_reason_code
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "fallback_reason_message".to_string(),
        report
            .perf
            .fallback_reason_message
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "frames_processed".to_string(),
        Value::from(report.perf.frames_processed),
    );
    map.insert(
        "native_total_elapsed_ms".to_string(),
        report
            .perf
            .total_elapsed_ms
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "frames_decoded".to_string(),
        report
            .perf
            .frames_decoded
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "frames_preprocessed".to_string(),
        report
            .perf
            .frames_preprocessed
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "frames_inferred".to_string(),
        report
            .perf
            .frames_inferred
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "frames_encoded".to_string(),
        report
            .perf
            .frames_encoded
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "preprocess_avg_us".to_string(),
        report
            .perf
            .preprocess_avg_us
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "inference_frame_avg_us".to_string(),
        report
            .perf
            .inference_frame_avg_us
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "inference_dispatch_avg_us".to_string(),
        report
            .perf
            .inference_dispatch_avg_us
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "postprocess_frame_avg_us".to_string(),
        report
            .perf
            .postprocess_frame_avg_us
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "postprocess_dispatch_avg_us".to_string(),
        report
            .perf
            .postprocess_dispatch_avg_us
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "encode_avg_us".to_string(),
        report
            .perf
            .encode_avg_us
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "vram_current_mb".to_string(),
        report
            .perf
            .vram_current_mb
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "vram_peak_mb".to_string(),
        report
            .perf
            .vram_peak_mb
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "effective_max_batch".to_string(),
        Value::from(report.perf.effective_max_batch),
    );
    map.insert(
        "trt_cache_enabled".to_string(),
        Value::Bool(report.perf.trt_cache_enabled),
    );
    map.insert(
        "trt_cache_dir".to_string(),
        report
            .perf
            .trt_cache_dir
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "runtime_snapshot".to_string(),
        report
            .runtime_snapshot
            .as_ref()
            .map(|snapshot| serde_json::to_value(snapshot).unwrap_or(Value::Null))
            .unwrap_or(Value::Null),
    );
    map.insert(
        "observed_metrics".to_string(),
        report
            .observed_metrics
            .as_ref()
            .map(|metrics| serde_json::to_value(metrics).unwrap_or(Value::Null))
            .unwrap_or(Value::Null),
    );
    map
}

#[cfg(feature = "native_engine")]
pub fn native_benchmark_warmup_start_json(
    request: &NativeToolRunRequest,
    index: u32,
    output: String,
) -> serde_json::Value {
    serde_json::json!({
        "event": "warmup_start",
        "index": index,
        "output": output,
        "native_direct": request.native_direct,
        "trt_cache_enabled": request.trt_cache_dir.is_some(),
    })
}

#[cfg(feature = "native_engine")]
pub fn native_benchmark_result_json(
    report: &NativeUpscaleResult,
    event: &'static str,
    elapsed_ms: u128,
) -> serde_json::Value {
    let mut payload = native_result_summary_json(report);
    payload.insert("event".to_string(), serde_json::json!(event));
    payload.insert("elapsed_ms".to_string(), serde_json::json!(elapsed_ms));
    serde_json::Value::Object(payload)
}

#[cfg(feature = "native_engine")]
pub fn native_benchmark_done_json(
    report: &NativeUpscaleResult,
    elapsed_ms: u128,
    request: &NativeToolRunRequest,
    warmup_runs: u32,
) -> serde_json::Value {
    let mut payload = native_result_summary_json(report);
    payload.insert("event".to_string(), serde_json::json!("done"));
    payload.insert("elapsed_ms".to_string(), serde_json::json!(elapsed_ms));
    payload.insert("mode".to_string(), serde_json::json!("native"));
    payload.insert(
        "native_direct".to_string(),
        serde_json::json!(request.native_direct),
    );
    payload.insert(
        "requested_max_batch".to_string(),
        serde_json::json!(request.max_batch),
    );
    payload.insert("warmup_runs".to_string(), serde_json::json!(warmup_runs));
    serde_json::Value::Object(payload)
}

#[cfg(feature = "native_engine")]
pub fn native_result_summary_lines(report: &NativeUpscaleResult) -> Vec<String> {
    let mut lines = vec![format!(
        "frames={} encoder_mode={} encoder_detail={}",
        report.perf.frames_processed,
        report.encoder_mode,
        report.encoder_detail.as_deref().unwrap_or("none")
    )];
    if let Some(elapsed_ms) = report.perf.total_elapsed_ms {
        lines.push(format!("native elapsed ms: {elapsed_ms}"));
    }
    if let Some(vram_peak_mb) = report.perf.vram_peak_mb {
        lines.push(format!("native peak vram mb: {vram_peak_mb}"));
    }
    if let Some(requested) = &report.perf.requested_executor {
        lines.push(format!("requested executor: {requested}"));
    }
    if let Some(executed) = &report.perf.executed_executor {
        lines.push(format!("executed executor: {executed}"));
    }
    if report.perf.fallback_used {
        lines.push(format!(
            "fallback: {} {}",
            report
                .perf
                .fallback_reason_code
                .as_deref()
                .unwrap_or("unknown"),
            report
                .perf
                .fallback_reason_message
                .as_deref()
                .unwrap_or("no message")
        ));
    }
    lines
}

#[cfg(feature = "native_engine")]
pub fn native_tool_run_banner(request: &NativeToolRunRequest) -> String {
    format!(
        "  Running native pipeline via {} (this may take time)...",
        request.route_label()
    )
}

#[cfg(feature = "native_engine")]
pub fn native_smoke_success_lines(report: &NativeUpscaleResult) -> Vec<String> {
    let mut lines = native_result_summary_lines(report);
    lines.push(format!("encoder_mode={}", report.encoder_mode));
    if let Some(detail) = &report.encoder_detail {
        lines.push(format!("encoder_detail={detail}"));
    }
    lines
}
