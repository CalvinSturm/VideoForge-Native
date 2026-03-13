use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::python_env::WorkerCaps;
use crate::runtime_truth::{RunObservedMetrics, RunStatus, RuntimeConfigSnapshot};

pub const RUN_MANIFEST_SCHEMA_V1: &str = "videoforge.run_manifest.v1";
const RUN_MANIFEST_FILENAME_V1: &str = "videoforge.run_manifest.v1.json";
const RUNTIME_CONFIG_SNAPSHOT_FILENAME_V1: &str = "videoforge.runtime_config_snapshot.v1.json";
const RUN_OBSERVED_METRICS_FILENAME_V1: &str = "videoforge.run_observed_metrics.v1.json";
const RUNSCOPE_REPORT_FILENAME_V1: &str = "videoforge_run.json";

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkerCapsSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_level: Option<String>,
    pub use_typed_ipc: bool,
    pub use_shm_proto_v2: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shm_ring_size: Option<u32>,
    pub use_events: bool,
    pub prealloc_tensors: bool,
    pub deterministic: bool,
}

impl From<&WorkerCaps> for WorkerCapsSnapshot {
    fn from(value: &WorkerCaps) -> Self {
        Self {
            log_level: value.log_level.clone(),
            use_typed_ipc: value.use_typed_ipc,
            use_shm_proto_v2: value.use_shm_proto_v2,
            shm_ring_size: value.shm_ring_size,
            use_events: value.use_events,
            prealloc_tensors: value.prealloc_tensors,
            deterministic: value.deterministic,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct ProtocolSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipc_protocol_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shm_protocol_version: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RunManifestV1 {
    pub schema_version: String,
    pub created_at_utc: String,
    pub job_id: String,
    pub input_path: String,
    pub output_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route_id: Option<String>,
    pub scale: u32,
    pub precision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    pub worker_caps: WorkerCapsSnapshot,
    pub protocol: ProtocolSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_executor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executed_executor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_preserved: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trt_cache_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trt_cache_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_version: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RunManifestInputs<'a> {
    pub input_path: &'a str,
    pub output_path: &'a str,
    pub engine_family: Option<&'a str>,
    pub route_id: Option<&'a str>,
    pub scale: u32,
    pub precision: &'a str,
    pub model_key: Option<&'a str>,
    pub model_path: Option<&'a str>,
    pub worker_caps: WorkerCapsSnapshot,
    pub ipc_protocol_version: Option<u32>,
    pub shm_protocol_version: Option<u32>,
    pub requested_executor: Option<&'a str>,
    pub executed_executor: Option<&'a str>,
    pub audio_preserved: Option<bool>,
    pub trt_cache_enabled: Option<bool>,
    pub trt_cache_dir: Option<&'a str>,
    pub app_version: Option<&'a str>,
}

#[derive(Debug, Clone)]
pub struct RunArtifactFinalizeInputs<'a> {
    pub runtime_snapshot: &'a RuntimeConfigSnapshot,
    pub observed_metrics: Option<&'a RunObservedMetrics>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct VideoForgeRunScopeReport {
    pub producer: String,
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suite: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scenario: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command: Vec<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub metrics: std::collections::BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<String>,
    pub runtime_snapshot: RuntimeConfigSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_metrics: Option<RunObservedMetrics>,
}

pub fn run_artifacts_enabled_from_env() -> bool {
    match std::env::var("VIDEOFORGE_ENABLE_RUN_ARTIFACTS") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

pub fn normalize_path_for_id(path: &str) -> String {
    path.replace('\\', "/")
}

pub fn compute_job_id(
    input_path: &str,
    output_path: &str,
    scale: u32,
    precision: &str,
    model_key: Option<&str>,
    worker_caps: Option<&WorkerCapsSnapshot>,
) -> String {
    let mut hasher = Sha256::new();
    let caps_json = worker_caps
        .and_then(|c| serde_json::to_string(c).ok())
        .unwrap_or_default();
    let payload = format!(
        "input={}\noutput={}\nscale={}\nprecision={}\nmodel={}\nworker_caps={}\n",
        normalize_path_for_id(input_path),
        normalize_path_for_id(output_path),
        scale,
        precision,
        model_key.unwrap_or_default(),
        caps_json
    );
    hasher.update(payload.as_bytes());
    let digest = hasher.finalize();
    format!("{:x}", digest)
}

pub fn build_run_manifest_v1(inputs: &RunManifestInputs<'_>) -> Result<RunManifestV1> {
    let worker_caps = inputs.worker_caps.clone();
    let created_at_utc = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .context("formatting manifest timestamp")?;
    let job_id = compute_job_id(
        inputs.input_path,
        inputs.output_path,
        inputs.scale,
        inputs.precision,
        inputs.model_key,
        Some(&worker_caps),
    );

    Ok(RunManifestV1 {
        schema_version: RUN_MANIFEST_SCHEMA_V1.to_string(),
        created_at_utc,
        job_id,
        input_path: inputs.input_path.to_string(),
        output_path: inputs.output_path.to_string(),
        engine_family: inputs.engine_family.map(str::to_string),
        route_id: inputs.route_id.map(str::to_string),
        scale: inputs.scale,
        precision: inputs.precision.to_string(),
        model_key: inputs.model_key.map(str::to_string),
        model_path: inputs.model_path.map(str::to_string),
        worker_caps,
        protocol: ProtocolSnapshot {
            ipc_protocol_version: inputs.ipc_protocol_version,
            shm_protocol_version: inputs.shm_protocol_version,
        },
        requested_executor: inputs.requested_executor.map(str::to_string),
        executed_executor: inputs.executed_executor.map(str::to_string),
        audio_preserved: inputs.audio_preserved,
        trt_cache_enabled: inputs.trt_cache_enabled,
        trt_cache_dir: inputs.trt_cache_dir.map(str::to_string),
        app_version: inputs.app_version.map(str::to_string),
    })
}

pub fn maybe_write_run_manifest(
    enable: bool,
    inputs: &RunManifestInputs<'_>,
) -> Result<Option<PathBuf>> {
    if !enable {
        return Ok(None);
    }

    let manifest = build_run_manifest_v1(inputs)?;
    let output_dir = Path::new(inputs.output_path)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let artifacts_dir = output_dir.join(".videoforge_runs").join(&manifest.job_id);
    fs::create_dir_all(&artifacts_dir)
        .with_context(|| format!("creating artifacts dir '{}'", artifacts_dir.display()))?;

    let final_path = artifacts_dir.join(RUN_MANIFEST_FILENAME_V1);
    write_json_atomic(&final_path, &manifest)
        .with_context(|| format!("writing run manifest artifact '{}'", final_path.display()))?;

    Ok(Some(final_path))
}

pub fn maybe_finalize_run_artifacts(
    artifacts_root: Option<&Path>,
    inputs: &RunArtifactFinalizeInputs<'_>,
) -> Result<Option<PathBuf>> {
    let Some(root) = artifacts_root else {
        return Ok(None);
    };

    fs::create_dir_all(root)
        .with_context(|| format!("creating finalized artifacts dir '{}'", root.display()))?;

    let runtime_snapshot_path = root.join(RUNTIME_CONFIG_SNAPSHOT_FILENAME_V1);
    write_json_atomic(&runtime_snapshot_path, inputs.runtime_snapshot).with_context(|| {
        format!(
            "writing runtime snapshot artifact '{}'",
            runtime_snapshot_path.display()
        )
    })?;

    if let Some(observed_metrics) = inputs.observed_metrics {
        let observed_metrics_path = root.join(RUN_OBSERVED_METRICS_FILENAME_V1);
        write_json_atomic(&observed_metrics_path, observed_metrics).with_context(|| {
            format!(
                "writing observed metrics artifact '{}'",
                observed_metrics_path.display()
            )
        })?;
    }

    let runscope_report = build_runscope_report(inputs);
    let runscope_report_path = root.join(RUNSCOPE_REPORT_FILENAME_V1);
    write_json_atomic(&runscope_report_path, &runscope_report).with_context(|| {
        format!(
            "writing RunScope report artifact '{}'",
            runscope_report_path.display()
        )
    })?;

    Ok(Some(runscope_report_path))
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("artifact.json");
    let tmp_path = path.with_file_name(format!("{file_name}.tmp"));
    let json = serde_json::to_vec_pretty(value).context("serializing JSON artifact")?;
    fs::write(&tmp_path, json)
        .with_context(|| format!("writing temp artifact '{}'", tmp_path.display()))?;
    fs::rename(&tmp_path, path).with_context(|| {
        format!(
            "renaming temp artifact '{}' -> '{}'",
            tmp_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn build_runscope_report(inputs: &RunArtifactFinalizeInputs<'_>) -> VideoForgeRunScopeReport {
    let runtime_snapshot = inputs.runtime_snapshot.clone();
    let observed_metrics = inputs.observed_metrics.cloned();
    let route_id = runtime_snapshot.route_id.clone();
    let model = runtime_snapshot
        .model_key
        .clone()
        .or_else(|| basename_without_extension(runtime_snapshot.model_path.as_deref()));
    let scenario = build_runscope_scenario(&runtime_snapshot, model.as_deref());
    let status = observed_metrics
        .as_ref()
        .map(|metrics| runscope_status(&metrics.status))
        .unwrap_or("unknown")
        .to_string();
    let started_at = Some(OffsetDateTime::now_utc().format(&Rfc3339).unwrap());
    let finished_at = terminal_finished_at(observed_metrics.as_ref());
    let duration_ms = observed_metrics
        .as_ref()
        .and_then(|metrics| metrics.total_elapsed_ms);
    let exit_code = observed_metrics
        .as_ref()
        .map(|metrics| match metrics.status {
            RunStatus::Succeeded => 0,
            RunStatus::Running => 2,
            RunStatus::Cancelled => 130,
            RunStatus::Failed => 1,
        });

    VideoForgeRunScopeReport {
        producer: "videoforge".to_string(),
        run_id: runtime_snapshot.run_id.clone(),
        suite: runtime_snapshot
            .media_kind
            .map(|kind| match kind {
                crate::runtime_truth::RuntimeMediaKind::Image => "image_upscale".to_string(),
                crate::runtime_truth::RuntimeMediaKind::Video => "video_upscale".to_string(),
            })
            .or_else(|| Some("upscale".to_string())),
        scenario,
        label: Some(route_id.clone()),
        status,
        started_at,
        finished_at,
        duration_ms,
        exit_code,
        backend: Some(route_id.clone()),
        model,
        precision: runtime_snapshot.precision.clone(),
        dataset: None,
        input_count: Some(1),
        cwd: std::env::current_dir()
            .ok()
            .map(|path| path.to_string_lossy().to_string()),
        command: build_runscope_command(&runtime_snapshot),
        metrics: build_runscope_metrics(observed_metrics.as_ref()),
        engine: Some(format!("{:?}", runtime_snapshot.engine_family).to_ascii_lowercase()),
        pipeline: Some(route_id),
        runtime_snapshot,
        observed_metrics,
    }
}

fn build_runscope_scenario(
    runtime_snapshot: &RuntimeConfigSnapshot,
    model: Option<&str>,
) -> Option<String> {
    let mut parts = Vec::new();
    parts.push(runtime_snapshot.route_id.clone());
    if let Some(model) = model.filter(|model| !model.trim().is_empty()) {
        parts.push(model.to_string());
    }
    if let Some(precision) = runtime_snapshot
        .precision
        .as_deref()
        .filter(|precision| !precision.trim().is_empty())
    {
        parts.push(precision.to_string());
    }
    if let Some(scale) = runtime_snapshot.scale {
        parts.push(format!("x{scale}"));
    }
    (!parts.is_empty()).then_some(parts.join("_"))
}

fn build_runscope_command(runtime_snapshot: &RuntimeConfigSnapshot) -> Vec<String> {
    let mut command = vec!["videoforge".to_string(), "upscale".to_string()];
    if matches!(
        runtime_snapshot.engine_family,
        crate::runtime_truth::RuntimeEngineFamily::Native
    ) {
        command.push("--native".to_string());
    }
    if let Some(route_id) = runtime_snapshot
        .executed_executor
        .as_deref()
        .filter(|route_id| !route_id.trim().is_empty())
    {
        command.push(format!("--executor={route_id}"));
    }
    command
}

fn build_runscope_metrics(
    observed_metrics: Option<&RunObservedMetrics>,
) -> std::collections::BTreeMap<String, f64> {
    let mut metrics = std::collections::BTreeMap::new();
    let Some(observed_metrics) = observed_metrics else {
        return metrics;
    };

    if let Some(total_elapsed_ms) = observed_metrics.total_elapsed_ms {
        metrics.insert("total_elapsed_ms".to_string(), total_elapsed_ms as f64);
    }
    if let Some(work_units_processed) = observed_metrics.work_units_processed {
        metrics.insert(
            "work_units_processed".to_string(),
            work_units_processed as f64,
        );
        if let Some(total_elapsed_ms) = observed_metrics.total_elapsed_ms.filter(|value| *value > 0)
        {
            let fps = work_units_processed as f64 / (total_elapsed_ms as f64 / 1000.0);
            metrics.insert("fps".to_string(), fps);
        }
    }

    if let Some(python) = observed_metrics.extensions.python.as_ref() {
        if let Some(frames_decoded) = python.frames_decoded {
            metrics.insert("frames_decoded".to_string(), frames_decoded as f64);
        }
        if let Some(frames_processed) = python.frames_processed {
            metrics.insert("frames_processed".to_string(), frames_processed as f64);
        }
        if let Some(frames_encoded) = python.frames_encoded {
            metrics.insert("frames_encoded".to_string(), frames_encoded as f64);
        }
    }

    if let Some(native) = observed_metrics.extensions.native.as_ref() {
        insert_native_metric(&mut metrics, "frames_decoded", native.frames_decoded);
        insert_native_metric(
            &mut metrics,
            "frames_preprocessed",
            native.frames_preprocessed,
        );
        insert_native_metric(&mut metrics, "frames_inferred", native.frames_inferred);
        insert_native_metric(&mut metrics, "frames_encoded", native.frames_encoded);
        insert_native_metric(&mut metrics, "preprocess_avg_us", native.preprocess_avg_us);
        insert_native_metric(
            &mut metrics,
            "inference_frame_avg_us",
            native.inference_frame_avg_us,
        );
        insert_native_metric(
            &mut metrics,
            "inference_dispatch_avg_us",
            native.inference_dispatch_avg_us,
        );
        insert_native_metric(
            &mut metrics,
            "postprocess_frame_avg_us",
            native.postprocess_frame_avg_us,
        );
        insert_native_metric(
            &mut metrics,
            "postprocess_dispatch_avg_us",
            native.postprocess_dispatch_avg_us,
        );
        insert_native_metric(&mut metrics, "encode_avg_us", native.encode_avg_us);
        insert_native_metric(&mut metrics, "vram_current_mb", native.vram_current_mb);
        insert_native_metric(&mut metrics, "vram_peak_mb", native.vram_peak_mb);
    }

    metrics
}

fn insert_native_metric(
    metrics: &mut std::collections::BTreeMap<String, f64>,
    key: &str,
    value: Option<u64>,
) {
    if let Some(value) = value {
        metrics.insert(key.to_string(), value as f64);
    }
}

fn basename_without_extension(path: Option<&str>) -> Option<String> {
    let path = path?;
    Path::new(path)
        .file_stem()
        .and_then(|value| value.to_str())
        .map(ToString::to_string)
}

fn runscope_status(status: &RunStatus) -> &'static str {
    match status {
        RunStatus::Running => "unknown",
        RunStatus::Succeeded => "pass",
        RunStatus::Failed => "fail",
        RunStatus::Cancelled => "error",
    }
}

fn terminal_finished_at(observed_metrics: Option<&RunObservedMetrics>) -> Option<String> {
    let observed_metrics = observed_metrics?;
    if matches!(observed_metrics.status, RunStatus::Running) {
        None
    } else {
        Some(OffsetDateTime::now_utc().format(&Rfc3339).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_truth::{
        NativeRuntimeMetricsExtension, RunObservedMetrics, RunStatus, RuntimeConfigSnapshot,
        RuntimeEngineFamily, RuntimeMediaKind, RuntimeMetricsExtensions,
    };
    use serde_json::Value;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_caps() -> WorkerCaps {
        WorkerCaps::default()
    }

    fn test_inputs<'a>(output_path: &'a str, caps: &'a WorkerCaps) -> RunManifestInputs<'a> {
        RunManifestInputs {
            input_path: r"C:\input\clip.mp4",
            output_path,
            engine_family: Some("python"),
            route_id: Some("python_sidecar"),
            scale: 4,
            precision: "fp16",
            model_key: Some("RCAN_x4"),
            model_path: None,
            worker_caps: WorkerCapsSnapshot::from(caps),
            ipc_protocol_version: Some(1),
            shm_protocol_version: None,
            requested_executor: None,
            executed_executor: None,
            audio_preserved: None,
            trt_cache_enabled: None,
            trt_cache_dir: None,
            app_version: Some("0.1.0"),
        }
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), nanos))
    }

    fn test_runtime_snapshot() -> RuntimeConfigSnapshot {
        let mut snapshot = RuntimeConfigSnapshot::new(
            "run-123",
            "native_direct",
            RuntimeEngineFamily::Native,
            r"C:\input\clip.mp4",
            r"C:\output\clip_x2.mp4",
        );
        snapshot.media_kind = Some(RuntimeMediaKind::Video);
        snapshot.model_key = Some("2x_SPAN_soft".to_string());
        snapshot.model_path = Some(r"C:\weights\2x_SPAN_soft.onnx".to_string());
        snapshot.scale = Some(2);
        snapshot.precision = Some("fp16".to_string());
        snapshot.requested_executor = Some("direct".to_string());
        snapshot.executed_executor = Some("direct".to_string());
        snapshot
    }

    fn test_observed_metrics() -> RunObservedMetrics {
        let mut metrics = RunObservedMetrics::new("run-123", "native_direct", RunStatus::Succeeded);
        metrics.total_elapsed_ms = Some(2_000);
        metrics.work_units_processed = Some(100);
        metrics.extensions = RuntimeMetricsExtensions {
            python: None,
            native: Some(NativeRuntimeMetricsExtension {
                frames_decoded: Some(100),
                frames_preprocessed: Some(100),
                frames_inferred: Some(100),
                frames_encoded: Some(100),
                preprocess_avg_us: Some(1100),
                inference_frame_avg_us: Some(2200),
                inference_dispatch_avg_us: Some(2200),
                postprocess_frame_avg_us: Some(3300),
                postprocess_dispatch_avg_us: Some(3300),
                encode_avg_us: Some(4400),
                vram_current_mb: Some(512),
                vram_peak_mb: Some(1024),
            }),
        };
        metrics
    }

    #[test]
    fn test_manifest_schema_and_required_fields() {
        let caps = test_caps();
        let manifest =
            build_run_manifest_v1(&test_inputs("C:/out/final.mp4", &caps)).expect("manifest");
        let value = serde_json::to_value(&manifest).expect("serialize manifest");

        assert_eq!(manifest.schema_version, RUN_MANIFEST_SCHEMA_V1);
        assert_eq!(
            value.get("schema_version").and_then(Value::as_str),
            Some(RUN_MANIFEST_SCHEMA_V1)
        );
        for key in [
            "schema_version",
            "created_at_utc",
            "job_id",
            "input_path",
            "output_path",
        ] {
            assert!(value.get(key).is_some(), "missing key: {key}");
        }
    }

    #[test]
    fn test_artifacts_gated_off_creates_nothing() {
        let caps = test_caps();
        let root = unique_temp_dir("vf_manifest_off");
        let output_dir = root.join("renders");
        fs::create_dir_all(&output_dir).expect("create output dir");
        let output_file = output_dir.join("out.mp4");
        let output_file_str = output_file.to_string_lossy().to_string();
        let inputs = test_inputs(&output_file_str, &caps);

        let written = maybe_write_run_manifest(false, &inputs).expect("helper returns");
        assert!(written.is_none());
        assert!(!output_dir.join(".videoforge_runs").exists());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_job_id_deterministic() {
        let caps = WorkerCapsSnapshot::default();
        let a = compute_job_id(
            r"C:\in.mp4",
            r"C:\out.mp4",
            4,
            "fp32",
            Some("RCAN_x4"),
            Some(&caps),
        );
        let b = compute_job_id(
            r"C:\in.mp4",
            r"C:\out.mp4",
            4,
            "fp32",
            Some("RCAN_x4"),
            Some(&caps),
        );
        let c = compute_job_id(
            r"C:\in.mp4",
            r"C:\out.mp4",
            2,
            "fp32",
            Some("RCAN_x4"),
            Some(&caps),
        );

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn finalize_run_artifacts_writes_runscope_ingest_bundle() {
        let root = unique_temp_dir("vf_runscope_bundle");
        fs::create_dir_all(&root).expect("create root");
        let snapshot = test_runtime_snapshot();
        let metrics = test_observed_metrics();

        let written = maybe_finalize_run_artifacts(
            Some(&root),
            &RunArtifactFinalizeInputs {
                runtime_snapshot: &snapshot,
                observed_metrics: Some(&metrics),
            },
        )
        .expect("finalize artifacts")
        .expect("bundle path");

        assert!(written.ends_with(RUNSCOPE_REPORT_FILENAME_V1));
        assert!(root.join(RUNTIME_CONFIG_SNAPSHOT_FILENAME_V1).exists());
        assert!(root.join(RUN_OBSERVED_METRICS_FILENAME_V1).exists());
        assert!(root.join(RUNSCOPE_REPORT_FILENAME_V1).exists());

        let report: Value = serde_json::from_str(
            &fs::read_to_string(root.join(RUNSCOPE_REPORT_FILENAME_V1)).expect("read report"),
        )
        .expect("parse report");
        assert_eq!(report["producer"], "videoforge");
        assert_eq!(report["run_id"], "run-123");
        assert_eq!(report["backend"], "native_direct");
        assert_eq!(report["engine"], "native");
        assert_eq!(report["metrics"]["fps"], 50.0);
        assert_eq!(report["runtime_snapshot"]["route_id"], "native_direct");
        assert_eq!(report["observed_metrics"]["status"], "succeeded");

        let _ = fs::remove_dir_all(&root);
    }
}
