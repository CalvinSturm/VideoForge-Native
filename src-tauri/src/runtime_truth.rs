//! Shared runtime-truth contracts.
//!
//! These types define the runtime-owned schemas that later PRs will use for:
//! - run-start resolved config snapshots
//! - observed execution metrics
//! - stable serialization across logs, tools, and persisted artifacts
//!
//! Contract rules:
//! - `RuntimeConfigSnapshot` contains resolved execution truth only
//! - `RunObservedMetrics` contains observed execution facts only
//! - shared fields exist only when they mean the same thing across routes
//! - route-specific truth lives in optional namespaced extension blocks
//! - missing metrics are omitted, never encoded as zero-as-unknown
//! - only approved ambiguous fields carry `FieldOrigin`
//!
//! Schema evolution rules:
//! - additive fields are allowed within a schema version
//! - renames, removals, or semantic reinterpretation require a version bump
//! - tool output, logs, and persisted artifacts must serialize these same structs

use serde::{Deserialize, Serialize};

use crate::run_manifest::WorkerCapsSnapshot;

pub const RUNTIME_CONFIG_SNAPSHOT_SCHEMA_V1: &str = "videoforge.runtime_config_snapshot.v1";
pub const RUN_OBSERVED_METRICS_SCHEMA_V1: &str = "videoforge.run_observed_metrics.v1";

/// Tracks where a resolved value came from when user intent can drift from
/// runtime-effective truth.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FieldOrigin {
    UserRequested,
    Defaulted,
    EnvironmentDerived,
    Inferred,
    FallbackAdjusted,
    RuntimeForced,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeSnapshotKind {
    RunStart,
    RouteFallback,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeEngineFamily {
    Python,
    Native,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMediaKind {
    Image,
    Video,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

/// The only fields that carry origin metadata in v1. Keeping this bounded
/// avoids turning the snapshot into a noisy mirror of every resolved field.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct RuntimeConfigFieldOrigins {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executor: Option<FieldOrigin>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<FieldOrigin>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision: Option<FieldOrigin>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<FieldOrigin>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_policy: Option<FieldOrigin>,
}

impl RuntimeConfigFieldOrigins {
    pub fn is_empty(&self) -> bool {
        self.executor.is_none()
            && self.batch_size.is_none()
            && self.precision.is_none()
            && self.scale.is_none()
            && self.cache_policy.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeFallbackInfo {
    pub from_route_id: String,
    pub to_route_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct RuntimeConfigExtensions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub python: Option<PythonRuntimeConfigExtension>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native: Option<NativeRuntimeConfigExtension>,
}

impl RuntimeConfigExtensions {
    pub fn is_empty(&self) -> bool {
        self.python.is_none() && self.native.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct RuntimeMetricsExtensions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub python: Option<PythonRuntimeMetricsExtension>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native: Option<NativeRuntimeMetricsExtension>,
}

impl RuntimeMetricsExtensions {
    pub fn is_empty(&self) -> bool {
        self.python.is_none() && self.native.is_none()
    }
}

/// Canonical runtime-owned resolved config at a run-state boundary.
///
/// This schema must not include observed timings, throughput, or post-run
/// counters. Those belong in `RunObservedMetrics`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeConfigSnapshot {
    pub schema_version: String,
    pub run_id: String,
    pub snapshot_kind: RuntimeSnapshotKind,
    pub route_id: String,
    pub engine_family: RuntimeEngineFamily,
    pub input_path: String,
    pub output_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_kind: Option<RuntimeMediaKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_executor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executed_executor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
    #[serde(default, skip_serializing_if = "RuntimeConfigFieldOrigins::is_empty")]
    pub field_origins: RuntimeConfigFieldOrigins,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<RuntimeFallbackInfo>,
    #[serde(default, skip_serializing_if = "RuntimeConfigExtensions::is_empty")]
    pub extensions: RuntimeConfigExtensions,
}

impl RuntimeConfigSnapshot {
    pub fn new(
        run_id: impl Into<String>,
        route_id: impl Into<String>,
        engine_family: RuntimeEngineFamily,
        input_path: impl Into<String>,
        output_path: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: RUNTIME_CONFIG_SNAPSHOT_SCHEMA_V1.to_string(),
            run_id: run_id.into(),
            snapshot_kind: RuntimeSnapshotKind::RunStart,
            route_id: route_id.into(),
            engine_family,
            input_path: input_path.into(),
            output_path: output_path.into(),
            media_kind: None,
            requested_executor: None,
            executed_executor: None,
            model_key: None,
            model_path: None,
            model_format: None,
            scale: None,
            precision: None,
            field_origins: RuntimeConfigFieldOrigins::default(),
            fallback: None,
            extensions: RuntimeConfigExtensions::default(),
        }
    }
}

/// Canonical runtime-owned observed execution facts.
///
/// This schema must not include desired config, resolution rationale, or any
/// values that were only planned rather than observed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunObservedMetrics {
    pub schema_version: String,
    pub run_id: String,
    pub route_id: String,
    pub status: RunStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_elapsed_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub work_units_processed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(default, skip_serializing_if = "RuntimeMetricsExtensions::is_empty")]
    pub extensions: RuntimeMetricsExtensions,
}

impl RunObservedMetrics {
    pub fn new(run_id: impl Into<String>, route_id: impl Into<String>, status: RunStatus) -> Self {
        Self {
            schema_version: RUN_OBSERVED_METRICS_SCHEMA_V1.to_string(),
            run_id: run_id.into(),
            route_id: route_id.into(),
            status,
            total_elapsed_ms: None,
            work_units_processed: None,
            error_code: None,
            error_message: None,
            extensions: RuntimeMetricsExtensions::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PythonRuntimeConfigExtension {
    pub python_bin: String,
    pub script_path: String,
    pub zenoh_timeout_secs: u64,
    pub enable_run_artifacts: bool,
    pub use_shm_proto_v2: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shm_ring_size_override: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved_shm_ring_size: Option<u32>,
    pub worker_caps: WorkerCapsSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipc_protocol_version: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NativeRuntimeConfigExtension {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_output_path: Option<String>,
    pub native_runtime_enabled: bool,
    pub native_direct_enabled: bool,
    pub preserve_audio: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_batch: Option<u32>,
    pub trt_cache_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trt_cache_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PythonRuntimeMetricsExtension {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_decoded: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_processed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_encoded: Option<u64>,
}

impl PythonRuntimeMetricsExtension {
    pub fn is_empty(&self) -> bool {
        self.frames_decoded.is_none()
            && self.frames_processed.is_none()
            && self.frames_encoded.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NativeRuntimeMetricsExtension {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_decoded: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_preprocessed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_inferred: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_encoded: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preprocess_avg_us: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_frame_avg_us: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_dispatch_avg_us: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocess_frame_avg_us: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocess_dispatch_avg_us: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encode_avg_us: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_current_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_peak_mb: Option<u64>,
}

impl NativeRuntimeMetricsExtension {
    pub fn is_empty(&self) -> bool {
        self.frames_decoded.is_none()
            && self.frames_preprocessed.is_none()
            && self.frames_inferred.is_none()
            && self.frames_encoded.is_none()
            && self.preprocess_avg_us.is_none()
            && self.inference_frame_avg_us.is_none()
            && self.inference_dispatch_avg_us.is_none()
            && self.postprocess_frame_avg_us.is_none()
            && self.postprocess_dispatch_avg_us.is_none()
            && self.encode_avg_us.is_none()
            && self.vram_current_mb.is_none()
            && self.vram_peak_mb.is_none()
    }
}

pub fn log_runtime_config_snapshot(snapshot: &RuntimeConfigSnapshot) {
    match serde_json::to_string(snapshot) {
        Ok(json) => tracing::info!(
            run_id = %snapshot.run_id,
            route_id = %snapshot.route_id,
            runtime_snapshot = %json,
            "Runtime config snapshot"
        ),
        Err(err) => tracing::warn!(
            run_id = %snapshot.run_id,
            error = %err,
            "Failed to serialize runtime config snapshot"
        ),
    }
}

pub fn log_run_observed_metrics(metrics: &RunObservedMetrics) {
    match serde_json::to_string(metrics) {
        Ok(json) => tracing::info!(
            run_id = %metrics.run_id,
            route_id = %metrics.route_id,
            run_observed_metrics = %json,
            "Run observed metrics"
        ),
        Err(err) => tracing::warn!(
            run_id = %metrics.run_id,
            error = %err,
            "Failed to serialize run observed metrics"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn runtime_config_snapshot_serializes_required_fields_and_omits_empty_optional_blocks() {
        let snapshot = RuntimeConfigSnapshot::new(
            "run-123",
            "python_sidecar",
            RuntimeEngineFamily::Python,
            "in.mp4",
            "out.mp4",
        );

        let value = serde_json::to_value(&snapshot).expect("serialize runtime config snapshot");
        assert_eq!(
            value.get("schema_version").and_then(Value::as_str),
            Some(RUNTIME_CONFIG_SNAPSHOT_SCHEMA_V1)
        );
        assert_eq!(value.get("run_id").and_then(Value::as_str), Some("run-123"));
        assert_eq!(
            value.get("snapshot_kind").and_then(Value::as_str),
            Some("run_start")
        );
        assert_eq!(
            value.get("route_id").and_then(Value::as_str),
            Some("python_sidecar")
        );
        assert_eq!(
            value.get("engine_family").and_then(Value::as_str),
            Some("python")
        );
        assert_eq!(
            value.get("input_path").and_then(Value::as_str),
            Some("in.mp4")
        );
        assert_eq!(
            value.get("output_path").and_then(Value::as_str),
            Some("out.mp4")
        );
        assert!(value.get("field_origins").is_none());
        assert!(value.get("fallback").is_none());
        assert!(value.get("extensions").is_none());
        assert!(value.get("total_elapsed_ms").is_none());
    }

    #[test]
    fn runtime_config_snapshot_serializes_field_origins_and_namespaced_extensions() {
        let mut snapshot = RuntimeConfigSnapshot::new(
            "run-456",
            "native_direct",
            RuntimeEngineFamily::Native,
            "clip.mp4",
            "clip_x4.mp4",
        );
        snapshot.media_kind = Some(RuntimeMediaKind::Video);
        snapshot.scale = Some(4);
        snapshot.precision = Some("fp16".to_string());
        snapshot.field_origins.executor = Some(FieldOrigin::EnvironmentDerived);
        snapshot.field_origins.batch_size = Some(FieldOrigin::RuntimeForced);
        snapshot.extensions.native = Some(NativeRuntimeConfigExtension {
            requested_output_path: Some("requested.mp4".to_string()),
            native_runtime_enabled: true,
            native_direct_enabled: true,
            preserve_audio: true,
            max_batch: Some(4),
            trt_cache_enabled: true,
            trt_cache_dir: Some("artifacts/trt".to_string()),
        });

        let value = serde_json::to_value(&snapshot).expect("serialize config snapshot");
        assert_eq!(
            value["field_origins"]["executor"],
            Value::String("environment_derived".to_string())
        );
        assert_eq!(
            value["field_origins"]["batch_size"],
            Value::String("runtime_forced".to_string())
        );
        assert!(value.get("fallback").is_none());
        assert!(value["extensions"].get("python").is_none());
        assert_eq!(value["extensions"]["native"]["max_batch"], 4);
        assert_eq!(
            value["extensions"]["native"]["requested_output_path"],
            "requested.mp4"
        );
    }

    #[test]
    fn run_observed_metrics_serializes_required_fields_and_omits_missing_metrics() {
        let metrics = RunObservedMetrics::new("run-123", "python_sidecar", RunStatus::Running);
        let value = serde_json::to_value(&metrics).expect("serialize observed metrics");

        assert_eq!(
            value.get("schema_version").and_then(Value::as_str),
            Some(RUN_OBSERVED_METRICS_SCHEMA_V1)
        );
        assert_eq!(value.get("run_id").and_then(Value::as_str), Some("run-123"));
        assert_eq!(
            value.get("route_id").and_then(Value::as_str),
            Some("python_sidecar")
        );
        assert_eq!(value.get("status").and_then(Value::as_str), Some("running"));
        assert!(value.get("total_elapsed_ms").is_none());
        assert!(value.get("work_units_processed").is_none());
        assert!(value.get("extensions").is_none());
        assert!(value.get("input_path").is_none());
    }

    #[test]
    fn run_observed_metrics_serializes_namespaced_metric_extensions_without_zero_fill() {
        let mut metrics = RunObservedMetrics::new("run-789", "native_direct", RunStatus::Succeeded);
        metrics.total_elapsed_ms = Some(1400);
        metrics.work_units_processed = Some(240);
        metrics.extensions.native = Some(NativeRuntimeMetricsExtension {
            frames_decoded: Some(240),
            frames_preprocessed: Some(240),
            frames_inferred: Some(240),
            frames_encoded: Some(240),
            preprocess_avg_us: None,
            inference_frame_avg_us: Some(5000),
            inference_dispatch_avg_us: None,
            postprocess_frame_avg_us: None,
            postprocess_dispatch_avg_us: None,
            encode_avg_us: Some(1400),
            vram_current_mb: None,
            vram_peak_mb: Some(4096),
        });

        let value = serde_json::to_value(&metrics).expect("serialize observed metrics");
        assert_eq!(value["total_elapsed_ms"], 1400);
        assert_eq!(value["work_units_processed"], 240);
        assert_eq!(value["extensions"]["native"]["frames_inferred"], 240);
        assert!(value["extensions"]["native"]
            .get("preprocess_avg_us")
            .is_none());
        assert!(value["extensions"]["native"]
            .get("vram_current_mb")
            .is_none());
        assert!(value.get("precision").is_none());
    }

    #[test]
    fn field_origin_serializes_to_stable_snake_case_values() {
        let value =
            serde_json::to_value(FieldOrigin::FallbackAdjusted).expect("serialize field origin");
        assert_eq!(value, Value::String("fallback_adjusted".to_string()));
    }
}
