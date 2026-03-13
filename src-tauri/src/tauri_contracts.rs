use serde::{Deserialize, Serialize};

use crate::edit_config::EditConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpscaleRequest {
    pub input_path: String,
    pub output_path: String,
    pub model: String,
    pub edit_config: EditConfig,
    pub scale: u32,
    pub precision: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRequest {
    pub input_path: String,
    pub output_path: String,
    pub edit_config: EditConfig,
    pub scale: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeUpscaleRequest {
    pub input_path: String,
    pub output_path: String,
    pub model_path: String,
    pub scale: u32,
    pub precision: Option<String>,
    pub audio: Option<bool>,
    pub max_batch: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaveValidateRequest {
    pub fixture: Option<String>,
    pub profile: Option<String>,
    pub best_effort: Option<bool>,
    pub strict_audit: Option<bool>,
    pub mock_run: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaveBenchmarkRequest {
    pub args: Vec<String>,
    pub strict_audit: Option<bool>,
    pub mock_run: Option<bool>,
    pub ui_opt_in: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpscaleProgressEventPayload {
    #[serde(rename = "jobId")]
    pub job_id: String,
    pub progress: u32,
    pub message: String,
    #[serde(rename = "outputPath", skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
    pub eta: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_decoded: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_processed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_encoded: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_ms: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatsPayload {
    pub cpu: f32,
    #[serde(rename = "ramUsed")]
    pub ram_used: u64,
    #[serde(rename = "ramTotal")]
    pub ram_total: u64,
    #[serde(rename = "gpuName")]
    pub gpu_name: String,
}
