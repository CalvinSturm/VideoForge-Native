//! Zenoh Control Channel for VideoForge Research Pipeline
//!
//! Listens on `vf/control/**` for real-time parameter updates from the UI.
//! Publishes status/diagnostics back to `vf/control/status`.
//!
//! JSON protocol:
//!
//! **vf/control/params** (UI → Engine)
//! ```json
//! { "alpha_structure": 0.5, "hf_method": "sobel", ... }
//! ```
//!
//! **vf/control/blend_control** (UI → Engine)
//! ```json
//! { "primary": "structure", "secondary": "texture", "alpha": 0.6, "hallucination_view": false }
//! ```
//!
//! **vf/control/model_enable** (UI → Engine)
//! ```json
//! { "role": "texture", "enabled": true }
//! ```
//!
//! **vf/control/status** (Engine → UI)
//! ```json
//! { "status": "params_updated", "data": { ... } }
//! ```

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::sync::Mutex;

// =============================================================================
// RESEARCH CONFIG (mirrors Python BlendParameters)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchConfig {
    // Global model weights
    pub alpha_structure: f64,
    pub alpha_texture: f64,
    pub alpha_perceptual: f64,
    pub alpha_diffusion: f64,

    // Frequency band weights
    pub low_freq_strength: f64,
    pub mid_freq_strength: f64,
    pub high_freq_strength: f64,

    // Hallucination controls
    pub h_sensitivity: f64,
    pub h_blend_reduction: f64,

    // Spatial routing
    pub edge_model_bias: f64,
    pub texture_model_bias: f64,
    pub flat_region_suppression: f64,

    // Analysis method
    pub hf_method: String,

    // Performance preset
    pub preset: String,

    // Frequency band sigmas
    pub freq_low_sigma: f64,
    pub freq_mid_sigma: f64,

    // Spatial thresholds
    pub edge_threshold: f64,
    pub texture_threshold: f64,

    // Pipeline mix
    pub spatial_freq_mix: f64,

    // SR Pipeline — Detail Enhancement
    pub adr_enabled: bool,
    pub detail_strength: f64,

    // SR Pipeline — Blending
    pub luma_only: bool,
    pub edge_strength: f64,
    pub sharpen_strength: f64,

    // SR Pipeline — Temporal
    pub temporal_enabled: bool,
    pub temporal_alpha: f64,

    // SR Pipeline — Secondary model
    pub secondary_model: String,

    // SR Pipeline — Advanced
    pub return_gpu_tensor: bool,

    // Transient — cleared after being sent to Python
    #[serde(default)]
    pub reset_temporal: bool,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            alpha_structure: 0.5,
            alpha_texture: 0.3,
            alpha_perceptual: 0.15,
            alpha_diffusion: 0.05,
            low_freq_strength: 1.0,
            mid_freq_strength: 1.0,
            high_freq_strength: 1.0,
            h_sensitivity: 1.0,
            h_blend_reduction: 0.5,
            edge_model_bias: 0.7,
            texture_model_bias: 0.7,
            flat_region_suppression: 0.3,
            hf_method: "laplacian".to_string(),
            preset: "balanced".to_string(),
            freq_low_sigma: 4.0,
            freq_mid_sigma: 1.5,
            edge_threshold: 0.5,
            texture_threshold: 0.2,
            spatial_freq_mix: 0.5,

            adr_enabled: false,
            detail_strength: 0.5,
            luma_only: true,
            edge_strength: 0.3,
            sharpen_strength: 0.0,
            temporal_enabled: true,
            temporal_alpha: 0.9,
            secondary_model: "None".to_string(),
            return_gpu_tensor: true,
            reset_temporal: false,
        }
    }
}

// =============================================================================
// BLEND CONTROL MESSAGE (from UI)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendControlMessage {
    pub primary: String,
    pub secondary: String,
    pub alpha: f64,
    pub hallucination_view: bool,
}

// =============================================================================
// MODEL TOGGLE MESSAGE (from UI)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelToggleMessage {
    pub role: String,
    pub enabled: bool,
}

// =============================================================================
// CONTROL CHANNEL STATE
// =============================================================================

pub struct ControlChannel {
    config: Arc<Mutex<ResearchConfig>>,
    zenoh_prefix: String,
}

impl ControlChannel {
    pub fn new(prefix: &str) -> Self {
        Self {
            config: Arc::new(Mutex::new(ResearchConfig::default())),
            zenoh_prefix: prefix.to_string(),
        }
    }

    pub fn config(&self) -> Arc<Mutex<ResearchConfig>> {
        Arc::clone(&self.config)
    }

    /// Start the Zenoh control listener. Spawns an async task that listens
    /// on `{prefix}/**` and forwards parameter updates to the Python worker
    /// via the existing Zenoh IPC publisher.
    pub async fn start(
        &self,
        session: &zenoh::Session,
        python_publisher: &zenoh::pubsub::Publisher<'_>,
        app: Option<AppHandle>,
    ) -> anyhow::Result<()> {
        let sub_key = format!("{}/**", self.zenoh_prefix);
        let subscriber = session
            .declare_subscriber(&sub_key)
            .await
            .map_err(|e| anyhow::anyhow!("Zenoh subscribe failed: {}", e))?;

        let config = Arc::clone(&self.config);
        let prefix = self.zenoh_prefix.clone();

        // Status publisher for ack messages back to UI
        // Pass owned String so the publisher doesn't borrow a local — it gets moved into tokio::spawn
        let status_pub = session
            .declare_publisher(format!("{}/status", prefix))
            .await
            .map_err(|e| anyhow::anyhow!("Zenoh status pub failed: {}", e))?;

        println!("Control: Listening on {}", sub_key);

        // Process incoming messages
        tokio::spawn(async move {
            loop {
                match subscriber.recv_async().await {
                    Ok(sample) => {
                        let topic = sample.key_expr().as_str().to_string();
                        let payload = match sample.payload().to_bytes().to_vec().as_slice() {
                            bytes => match String::from_utf8(bytes.to_vec()) {
                                Ok(s) => s,
                                Err(_) => continue,
                            },
                        };

                        // Route by topic suffix
                        if topic.ends_with("/params") {
                            Self::handle_params_update(&config, &payload, &status_pub).await;
                            // Emit to UI for reactivity
                            if let Some(ref app) = app {
                                let guard = config.lock().await;
                                let _ = app.emit("research-config-updated",
                                    serde_json::to_value(&*guard).unwrap_or_default());
                            }
                        } else if topic.ends_with("/blend_control") {
                            Self::handle_blend_control(&config, &payload, &status_pub).await;
                        } else if topic.ends_with("/model_enable") {
                            Self::handle_model_toggle(&payload, &status_pub).await;
                        } else if topic.ends_with("/get_params") {
                            Self::handle_get_params(&config, &status_pub).await;
                        }
                    }
                    Err(e) => {
                        eprintln!("Control: Zenoh recv error: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    async fn handle_params_update(
        config: &Arc<Mutex<ResearchConfig>>,
        payload: &str,
        status_pub: &zenoh::pubsub::Publisher<'_>,
    ) {
        match serde_json::from_str::<serde_json::Value>(payload) {
            Ok(updates) => {
                let mut guard = config.lock().await;

                // Apply each field if present
                if let Some(v) = updates.get("alpha_structure").and_then(|v| v.as_f64()) {
                    guard.alpha_structure = v;
                }
                if let Some(v) = updates.get("alpha_texture").and_then(|v| v.as_f64()) {
                    guard.alpha_texture = v;
                }
                if let Some(v) = updates.get("alpha_perceptual").and_then(|v| v.as_f64()) {
                    guard.alpha_perceptual = v;
                }
                if let Some(v) = updates.get("alpha_diffusion").and_then(|v| v.as_f64()) {
                    guard.alpha_diffusion = v;
                }
                if let Some(v) = updates.get("low_freq_strength").and_then(|v| v.as_f64()) {
                    guard.low_freq_strength = v;
                }
                if let Some(v) = updates.get("mid_freq_strength").and_then(|v| v.as_f64()) {
                    guard.mid_freq_strength = v;
                }
                if let Some(v) = updates.get("high_freq_strength").and_then(|v| v.as_f64()) {
                    guard.high_freq_strength = v;
                }
                if let Some(v) = updates.get("h_sensitivity").and_then(|v| v.as_f64()) {
                    guard.h_sensitivity = v;
                }
                if let Some(v) = updates.get("h_blend_reduction").and_then(|v| v.as_f64()) {
                    guard.h_blend_reduction = v;
                }
                if let Some(v) = updates.get("edge_model_bias").and_then(|v| v.as_f64()) {
                    guard.edge_model_bias = v;
                }
                if let Some(v) = updates.get("texture_model_bias").and_then(|v| v.as_f64()) {
                    guard.texture_model_bias = v;
                }
                if let Some(v) = updates.get("flat_region_suppression").and_then(|v| v.as_f64()) {
                    guard.flat_region_suppression = v;
                }
                if let Some(v) = updates.get("hf_method").and_then(|v| v.as_str()) {
                    guard.hf_method = v.to_string();
                }
                if let Some(v) = updates.get("preset").and_then(|v| v.as_str()) {
                    guard.preset = v.to_string();
                }
                if let Some(v) = updates.get("freq_low_sigma").and_then(|v| v.as_f64()) {
                    guard.freq_low_sigma = v;
                }
                if let Some(v) = updates.get("freq_mid_sigma").and_then(|v| v.as_f64()) {
                    guard.freq_mid_sigma = v;
                }
                if let Some(v) = updates.get("edge_threshold").and_then(|v| v.as_f64()) {
                    guard.edge_threshold = v;
                }
                if let Some(v) = updates.get("texture_threshold").and_then(|v| v.as_f64()) {
                    guard.texture_threshold = v;
                }
                if let Some(v) = updates.get("spatial_freq_mix").and_then(|v| v.as_f64()) {
                    guard.spatial_freq_mix = v;
                }
                // SR Pipeline params
                if let Some(v) = updates.get("adr_enabled").and_then(|v| v.as_bool()) {
                    guard.adr_enabled = v;
                }
                if let Some(v) = updates.get("detail_strength").and_then(|v| v.as_f64()) {
                    guard.detail_strength = v;
                }
                if let Some(v) = updates.get("luma_only").and_then(|v| v.as_bool()) {
                    guard.luma_only = v;
                }
                if let Some(v) = updates.get("edge_strength").and_then(|v| v.as_f64()) {
                    guard.edge_strength = v;
                }
                if let Some(v) = updates.get("sharpen_strength").and_then(|v| v.as_f64()) {
                    guard.sharpen_strength = v;
                }
                if let Some(v) = updates.get("temporal_enabled").and_then(|v| v.as_bool()) {
                    guard.temporal_enabled = v;
                }
                if let Some(v) = updates.get("temporal_alpha").and_then(|v| v.as_f64()) {
                    guard.temporal_alpha = v;
                }
                if let Some(v) = updates.get("secondary_model").and_then(|v| v.as_str()) {
                    guard.secondary_model = v.to_string();
                }
                if let Some(v) = updates.get("return_gpu_tensor").and_then(|v| v.as_bool()) {
                    guard.return_gpu_tensor = v;
                }

                let ack = serde_json::json!({
                    "status": "params_updated",
                    "data": serde_json::to_value(&*guard).unwrap_or_default()
                });
                let _ = status_pub.put(ack.to_string()).await;

                println!("Control: Params updated");
            }
            Err(e) => {
                eprintln!("Control: Invalid params JSON: {}", e);
            }
        }
    }

    async fn handle_blend_control(
        config: &Arc<Mutex<ResearchConfig>>,
        payload: &str,
        status_pub: &zenoh::pubsub::Publisher<'_>,
    ) {
        match serde_json::from_str::<BlendControlMessage>(payload) {
            Ok(msg) => {
                // Warn if both models are diffusion (12GB VRAM risk)
                if msg.primary == "diffusion" && msg.secondary == "diffusion" {
                    println!(
                        "Control: WARNING — Both model slots are diffusion. \
                         Estimated VRAM >12GB. High risk of OOM."
                    );
                    let warn = serde_json::json!({
                        "status": "vram_warning",
                        "data": {
                            "message": "Both model slots are diffusion. >12GB VRAM risk.",
                            "primary": msg.primary,
                            "secondary": msg.secondary
                        }
                    });
                    let _ = status_pub.put(warn.to_string()).await;
                }

                // Map blend control to config params
                let mut guard = config.lock().await;
                if msg.primary == "structure" {
                    guard.alpha_structure = msg.alpha;
                    guard.alpha_texture = 1.0 - msg.alpha;
                } else if msg.primary == "texture" {
                    guard.alpha_texture = msg.alpha;
                    guard.alpha_structure = 1.0 - msg.alpha;
                }

                // Set preset based on which models are active
                if !msg.primary.is_empty() && !msg.secondary.is_empty() {
                    guard.preset = "balanced".to_string();
                }

                let ack = serde_json::json!({
                    "status": "blend_control_applied",
                    "data": msg
                });
                let _ = status_pub.put(ack.to_string()).await;

                println!(
                    "Control: Blend updated — primary={}, secondary={}, alpha={:.2}",
                    msg.primary, msg.secondary, msg.alpha
                );
            }
            Err(e) => {
                eprintln!("Control: Invalid blend_control JSON: {}", e);
            }
        }
    }

    async fn handle_model_toggle(
        payload: &str,
        status_pub: &zenoh::pubsub::Publisher<'_>,
    ) {
        match serde_json::from_str::<ModelToggleMessage>(payload) {
            Ok(msg) => {
                let valid_roles = ["structure", "texture", "perceptual", "diffusion"];
                if !valid_roles.contains(&msg.role.as_str()) {
                    eprintln!("Control: Unknown model role: {}", msg.role);
                    return;
                }

                let ack = serde_json::json!({
                    "status": "model_toggled",
                    "data": msg
                });
                let _ = status_pub.put(ack.to_string()).await;

                println!(
                    "Control: Model {} {}",
                    msg.role,
                    if msg.enabled { "enabled" } else { "disabled" }
                );
            }
            Err(e) => {
                eprintln!("Control: Invalid model_enable JSON: {}", e);
            }
        }
    }

    async fn handle_get_params(
        config: &Arc<Mutex<ResearchConfig>>,
        status_pub: &zenoh::pubsub::Publisher<'_>,
    ) {
        let guard = config.lock().await;
        let ack = serde_json::json!({
            "status": "current_params",
            "data": serde_json::to_value(&*guard).unwrap_or_default()
        });
        let _ = status_pub.put(ack.to_string()).await;
    }
}

// =============================================================================
// TAURI COMMAND: Get/Set Research Config
// =============================================================================

/// Tauri command to get current research config from the Rust side.
/// The UI calls this to initialize slider values.
#[tauri::command]
pub async fn get_research_config(
    state: tauri::State<'_, Arc<Mutex<ResearchConfig>>>,
) -> Result<ResearchConfig, String> {
    let guard = state.lock().await;
    Ok(guard.clone())
}

/// Tauri command to update research config from the UI.
/// The UI calls this when sliders change. The config is stored in Rust state
/// and forwarded to Python via Zenoh on the next frame.
#[tauri::command]
pub async fn set_research_config(
    app: AppHandle,
    state: tauri::State<'_, Arc<Mutex<ResearchConfig>>>,
    config: ResearchConfig,
) -> Result<(), String> {
    let mut guard = state.lock().await;
    *guard = config;
    let snapshot = serde_json::to_value(&*guard).unwrap_or_default();
    drop(guard);
    let _ = app.emit("research-params-changed", snapshot);
    Ok(())
}

/// Tauri command to update a single research parameter.
#[tauri::command]
pub async fn update_research_param(
    app: AppHandle,
    state: tauri::State<'_, Arc<Mutex<ResearchConfig>>>,
    key: String,
    value: serde_json::Value,
) -> Result<(), String> {
    let mut guard = state.lock().await;

    match key.as_str() {
        "alpha_structure" => guard.alpha_structure = value.as_f64().ok_or("Invalid f64")?,
        "alpha_texture" => guard.alpha_texture = value.as_f64().ok_or("Invalid f64")?,
        "alpha_perceptual" => guard.alpha_perceptual = value.as_f64().ok_or("Invalid f64")?,
        "alpha_diffusion" => guard.alpha_diffusion = value.as_f64().ok_or("Invalid f64")?,
        "low_freq_strength" => guard.low_freq_strength = value.as_f64().ok_or("Invalid f64")?,
        "mid_freq_strength" => guard.mid_freq_strength = value.as_f64().ok_or("Invalid f64")?,
        "high_freq_strength" => guard.high_freq_strength = value.as_f64().ok_or("Invalid f64")?,
        "h_sensitivity" => guard.h_sensitivity = value.as_f64().ok_or("Invalid f64")?,
        "h_blend_reduction" => guard.h_blend_reduction = value.as_f64().ok_or("Invalid f64")?,
        "edge_model_bias" => guard.edge_model_bias = value.as_f64().ok_or("Invalid f64")?,
        "texture_model_bias" => guard.texture_model_bias = value.as_f64().ok_or("Invalid f64")?,
        "flat_region_suppression" => guard.flat_region_suppression = value.as_f64().ok_or("Invalid f64")?,
        "hf_method" => guard.hf_method = value.as_str().ok_or("Invalid string")?.to_string(),
        "preset" => guard.preset = value.as_str().ok_or("Invalid string")?.to_string(),
        "freq_low_sigma" => guard.freq_low_sigma = value.as_f64().ok_or("Invalid f64")?,
        "freq_mid_sigma" => guard.freq_mid_sigma = value.as_f64().ok_or("Invalid f64")?,
        "edge_threshold" => guard.edge_threshold = value.as_f64().ok_or("Invalid f64")?,
        "texture_threshold" => guard.texture_threshold = value.as_f64().ok_or("Invalid f64")?,
        "spatial_freq_mix" => guard.spatial_freq_mix = value.as_f64().ok_or("Invalid f64")?,
        // SR Pipeline params
        "adr_enabled" => guard.adr_enabled = value.as_bool().ok_or("Invalid bool")?,
        "detail_strength" => guard.detail_strength = value.as_f64().ok_or("Invalid f64")?,
        "luma_only" => guard.luma_only = value.as_bool().ok_or("Invalid bool")?,
        "edge_strength" => guard.edge_strength = value.as_f64().ok_or("Invalid f64")?,
        "sharpen_strength" => guard.sharpen_strength = value.as_f64().ok_or("Invalid f64")?,
        "temporal_enabled" => guard.temporal_enabled = value.as_bool().ok_or("Invalid bool")?,
        "temporal_alpha" => guard.temporal_alpha = value.as_f64().ok_or("Invalid f64")?,
        "secondary_model" => guard.secondary_model = value.as_str().ok_or("Invalid string")?.to_string(),
        "return_gpu_tensor" => guard.return_gpu_tensor = value.as_bool().ok_or("Invalid bool")?,
        _ => return Err(format!("Unknown parameter: {}", key)),
    }

    let snapshot = serde_json::to_value(&*guard).unwrap_or_default();
    drop(guard);
    let _ = app.emit("research-params-changed", snapshot);

    Ok(())
}

/// Tauri command to reset temporal EMA buffers.
/// Called from the UI "Reset Temporal Buffer" button.
/// Sets a transient flag on ResearchConfig that Python reads on the next frame,
/// then Python clears its temporal buffers and the flag auto-resets.
#[tauri::command]
pub async fn reset_temporal_buffer(
    state: tauri::State<'_, Arc<Mutex<ResearchConfig>>>,
) -> Result<(), String> {
    let mut guard = state.lock().await;
    guard.reset_temporal = true;
    println!("Control: Temporal buffer reset flag set");
    Ok(())
}
