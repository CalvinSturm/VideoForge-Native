//! VideoForge Tauri backend — module wiring and application entry point.
//!
//! This file only contains:
//! - Module declarations
//! - The system monitor background task
//! - The `run()` entry point that wires Tauri
//!
//! All business logic lives in the sub-modules listed below.

use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::sync::Mutex;
use tokio::time::Duration;

// --- MODULES ---
pub mod commands;
pub mod control;
pub mod edit_config;
pub mod ipc;
pub mod models;
pub mod python_env;
pub mod rave_cli;
pub mod runtime_truth;
pub mod run_manifest;
pub mod shm;
pub mod spatial_map;
pub mod spatial_publisher;
mod utils;
mod video_pipeline;
pub mod win_events;

// --- SYSTEM MONITOR (background thread) ---

fn spawn_system_monitor(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        let mut sys = sysinfo::System::new_all();
        loop {
            sys.refresh_cpu();
            sys.refresh_memory();

            let cpu_usage = sys.global_cpu_info().cpu_usage();
            let ram_used = sys.used_memory();
            let ram_total = sys.total_memory();

            let _ = app.emit(
                "system-stats",
                serde_json::json!({
                    "cpu": cpu_usage,
                    "ramUsed": ram_used,
                    "ramTotal": ram_total,
                    "gpuName": "NVIDIA CUDA:0"
                }),
            );

            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });
}

// --- ENTRY POINT ---

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialise structured logging.  Uses RUST_LOG env var for filter control.
    // Falls back silently if another subscriber is already registered.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("videoforge=info")),
        )
        .try_init();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(Arc::new(Mutex::new(control::ResearchConfig::default())))
        .manage(Arc::new(spatial_map::SpatialMapState::new()))
        .setup(|app| {
            spawn_system_monitor(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // AI upscaling
            commands::upscale::upscale_request,
            // Native engine (always registered; returns FEATURE_DISABLED if not compiled in)
            commands::native_engine::upscale_request_native,
            // RAVE pipeline integration
            commands::rave::rave_validate,
            commands::rave::rave_upscale,
            commands::rave::rave_benchmark,
            // Transcode-only export
            commands::export::export_request,
            // Engine management
            commands::engine::check_engine_status,
            commands::engine::install_engine,
            commands::engine::reset_engine,
            commands::engine::get_models,
            // OS helpers
            commands::engine::show_in_folder,
            commands::engine::open_media,
            // Research parameter control
            control::get_research_config,
            control::set_research_config,
            control::update_research_param,
            control::reset_temporal_buffer,
            // Spatial map
            spatial_map::fetch_spatial_frame,
            spatial_map::mark_frame_complete
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
