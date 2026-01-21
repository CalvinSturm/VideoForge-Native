use anyhow::Result;
use tokio::sync::watch;
use tracing::info;
use std::path::Path;
use tauri::{AppHandle, Emitter}; // For sending progress to React

mod models;
mod video;
mod inference;

// This is the function React calls via invoke("upscale_request")
#[tauri::command]
async fn upscale_request(
    app: AppHandle,
    input_path: String,
    output_path: String,
    model: String,
    scale: u32,
) -> Result<(), String> {
    info!("Upscale request received for: {}", input_path);

    // 1. Initialize Python Bridge
    let python_path = r"C:\Users\Calvin\videoForge2\venv\Scripts\python.exe";
    let python_script = r"C:\Users\Calvin\videoForge2\python\realesrgan_worker.py";
    
    let mut bridge = inference::InferenceBridge::spawn(python_path, python_script)
        .await
        .map_err(|e| e.to_string())?;

    // 2. Setup channels
    let (cancel_tx, cancel_rx) = watch::channel(false);
    
    let mut decoder = video::VideoDecoder::new(&input_path, 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    
    let mut encoder = video::VideoEncoder::new(
        &output_path,
        30,
        "prores_ks",
        "yuva444p10le",
    ).map_err(|e| e.to_string())?;

    let mut idx = 0;
    // You'd need a way to know total frames for progress; let's assume 100 for now
    while let Some(frame) = decoder.read_frame().await.map_err(|e| e.to_string())? {
        // Check for cancellation
        if *cancel_rx.borrow() { break; }

        let upscaled = video::upscale_frame(
            &mut bridge.stdin,
            &mut bridge.stdout,
            frame,
            &format!("frame-{idx}"),
            cancel_rx.clone(),
        ).await.map_err(|e| e.to_string())?;

        encoder.write_frame(&upscaled).await.map_err(|e| e.to_string())?;
        
        // 3. SEND PROGRESS TO REACT
        // This is what makes your progress bars move!
        app.emit("upscale-progress", serde_json::json!({
            "jobId": "current-job", // You can pass this from JS
            "progress": idx, // Should be (idx / total_frames) * 100
            "message": format!("Processing frame {}", idx)
        })).unwrap();

        idx += 1;
    }

    encoder.finish().await.map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    tauri::Builder::default()
        // Register the plugins your twin mentioned
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![upscale_request]) // Link the command
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}