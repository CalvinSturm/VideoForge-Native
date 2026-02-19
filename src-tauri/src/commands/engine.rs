//! Engine management commands — install, status, reset, model listing,
//! and OS-level file/folder helpers.

use std::fs::{self, File};
use std::io::{Cursor, Write};
use std::path::Path;

use futures_util::StreamExt;
use tauri::{AppHandle, Emitter};

use crate::models::ModelInfo;
use crate::python_env::{get_python_install_dir, resolve_python_environment, PYTHON_PIDS};

const ENGINE_URL: &str =
    "https://github.com/YourRepo/releases/download/v1.0/engine.zip";

// ─── check_engine_status ─────────────────────────────────────────────────────

#[tauri::command]
pub async fn check_engine_status() -> bool {
    resolve_python_environment().is_ok()
}

// ─── install_engine ──────────────────────────────────────────────────────────

#[tauri::command]
pub async fn install_engine(app: AppHandle) -> Result<(), String> {
    let install_dir = get_python_install_dir();
    let parent_dir = install_dir.parent().unwrap();

    if !parent_dir.exists() {
        fs::create_dir_all(parent_dir).map_err(|e| e.to_string())?;
    }

    tracing::info!(url = ENGINE_URL, "Downloading engine");
    let response = reqwest::get(ENGINE_URL).await.map_err(|e| e.to_string())?;
    let total_size = response.content_length().unwrap_or(0);

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut buffer = Vec::new();

    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| e.to_string())?;
        downloaded += chunk.len() as u64;
        buffer.extend_from_slice(&chunk);

        if total_size > 0 {
            let pct = (downloaded as f64 / total_size as f64) * 100.0;
            app.emit("install-progress", pct).unwrap();
        }
    }

    tracing::info!(dest = %parent_dir.display(), "Extracting engine archive");
    let reader = Cursor::new(buffer);
    let mut archive = zip::ZipArchive::new(reader).map_err(|e| e.to_string())?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
        let outpath = match file.enclosed_name() {
            Some(path) => parent_dir.join(path),
            None => continue,
        };

        if (*file.name()).ends_with('/') {
            fs::create_dir_all(&outpath).map_err(|e| e.to_string())?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p).map_err(|e| e.to_string())?;
                }
            }
            let mut outfile = File::create(&outpath).map_err(|e| e.to_string())?;
            std::io::copy(&mut file, &mut outfile).map_err(|e| e.to_string())?;
        }
    }

    tracing::info!("Engine installation complete");
    Ok(())
}

// ─── get_models ──────────────────────────────────────────────────────────────

#[tauri::command]
pub fn get_models() -> Result<Vec<ModelInfo>, String> {
    Ok(crate::models::list_models())
}

// ─── reset_engine ────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn reset_engine() -> Result<(), String> {
    tracing::warn!("Panic button: forcing engine shutdown");

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        use std::process::Command as StdCommand;

        let pids: Vec<u32> = {
            let guard = PYTHON_PIDS.lock().unwrap();
            guard.iter().copied().collect()
        };

        for pid in pids {
            tracing::info!(pid, "Sending taskkill to tracked Python process");
            let _ = StdCommand::new("taskkill")
                .args(["/F", "/PID", &pid.to_string()])
                .creation_flags(0x08000000)
                .output();
        }

        PYTHON_PIDS.lock().unwrap().clear();
    }

    Ok(())
}

// ─── show_in_folder ──────────────────────────────────────────────────────────

#[tauri::command]
pub async fn show_in_folder(path: String) -> Result<(), String> {
    let file_path = Path::new(&path);
    if !file_path.exists() {
        return Err(format!("Path does not exist: {}", path));
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        use std::process::Command as StdCommand;
        StdCommand::new("explorer")
            .arg(format!("/select,{}", path))
            .creation_flags(0x08000000)
            .spawn()
            .map_err(|e| format!("Failed to open explorer: {}", e))?;
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Some(parent) = file_path.parent() {
            opener::open(parent).map_err(|e| format!("Failed to open folder: {}", e))?;
        }
    }

    Ok(())
}

// ─── open_media ──────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn open_media(path: String) -> Result<(), String> {
    if !Path::new(&path).exists() {
        return Err(format!("File does not exist: {}", path));
    }
    opener::open(&path).map_err(|e| format!("Failed to open file: {}", e))
}
