use anyhow::{anyhow, Context, Result};
use dirs::data_local_dir;
use futures_util::StreamExt;
use std::env;
use std::fs::{self, File};
use std::io::{Cursor, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use sysinfo::System;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::time::{timeout, Duration, Instant};
use zenoh::Config;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

// --- MODULES ---
mod edit_config;
mod models;
mod shm;
mod video_pipeline;

use crate::edit_config::{build_ffmpeg_filters, calculate_output_dimensions, EditConfig};
use crate::models::ModelInfo;

// --- CONSTANTS ---
const ENGINE_URL: &str = "https://github.com/YourRepo/releases/download/v1.0/engine.zip";

// --- RAII GUARD FOR PYTHON PROCESS ---
struct ProcessGuard {
    child: Option<Child>,
}
impl ProcessGuard {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }
    fn disarm(&mut self) -> Option<Child> {
        self.child.take()
    }
}
impl Drop for ProcessGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            println!("Guard: Killing Python process (Cleanup)...");
            let _ = child.start_kill();
        }
    }
}

// --- HELPER FUNCTIONS ---

fn get_python_install_dir() -> PathBuf {
    let mut path = data_local_dir().expect("Could not find AppData");
    path.push("VideoForge");
    path.push("python");
    path
}

fn resolve_python_environment() -> Result<(String, String)> {
    let install_dir = get_python_install_dir();
    let installed_python = install_dir.join("python.exe");
    let installed_script = install_dir.join("shm_worker.py");

    if installed_python.exists() && installed_script.exists() {
        return Ok((
            installed_python.to_string_lossy().to_string(),
            installed_script.to_string_lossy().to_string(),
        ));
    }

    // Dev Fallback
    let local_venv = Path::new(r"C:\Users\Calvin\VideoForge\venv310\Scripts\python.exe");
    if local_venv.exists() {
        if let Ok(cwd) = env::current_dir() {
            let script_local = cwd.join("python").join("shm_worker.py");
            if script_local.exists() {
                return Ok((
                    local_venv.to_string_lossy().to_string(),
                    script_local.to_string_lossy().to_string(),
                ));
            }
            let script_up = cwd
                .parent()
                .unwrap_or(Path::new(".."))
                .join("python")
                .join("shm_worker.py");
            if script_up.exists() {
                return Ok((
                    local_venv.to_string_lossy().to_string(),
                    script_up.to_string_lossy().to_string(),
                ));
            }
        }
    }

    Err(anyhow!("AI Engine not found. Please run the installer."))
}

fn get_smart_output_path(input: &str, is_video: bool) -> String {
    let input_path = Path::new(input);
    let parent = input_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let extension = if is_video { "mp4" } else { "png" };
    parent
        .join(format!("{}_{}_upscaled.{}", stem, timestamp, extension))
        .to_string_lossy()
        .to_string()
}

fn is_image_file(path: &str) -> bool {
    if let Some(ext) = Path::new(path).extension() {
        let e = ext.to_string_lossy().to_lowercase();
        return matches!(
            e.as_str(),
            "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tif" | "tiff"
        );
    }
    false
}

fn get_free_port() -> u16 {
    if let Ok(listener) = TcpListener::bind("127.0.0.1:0") {
        if let Ok(addr) = listener.local_addr() {
            return addr.port();
        }
    }
    7447
}

// Simple parser for "frame= 123" lines
fn parse_frame_from_line(line: &str) -> Option<u64> {
    if let Some(pos) = line.find("frame=") {
        let remainder = &line[pos + 6..];
        let num_str = remainder.split_whitespace().next()?;
        return num_str.parse::<u64>().ok();
    }
    None
}

// --- SYSTEM MONITOR (Background Thread) ---
fn spawn_system_monitor(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        let mut sys = System::new_all();
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

// -----------------------------------------------------------------------------
// COMMANDS
// -----------------------------------------------------------------------------

#[tauri::command]
async fn check_engine_status() -> bool {
    resolve_python_environment().is_ok()
}

#[tauri::command]
async fn install_engine(app: AppHandle) -> Result<(), String> {
    let install_dir = get_python_install_dir();
    let parent_dir = install_dir.parent().unwrap();

    if !parent_dir.exists() {
        fs::create_dir_all(parent_dir).map_err(|e| e.to_string())?;
    }

    println!("Downloading Engine from: {}", ENGINE_URL);
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

    println!("Extracting to {:?}", parent_dir);
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

    println!("Installation Complete!");
    Ok(())
}

#[tauri::command]
fn get_models() -> Result<Vec<ModelInfo>, String> {
    Ok(models::list_models())
}

#[tauri::command]
async fn reset_engine() -> Result<(), String> {
    println!("Panic Button: Forcing Engine Shutdown...");
    #[cfg(target_os = "windows")]
    {
        let _ = StdCommand::new("taskkill")
            .args(["/F", "/IM", "python.exe"])
            .creation_flags(0x08000000)
            .output();
    }
    Ok(())
}

#[tauri::command]
async fn show_in_folder(path: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        StdCommand::new("explorer")
            .args(["/select,", &path])
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
async fn open_media(path: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        StdCommand::new("cmd")
            .args(["/C", "start", "", &path])
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// EXPORT REQUEST (Transcode Only)
// -----------------------------------------------------------------------------
#[tauri::command]
async fn export_request(
    app: AppHandle,
    input_path: String,
    mut output_path: String,
    edit_config: EditConfig,
    _scale: u32,
) -> Result<String, String> {
    let is_img = is_image_file(&input_path);

    // Auto-generate output path if empty
    if output_path.trim().is_empty() {
        output_path = get_smart_output_path(&input_path, !is_img);
    }

    let probe_res = video_pipeline::probe_video(&input_path).map_err(|e| e.to_string())?;
    let (width, height, duration, fps, _total_frames) = probe_res;

    let filter_str = build_ffmpeg_filters(&edit_config, width, height);

    // --- IMAGE EXPORT PATH ---
    if is_img {
        let mut args = vec![
            "-y".to_string(),
            "-hide_banner".to_string(),
            "-loglevel".to_string(),
            "warning".to_string(),
            "-i".to_string(),
            input_path,
        ];

        if !filter_str.is_empty() {
            args.push("-vf".to_string());
            args.push(filter_str);
        }

        // Ensure output path ends with .png for image export
        if !output_path.to_lowercase().ends_with(".png")
            && !output_path.to_lowercase().ends_with(".jpg")
            && !output_path.to_lowercase().ends_with(".jpeg")
        {
            output_path = output_path.replace(".mp4", ".png");
            if !output_path.ends_with(".png") {
                output_path.push_str(".png");
            }
        }

        args.push(output_path.clone());

        let output = Command::new("ffmpeg")
            .args(&args)
            .output()
            .await
            .map_err(|e| format!("Failed to spawn ffmpeg: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Image export failed: {}", stderr));
        }

        app.emit(
            "upscale-progress",
            serde_json::json!({
                "jobId": "export",
                "progress": 100,
                "message": "Export Complete",
                "outputPath": output_path,
                "eta": 0
            }),
        )
        .unwrap();

        return Ok(output_path);
    }

    // --- VIDEO EXPORT PATH ---
    let start_time = edit_config.trim_start;
    let end_time = if edit_config.trim_end > 0.0 {
        edit_config.trim_end
    } else {
        duration
    };
    let export_duration = (end_time - start_time).max(0.1);

    // Calculate expected frames based on TARGET FPS if interpolation is active
    let target_fps_val = if edit_config.fps > 0 {
        edit_config.fps as f64
    } else {
        fps
    };
    let expected_frames = (export_duration * target_fps_val).round() as u64;

    let mut video_filter_str = filter_str;

    let target_fps = edit_config.fps;
    let source_fps = fps as u32;
    if target_fps > 0 && target_fps != source_fps {
        let interp = format!(
            "minterpolate=fps={}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
            target_fps
        );
        video_filter_str = if video_filter_str.is_empty() {
            interp
        } else {
            format!("{},{}", video_filter_str, interp)
        };
    }

    let mut args = vec![
        "-y".to_string(),
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "info".to_string(),
    ];

    if start_time > 0.0 {
        args.push("-ss".to_string());
        args.push(format!("{:.3}", start_time));
    }
    if export_duration > 0.0 {
        args.push("-t".to_string());
        args.push(format!("{:.3}", export_duration));
    }

    args.push("-i".to_string());
    args.push(input_path);

    if !video_filter_str.is_empty() {
        args.push("-vf".to_string());
        args.push(video_filter_str);
    }

    args.extend_from_slice(&[
        "-c:v".to_string(),
        "h264_nvenc".to_string(),
        "-preset".to_string(),
        "p7".to_string(),
        "-cq".to_string(),
        "18".to_string(),
        "-movflags".to_string(),
        "+faststart".to_string(),
        output_path.clone(),
    ]);

    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn ffmpeg: {}", e))?;

    let stderr = child.stderr.take().ok_or("Failed to open stderr")?;
    let mut reader = BufReader::new(stderr).lines();

    let process_start_time = Instant::now();

    while let Ok(Some(line)) = reader.next_line().await {
        if line.contains("frame=") {
            if let Some(current_frame) = parse_frame_from_line(&line) {
                let elapsed = process_start_time.elapsed().as_secs_f64();
                let rate = if elapsed > 0.0 {
                    current_frame as f64 / elapsed
                } else {
                    0.0
                };
                let remaining_frames = expected_frames.saturating_sub(current_frame);
                let eta = if rate > 0.0 {
                    (remaining_frames as f64 / rate) as u64
                } else {
                    0
                };

                let pct =
                    ((current_frame as f64 / expected_frames as f64) * 100.0).min(99.0) as u32;

                let _ = app.emit(
                    "upscale-progress",
                    serde_json::json!({
                        "jobId": "export",
                        "progress": pct,
                        "message": format!("Processing Frame {}/{}", current_frame, expected_frames),
                        "eta": eta,
                        "outputPath": null
                    }),
                );
            }
        }
    }

    let status = child
        .wait()
        .await
        .map_err(|e| format!("Wait failed: {}", e))?;

    if !status.success() {
        return Err(format!("Export failed with code {:?}", status.code()));
    }

    app.emit(
        "upscale-progress",
        serde_json::json!({
            "jobId": "export",
            "progress": 100,
            "message": "Export Complete",
            "outputPath": output_path,
            "eta": 0
        }),
    )
    .unwrap();

    Ok(output_path)
}

// -----------------------------------------------------------------------------
// UPSCALE REQUEST (AI + Edits)
// -----------------------------------------------------------------------------
#[tauri::command]
async fn upscale_request(
    app: AppHandle,
    input_path: String,
    mut output_path: String,
    model: String,
    edit_config: EditConfig,
    scale: u32,
) -> Result<String, String> {
    if !Path::new(&input_path).exists() {
        return Err(format!("Input file not found: {}", input_path));
    }

    let is_img = is_image_file(&input_path);
    if output_path.trim().is_empty() {
        output_path = get_smart_output_path(&input_path, !is_img);
    }

    let port = get_free_port();
    let ipc_endpoint = format!("tcp/127.0.0.1:{}", port);
    let zenoh_prefix = format!("videoforge/ipc/{}", port);

    let mut config = Config::default();
    config
        .insert_json5("listen/endpoints", &format!("[\"{}\"]", ipc_endpoint))
        .map_err(|e| e.to_string())?;
    let session = zenoh::open(config)
        .await
        .map_err(|e: zenoh::Error| e.to_string())?;

    let publisher = session
        .declare_publisher(format!("{}/req", zenoh_prefix))
        .await
        .map_err(|e: zenoh::Error| e.to_string())?;
    let subscriber = session
        .declare_subscriber(format!("{}/res", zenoh_prefix))
        .await
        .map_err(|e: zenoh::Error| e.to_string())?;

    let (python_bin, script_path) = resolve_python_environment().map_err(|e| e.to_string())?;
    println!("Spawning Worker: {} -> {}", python_bin, script_path);

    let mut cmd = Command::new(python_bin);
    cmd.arg(script_path);
    cmd.arg("--port");
    cmd.arg(port.to_string());
    cmd.arg("--parent-pid");
    cmd.arg(std::process::id().to_string());
    cmd.stdout(Stdio::null()); // Prevent blocking on pipe buffer
    cmd.stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000);

    let python_child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn python: {}", e))?;
    let mut python_guard = ProcessGuard::new(python_child);

    if timeout(Duration::from_secs(60), subscriber.recv_async())
        .await
        .is_err()
    {
        return Err("Python Handshake Timeout".to_string());
    }

    let load_req =
        serde_json::json!({ "command": "load_model", "params": { "model_name": model } });
    publisher
        .put(load_req.to_string())
        .await
        .map_err(|e: zenoh::Error| e.to_string())?;

    let load_msg = timeout(Duration::from_secs(30), subscriber.recv_async())
        .await
        .map_err(|_| "Model Load Timeout")?
        .map_err(|e: zenoh::Error| e.to_string())?;
    let load_data =
        String::from_utf8(load_msg.payload().to_bytes().to_vec()).map_err(|e| e.to_string())?;

    if !load_data.contains("MODEL_LOADED") {
        return Err(format!("Failed to load model: {}", load_data));
    }

    // --- IMAGE PIPELINE ---
    if is_img {
        let req_payload = serde_json::json!({
            "command": "upscale_image_file",
            "id": "single_shot",
            "params": {
                "input_path": input_path,
                "output_path": output_path,
                "config": edit_config
            }
        });
        publisher
            .put(req_payload.to_string())
            .await
            .map_err(|e: zenoh::Error| e.to_string())?;

        // PROGRESS LOOP FOR IMAGES
        loop {
            let msg = timeout(Duration::from_secs(300), subscriber.recv_async())
                .await
                .map_err(|_| "Image Upscale Timeout")?
                .map_err(|e: zenoh::Error| e.to_string())?;

            let payload_str =
                String::from_utf8(msg.payload().to_bytes().to_vec()).map_err(|e| e.to_string())?;
            let resp: serde_json::Value =
                serde_json::from_str(&payload_str).map_err(|e| e.to_string())?;

            if resp["status"] == "progress" {
                let current = resp["current"].as_u64().unwrap_or(0);
                let total = resp["total"].as_u64().unwrap_or(1);
                let pct = (current as f64 / total as f64) * 100.0;

                let _ = app.emit(
                    "upscale-progress",
                    serde_json::json!({
                        "jobId": "active", // UI uses this ID for active task
                        "progress": pct as u32,
                        "message": format!("Processing Tile {}/{}", current, total)
                    }),
                );
            } else if resp["status"] == "ok" {
                break; // Done
            } else if resp["status"] == "error" {
                return Err(format!("Python Error: {}", resp["message"]));
            }
        }

        let _ = publisher
            .put(serde_json::json!({ "command": "shutdown" }).to_string())
            .await;
        if let Some(mut child) = python_guard.disarm() {
            let _ = child.wait().await;
        }

        return Ok(output_path);
    }

    // --- VIDEO PIPELINE ---
    let probe_res = video_pipeline::probe_video(&input_path).map_err(|e| e.to_string())?;
    let (input_w, input_h, duration, fps, _total_frames) = probe_res;

    let (process_w, process_h) = calculate_output_dimensions(&edit_config, input_w, input_h);
    let start_time = edit_config.trim_start;
    let end_time = if edit_config.trim_end > 0.0 {
        edit_config.trim_end
    } else {
        duration
    };
    let process_duration = (end_time - start_time).max(0.1);
    let process_frames = (process_duration * fps).round() as u64;
    let scale_factor = scale as usize;

    let create_req = serde_json::json!({
        "command": "create_shm", "width": process_w, "height": process_h,
        "scale": scale_factor, "ring_size": shm::RING_SIZE
    });
    publisher
        .put(create_req.to_string())
        .await
        .map_err(|e: zenoh::Error| e.to_string())?;

    let shm_msg = timeout(Duration::from_secs(10), subscriber.recv_async())
        .await
        .map_err(|_| "SHM Creation Timeout")?
        .map_err(|e: zenoh::Error| e.to_string())?;
    let shm_data =
        String::from_utf8(shm_msg.payload().to_bytes().to_vec()).map_err(|e| e.to_string())?;
    let shm_resp: serde_json::Value = serde_json::from_str(&shm_data).map_err(|e| e.to_string())?;

    if shm_resp["status"] != "SHM_CREATED" {
        return Err(format!("SHM Init Failed: {}", shm_resp["message"]));
    }
    let shm_path = shm_resp
        .get("shm_path")
        .and_then(|s| s.as_str())
        .unwrap()
        .to_string();

    let shm = shm::VideoShm::open(&shm_path, process_w, process_h, scale_factor)
        .map_err(|e| e.to_string())?;
    let shared_shm = Mutex::new(shm);

    let (free_tx, mut free_rx) = mpsc::channel::<usize>(shm::RING_SIZE);
    let (ai_tx, mut ai_rx) = mpsc::channel::<usize>(shm::RING_SIZE);
    let (enc_tx, mut enc_rx) = mpsc::channel::<usize>(shm::RING_SIZE);

    for i in 0..shm::RING_SIZE {
        free_tx.send(i).await.map_err(|e| e.to_string())?;
    }

    let filters = build_ffmpeg_filters(&edit_config, input_w, input_h);

    let decoder_task = async {
        let mut decoder =
            video_pipeline::VideoDecoder::new(&input_path, start_time, process_duration, &filters)
                .await
                .map_err(|e| e.to_string())?;
        while let Some(slot_idx) = free_rx.recv().await {
            let mut shm_guard = shared_shm.lock().await;
            if decoder
                .read_raw_frame_into(shm_guard.input_slot_mut(slot_idx))
                .await
                .unwrap_or(false)
            {
                drop(shm_guard);
                if ai_tx.send(slot_idx).await.is_err() {
                    break;
                }
            } else {
                break;
            }
        }
        Ok::<(), String>(())
    };

    let ai_task = async {
        while let Some(slot_idx) = ai_rx.recv().await {
            let req = serde_json::json!({ "command": "process_frame", "slot": slot_idx });
            if publisher.put(req.to_string()).await.is_err() {
                break;
            }
            match timeout(Duration::from_secs(30), subscriber.recv_async()).await {
                Ok(Ok(msg)) => {
                    let s =
                        String::from_utf8(msg.payload().to_bytes().to_vec()).unwrap_or_default();
                    if s.contains("FRAME_DONE") {
                        if enc_tx.send(slot_idx).await.is_err() {
                            break;
                        }
                    } else {
                        return Err(format!("AI Error: {}", s));
                    }
                }
                _ => return Err("AI Timeout".to_string()),
            }
        }
        Ok(())
    };

    let target_fps = edit_config.fps;
    let encoder_task = async {
        let mut encoder = video_pipeline::VideoEncoder::new(
            &output_path,
            fps as u32,
            target_fps,
            process_w * scale_factor,
            process_h * scale_factor,
        )
        .await
        .map_err(|e| e.to_string())?;
        let mut processed_count = 0u64;
        let eta_start = Instant::now();

        while let Some(slot_idx) = enc_rx.recv().await {
            let shm_guard = shared_shm.lock().await;
            if let Err(e) = encoder
                .write_raw_frame(shm_guard.output_slot(slot_idx))
                .await
            {
                eprintln!("Encoder Write Error: {}", e);
                break;
            }
            drop(shm_guard);
            let _ = free_tx.send(slot_idx).await;

            processed_count += 1;

            if processed_count % 5 == 0 || processed_count == process_frames {
                let pct =
                    ((processed_count as f64 / process_frames as f64) * 100.0).min(100.0) as u32;

                let elapsed = eta_start.elapsed().as_secs_f64();
                let fps_proc = processed_count as f64 / elapsed;
                let eta = if fps_proc > 0.0 {
                    (process_frames.saturating_sub(processed_count) as f64 / fps_proc) as u64
                } else {
                    0
                };

                let _ = app.emit("upscale-progress", serde_json::json!({
                    "jobId": "active", "progress": pct,
                    "message": format!("Processing Frame {}/{}", processed_count, process_frames),
                    "eta": eta
                }));
            }
            if processed_count >= process_frames {
                break;
            }
        }
        encoder.finish().await.map_err(|e| e.to_string())?;
        let _ = app.emit("upscale-progress", serde_json::json!({
            "jobId": "active", "progress": 100, "message": "Finalizing...", "outputPath": output_path, "eta": 0
        }));
        Ok(())
    };

    if let Err(e) = tokio::try_join!(decoder_task, ai_task, encoder_task) {
        return Err(e);
    }

    let _ = publisher
        .put(serde_json::json!({ "command": "shutdown" }).to_string())
        .await;
    // Targeted Cleanup
    if let Some(mut child) = python_guard.disarm() {
        // Try graceful shutdown first
        if timeout(Duration::from_secs(3), child.wait()).await.is_err() {
            println!("Engine did not exit in time. Forcing kill...");
            let _ = child.start_kill(); // Tokio's kill
            
            // Double-tap with Windows taskkill by PID to be sure
            #[cfg(target_os = "windows")]
            if let Some(pid) = child.id() {
                 let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/PID", &pid.to_string()])
                    .creation_flags(0x08000000)
                    .output();
            }
        }
    }

    Ok(output_path)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            spawn_system_monitor(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            upscale_request,
            export_request,
            show_in_folder,
            open_media,
            reset_engine,
            get_models,
            check_engine_status,
            install_engine
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
