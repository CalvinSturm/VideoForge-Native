//! Export command — FFmpeg transcode (no AI upscaling).

use std::process::Stdio;
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::process::Command;
use tokio::process::Command as TokioCommand;
use tokio::time::Instant;

use crate::edit_config::{build_ffmpeg_filters, EditConfig};
use crate::tauri_contracts::ExportRequest;
use crate::video_pipeline;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Parse the current frame number from an FFmpeg progress line like
/// `frame=  123 fps= 30 ...`.
fn parse_frame_from_line(line: &str) -> Option<u64> {
    if let Some(pos) = line.find("frame=") {
        let remainder = &line[pos + 6..];
        let num_str = remainder.split_whitespace().next()?;
        return num_str.parse::<u64>().ok();
    }
    None
}

pub fn get_smart_output_path(input: &str, is_video: bool) -> String {
    use std::path::Path;
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

pub fn is_image_file(path: &str) -> bool {
    use std::path::Path;
    if let Some(ext) = Path::new(path).extension() {
        let e = ext.to_string_lossy().to_lowercase();
        return matches!(
            e.as_str(),
            "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tif" | "tiff"
        );
    }
    false
}

// ─── export_request ──────────────────────────────────────────────────────────

#[tauri::command]
pub async fn export_request(
    app: AppHandle,
    input_path: String,
    output_path: String,
    edit_config: EditConfig,
    _scale: u32,
) -> Result<String, String> {
    let request = ExportRequest {
        input_path,
        output_path,
        edit_config,
        scale: _scale,
    };
    let is_img = is_image_file(&request.input_path);
    let mut output_path = request.output_path;

    if output_path.trim().is_empty() {
        output_path = get_smart_output_path(&request.input_path, !is_img);
    }

    tracing::info!(
        input = %request.input_path,
        output = %output_path,
        is_image = is_img,
        "Export request"
    );

    let probe_res = video_pipeline::probe_video(&request.input_path).map_err(|e| e.to_string())?;
    let (width, height, duration, fps, _total_frames) = probe_res;

    let filter_str = build_ffmpeg_filters(&request.edit_config, width, height);

    // ── Image export ─────────────────────────────────────────────────────────
    if is_img {
        let mut args = vec![
            "-y".to_string(),
            "-hide_banner".to_string(),
            "-loglevel".to_string(),
            "warning".to_string(),
            "-i".to_string(),
            request.input_path.clone(),
        ];

        if !filter_str.is_empty() {
            args.push("-vf".to_string());
            args.push(filter_str);
        }

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

        let output = TokioCommand::new("ffmpeg")
            .args(&args)
            .output()
            .await
            .map_err(|e| format!("Failed to spawn ffmpeg: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!(stderr = %stderr, "Image export failed");
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

    // ── Video export ─────────────────────────────────────────────────────────
    let start_time = request.edit_config.trim_start;
    let end_time = if request.edit_config.trim_end > 0.0 {
        request.edit_config.trim_end
    } else {
        duration
    };
    let export_duration = (end_time - start_time).max(0.1);

    let target_fps_val = if request.edit_config.fps > 0 {
        request.edit_config.fps as f64
    } else {
        fps
    };
    let expected_frames = (export_duration * target_fps_val).round() as u64;

    let mut video_filter_str = filter_str;
    let target_fps = request.edit_config.fps;
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
    args.push(request.input_path.clone());

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
        "-c:a".to_string(),
        "copy".to_string(),
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
        tracing::error!(exit_code = ?status.code(), "Video export failed");
        return Err(format!("Export failed with code {:?}", status.code()));
    }

    tracing::info!(output = %output_path, "Export complete");
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
