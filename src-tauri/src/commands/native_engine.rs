//! Native engine MVP — GPU-resident upscaling via engine-v2.
//!
//! This command is gated behind the `native_engine` Cargo feature.
//! When the feature is **disabled** (the default), the command returns a
//! structured `FEATURE_DISABLED` error immediately.
//!
//! When the feature is **enabled** and runtime-gated, the pipeline is:
//! ```text
//! FFmpeg demux → Annex B elementary stream (.h264/.hevc temp file)
//!      ↓
//! engine-v2 NVDEC → TensorRT → NVENC → compressed output temp file
//!      ↓
//! FFmpeg mux: video_output + audio_from_original → final .mp4
//! ```
//!
//! Runtime routing:
//!
//! - `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1` is required before the command runs.
//! - `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` routes into the true in-process path.
//! - without the direct flag, the native command stays on the CLI-backed path.
//!
//! This file owns the app-facing runtime gating, FFmpeg file-boundary steps,
//! and the opt-in engine-v2 direct path.

use serde::{Deserialize, Serialize};
use std::path::Path;
#[cfg(feature = "native_engine")]
use std::path::PathBuf;
#[cfg(feature = "native_engine")]
use std::sync::Arc;
#[cfg(feature = "native_engine")]
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
#[cfg(feature = "native_engine")]
use std::sync::Mutex;
#[cfg(feature = "native_engine")]
use std::sync::OnceLock;

#[cfg(all(feature = "native_engine", windows))]
unsafe extern "system" {
    fn LoadLibraryW(lpLibFileName: *const u16) -> *mut std::ffi::c_void;
}

#[cfg(all(feature = "native_engine", windows))]
fn preload_windows_dll(path: &Path) {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: Calling system loader with a null-terminated UTF-16 path.
    let _ = unsafe { LoadLibraryW(wide.as_ptr()) };
}

#[cfg(feature = "native_engine")]
fn workspace_root() -> Option<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .ok()
}

#[cfg(feature = "native_engine")]
fn prepend_path_dirs(dirs: &[PathBuf]) {
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut paths: Vec<PathBuf> = std::env::split_paths(&current).collect();
    for dir in dirs.iter().filter(|d| d.is_dir()) {
        if !paths.iter().any(|p| p == dir) {
            paths.insert(0, dir.clone());
        }
    }
    if let Ok(joined) = std::env::join_paths(paths) {
        // SAFETY: process-local env mutation before worker/process launches.
        unsafe { std::env::set_var("PATH", joined) };
    }
}

#[cfg(feature = "native_engine")]
fn find_file_under(root: &Path, file_name: &str, max_depth: usize) -> Option<PathBuf> {
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        if depth > max_depth {
            continue;
        }
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some(file_name) {
                return Some(path);
            }
            if path.is_dir() {
                stack.push((path, depth + 1));
            }
        }
    }
    None
}

#[cfg(feature = "native_engine")]
struct NativeRuntimeEnv {
    ffmpeg_cmd: String,
}

#[cfg(feature = "native_engine")]
static NATIVE_RUNTIME_ENV: OnceLock<NativeRuntimeEnv> = OnceLock::new();

#[cfg(feature = "native_engine")]
fn discover_native_runtime_env() -> NativeRuntimeEnv {
    let ffmpeg_exe = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    let ffprobe_exe = if cfg!(windows) {
        "ffprobe.exe"
    } else {
        "ffprobe"
    };

    let mut ffmpeg_bin: Option<PathBuf> = None;
    let mut tensorrt_bin: Option<PathBuf> = None;

    if let Some(root) = workspace_root() {
        let known_ffmpeg = [
            root.join("third_party").join("ffmpeg").join("bin"),
            root.join("third_party").join("ffmpeg"),
        ];
        for dir in &known_ffmpeg {
            if dir.join(ffmpeg_exe).exists() && dir.join(ffprobe_exe).exists() {
                ffmpeg_bin = Some(dir.clone());
                break;
            }
        }

        if ffmpeg_bin.is_none() {
            for scan_root in [root.join("artifacts"), root.join("third_party")] {
                if !scan_root.exists() {
                    continue;
                }
                if let Some(ffmpeg_path) = find_file_under(&scan_root, ffmpeg_exe, 5) {
                    let bin_dir = ffmpeg_path.parent().map(|p| p.to_path_buf());
                    if let Some(bin_dir) = bin_dir {
                        if bin_dir.join(ffprobe_exe).exists() {
                            ffmpeg_bin = Some(bin_dir);
                            break;
                        }
                    }
                }
            }
        }

        let known_trt = root.join("third_party").join("tensorrt");
        if known_trt.join("nvinfer_10.dll").exists() {
            tensorrt_bin = Some(known_trt);
        } else if let Some(nvinfer) =
            find_file_under(&root.join("third_party"), "nvinfer_10.dll", 4)
        {
            tensorrt_bin = nvinfer.parent().map(|p| p.to_path_buf());
        }
    }

    let mut path_additions = Vec::new();
    if let Some(dir) = &ffmpeg_bin {
        path_additions.push(dir.clone());
    }
    if let Some(dir) = &tensorrt_bin {
        path_additions.push(dir.clone());

        #[cfg(windows)]
        {
            // Preload core TensorRT DLLs from absolute paths so ORT provider
            // registration does not depend on PATH search behavior.
            for dll in [
                "nvinfer_10.dll",
                "nvinfer_plugin_10.dll",
                "nvinfer_dispatch_10.dll",
                "nvonnxparser_10.dll",
                "cudnn64_9.dll",
            ] {
                let p = dir.join(dll);
                if p.exists() {
                    preload_windows_dll(&p);
                }
            }
        }

        // Stage TensorRT runtime DLLs next to the executable as a robust loader path.
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let src = entry.path();
                        let is_dll = src
                            .extension()
                            .and_then(|e| e.to_str())
                            .is_some_and(|e| e.eq_ignore_ascii_case("dll"));
                        if !is_dll {
                            continue;
                        }
                        let Some(name) = src.file_name() else {
                            continue;
                        };
                        let dst = exe_dir.join(name);
                        if !dst.exists() {
                            let _ = std::fs::copy(&src, &dst);
                        }
                    }
                }
            }
        }
    }
    prepend_path_dirs(&path_additions);

    let ffmpeg_cmd = if let Some(bin) = ffmpeg_bin {
        bin.join(ffmpeg_exe).to_string_lossy().to_string()
    } else {
        "ffmpeg".to_string()
    };

    NativeRuntimeEnv { ffmpeg_cmd }
}

#[cfg(feature = "native_engine")]
fn configure_native_runtime_env() -> String {
    NATIVE_RUNTIME_ENV
        .get_or_init(discover_native_runtime_env)
        .ffmpeg_cmd
        .clone()
}

#[cfg(feature = "native_engine")]
fn probe_video_coded_geometry(path: &str) -> Result<(usize, usize, f64, f64, u64), String> {
    let output = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,coded_width,coded_height,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ])
        .output()
        .map_err(|e| format!("ffprobe launch failed: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffprobe failed: {stderr}"));
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("ffprobe JSON parse failed: {e}"))?;
    let stream = json
        .get("streams")
        .and_then(|s| s.get(0))
        .ok_or_else(|| "No video stream found".to_string())?;

    let display_width = stream.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let display_height = stream.get("height").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let coded_width = stream
        .get("coded_width")
        .and_then(|v| v.as_u64())
        .unwrap_or(display_width as u64) as usize;
    let coded_height = stream
        .get("coded_height")
        .and_then(|v| v.as_u64())
        .unwrap_or(display_height as u64) as usize;

    let fps_str = stream
        .get("r_frame_rate")
        .and_then(|v| v.as_str())
        .unwrap_or("30/1");
    let fps = if let Some((num, den)) = fps_str.split_once('/') {
        let num = num.parse::<f64>().unwrap_or(30.0);
        let den = den.parse::<f64>().unwrap_or(1.0);
        if den == 0.0 { 30.0 } else { num / den }
    } else {
        fps_str.parse::<f64>().unwrap_or(30.0)
    };

    let duration = json
        .get("format")
        .and_then(|f| f.get("duration"))
        .and_then(|d| d.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let total_frames = (duration * fps).round() as u64;

    tracing::info!(
        display_w = display_width,
        display_h = display_height,
        coded_w = coded_width,
        coded_h = coded_height,
        "Resolved native coded geometry"
    );

    Ok((coded_width, coded_height, duration, fps, total_frames))
}

#[cfg(feature = "native_engine")]
static NATIVE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "native_engine")]
fn native_temp_token() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let pid = std::process::id();
    let seq = NATIVE_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{pid}_{ts}_{seq}")
}

#[cfg(feature = "native_engine")]
fn stderr_tail(bytes: &[u8], lines: usize) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .rev()
        .take(lines)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

/// Structured response returned by `upscale_request_native`.
#[derive(Debug, Serialize, Deserialize)]
pub struct NativeUpscaleResult {
    pub output_path: String,
    pub engine: String,
    pub encoder_mode: String,
    pub encoder_detail: Option<String>,
    pub frames_processed: u64,
    pub audio_preserved: bool,
    pub trt_cache_enabled: bool,
    pub trt_cache_dir: Option<String>,
}

/// Structured error returned by `upscale_request_native`.
#[derive(Debug, Serialize, Deserialize)]
pub struct NativeUpscaleError {
    pub code: String,
    pub message: String,
}

impl NativeUpscaleError {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
        }
    }
}

#[cfg(feature = "native_engine")]
fn native_engine_runtime_enabled() -> bool {
    match std::env::var("VIDEOFORGE_ENABLE_NATIVE_ENGINE") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

#[cfg(feature = "native_engine")]
fn native_engine_direct_enabled() -> bool {
    match std::env::var("VIDEOFORGE_NATIVE_ENGINE_DIRECT") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

#[cfg(feature = "native_engine")]
fn trt_cache_runtime(model_path: &str) -> (bool, Option<String>) {
    let enabled = std::env::var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE")
        .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "True"))
        .unwrap_or(false);
    if !enabled {
        return (false, None);
    }

    let cache_root = std::env::var_os("VIDEOFORGE_TRT_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("videoforge").join("trt_cache"));
    let model_tag = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let cache_dir = cache_root.join(model_tag);
    (true, Some(cache_dir.to_string_lossy().to_string()))
}

#[cfg(feature = "native_engine")]
fn infer_model_scale(model_path: &str) -> Option<u32> {
    let stem = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())?
        .to_ascii_lowercase();

    for scale in [8u32, 4, 3, 2] {
        let prefix = format!("{scale}x");
        if stem.starts_with(&prefix) {
            return Some(scale);
        }
    }

    None
}

#[cfg(feature = "native_engine")]
async fn run_native_via_rave_cli(
    input_path: String,
    output_path: String,
    model_path: String,
    _scale: u32,
    precision: String,
    _preserve_audio: bool,
    max_batch: u32,
    encoder_detail: Option<String>,
) -> Result<NativeUpscaleResult, String> {
    let make_err =
        |code: &str, msg: &str| serde_json::to_string(&NativeUpscaleError::new(code, msg)).unwrap();

    if !(1..=8).contains(&max_batch) {
        return Err(make_err(
            "INVALID_BATCH",
            &format!("Invalid max_batch value '{max_batch}'. Must be in range 1-8."),
        ));
    }

    let resolved_output = if output_path.trim().is_empty() {
        let p = Path::new(&input_path);
        let stem = p
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let ext = p
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if ext.is_empty() {
            format!("{stem}_rave_upscaled.mp4")
        } else {
            input_path.replace(&format!(".{ext}"), "_rave_upscaled.mp4")
        }
    } else {
        output_path
    };

    let (trt_cache_enabled, trt_cache_dir) = trt_cache_runtime(&model_path);
    let mut args = vec![
        "-i".to_string(),
        input_path,
        "-m".to_string(),
        model_path,
        "-o".to_string(),
        resolved_output.clone(),
        "--precision".to_string(),
        precision,
        "--progress".to_string(),
        "jsonl".to_string(),
    ];
    if max_batch > 1 {
        args.push("--max-batch".to_string());
        args.push(max_batch.to_string());
    }

    let res =
        crate::commands::rave::rave_upscale(args, Some(true), Some(false), Some(true)).await?;
    let output = res
        .get("output")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            make_err(
                "RAVE_CONTRACT",
                "rave_upscale did not return a valid output path",
            )
        })?
        .to_string();

    Ok(NativeUpscaleResult {
        output_path: output,
        engine: "native_via_rave_cli".to_string(),
        encoder_mode: "rave_cli".to_string(),
        encoder_detail,
        frames_processed: 0,
        audio_preserved: true,
        trt_cache_enabled,
        trt_cache_dir,
    })
}

#[cfg(feature = "native_engine")]
fn decode_native_error(err_json: &str) -> Option<NativeUpscaleError> {
    serde_json::from_str(err_json).ok()
}

#[cfg(feature = "native_engine")]
fn should_fallback_to_rave_cli(err: &NativeUpscaleError) -> bool {
    matches!(err.code.as_str(), "ENCODER_INIT" | "PIPELINE")
        && (err.message.contains("NVENC")
            || err.message.contains("nvEnc")
            || err.message.contains("Software fallback"))
}

// =============================================================================
// Command (always present — returns FEATURE_DISABLED when feature is off)
// =============================================================================

/// Run the native engine-v2 GPU upscaling pipeline.
///
/// # Parameters
/// - `input_path`:   Source video file (MP4, MKV, …).
/// - `output_path`:  Destination MP4.  Auto-generated if empty.
/// - `model_path`:   Path to a TensorRT-compatible ONNX model file.
/// - `scale`:        Upscale factor (e.g. 4 for 4×).
/// - `precision`:    `"fp32"` | `"fp16"` (default `"fp32"`).
/// - `audio`:        Whether to preserve audio from input (default `true`).
///
/// # Returns
/// `Ok(NativeUpscaleResult)` on success, `Err(NativeUpscaleError)` on failure.
#[tauri::command]
pub async fn upscale_request_native(
    input_path: String,
    output_path: String,
    model_path: String,
    scale: u32,
    precision: Option<String>,
    audio: Option<bool>,
    max_batch: Option<u32>,
) -> Result<NativeUpscaleResult, String> {
    // Validate inputs regardless of feature flag.
    if !Path::new(&input_path).exists() {
        return Err(serde_json::to_string(&NativeUpscaleError::new(
            "INPUT_NOT_FOUND",
            format!("Input file not found: {}", input_path),
        ))
        .unwrap());
    }

    #[cfg(not(feature = "native_engine"))]
    {
        let _ = (
            &output_path,
            &model_path,
            scale,
            &precision,
            &audio,
            &max_batch,
        );
        return Err(serde_json::to_string(&NativeUpscaleError::new(
            "FEATURE_DISABLED",
            "The native_engine feature is not compiled in. \
             Rebuild with `cargo build --features native_engine` or use the \
             Python pipeline (upscale_request).",
        ))
        .unwrap());
    }

    #[cfg(feature = "native_engine")]
    {
        if !native_engine_runtime_enabled() {
            return Err(serde_json::to_string(&NativeUpscaleError::new(
                "NATIVE_ENGINE_DISABLED",
                "Native engine is disabled by default for stability. \
                 Set VIDEOFORGE_ENABLE_NATIVE_ENGINE=1 to opt in explicitly.",
            ))
            .unwrap());
        }

        let precision = precision.unwrap_or_else(|| "fp32".to_string());
        let audio = audio.unwrap_or(true);
        let max_batch = max_batch.unwrap_or(1);
        let effective_scale = infer_model_scale(&model_path).unwrap_or(scale);
        if effective_scale != scale {
            tracing::warn!(
                requested_scale = scale,
                inferred_scale = effective_scale,
                model = %model_path,
                "Requested native scale does not match model filename; overriding to inferred model scale"
            );
        }

        if native_engine_direct_enabled() {
            let direct_result = run_native_pipeline(
                input_path.clone(),
                output_path.clone(),
                model_path.clone(),
                effective_scale,
                precision.clone(),
                audio,
                max_batch,
            )
            .await;

            match direct_result {
                Ok(result) => Ok(result),
                Err(err_json) => {
                    let Some(err) = decode_native_error(&err_json) else {
                        return Err(err_json);
                    };
                    if should_fallback_to_rave_cli(&err) {
                        tracing::warn!(
                            code = %err.code,
                            message = %err.message,
                            "Direct native path failed; falling back to CLI-backed native path"
                        );
                        run_native_via_rave_cli(
                            input_path,
                            output_path,
                            model_path,
                            effective_scale,
                            precision,
                            audio,
                            max_batch,
                            Some(format!("direct_native_failed: {}: {}", err.code, err.message)),
                        )
                        .await
                    } else {
                        Err(err_json)
                    }
                }
            }
        } else {
            run_native_via_rave_cli(
                input_path,
                output_path,
                model_path,
                effective_scale,
                precision,
                audio,
                max_batch,
                None,
            )
            .await
        }
    }
}

// =============================================================================
// Native pipeline implementation (compiled only with native_engine feature)
// =============================================================================

#[cfg(feature = "native_engine")]
async fn run_native_pipeline(
    input_path: String,
    mut output_path: String,
    model_path: String,
    scale: u32,
    precision: String,
    preserve_audio: bool,
    max_batch: u32,
) -> Result<NativeUpscaleResult, String> {
    use std::process::Stdio;
    use tokio::process::Command;

    use videoforge_engine::codecs::sys::cudaVideoCodec as CudaCodec;

    let make_err =
        |code: &str, msg: &str| serde_json::to_string(&NativeUpscaleError::new(code, msg)).unwrap();
    let ffmpeg_cmd = configure_native_runtime_env();

    if !(1..=8).contains(&max_batch) {
        return Err(make_err(
            "INVALID_BATCH",
            &format!("Invalid max_batch value '{max_batch}'. Must be in range 1-8."),
        ));
    }

    // Validate model path.
    if !Path::new(&model_path).exists() {
        return Err(make_err(
            "MODEL_NOT_FOUND",
            &format!("Model not found: {}", model_path),
        ));
    }

    // Generate output path.
    if output_path.trim().is_empty() {
        let stem = Path::new(&input_path)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let dir = Path::new(&input_path)
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();
        let file_name = format!("{}_{}_native_upscaled.mp4", stem, native_temp_token());
        output_path = dir.join(file_name).to_string_lossy().to_string();
    }

    tracing::info!(
        input = %input_path,
        output = %output_path,
        model = %model_path,
        scale,
        precision = %precision,
        max_batch,
        "Native engine pipeline starting"
    );

    // ── Step 1: FFmpeg extract elementary stream ──────────────────────────────
    let tmp_dir = std::env::temp_dir();
    let temp_token = native_temp_token();
    let elementary_stream_path = tmp_dir.join(format!("vf_native_input_{temp_token}.h264"));
    let native_output_path_h264 = tmp_dir.join(format!("vf_native_output_{temp_token}.h264"));

    tracing::info!(
        stream_path = %elementary_stream_path.display(),
        "Extracting H.264 elementary stream with FFmpeg"
    );

    let demux_output = Command::new(&ffmpeg_cmd)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            &input_path,
            "-vcodec",
            "copy",
            "-an", // no audio in elementary stream
            "-bsf:v",
            "h264_mp4toannexb", // convert to Annex B
            elementary_stream_path.to_str().unwrap(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| make_err("FFMPEG_SPAWN", &format!("FFmpeg spawn failed: {}", e)))?;

    if !demux_output.status.success() {
        // Attempt HEVC fallback
        let hevc_path = tmp_dir.join(format!("vf_native_input_{temp_token}.hevc"));
        let native_output_path_hevc = tmp_dir.join(format!("vf_native_output_{temp_token}.hevc"));
        let hevc_output = Command::new(&ffmpeg_cmd)
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-i",
                &input_path,
                "-vcodec",
                "copy",
                "-an",
                "-bsf:v",
                "hevc_mp4toannexb",
                hevc_path.to_str().unwrap(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .await
            .ok();

        if hevc_output.as_ref().is_some_and(|o| o.status.success()) {
            tracing::info!("Falling back to HEVC elementary stream");
            return run_engine_pipeline(
                hevc_path,
                native_output_path_hevc,
                output_path,
                input_path,
                model_path,
                scale,
                precision,
                preserve_audio,
                max_batch,
                ffmpeg_cmd.clone(),
                CudaCodec::HEVC,
            )
            .await;
        }

        let h264_stderr = stderr_tail(&demux_output.stderr, 12);
        let hevc_stderr = hevc_output
            .as_ref()
            .map(|o| stderr_tail(&o.stderr, 12))
            .unwrap_or_else(|| "FFmpeg HEVC fallback did not produce output.".to_string());
        return Err(make_err(
            "DEMUX_FAILED",
            &format!(
                "FFmpeg could not extract an H.264 or HEVC elementary stream from the input. \
                 Only H.264 and HEVC containers are supported by the native engine MVP.\n\
                 h264 stderr:\n{}\n\
                 hevc stderr:\n{}",
                h264_stderr, hevc_stderr
            ),
        ));
    }

    run_engine_pipeline(
        elementary_stream_path,
        native_output_path_h264,
        output_path,
        input_path,
        model_path,
        scale,
        precision,
        preserve_audio,
        max_batch,
        ffmpeg_cmd,
        CudaCodec::H264,
    )
    .await
}

#[cfg(feature = "native_engine")]
#[allow(clippy::too_many_arguments)] // TODO(clippy): this pipeline entry mirrors explicit stage inputs; keep stable for now.
async fn run_engine_pipeline(
    input_stream: PathBuf,
    engine_output: PathBuf,
    final_output: String,
    original_input: String,
    model_path: String,
    scale: u32,
    precision: String,
    preserve_audio: bool,
    max_batch: u32,
    ffmpeg_cmd: String,
    codec: videoforge_engine::codecs::sys::cudaVideoCodec,
) -> Result<NativeUpscaleResult, String> {
    use std::process::Stdio;
    use std::sync::Arc;
    use tokio::process::Command;

    use videoforge_engine::backends::tensorrt::{BatchConfig, PrecisionPolicy, TensorRtBackend};
    use videoforge_engine::codecs::nvdec::NvDecoder;
    use videoforge_engine::codecs::nvenc::NvEncConfig;
    use videoforge_engine::core::backend::UpscaleBackend;
    use videoforge_engine::core::context::GpuContext;
    use videoforge_engine::core::kernels::{ModelPrecision, PreprocessKernels};
    use videoforge_engine::engine::pipeline::{PipelineConfig, UpscalePipeline};

    let make_err =
        |code: &str, msg: &str| serde_json::to_string(&NativeUpscaleError::new(code, msg)).unwrap();
    let (trt_cache_enabled, trt_cache_dir) = trt_cache_runtime(&model_path);

    // ── Step 1.5: Probe input for dimensions ──────────────────────────────────
    // encoder_nv12_pitch and encoder width/height must be set before
    // UpscalePipeline::new() — the pipeline asserts pitch > 0.
    tracing::info!(path = %original_input, "Probing input video dimensions");
    let (input_w, input_h, _duration, fps, _) = probe_video_coded_geometry(&original_input)
        .map_err(|e| make_err("PROBE_FAILED", &e))?;
    let output_w = input_w.saturating_mul(scale as usize);
    let output_h = input_h.saturating_mul(scale as usize);
    // NV12 row stride must be 256-byte aligned (NVENC hardware requirement).
    let encoder_nv12_pitch = output_w.div_ceil(256) * 256;
    // Express fps as a rational with 1000 as denominator — handles 23.976, 29.97, etc.
    let fps_num = (fps * 1000.0).round() as u32;
    let fps_den = 1000u32;
    tracing::info!(
        input_w,
        input_h,
        output_w,
        output_h,
        encoder_nv12_pitch,
        fps,
        "Video dimensions resolved"
    );

    // ── Step 2: Initialise GPU context ────────────────────────────────────────
    tracing::info!("Initialising GPU context (device 0)");
    // GpuContext::new already returns Arc<GpuContext>.
    let ctx = GpuContext::new(0)
        .map_err(|e| make_err("GPU_INIT", &format!("GPU context creation failed: {}", e)))?;

    // ── Step 3: Load TensorRT backend ─────────────────────────────────────────
    tracing::info!(model = %model_path, "Loading TensorRT backend");
    let precision_policy = match precision.as_str() {
        "fp16" => PrecisionPolicy::Fp16,
        _ => PrecisionPolicy::Fp32,
    };
    // TensorRtBackend::new(model_path, ctx, device_id, ring_size, downstream_capacity).
    // Use with_precision to apply the precision policy.
    let backend = TensorRtBackend::with_precision(
        std::path::PathBuf::from(&model_path),
        ctx.clone(),
        0, // device_id
        8, // ring_size (≥ downstream_capacity + 2)
        4, // downstream_capacity
        precision_policy,
        BatchConfig {
            max_batch: max_batch as usize,
            ..BatchConfig::default()
        },
    );
    let backend = Arc::new(backend);
    backend.initialize().await.map_err(|e| {
        make_err(
            "BACKEND_INIT",
            &format!("TensorRT backend init failed: {}", e),
        )
    })?;

    // ── Step 4: Compile kernels ───────────────────────────────────────────────
    let kernels = PreprocessKernels::compile(ctx.device())
        .map_err(|e| make_err("KERNEL_COMPILE", &format!("Kernel compile failed: {}", e)))?;
    let kernels = Arc::new(kernels);

    // ── Step 5: Create decoder with FileBitstreamSource ───────────────────────
    tracing::info!(path = %input_stream.display(), "Creating NVDEC decoder");
    let model_prec = match backend
        .metadata()
        .map_err(|e| make_err("BACKEND_INIT", &format!("Model metadata unavailable: {}", e)))?
        .input_format
    {
        videoforge_engine::core::types::PixelFormat::RgbPlanarF16 => ModelPrecision::F16,
        _ => ModelPrecision::F32,
    };
    let source = FileBitstreamSource::new(&input_stream).map_err(|e| {
        make_err(
            "SOURCE_OPEN",
            &format!("Cannot open elementary stream: {}", e),
        )
    })?;
    let decoder = NvDecoder::new(ctx.clone(), Box::new(source), codec)
        .map_err(|e| make_err("DECODER_INIT", &format!("NVDEC decoder init failed: {}", e)))?;

    // ── Step 6: Create encoder with FileBitstreamSink ─────────────────────────
    tracing::info!(path = %engine_output.display(), "Creating NVENC encoder");

    let enc_config = NvEncConfig {
        width: output_w as u32,
        height: output_h as u32,
        fps_num,
        fps_den,
        bitrate: 8_000_000,
        max_bitrate: 0,
        gop_length: 30,
        b_frames: 0,
        nv12_pitch: encoder_nv12_pitch as u32,
    };
    let sink = FileBitstreamSink::new(&engine_output)
        .map_err(|e| make_err("SINK_OPEN", &format!("Cannot create output stream: {}", e)))?;
    // NvEncoder::new takes (raw_cuda_context: *mut c_void, sink, config).
    // Bind the primary context to this thread, then retrieve whatever is
    // current via cuCtxGetCurrent — the NVENC-canonical approach (matches
    // NVIDIA SDK samples).
    ctx.device().bind_to_thread().map_err(|e| {
        make_err(
            "ENCODER_INIT",
            &format!("Failed to bind CUDA context: {:?}", e),
        )
    })?;
    let cuda_ctx = ctx
        .current_context_ptr()
        .map_err(|e| make_err("ENCODER_INIT", &format!("cuCtxGetCurrent failed: {}", e)))?;
    let encoder = NativeVideoEncoderWrapper::new(
        ctx.clone(),
        ffmpeg_cmd.clone(),
        cuda_ctx,
        Box::new(sink),
        enc_config,
        engine_output.clone(),
    )
    .map_err(|e| make_err("ENCODER_INIT", &format!("Encoder init failed: {}", e)))?;
    let encoder_mode = encoder.mode_handle();
    let encoder_detail = encoder.detail_handle();

    // ── Step 7: Run the pipeline ──────────────────────────────────────────────
    let config = PipelineConfig {
        model_precision: model_prec,
        encoder_nv12_pitch,
        inference_max_batch: max_batch as usize,
        ..PipelineConfig::default()
    };
    let pipeline = UpscalePipeline::new(ctx.clone(), kernels, config);

    tracing::info!("Running engine-v2 pipeline");
    pipeline
        .run(decoder, backend, encoder)
        .await
        .map_err(|e| make_err("PIPELINE", &format!("Pipeline error: {}", e)))?;

    let frames = pipeline
        .metrics()
        .frames_encoded
        .load(std::sync::atomic::Ordering::Relaxed);
    let encoder_mode = encoder_mode.as_str().to_string();
    let encoder_detail = encoder_detail.get();
    tracing::info!(
        frames_encoded = frames,
        encoder_mode = %encoder_mode,
        encoder_detail = encoder_detail.as_deref().unwrap_or("none"),
        "engine-v2 pipeline complete"
    );

    // ── Step 8: FFmpeg mux video + audio ──────────────────────────────────────
    tracing::info!(
        video = %engine_output.display(),
        audio_src = %original_input,
        output = %final_output,
        "Muxing video and audio with FFmpeg"
    );

    let mut mux_args = vec![
        "-y".to_string(),
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "warning".to_string(),
        // video from native engine output
        "-i".to_string(),
        engine_output.to_str().unwrap().to_string(),
    ];

    if preserve_audio {
        // Audio from original input
        mux_args.push("-i".to_string());
        mux_args.push(original_input.clone());
    }

    mux_args.extend(["-c:v".to_string(), "copy".to_string()]);

    if preserve_audio {
        mux_args.extend([
            "-c:a".to_string(),
            "copy".to_string(),
            "-map".to_string(),
            "0:v:0".to_string(),
            "-map".to_string(),
            "1:a?".to_string(), // optional audio — won't fail if absent
        ]);
    } else {
        mux_args.push("-an".to_string());
    }

    mux_args.extend([
        "-movflags".to_string(),
        "+faststart".to_string(),
        final_output.clone(),
    ]);

    let mux_output = Command::new(&ffmpeg_cmd)
        .args(&mux_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| make_err("FFMPEG_MUX_SPAWN", &e.to_string()))?;

    if !mux_output.status.success() {
        let stderr_tail = stderr_tail(&mux_output.stderr, 12);
        tracing::error!(
            status = ?mux_output.status.code(),
            input_stream = %input_stream.display(),
            engine_output = %engine_output.display(),
            final_output = %final_output,
            stderr = %stderr_tail,
            "FFmpeg mux step failed"
        );
        return Err(make_err(
            "MUX_FAILED",
            &format!(
                "FFmpeg mux step failed. Temporary files were preserved for inspection. stderr:\n{}",
                stderr_tail
            ),
        ));
    }

    // Clean up temp files after a successful mux.
    let _ = std::fs::remove_file(&input_stream);
    let _ = std::fs::remove_file(&engine_output);

    tracing::info!(output = %final_output, "Native engine upscale complete");

    Ok(NativeUpscaleResult {
        output_path: final_output,
        engine: "native_v2".to_string(),
        encoder_mode,
        encoder_detail,
        frames_processed: frames,
        audio_preserved: preserve_audio,
        trt_cache_enabled,
        trt_cache_dir,
    })
}

// =============================================================================
// FileBitstreamSource — reads Annex B elementary stream from disk
// =============================================================================

#[cfg(feature = "native_engine")]
struct FileBitstreamSource {
    file: std::fs::File,
    eof: bool,
    pts_counter: i64,
}

#[cfg(feature = "native_engine")]
impl FileBitstreamSource {
    fn new(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            file: std::fs::File::open(path)?,
            eof: false,
            pts_counter: 0,
        })
    }
}

#[cfg(feature = "native_engine")]
impl videoforge_engine::codecs::nvdec::BitstreamSource for FileBitstreamSource {
    fn read_packet(
        &mut self,
    ) -> videoforge_engine::error::Result<Option<videoforge_engine::codecs::nvdec::BitstreamPacket>>
    {
        use std::io::Read;

        if self.eof {
            return Ok(None);
        }

        // Feed in 1 MiB chunks.  The NVDEC parser handles cross-chunk NAL
        // boundaries internally, so we can stream reads directly from disk.
        const CHUNK: usize = 1024 * 1024;
        let mut chunk = vec![0_u8; CHUNK];
        let bytes_read = self.file.read(&mut chunk).map_err(|e| {
            videoforge_engine::error::EngineError::Decode(format!(
                "read elementary stream: {e}"
            ))
        })?;
        if bytes_read == 0 {
            self.eof = true;
            return Ok(None);
        }
        chunk.truncate(bytes_read);
        if bytes_read < CHUNK {
            self.eof = true;
        }

        // Heuristic keyframe detection: IDR NAL type 0x65 after a start code.
        let is_keyframe = chunk.windows(5).any(|w| {
            (w[0] == 0 && w[1] == 0 && w[2] == 0 && w[3] == 1 && (w[4] & 0x1F) == 5)
                || (w[0] == 0 && w[1] == 0 && w[2] == 1 && (w[3] & 0x1F) == 5)
        });

        let pts = self.pts_counter;
        self.pts_counter += 1;

        Ok(Some(videoforge_engine::codecs::nvdec::BitstreamPacket {
            data: chunk,
            pts,
            is_keyframe,
        }))
    }
}

// =============================================================================
// FileBitstreamSink — writes NVENC output to disk
// =============================================================================

#[cfg(feature = "native_engine")]
struct FileBitstreamSink {
    file: std::fs::File,
}

#[cfg(feature = "native_engine")]
impl FileBitstreamSink {
    fn new(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            file: std::fs::File::create(path)?,
        })
    }
}

#[cfg(feature = "native_engine")]
impl videoforge_engine::codecs::nvenc::BitstreamSink for FileBitstreamSink {
    fn write_packet(
        &mut self,
        data: &[u8],
        _pts: i64,
        _is_keyframe: bool,
    ) -> videoforge_engine::error::Result<()> {
        use std::io::Write;
        self.file
            .write_all(data)
            .map_err(|e| videoforge_engine::error::EngineError::Encode(e.to_string()))
    }

    fn flush(&mut self) -> videoforge_engine::error::Result<()> {
        use std::io::Write;
        self.file
            .flush()
            .map_err(|e| videoforge_engine::error::EngineError::Encode(e.to_string()))
    }
}

#[cfg(feature = "native_engine")]
struct SoftwareBitstreamEncoder {
    child: Option<std::process::Child>,
    stdin: Option<std::process::ChildStdin>,
    ctx: Arc<videoforge_engine::core::context::GpuContext>,
    width: usize,
    height: usize,
    tight_nv12: Vec<u8>,
}

#[cfg(feature = "native_engine")]
impl SoftwareBitstreamEncoder {
    fn new(
        ctx: Arc<videoforge_engine::core::context::GpuContext>,
        ffmpeg_cmd: &str,
        output_path: &Path,
        width: u32,
        height: u32,
        fps_num: u32,
        fps_den: u32,
    ) -> videoforge_engine::error::Result<Self> {
        use std::process::{Command, Stdio};

        let fps_den = fps_den.max(1);
        let codec = match output_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref()
        {
            Some("h264") | Some("264") => "libx264",
            _ => "libx265",
        };
        let format = if codec == "libx264" { "h264" } else { "hevc" };

        let mut cmd = Command::new(ffmpeg_cmd);
        cmd.arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-y")
            .arg("-f")
            .arg("rawvideo")
            .arg("-pix_fmt")
            .arg("nv12")
            .arg("-s:v")
            .arg(format!("{width}x{height}"))
            .arg("-framerate")
            .arg(format!("{fps_num}/{fps_den}"))
            .arg("-i")
            .arg("-")
            .arg("-an")
            .arg("-c:v")
            .arg(codec)
            .arg("-preset")
            .arg("medium")
            .arg("-f")
            .arg(format)
            .arg(output_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            videoforge_engine::error::EngineError::Encode(format!(
                "Software encode fallback unavailable (failed to launch ffmpeg): {e}"
            ))
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            videoforge_engine::error::EngineError::Encode(
                "Software encode fallback stdin unavailable".into(),
            )
        })?;

        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
            ctx,
            width: width as usize,
            height: height as usize,
            tight_nv12: Vec::new(),
        })
    }

    fn tight_size_bytes(&self) -> usize {
        self.width * self.height * 3 / 2
    }

    fn repack_nv12_tight(
        &mut self,
        src: &[u8],
        src_pitch: usize,
    ) -> videoforge_engine::error::Result<&[u8]> {
        let y_size_tight = self.width * self.height;
        let uv_size_tight = y_size_tight / 2;
        let total_tight = y_size_tight + uv_size_tight;
        let src_required = src_pitch * self.height * 3 / 2;
        if src.len() < src_required {
            return Err(videoforge_engine::error::EngineError::Encode(format!(
                "Software encoder readback too small: got {} bytes, need at least {} bytes",
                src.len(),
                src_required
            )));
        }

        self.tight_nv12.resize(total_tight, 0);
        for row in 0..self.height {
            let src_off = row * src_pitch;
            let dst_off = row * self.width;
            self.tight_nv12[dst_off..dst_off + self.width]
                .copy_from_slice(&src[src_off..src_off + self.width]);
        }

        let uv_src_base = src_pitch * self.height;
        let uv_dst_base = y_size_tight;
        for row in 0..(self.height / 2) {
            let src_off = uv_src_base + row * src_pitch;
            let dst_off = uv_dst_base + row * self.width;
            self.tight_nv12[dst_off..dst_off + self.width]
                .copy_from_slice(&src[src_off..src_off + self.width]);
        }

        Ok(&self.tight_nv12)
    }
}

#[cfg(feature = "native_engine")]
impl videoforge_engine::engine::pipeline::FrameEncoder for SoftwareBitstreamEncoder {
    fn encode(
        &mut self,
        frame: videoforge_engine::core::types::FrameEnvelope,
    ) -> videoforge_engine::error::Result<()> {
        use std::io::Write;

        if frame.texture.format != videoforge_engine::core::types::PixelFormat::Nv12 {
            return Err(videoforge_engine::error::EngineError::Encode(format!(
                "Software fallback encoder expected NV12 frame, got {:?}",
                frame.texture.format
            )));
        }

        self.ctx.sync_all()?;
        let host = self.ctx.device().dtoh_sync_copy(&*frame.texture.data).map_err(|e| {
            videoforge_engine::error::EngineError::Encode(format!(
                "Software fallback DtoH readback failed: {e}"
            ))
        })?;

        let payload: Vec<u8> = if frame.texture.pitch == self.width {
            let tight = self.tight_size_bytes();
            if host.len() < tight {
                return Err(videoforge_engine::error::EngineError::Encode(format!(
                    "Software encoder readback too small for tight NV12: got {} bytes, need {} bytes",
                    host.len(),
                    tight
                )));
            }
            host[..tight].to_vec()
        } else {
            self.repack_nv12_tight(&host, frame.texture.pitch)?.to_vec()
        };

        let stdin = self.stdin.as_mut().ok_or_else(|| {
            videoforge_engine::error::EngineError::Encode(
                "Software fallback encoder stdin closed".into(),
            )
        })?;
        stdin.write_all(&payload).map_err(|e| {
            videoforge_engine::error::EngineError::Encode(format!(
                "Software fallback failed writing frame to ffmpeg stdin: {e}"
            ))
        })
    }

    fn flush(&mut self) -> videoforge_engine::error::Result<()> {
        drop(self.stdin.take());
        let child = self.child.take().ok_or_else(|| {
            videoforge_engine::error::EngineError::Encode(
                "Software fallback process missing".into(),
            )
        })?;
        let output = child.wait_with_output().map_err(|e| {
            videoforge_engine::error::EngineError::Encode(format!(
                "Software fallback failed waiting for ffmpeg exit: {e}"
            ))
        })?;
        if !output.status.success() {
            return Err(videoforge_engine::error::EngineError::Encode(format!(
                "Software fallback ffmpeg exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "native_engine")]
enum NativeVideoEncoder {
    Nvenc(videoforge_engine::codecs::nvenc::NvEncoder),
    Software(SoftwareBitstreamEncoder),
}

#[cfg(feature = "native_engine")]
#[derive(Clone)]
struct NativeEncoderModeHandle(Arc<AtomicU8>);

#[cfg(feature = "native_engine")]
impl NativeEncoderModeHandle {
    const NVENC: u8 = 1;
    const NVENC_LEGACY_STAGING: u8 = 2;
    const SOFTWARE: u8 = 3;

    fn new(initial: u8) -> Self {
        Self(Arc::new(AtomicU8::new(initial)))
    }

    fn set(&self, mode: u8) {
        self.0.store(mode, Ordering::Relaxed);
    }

    fn as_str(&self) -> &'static str {
        match self.0.load(Ordering::Relaxed) {
            Self::NVENC => "nvenc",
            Self::NVENC_LEGACY_STAGING => "nvenc_legacy_staging",
            Self::SOFTWARE => "software",
            _ => "unknown",
        }
    }
}

#[cfg(feature = "native_engine")]
#[derive(Clone, Default)]
struct NativeEncoderDetailHandle(Arc<Mutex<Option<String>>>);

#[cfg(feature = "native_engine")]
impl NativeEncoderDetailHandle {
    fn set(&self, detail: impl Into<String>) {
        let mut slot = self.0.lock().expect("encoder detail mutex poisoned");
        *slot = Some(detail.into());
    }

    fn get(&self) -> Option<String> {
        self.0
            .lock()
            .expect("encoder detail mutex poisoned")
            .clone()
    }
}

#[cfg(feature = "native_engine")]
struct NativeVideoEncoderConfig {
    ctx: Arc<videoforge_engine::core::context::GpuContext>,
    ffmpeg_cmd: String,
    output_path: PathBuf,
    enc_config: videoforge_engine::codecs::nvenc::NvEncConfig,
}

#[cfg(feature = "native_engine")]
struct NativeVideoEncoderWrapper {
    inner: Option<NativeVideoEncoder>,
    fallback: NativeVideoEncoderConfig,
    frames_encoded: u64,
    mode: NativeEncoderModeHandle,
    detail: NativeEncoderDetailHandle,
}

#[cfg(feature = "native_engine")]
impl NativeVideoEncoderWrapper {
    fn new(
        ctx: Arc<videoforge_engine::core::context::GpuContext>,
        ffmpeg_cmd: String,
        cuda_ctx: *mut std::ffi::c_void,
        sink: Box<dyn videoforge_engine::codecs::nvenc::BitstreamSink>,
        config: videoforge_engine::codecs::nvenc::NvEncConfig,
        output_path: PathBuf,
    ) -> videoforge_engine::error::Result<Self> {
        let fallback = NativeVideoEncoderConfig {
            ctx,
            ffmpeg_cmd,
            output_path,
            enc_config: config.clone(),
        };
        let detail = NativeEncoderDetailHandle::default();
        match videoforge_engine::codecs::nvenc::NvEncoder::new(cuda_ctx, sink, config) {
            Ok(enc) => Ok(Self {
                inner: Some(NativeVideoEncoder::Nvenc(enc)),
                fallback,
                frames_encoded: 0,
                mode: NativeEncoderModeHandle::new(NativeEncoderModeHandle::NVENC),
                detail,
            }),
            Err(err) => {
                detail.set(format!("nvenc_init_failed: {err}"));
                tracing::warn!(
                    error = %err,
                    output = %fallback.output_path.display(),
                    "NVENC init failed; refusing in-process software fallback"
                );
                Err(videoforge_engine::error::EngineError::Encode(format!(
                    "NVENC init failed for direct native path: {err}"
                )))
            }
        }
    }

    fn mode_handle(&self) -> NativeEncoderModeHandle {
        self.mode.clone()
    }

    fn detail_handle(&self) -> NativeEncoderDetailHandle {
        self.detail.clone()
    }
}

#[cfg(feature = "native_engine")]
impl videoforge_engine::engine::pipeline::FrameEncoder for NativeVideoEncoderWrapper {
    fn encode(
        &mut self,
        frame: videoforge_engine::core::types::FrameEnvelope,
    ) -> videoforge_engine::error::Result<()> {
        let inner = self.inner.take().ok_or_else(|| {
            videoforge_engine::error::EngineError::Encode("Native encoder missing".into())
        })?;
        match inner {
            NativeVideoEncoder::Nvenc(mut enc) => match enc.encode(frame.clone()) {
                Ok(()) => {
                    self.frames_encoded += 1;
                    match enc.runtime_mode() {
                        "nvenc_legacy_staging" => {
                            self.mode.set(NativeEncoderModeHandle::NVENC_LEGACY_STAGING)
                        }
                        _ => self.mode.set(NativeEncoderModeHandle::NVENC),
                    }
                    self.inner = Some(NativeVideoEncoder::Nvenc(enc));
                    Ok(())
                }
                Err(err) => {
                    if self.frames_encoded > 0 {
                        self.inner = Some(NativeVideoEncoder::Nvenc(enc));
                        return Err(videoforge_engine::error::EngineError::Encode(format!(
                            "NVENC encode failed after {} frame(s); refusing mid-stream software fallback: {}",
                            self.frames_encoded, err
                        )));
                    }
                    tracing::warn!(
                        error = %err,
                        output = %self.fallback.output_path.display(),
                        "NVENC encode failed before first frame; refusing in-process software fallback"
                    );
                    self.detail.set(format!("nvenc_first_frame_encode_failed: {err}"));
                    drop(enc);
                    Err(videoforge_engine::error::EngineError::Encode(format!(
                        "NVENC first-frame encode failed for direct native path: {err}"
                    )))
                }
            },
            NativeVideoEncoder::Software(mut enc) => {
                let result = enc.encode(frame);
                if result.is_ok() {
                    self.frames_encoded += 1;
                }
                self.inner = Some(NativeVideoEncoder::Software(enc));
                result
            }
        }
    }

    fn flush(&mut self) -> videoforge_engine::error::Result<()> {
        match self.inner.as_mut().ok_or_else(|| {
            videoforge_engine::error::EngineError::Encode("Native encoder missing".into())
        })? {
            NativeVideoEncoder::Nvenc(enc) => enc.flush(),
            NativeVideoEncoder::Software(enc) => enc.flush(),
        }
    }
}
