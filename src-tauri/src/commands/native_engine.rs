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
//! engine-v2 NVDEC → TensorRT → NVENC → FFmpeg mux stdin
//!      ↓
//! final .mp4 (+ optional copied audio from original input)
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
use std::collections::VecDeque;
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
fn probe_video_coded_geometry(
    path: &str,
) -> Result<
    (
        usize,
        usize,
        f64,
        f64,
        u64,
        videoforge_engine::codecs::sys::cudaVideoCodec,
    ),
    String,
> {
    let output = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,coded_width,coded_height,r_frame_rate,codec_name",
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
    let codec = match stream
        .get("codec_name")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "h264" => videoforge_engine::codecs::sys::cudaVideoCodec::H264,
        "hevc" | "h265" => videoforge_engine::codecs::sys::cudaVideoCodec::HEVC,
        other => {
            return Err(format!(
                "Unsupported native input codec '{other}'. Only H.264 and HEVC are supported."
            ))
        }
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
        codec = ?codec,
        "Resolved native coded geometry"
    );

    Ok((coded_width, coded_height, duration, fps, total_frames, codec))
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
fn classify_backend_init_error(model_path: &str, err: &str) -> String {
    if err.contains("Load model from")
        && err.contains("Type Error:")
        && (err.contains("tensor(float16)") || err.contains("tensor(float)"))
    {
        return format!(
            "Invalid ONNX artifact: {model_path} failed ORT model validation due to inconsistent tensor types. \
This is usually a broken or incompatible export, not a native runtime failure. Original error: {err}"
        );
    }

    format!("TensorRT backend init failed: {err}")
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
    preserve_audio: bool,
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
        audio_preserved: preserve_audio,
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

    run_engine_pipeline(
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
    use std::sync::Arc;

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
    let (input_w, input_h, _duration, fps, _, probed_codec) = probe_video_coded_geometry(&original_input)
        .map_err(|e| make_err("PROBE_FAILED", &e))?;
    if probed_codec != codec {
        tracing::warn!(
            requested = ?codec,
            probed = ?probed_codec,
            "Native codec routing differed from ffprobe result; using probed codec for streamed demux"
        );
    }
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
    let downstream_capacity = 4usize;
    let ring_size = TensorRtBackend::required_ring_slots(downstream_capacity, max_batch as usize);

    // TensorRtBackend::new(model_path, ctx, device_id, ring_size, downstream_capacity).
    // Use with_precision to apply the precision policy and keep output slots batch-safe.
    let backend = TensorRtBackend::with_precision(
        std::path::PathBuf::from(&model_path),
        ctx.clone(),
        0, // device_id
        ring_size,
        downstream_capacity,
        precision_policy,
        BatchConfig {
            max_batch: max_batch as usize,
            ..BatchConfig::default()
        },
    );
    let backend = Arc::new(backend);
    backend
        .initialize()
        .await
        .map_err(|e| make_err("BACKEND_INIT", &classify_backend_init_error(&model_path, &e.to_string())))?;

    // ── Step 4: Compile kernels ───────────────────────────────────────────────
    let kernels = PreprocessKernels::compile(ctx.device())
        .map_err(|e| make_err("KERNEL_COMPILE", &format!("Kernel compile failed: {}", e)))?;
    let kernels = Arc::new(kernels);

    // ── Step 5: Create decoder with FFmpeg-streamed elementary source ────────
    tracing::info!(path = %original_input, codec = ?probed_codec, "Creating NVDEC decoder");
    let model_prec = match backend
        .metadata()
        .map_err(|e| make_err("BACKEND_INIT", &format!("Model metadata unavailable: {}", e)))?
        .input_format
    {
        videoforge_engine::core::types::PixelFormat::RgbPlanarF16 => ModelPrecision::F16,
        _ => ModelPrecision::F32,
    };
    let source = FfmpegBitstreamSource::spawn(&ffmpeg_cmd, &original_input, probed_codec)
        .map_err(|e| make_err("SOURCE_OPEN", &format!("Cannot stream elementary input: {}", e)))?;
    let decoder = NvDecoder::new(ctx.clone(), Box::new(source), probed_codec)
        .map_err(|e| make_err("DECODER_INIT", &format!("NVDEC decoder init failed: {}", e)))?;

    // ── Step 6: Create encoder with streaming FFmpeg mux sink ────────────────
    tracing::info!(path = %final_output, "Creating NVENC encoder and mux sink");

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
    let mux_codec_hint = StreamingCodecHint::new(match probed_codec {
        videoforge_engine::codecs::sys::cudaVideoCodec::H264 => Some("h264"),
        videoforge_engine::codecs::sys::cudaVideoCodec::HEVC => Some("hevc"),
        _ => None,
    });
    let sink = StreamingMuxSink::new(
        &ffmpeg_cmd,
        &final_output,
        &original_input,
        preserve_audio,
        mux_codec_hint.clone(),
    )
        .map_err(|e| make_err("SINK_OPEN", &format!("Cannot create mux stream: {}", e)))?;
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
        PathBuf::from(&final_output),
        mux_codec_hint,
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
    let pipeline_backend = backend.clone();
    let pipeline_result = pipeline.run(decoder, pipeline_backend, encoder).await;
    let shutdown_result = backend.shutdown().await;

    if let Err(e) = pipeline_result {
        let shutdown_detail = shutdown_result
            .err()
            .map(|shutdown_err| format!(" Cleanup error after pipeline failure: {shutdown_err}"))
            .unwrap_or_default();
        return Err(make_err(
            "PIPELINE",
            &format!("Pipeline error: {e}.{shutdown_detail}"),
        ));
    }

    if let Err(e) = shutdown_result {
        return Err(make_err(
            "BACKEND_SHUTDOWN",
            &format!("TensorRT backend shutdown failed: {e}"),
        ));
    }

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
// FfmpegBitstreamSource — streams Annex B elementary video from FFmpeg stdout
// =============================================================================

#[cfg(feature = "native_engine")]
struct FfmpegBitstreamSource {
    child: std::process::Child,
    stdout: std::process::ChildStdout,
    stderr_tail: SharedStderrTail,
    eof: bool,
    pts_counter: i64,
}

#[cfg(feature = "native_engine")]
#[derive(Clone, Default)]
struct SharedStderrTail(Arc<Mutex<VecDeque<String>>>);

#[cfg(feature = "native_engine")]
impl SharedStderrTail {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(VecDeque::with_capacity(16))))
    }

    fn spawn_reader<R>(&self, reader: R)
    where
        R: std::io::Read + Send + 'static,
    {
        let tail = self.clone();
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};

            let buf = BufReader::new(reader);
            for line in buf.lines().map_while(Result::ok) {
                tail.push(line);
            }
        });
    }

    fn push(&self, line: String) {
        let mut guard = self.0.lock().expect("stderr tail mutex poisoned");
        if guard.len() >= 12 {
            guard.pop_front();
        }
        guard.push_back(line);
    }

    fn snapshot(&self) -> String {
        self.0
            .lock()
            .expect("stderr tail mutex poisoned")
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(feature = "native_engine")]
#[derive(Clone, Default)]
struct StreamingCodecHint(Arc<Mutex<Option<&'static str>>>);

#[cfg(feature = "native_engine")]
impl StreamingCodecHint {
    fn new(initial: Option<&'static str>) -> Self {
        Self(Arc::new(Mutex::new(initial)))
    }

    fn set(&self, format: &'static str) {
        let mut guard = self.0.lock().expect("codec hint mutex poisoned");
        *guard = Some(format);
    }

    fn get(&self) -> Option<&'static str> {
        *self.0.lock().expect("codec hint mutex poisoned")
    }
}

#[cfg(feature = "native_engine")]
impl FfmpegBitstreamSource {
    fn spawn(
        ffmpeg_cmd: &str,
        input_path: &str,
        codec: videoforge_engine::codecs::sys::cudaVideoCodec,
    ) -> std::io::Result<Self> {
        use std::process::{Command, Stdio};

        let bitstream_filter = match codec {
            videoforge_engine::codecs::sys::cudaVideoCodec::H264 => "h264_mp4toannexb",
            videoforge_engine::codecs::sys::cudaVideoCodec::HEVC => "hevc_mp4toannexb",
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Unsupported streamed demux codec: {codec:?}"),
                ));
            }
        };
        let output_format = match codec {
            videoforge_engine::codecs::sys::cudaVideoCodec::H264 => "h264",
            videoforge_engine::codecs::sys::cudaVideoCodec::HEVC => "hevc",
            _ => unreachable!(),
        };

        let mut child = Command::new(ffmpeg_cmd)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-vcodec",
                "copy",
                "-an",
                "-bsf:v",
                bitstream_filter,
                "-f",
                output_format,
                "-",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdout = child.stdout.take().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "FFmpeg demux stdout unavailable",
            )
        })?;
        let stderr_tail = SharedStderrTail::new();
        if let Some(stderr) = child.stderr.take() {
            stderr_tail.spawn_reader(stderr);
        }

        Ok(Self {
            child,
            stdout,
            stderr_tail,
            eof: false,
            pts_counter: 0,
        })
    }
}

#[cfg(feature = "native_engine")]
impl videoforge_engine::codecs::nvdec::BitstreamSource for FfmpegBitstreamSource {
    fn read_packet(
        &mut self,
    ) -> videoforge_engine::error::Result<Option<videoforge_engine::codecs::nvdec::BitstreamPacket>>
    {
        use std::io::Read;

        if self.eof {
            return Ok(None);
        }

        const CHUNK: usize = 1024 * 1024;
        let mut chunk = vec![0_u8; CHUNK];
        let bytes_read = self.stdout.read(&mut chunk).map_err(|e| {
            videoforge_engine::error::EngineError::Decode(format!(
                "read FFmpeg demux stream: {e}"
            ))
        })?;

        if bytes_read == 0 {
            self.eof = true;
            let status = self.child.wait().map_err(|e| {
                videoforge_engine::error::EngineError::Decode(format!(
                    "wait for FFmpeg demux process: {e}"
                ))
            })?;
            if !status.success() {
                return Err(videoforge_engine::error::EngineError::Decode(format!(
                    "FFmpeg demux failed with {status}: {}",
                    self.stderr_tail.snapshot()
                )));
            }
            return Ok(None);
        }

        chunk.truncate(bytes_read);
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
struct StreamingMuxSink {
    ffmpeg_cmd: String,
    output_path: String,
    original_input: String,
    preserve_audio: bool,
    codec_hint: StreamingCodecHint,
    stderr_tail: SharedStderrTail,
    child: Option<std::process::Child>,
    stdin: Option<std::process::ChildStdin>,
}

#[cfg(feature = "native_engine")]
impl StreamingMuxSink {
    fn new(
        ffmpeg_cmd: &str,
        output_path: &str,
        original_input: &str,
        preserve_audio: bool,
        codec_hint: StreamingCodecHint,
    ) -> std::io::Result<Self> {
        Ok(Self {
            ffmpeg_cmd: ffmpeg_cmd.to_string(),
            output_path: output_path.to_string(),
            original_input: original_input.to_string(),
            preserve_audio,
            codec_hint,
            stderr_tail: SharedStderrTail::new(),
            child: None,
            stdin: None,
        })
    }

    fn ensure_started(
        &mut self,
        bitstream_format: &'static str,
    ) -> videoforge_engine::error::Result<&mut std::process::ChildStdin> {
        use std::process::{Command, Stdio};

        if self.stdin.is_none() {
            let mut cmd = Command::new(&self.ffmpeg_cmd);
            cmd.arg("-y")
                .arg("-hide_banner")
                .arg("-loglevel")
                .arg("warning")
                .arg("-f")
                .arg(bitstream_format)
                .arg("-i")
                .arg("-");

            if self.preserve_audio {
                cmd.arg("-i").arg(&self.original_input);
            }

            cmd.arg("-c:v").arg("copy");

            if self.preserve_audio {
                cmd.arg("-c:a")
                    .arg("copy")
                    .arg("-map")
                    .arg("0:v:0")
                    .arg("-map")
                    .arg("1:a?");
            } else {
                cmd.arg("-an");
            }

            cmd.arg("-movflags")
                .arg("+faststart")
                .arg(&self.output_path)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());

            let mut child = cmd.spawn().map_err(|e| {
                videoforge_engine::error::EngineError::Encode(format!(
                    "FFmpeg mux spawn failed for {bitstream_format}: {e}"
                ))
            })?;
            if let Some(stderr) = child.stderr.take() {
                self.stderr_tail.spawn_reader(stderr);
            }
            let stdin = child.stdin.take().ok_or_else(|| {
                videoforge_engine::error::EngineError::Encode("Mux stdin unavailable".into())
            })?;
            self.stdin = Some(stdin);
            self.child = Some(child);
        }

        self.stdin.as_mut().ok_or_else(|| {
            videoforge_engine::error::EngineError::Encode("Mux stdin closed".into())
        })
    }
}

#[cfg(feature = "native_engine")]
fn infer_annex_b_bitstream_format(data: &[u8]) -> Option<&'static str> {
    let mut i = 0usize;
    while i + 4 < data.len() {
        let (nal_start, header_index) = if data[i..].starts_with(&[0, 0, 1]) {
            (i, i + 3)
        } else if data[i..].starts_with(&[0, 0, 0, 1]) {
            (i, i + 4)
        } else {
            i += 1;
            continue;
        };

        if header_index >= data.len() {
            break;
        }

        let header = data[header_index];
        let h264_type = header & 0x1F;
        if matches!(h264_type, 1 | 5 | 7 | 8) {
            return Some("h264");
        }

        if header_index + 1 < data.len() {
            let hevc_type = (header >> 1) & 0x3F;
            if matches!(hevc_type, 1 | 19 | 20 | 32 | 33 | 34) {
                return Some("hevc");
            }
        }

        i = nal_start + 3;
    }

    None
}

#[cfg(feature = "native_engine")]
fn pick_streaming_mux_format(data: &[u8]) -> &'static str {
    match infer_annex_b_bitstream_format(data) {
        Some(format) => format,
        None => {
            tracing::warn!(
                bytes = data.len(),
                "Could not infer Annex B bitstream format from packet; defaulting streaming mux to HEVC"
            );
            "hevc"
        }
    }
}

#[cfg(feature = "native_engine")]
impl videoforge_engine::codecs::nvenc::BitstreamSink for StreamingMuxSink {
    fn write_packet(
        &mut self,
        data: &[u8],
        _pts: i64,
        _is_keyframe: bool,
    ) -> videoforge_engine::error::Result<()> {
        use std::io::Write;
        let bitstream_format = self
            .codec_hint
            .get()
            .unwrap_or_else(|| pick_streaming_mux_format(data));
        let stdin = self.ensure_started(bitstream_format)?;
        stdin
            .write_all(data)
            .map_err(|e| videoforge_engine::error::EngineError::Encode(e.to_string()))
    }

    fn flush(&mut self) -> videoforge_engine::error::Result<()> {
        if self.child.is_none() {
            return Err(videoforge_engine::error::EngineError::Encode(
                "Streaming mux never received any packets".into(),
            ));
        }
        drop(self.stdin.take());
        let child = self.child.take().ok_or_else(|| {
            videoforge_engine::error::EngineError::Encode("Mux process missing".into())
        })?;
        let output = child.wait_with_output().map_err(|e| {
            videoforge_engine::error::EngineError::Encode(format!(
                "FFmpeg mux wait failed: {e}"
            ))
        })?;
        if !output.status.success() {
            return Err(videoforge_engine::error::EngineError::Encode(format!(
                "FFmpeg mux failed with {}: {}",
                output.status,
                self.stderr_tail.snapshot()
            )));
        }
        Ok(())
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
        let host = frame.texture.data.copy_to_host_sync(&self.ctx).map_err(|e| {
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
    mux_codec_hint: StreamingCodecHint,
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
        mux_codec_hint: StreamingCodecHint,
    ) -> videoforge_engine::error::Result<Self> {
        let fallback = NativeVideoEncoderConfig {
            ctx,
            ffmpeg_cmd,
            output_path,
            enc_config: config.clone(),
        };
        let detail = NativeEncoderDetailHandle::default();
        match videoforge_engine::codecs::nvenc::NvEncoder::new(cuda_ctx, sink, config) {
            Ok(enc) => {
                mux_codec_hint.set(enc.output_codec_name());
                Ok(Self {
                    inner: Some(NativeVideoEncoder::Nvenc(enc)),
                    fallback,
                    frames_encoded: 0,
                    mode: NativeEncoderModeHandle::new(NativeEncoderModeHandle::NVENC),
                    detail,
                    mux_codec_hint,
                })
            }
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
                    self.mux_codec_hint.set(enc.output_codec_name());
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
