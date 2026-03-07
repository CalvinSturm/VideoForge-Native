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
use std::ffi::OsString;
use std::path::{Path, PathBuf};
#[cfg(feature = "native_engine")]
use std::collections::VecDeque;
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

pub(crate) fn workspace_root() -> Option<PathBuf> {
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

pub(crate) fn find_file_under(root: &Path, file_name: &str, max_depth: usize) -> Option<PathBuf> {
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

#[derive(Debug, Clone)]
pub(crate) struct NativeRuntimePaths {
    pub ffmpeg_cmd: String,
    pub ffprobe_cmd: String,
    pub path_additions: Vec<PathBuf>,
    pub tensorrt_bin: Option<PathBuf>,
}

pub(crate) fn resolve_native_runtime_paths(
    workspace_root: Option<&Path>,
    extra_path_dir: Option<&Path>,
) -> NativeRuntimePaths {
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

    if let Some(root) = workspace_root {
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
        } else if let Some(nvinfer) = find_file_under(&root.join("third_party"), "nvinfer_10.dll", 4)
        {
            tensorrt_bin = nvinfer.parent().map(|p| p.to_path_buf());
        }
    }

    let mut path_additions = Vec::new();
    let mut push_unique = |path: PathBuf| {
        if path.exists() && !path_additions.iter().any(|p| p == &path) {
            path_additions.push(path);
        }
    };

    if let Some(dir) = &tensorrt_bin {
        push_unique(dir.clone());
    }
    if let Some(dir) = &ffmpeg_bin {
        push_unique(dir.clone());
    }
    if let Ok(vcpkg_root) = std::env::var("VCPKG_ROOT") {
        push_unique(
            PathBuf::from(vcpkg_root)
                .join("installed")
                .join("x64-windows")
                .join("bin"),
        );
    }
    push_unique(PathBuf::from(r"C:\tools\vcpkg\installed\x64-windows\bin"));
    if let Some(extra_dir) = extra_path_dir {
        push_unique(extra_dir.to_path_buf());
    }

    let ffmpeg_cmd = ffmpeg_bin
        .as_ref()
        .map(|bin| bin.join(ffmpeg_exe).to_string_lossy().to_string())
        .unwrap_or_else(|| "ffmpeg".to_string());
    let ffprobe_cmd = ffmpeg_bin
        .as_ref()
        .map(|bin| bin.join(ffprobe_exe).to_string_lossy().to_string())
        .unwrap_or_else(|| "ffprobe".to_string());

    NativeRuntimePaths {
        ffmpeg_cmd,
        ffprobe_cmd,
        path_additions,
        tensorrt_bin,
    }
}

#[cfg(feature = "native_engine")]
struct NativeRuntimeEnv {
    ffmpeg_cmd: String,
    ffprobe_cmd: String,
}

#[cfg(feature = "native_engine")]
static NATIVE_RUNTIME_ENV: OnceLock<NativeRuntimeEnv> = OnceLock::new();

#[cfg(feature = "native_engine")]
fn discover_native_runtime_env() -> NativeRuntimeEnv {
    let runtime_paths = resolve_native_runtime_paths(workspace_root().as_deref(), None);

    if let Some(dir) = &runtime_paths.tensorrt_bin {

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
    prepend_path_dirs(&runtime_paths.path_additions);

    NativeRuntimeEnv {
        ffmpeg_cmd: runtime_paths.ffmpeg_cmd,
        ffprobe_cmd: runtime_paths.ffprobe_cmd,
    }
}

#[cfg(feature = "native_engine")]
fn native_runtime_env() -> &'static NativeRuntimeEnv {
    NATIVE_RUNTIME_ENV.get_or_init(discover_native_runtime_env)
}

#[cfg(feature = "native_engine")]
fn configure_native_runtime_env() -> String {
    native_runtime_env().ffmpeg_cmd.clone()
}

#[cfg(feature = "native_engine")]
fn configure_native_probe_cmd() -> String {
    native_runtime_env().ffprobe_cmd.clone()
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
    let ffprobe_cmd = configure_native_probe_cmd();
    let output = std::process::Command::new(&ffprobe_cmd)
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
        .map_err(|e| format!("ffprobe launch failed via {ffprobe_cmd}: {e}"))?;

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
pub struct NativePerfReport {
    pub frames_processed: u64,
    pub effective_max_batch: u32,
    pub trt_cache_enabled: bool,
    pub trt_cache_dir: Option<String>,
    pub requested_executor: Option<String>,
    pub executed_executor: Option<String>,
    pub direct_attempted: bool,
    pub fallback_used: bool,
    pub fallback_reason_code: Option<String>,
    pub fallback_reason_message: Option<String>,
    pub total_elapsed_ms: Option<u64>,
    pub frames_decoded: Option<u64>,
    pub frames_preprocessed: Option<u64>,
    pub frames_inferred: Option<u64>,
    pub frames_encoded: Option<u64>,
    pub preprocess_avg_us: Option<u64>,
    pub inference_frame_avg_us: Option<u64>,
    pub inference_dispatch_avg_us: Option<u64>,
    pub postprocess_frame_avg_us: Option<u64>,
    pub postprocess_dispatch_avg_us: Option<u64>,
    pub encode_avg_us: Option<u64>,
    pub vram_current_mb: Option<u64>,
    pub vram_peak_mb: Option<u64>,
}

/// Structured response returned by `upscale_request_native`.
#[derive(Debug, Serialize, Deserialize)]
pub struct NativeUpscaleResult {
    pub output_path: String,
    pub engine: String,
    pub encoder_mode: String,
    pub encoder_detail: Option<String>,
    pub audio_preserved: bool,
    #[serde(flatten)]
    pub perf: NativePerfReport,
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

#[derive(Debug, Clone, Default)]
pub struct NativeRuntimeOverrides {
    pub enable_native: bool,
    pub direct: Option<bool>,
    pub trt_cache_enabled: Option<bool>,
    #[cfg(feature = "native_engine")]
    pub trt_cache_dir: Option<PathBuf>,
}

pub struct NativeRuntimeEnvGuard {
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl NativeRuntimeOverrides {
    pub fn native_command(direct: bool) -> Self {
        Self {
            enable_native: true,
            direct: Some(direct),
            trt_cache_enabled: None,
            #[cfg(feature = "native_engine")]
            trt_cache_dir: None,
        }
    }

    #[cfg(feature = "native_engine")]
    pub fn with_trt_cache(mut self, enabled: bool, dir: Option<PathBuf>) -> Self {
        self.trt_cache_enabled = Some(enabled);
        self.trt_cache_dir = dir;
        self
    }

    pub fn apply(&self) -> NativeRuntimeEnvGuard {
        let vars: Vec<(&'static str, Option<OsString>)> = vec![
            (
                "VIDEOFORGE_ENABLE_NATIVE_ENGINE",
                if self.enable_native {
                    Some(OsString::from("1"))
                } else {
                    None
                },
            ),
            (
                "VIDEOFORGE_NATIVE_ENGINE_DIRECT",
                self.direct
                    .and_then(|enabled| enabled.then(|| OsString::from("1"))),
            ),
            (
                "VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE",
                self.trt_cache_enabled
                    .and_then(|enabled| enabled.then(|| OsString::from("1"))),
            ),
            (
                "VIDEOFORGE_TRT_CACHE_DIR",
                {
                    #[cfg(feature = "native_engine")]
                    {
                        self.trt_cache_dir
                            .as_ref()
                            .map(|p| p.as_os_str().to_os_string())
                    }
                    #[cfg(not(feature = "native_engine"))]
                    {
                        None
                    }
                },
            ),
        ];

        let mut saved = Vec::with_capacity(vars.len());
        for (key, value) in vars {
            saved.push((key, std::env::var_os(key)));
            match value {
                Some(v) => unsafe { std::env::set_var(key, v) },
                None => unsafe { std::env::remove_var(key) },
            }
        }

        NativeRuntimeEnvGuard { saved }
    }
}

impl Drop for NativeRuntimeEnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.saved.drain(..).rev() {
            match value {
                Some(v) => unsafe { std::env::set_var(key, v) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
pub struct NativeToolRunRequest {
    pub input_path: String,
    pub output_path: String,
    pub model_path: String,
    pub scale: u32,
    pub precision: String,
    pub preserve_audio: bool,
    pub max_batch: Option<u32>,
    pub native_direct: bool,
    pub trt_cache_dir: Option<PathBuf>,
}

#[cfg(feature = "native_engine")]
impl NativeToolRunRequest {
    pub fn runtime_overrides(&self) -> NativeRuntimeOverrides {
        NativeRuntimeOverrides::native_command(self.native_direct).with_trt_cache(
            self.trt_cache_dir.is_some(),
            self.trt_cache_dir.clone(),
        )
    }
}

#[cfg(feature = "native_engine")]
pub async fn run_native_tool_request(
    request: NativeToolRunRequest,
) -> Result<NativeUpscaleResult, String> {
    let _runtime_env = request.runtime_overrides().apply();
    upscale_request_native(
        request.input_path,
        request.output_path,
        request.model_path,
        request.scale,
        Some(request.precision),
        Some(request.preserve_audio),
        request.max_batch,
    )
    .await
}

#[cfg(feature = "native_engine")]
pub fn native_result_summary_json(report: &NativeUpscaleResult) -> serde_json::Map<String, serde_json::Value> {
    use serde_json::{Map, Value};

    let mut map = Map::new();
    map.insert("output".to_string(), Value::String(report.output_path.clone()));
    map.insert("engine".to_string(), Value::String(report.engine.clone()));
    map.insert("encoder_mode".to_string(), Value::String(report.encoder_mode.clone()));
    map.insert(
        "encoder_detail".to_string(),
        report
            .encoder_detail
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "audio_preserved".to_string(),
        Value::Bool(report.audio_preserved),
    );
    map.insert(
        "requested_executor".to_string(),
        report
            .perf
            .requested_executor
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "executed_executor".to_string(),
        report
            .perf
            .executed_executor
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "direct_attempted".to_string(),
        Value::Bool(report.perf.direct_attempted),
    );
    map.insert(
        "fallback_used".to_string(),
        Value::Bool(report.perf.fallback_used),
    );
    map.insert(
        "fallback_reason_code".to_string(),
        report
            .perf
            .fallback_reason_code
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "fallback_reason_message".to_string(),
        report
            .perf
            .fallback_reason_message
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "frames_processed".to_string(),
        Value::from(report.perf.frames_processed),
    );
    map.insert(
        "native_total_elapsed_ms".to_string(),
        report
            .perf
            .total_elapsed_ms
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "frames_decoded".to_string(),
        report.perf.frames_decoded.map(Value::from).unwrap_or(Value::Null),
    );
    map.insert(
        "frames_preprocessed".to_string(),
        report
            .perf
            .frames_preprocessed
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "frames_inferred".to_string(),
        report.perf.frames_inferred.map(Value::from).unwrap_or(Value::Null),
    );
    map.insert(
        "frames_encoded".to_string(),
        report.perf.frames_encoded.map(Value::from).unwrap_or(Value::Null),
    );
    map.insert(
        "preprocess_avg_us".to_string(),
        report.perf.preprocess_avg_us.map(Value::from).unwrap_or(Value::Null),
    );
    map.insert(
        "inference_frame_avg_us".to_string(),
        report
            .perf
            .inference_frame_avg_us
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "inference_dispatch_avg_us".to_string(),
        report
            .perf
            .inference_dispatch_avg_us
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "postprocess_frame_avg_us".to_string(),
        report
            .perf
            .postprocess_frame_avg_us
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "postprocess_dispatch_avg_us".to_string(),
        report
            .perf
            .postprocess_dispatch_avg_us
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    map.insert(
        "encode_avg_us".to_string(),
        report.perf.encode_avg_us.map(Value::from).unwrap_or(Value::Null),
    );
    map.insert(
        "vram_current_mb".to_string(),
        report.perf.vram_current_mb.map(Value::from).unwrap_or(Value::Null),
    );
    map.insert(
        "vram_peak_mb".to_string(),
        report.perf.vram_peak_mb.map(Value::from).unwrap_or(Value::Null),
    );
    map.insert(
        "effective_max_batch".to_string(),
        Value::from(report.perf.effective_max_batch),
    );
    map.insert(
        "trt_cache_enabled".to_string(),
        Value::Bool(report.perf.trt_cache_enabled),
    );
    map.insert(
        "trt_cache_dir".to_string(),
        report
            .perf
            .trt_cache_dir
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    map
}

#[cfg(feature = "native_engine")]
pub fn native_result_summary_lines(report: &NativeUpscaleResult) -> Vec<String> {
    let mut lines = vec![
        format!(
            "frames={} encoder_mode={} encoder_detail={}",
            report.perf.frames_processed,
            report.encoder_mode,
            report.encoder_detail.as_deref().unwrap_or("none")
        ),
    ];
    if let Some(elapsed_ms) = report.perf.total_elapsed_ms {
        lines.push(format!("native elapsed ms: {elapsed_ms}"));
    }
    if let Some(vram_peak_mb) = report.perf.vram_peak_mb {
        lines.push(format!("native peak vram mb: {vram_peak_mb}"));
    }
    if let Some(requested) = &report.perf.requested_executor {
        lines.push(format!("requested executor: {requested}"));
    }
    if let Some(executed) = &report.perf.executed_executor {
        lines.push(format!("executed executor: {executed}"));
    }
    if report.perf.fallback_used {
        lines.push(format!(
            "fallback: {} {}",
            report
                .perf
                .fallback_reason_code
                .as_deref()
                .unwrap_or("unknown"),
            report
                .perf
                .fallback_reason_message
                .as_deref()
                .unwrap_or("no message")
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::{
        NativeRuntimeOverrides, NativeToolRunRequest, NativeVideoOutputProfile,
        NativeVideoSourceProfile,
    };
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};

    fn env_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn native_runtime_env_guard_restores_previous_values() {
        let _guard = env_test_lock().lock().expect("env test lock");
        let direct_key = "VIDEOFORGE_NATIVE_ENGINE_DIRECT";
        let cache_dir_key = "VIDEOFORGE_TRT_CACHE_DIR";

        unsafe {
            std::env::set_var("VIDEOFORGE_ENABLE_NATIVE_ENGINE", "0");
            std::env::set_var(direct_key, "0");
            std::env::set_var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE", "0");
            std::env::set_var(cache_dir_key, "before");
        }

        {
            #[cfg(feature = "native_engine")]
            let _runtime = NativeRuntimeOverrides::native_command(true)
                .with_trt_cache(true, Some(PathBuf::from("cache-dir")))
                .apply();

            assert_eq!(
                std::env::var("VIDEOFORGE_ENABLE_NATIVE_ENGINE").as_deref(),
                Ok("1")
            );
            assert_eq!(std::env::var(direct_key).as_deref(), Ok("1"));
            assert_eq!(
                std::env::var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE").as_deref(),
                Ok("1")
            );
            assert_eq!(std::env::var(cache_dir_key).as_deref(), Ok("cache-dir"));
        }

        assert_eq!(
            std::env::var("VIDEOFORGE_ENABLE_NATIVE_ENGINE").as_deref(),
            Ok("0")
        );
        assert_eq!(std::env::var(direct_key).as_deref(), Ok("0"));
        assert_eq!(
            std::env::var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE").as_deref(),
            Ok("0")
        );
        assert_eq!(std::env::var(cache_dir_key).as_deref(), Ok("before"));
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn source_profile_derives_expected_output_profile() {
        let source = NativeVideoSourceProfile {
            coded_width: 1920,
            coded_height: 1080,
            fps: 23.976,
            total_frames: 240,
            codec: videoforge_engine::codecs::sys::cudaVideoCodec::H264,
        };

        let output: NativeVideoOutputProfile = source.scaled_output(2);
        assert_eq!(output.width, 3840);
        assert_eq!(output.height, 2160);
        assert_eq!(output.nv12_pitch, 3840);
        assert_eq!(output.fps_num, 23_976);
        assert_eq!(output.fps_den, 1000);
        assert_eq!(output.frame_rate_arg(), "23976/1000");
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn tool_request_runtime_overrides_follow_cache_presence() {
        let _guard = env_test_lock().lock().expect("env test lock");
        let req = NativeToolRunRequest {
            input_path: "in.mp4".to_string(),
            output_path: "out.mp4".to_string(),
            model_path: "model.onnx".to_string(),
            scale: 2,
            precision: "fp16".to_string(),
            preserve_audio: true,
            max_batch: Some(4),
            native_direct: true,
            trt_cache_dir: Some(PathBuf::from("cache-dir")),
        };

        {
            let _runtime = req.runtime_overrides().apply();
            assert_eq!(
                std::env::var("VIDEOFORGE_ENABLE_NATIVE_ENGINE").as_deref(),
                Ok("1")
            );
            assert_eq!(
                std::env::var("VIDEOFORGE_NATIVE_ENGINE_DIRECT").as_deref(),
                Ok("1")
            );
            assert_eq!(
                std::env::var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE").as_deref(),
                Ok("1")
            );
            assert_eq!(std::env::var("VIDEOFORGE_TRT_CACHE_DIR").as_deref(), Ok("cache-dir"));
        }
    }
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
struct NativeExecutionRoute {
    requested_executor: &'static str,
    executed_executor: &'static str,
    direct_attempted: bool,
    fallback_reason_code: Option<String>,
    fallback_reason_message: Option<String>,
}

#[cfg(feature = "native_engine")]
impl NativeExecutionRoute {
    fn direct() -> Self {
        Self {
            requested_executor: "direct",
            executed_executor: "direct",
            direct_attempted: true,
            fallback_reason_code: None,
            fallback_reason_message: None,
        }
    }

    fn cli_requested() -> Self {
        Self {
            requested_executor: "cli",
            executed_executor: "cli",
            direct_attempted: false,
            fallback_reason_code: None,
            fallback_reason_message: None,
        }
    }

    fn cli_fallback(err: &NativeUpscaleError) -> Self {
        Self {
            requested_executor: "direct",
            executed_executor: "cli",
            direct_attempted: true,
            fallback_reason_code: Some(err.code.clone()),
            fallback_reason_message: Some(err.message.clone()),
        }
    }
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeRequestedExecutor {
    Direct,
    Cli,
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeOutputPathStyle {
    DirectTemp,
    CliStable,
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
struct NativeCliInvocation {
    output_path: String,
    args: Vec<String>,
}

#[cfg(feature = "native_engine")]
struct NativeCliExecutionPlan {
    output_path: String,
    prepared_command: crate::commands::rave::PreparedRaveCliCommand,
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
struct NativeJobSpec {
    input_path: String,
    requested_output_path: String,
    model_path: String,
    scale: u32,
    precision: String,
    preserve_audio: bool,
    max_batch: u32,
    trt_cache_enabled: bool,
    trt_cache_dir: Option<String>,
}

#[cfg(feature = "native_engine")]
impl NativeJobSpec {
    fn resolve(
        input_path: String,
        output_path: String,
        model_path: String,
        requested_scale: u32,
        precision: Option<String>,
        audio: Option<bool>,
        requested_max_batch: Option<u32>,
    ) -> Result<Self, String> {
        let make_err =
            |code: &str, msg: &str| serde_json::to_string(&NativeUpscaleError::new(code, msg)).unwrap();

        if !Path::new(&model_path).exists() {
            return Err(make_err(
                "MODEL_NOT_FOUND",
                &format!("Model not found: {}", model_path),
            ));
        }

        let batch_policy = crate::models::native_batch_policy_for_path(&model_path);
        let max_batch = requested_max_batch.unwrap_or(batch_policy.default_max_batch);
        if !(1..=8).contains(&max_batch) {
            return Err(make_err(
                "INVALID_BATCH",
                &format!("Invalid max_batch value '{max_batch}'. Must be in range 1-8."),
            ));
        }

        let effective_scale = infer_model_scale(&model_path).unwrap_or(requested_scale);
        if effective_scale != requested_scale {
            tracing::warn!(
                requested_scale,
                inferred_scale = effective_scale,
                model = %model_path,
                "Requested native scale does not match model filename; overriding to inferred model scale"
            );
        }
        if requested_max_batch.is_none() {
            tracing::info!(
                model = %model_path,
                default_max_batch = batch_policy.default_max_batch,
                max_validated_batch = batch_policy.max_validated_batch,
                "Applying model-aware native batching default"
            );
        } else {
            tracing::info!(
                model = %model_path,
                requested_max_batch = max_batch,
                default_max_batch = batch_policy.default_max_batch,
                max_validated_batch = batch_policy.max_validated_batch,
                "Using explicit native batching override"
            );
        }

        let (trt_cache_enabled, trt_cache_dir) = trt_cache_runtime(&model_path);

        Ok(Self {
            input_path,
            requested_output_path: output_path,
            model_path,
            scale: effective_scale,
            precision: precision.unwrap_or_else(|| "fp32".to_string()),
            preserve_audio: audio.unwrap_or(true),
            max_batch,
            trt_cache_enabled,
            trt_cache_dir,
        })
    }

    fn resolved_output_path(&self, style: NativeOutputPathStyle) -> String {
        if !self.requested_output_path.trim().is_empty() {
            return self.requested_output_path.clone();
        }

        match style {
            NativeOutputPathStyle::CliStable => self.default_cli_output_path(),
            NativeOutputPathStyle::DirectTemp => self.default_direct_output_path(),
        }
    }

    fn default_cli_output_path(&self) -> String {
        let p = Path::new(&self.input_path);
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
            self.input_path
                .replace(&format!(".{ext}"), "_rave_upscaled.mp4")
        }
    }

    fn default_direct_output_path(&self) -> String {
        let stem = Path::new(&self.input_path)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let dir = Path::new(&self.input_path)
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();
        let file_name = format!("{}_{}_native_upscaled.mp4", stem, native_temp_token());
        dir.join(file_name).to_string_lossy().to_string()
    }

    fn prepare_cli_invocation(&self) -> NativeCliInvocation {
        let output_path = self.resolved_output_path(NativeOutputPathStyle::CliStable);
        let mut args = vec![
            "-i".to_string(),
            self.input_path.clone(),
            "-m".to_string(),
            self.model_path.clone(),
            "-o".to_string(),
            output_path.clone(),
            "--precision".to_string(),
            self.precision.clone(),
            "--progress".to_string(),
            "jsonl".to_string(),
        ];
        if self.max_batch > 1 {
            args.push("--max-batch".to_string());
            args.push(self.max_batch.to_string());
        }
        NativeCliInvocation { output_path, args }
    }

    fn prepare_cli_execution(&self) -> Result<NativeCliExecutionPlan, String> {
        let invocation = self.prepare_cli_invocation();
        let prepared_command = crate::commands::rave::prepare_rave_upscale_command(
            invocation.args,
            true,
            false,
            true,
        )?;
        Ok(NativeCliExecutionPlan {
            output_path: invocation.output_path,
            prepared_command,
        })
    }

    fn prepare_direct_plan(&self, ffmpeg_cmd: String) -> Result<NativeDirectPlan, String> {
        use videoforge_engine::codecs::sys::cudaVideoCodec as CudaCodec;

        NativeDirectPlan::prepare(self, ffmpeg_cmd, CudaCodec::H264)
    }

    fn base_perf(&self, frames_processed: u64) -> NativePerfReport {
        NativePerfReport {
            frames_processed,
            effective_max_batch: self.max_batch,
            trt_cache_enabled: self.trt_cache_enabled,
            trt_cache_dir: self.trt_cache_dir.clone(),
            requested_executor: None,
            executed_executor: None,
            direct_attempted: false,
            fallback_used: false,
            fallback_reason_code: None,
            fallback_reason_message: None,
            total_elapsed_ms: None,
            frames_decoded: None,
            frames_preprocessed: None,
            frames_inferred: None,
            frames_encoded: None,
            preprocess_avg_us: None,
            inference_frame_avg_us: None,
            inference_dispatch_avg_us: None,
            postprocess_frame_avg_us: None,
            postprocess_dispatch_avg_us: None,
            encode_avg_us: None,
            vram_current_mb: None,
            vram_peak_mb: None,
        }
    }

    fn build_result(
        &self,
        output_path: String,
        engine: impl Into<String>,
        encoder_mode: impl Into<String>,
        encoder_detail: Option<String>,
        mut perf: NativePerfReport,
        route: NativeExecutionRoute,
    ) -> NativeUpscaleResult {
        perf.requested_executor = Some(route.requested_executor.to_string());
        perf.executed_executor = Some(route.executed_executor.to_string());
        perf.direct_attempted = route.direct_attempted;
        perf.fallback_used = route.fallback_reason_code.is_some();
        perf.fallback_reason_code = route.fallback_reason_code;
        perf.fallback_reason_message = route.fallback_reason_message;

        NativeUpscaleResult {
            output_path,
            engine: engine.into(),
            encoder_mode: encoder_mode.into(),
            encoder_detail,
            audio_preserved: self.preserve_audio,
            perf,
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

pub(crate) fn native_engine_runtime_enabled() -> bool {
    match std::env::var("VIDEOFORGE_ENABLE_NATIVE_ENGINE") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

#[cfg(feature = "native_engine")]
pub(crate) fn native_engine_direct_enabled() -> bool {
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
fn build_cli_perf_report(
    job: &NativeJobSpec,
    res: &serde_json::Value,
    progress: Option<&crate::rave_cli::RaveProgressSummary>,
) -> NativePerfReport {
    let mut perf = job.base_perf(progress.map(|p| p.frames_encoded).unwrap_or(0));
    perf.total_elapsed_ms = res
        .get("elapsed_ms")
        .and_then(|v| v.as_u64())
        .or_else(|| progress.map(|p| p.elapsed_ms));
    perf.frames_decoded = progress.map(|p| p.frames_decoded);
    perf.frames_inferred = progress.map(|p| p.frames_inferred);
    perf.frames_encoded = progress.map(|p| p.frames_encoded);
    perf.vram_current_mb = res.get("vram_current_mb").and_then(|v| v.as_u64());
    perf.vram_peak_mb = res.get("vram_peak_mb").and_then(|v| v.as_u64());
    perf
}

#[cfg(feature = "native_engine")]
fn build_direct_perf_report(
    job: &NativeJobSpec,
    metrics: &videoforge_engine::engine::pipeline::PipelineMetrics,
    total_elapsed_ms: u64,
    vram_current_bytes: usize,
    vram_peak_bytes: usize,
) -> NativePerfReport {
    use std::sync::atomic::Ordering;

    let frames_decoded = metrics.frames_decoded.load(Ordering::Relaxed);
    let frames_preprocessed = metrics.frames_preprocessed.load(Ordering::Relaxed);
    let frames_inferred = metrics.frames_inferred.load(Ordering::Relaxed);
    let frames_encoded = metrics.frames_encoded.load(Ordering::Relaxed);
    let inference_dispatches = metrics.inference_dispatches.load(Ordering::Relaxed);
    let postprocess_dispatches = metrics.postprocess_dispatches.load(Ordering::Relaxed);
    let avg = |total: &std::sync::atomic::AtomicU64, count: u64| -> Option<u64> {
        if count > 0 {
            Some(total.load(Ordering::Relaxed) / count)
        } else {
            None
        }
    };

    let mut perf = job.base_perf(frames_encoded);
    perf.total_elapsed_ms = Some(total_elapsed_ms);
    perf.frames_decoded = Some(frames_decoded);
    perf.frames_preprocessed = Some(frames_preprocessed);
    perf.frames_inferred = Some(frames_inferred);
    perf.frames_encoded = Some(frames_encoded);
    perf.preprocess_avg_us = avg(&metrics.preprocess_total_us, frames_preprocessed);
    perf.inference_frame_avg_us = avg(&metrics.inference_total_us, frames_inferred);
    perf.inference_dispatch_avg_us = avg(&metrics.inference_total_us, inference_dispatches);
    perf.postprocess_frame_avg_us = avg(&metrics.postprocess_total_us, frames_inferred);
    perf.postprocess_dispatch_avg_us = avg(&metrics.postprocess_total_us, postprocess_dispatches);
    perf.encode_avg_us = avg(&metrics.encode_total_us, frames_encoded);
    perf.vram_current_mb = Some((vram_current_bytes / (1024 * 1024)) as u64);
    perf.vram_peak_mb = Some((vram_peak_bytes / (1024 * 1024)) as u64);
    perf
}

#[cfg(feature = "native_engine")]
fn requested_native_executor() -> NativeRequestedExecutor {
    if native_engine_direct_enabled() {
        NativeRequestedExecutor::Direct
    } else {
        NativeRequestedExecutor::Cli
    }
}

#[cfg(feature = "native_engine")]
async fn run_native_via_rave_cli(
    job: &NativeJobSpec,
    route: NativeExecutionRoute,
) -> Result<NativeUpscaleResult, String> {
    let make_err =
        |code: &str, msg: &str| serde_json::to_string(&NativeUpscaleError::new(code, msg)).unwrap();
    let cli = job.prepare_cli_execution()?;
    let res = crate::commands::rave::run_prepared_rave_upscale(cli.prepared_command).await?;
    let output = res
        .json
        .get("output")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            make_err(
                "RAVE_CONTRACT",
                "rave_upscale did not return a valid output path",
            )
        })?
        .to_string();
    if output != cli.output_path {
        tracing::warn!(
            expected_output = %cli.output_path,
            actual_output = %output,
            "CLI-native returned an output path that differs from the prepared native job output"
        );
    }

    let perf = build_cli_perf_report(job, &res.json, res.progress.as_ref());

    Ok(job.build_result(
        output,
        "native_via_rave_cli",
        "rave_cli",
        None,
        perf,
        route,
    ))
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

#[cfg(feature = "native_engine")]
async fn run_native_job(job: NativeJobSpec) -> Result<NativeUpscaleResult, String> {
    match requested_native_executor() {
        NativeRequestedExecutor::Direct => run_direct_with_fallback(job).await,
        NativeRequestedExecutor::Cli => {
            run_native_via_rave_cli(&job, NativeExecutionRoute::cli_requested()).await
        }
    }
}

#[cfg(feature = "native_engine")]
async fn run_direct_with_fallback(job: NativeJobSpec) -> Result<NativeUpscaleResult, String> {
    let direct_result = run_native_pipeline(&job).await;

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
                run_native_via_rave_cli(&job, NativeExecutionRoute::cli_fallback(&err)).await
            } else {
                Err(err_json)
            }
        }
    }
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

        let job = NativeJobSpec::resolve(
            input_path,
            output_path,
            model_path,
            scale,
            precision,
            audio,
            max_batch,
        )?;

        run_native_job(job).await
    }
}

// =============================================================================
// Native pipeline implementation (compiled only with native_engine feature)
// =============================================================================

#[cfg(feature = "native_engine")]
async fn run_native_pipeline(
    job: &NativeJobSpec,
) -> Result<NativeUpscaleResult, String> {
    let ffmpeg_cmd = configure_native_runtime_env();
    let plan = job.prepare_direct_plan(ffmpeg_cmd)?;

    tracing::info!(
        input = %job.input_path,
        output = %plan.output_path,
        model = %job.model_path,
        scale = job.scale,
        precision = %job.precision,
        max_batch = job.max_batch,
        estimated_frames = plan.source.total_frames,
        "Native engine pipeline starting"
    );

    run_engine_pipeline(
        &job,
        &plan,
    )
    .await
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
struct NativeVideoSourceProfile {
    coded_width: usize,
    coded_height: usize,
    fps: f64,
    total_frames: u64,
    codec: videoforge_engine::codecs::sys::cudaVideoCodec,
}

#[cfg(feature = "native_engine")]
impl NativeVideoSourceProfile {
    fn probe(input_path: &str) -> Result<Self, String> {
        tracing::info!(path = %input_path, "Probing input video dimensions");
        let (coded_width, coded_height, _duration, fps, total_frames, codec) =
            probe_video_coded_geometry(input_path).map_err(|e| {
                serde_json::to_string(&NativeUpscaleError::new("PROBE_FAILED", &e)).unwrap()
            })?;

        tracing::info!(
            coded_width,
            coded_height,
            fps,
            total_frames,
            codec = ?codec,
            "Native video source profile resolved"
        );

        Ok(Self {
            coded_width,
            coded_height,
            fps,
            total_frames,
            codec,
        })
    }

    fn mux_codec_hint(&self) -> StreamingCodecHint {
        StreamingCodecHint::new(match self.codec {
            videoforge_engine::codecs::sys::cudaVideoCodec::H264 => Some("h264"),
            videoforge_engine::codecs::sys::cudaVideoCodec::HEVC => Some("hevc"),
            _ => None,
        })
    }

    fn scaled_output(&self, scale: u32) -> NativeVideoOutputProfile {
        let width = self.coded_width.saturating_mul(scale as usize);
        let height = self.coded_height.saturating_mul(scale as usize);
        NativeVideoOutputProfile {
            width,
            height,
            nv12_pitch: width.div_ceil(256) * 256,
            fps_num: (self.fps * 1000.0).round() as u32,
            fps_den: 1000u32,
        }
    }
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
struct NativeVideoOutputProfile {
    width: usize,
    height: usize,
    nv12_pitch: usize,
    fps_num: u32,
    fps_den: u32,
}

#[cfg(feature = "native_engine")]
impl NativeVideoOutputProfile {
    fn frame_rate_arg(&self) -> String {
        format!("{}/{}", self.fps_num, self.fps_den.max(1))
    }

    fn nvenc_config(&self) -> videoforge_engine::codecs::nvenc::NvEncConfig {
        videoforge_engine::codecs::nvenc::NvEncConfig {
            width: self.width as u32,
            height: self.height as u32,
            fps_num: self.fps_num,
            fps_den: self.fps_den,
            bitrate: 8_000_000,
            max_bitrate: 0,
            gop_length: 30,
            b_frames: 0,
            nv12_pitch: self.nv12_pitch as u32,
        }
    }

    fn pipeline_config(
        &self,
        model_precision: videoforge_engine::core::kernels::ModelPrecision,
        inference_max_batch: usize,
    ) -> videoforge_engine::engine::pipeline::PipelineConfig {
        videoforge_engine::engine::pipeline::PipelineConfig {
            model_precision,
            encoder_nv12_pitch: self.nv12_pitch,
            inference_max_batch,
            ..videoforge_engine::engine::pipeline::PipelineConfig::default()
        }
    }
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
struct NativeDirectPlan {
    ffmpeg_cmd: String,
    output_path: String,
    source: NativeVideoSourceProfile,
    output: NativeVideoOutputProfile,
    mux_codec_hint: StreamingCodecHint,
}

#[cfg(feature = "native_engine")]
impl NativeDirectPlan {
    fn prepare(
        job: &NativeJobSpec,
        ffmpeg_cmd: String,
        requested_codec: videoforge_engine::codecs::sys::cudaVideoCodec,
    ) -> Result<Self, String> {
        let output_path = job.resolved_output_path(NativeOutputPathStyle::DirectTemp);
        let source = NativeVideoSourceProfile::probe(&job.input_path)?;
        if source.codec != requested_codec {
            tracing::warn!(
                requested = ?requested_codec,
                probed = ?source.codec,
                "Native codec routing differed from ffprobe result; using probed codec for streamed demux"
            );
        }

        let output = source.scaled_output(job.scale);

        tracing::info!(
            input_w = source.coded_width,
            input_h = source.coded_height,
            output_w = output.width,
            output_h = output.height,
            encoder_nv12_pitch = output.nv12_pitch,
            fps = source.fps,
            "Video dimensions resolved"
        );

        let mux_codec_hint = source.mux_codec_hint();

        Ok(Self {
            ffmpeg_cmd,
            output_path,
            source,
            output,
            mux_codec_hint,
        })
    }

    fn nvenc_config(&self) -> videoforge_engine::codecs::nvenc::NvEncConfig {
        self.output.nvenc_config()
    }

    fn pipeline_config(
        &self,
        model_precision: videoforge_engine::core::kernels::ModelPrecision,
        inference_max_batch: usize,
    ) -> videoforge_engine::engine::pipeline::PipelineConfig {
        self.output
            .pipeline_config(model_precision, inference_max_batch)
    }
}

#[cfg(feature = "native_engine")]
async fn run_engine_pipeline(
    job: &NativeJobSpec,
    plan: &NativeDirectPlan,
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
    let started = std::time::Instant::now();

    // ── Step 2: Initialise GPU context ────────────────────────────────────────
    tracing::info!("Initialising GPU context (device 0)");
    // GpuContext::new already returns Arc<GpuContext>.
    let ctx = GpuContext::new(0)
        .map_err(|e| make_err("GPU_INIT", &format!("GPU context creation failed: {}", e)))?;

    // ── Step 3: Load TensorRT backend ─────────────────────────────────────────
    tracing::info!(model = %job.model_path, "Loading TensorRT backend");
    let precision_policy = match job.precision.as_str() {
        "fp16" => PrecisionPolicy::Fp16,
        _ => PrecisionPolicy::Fp32,
    };
    let downstream_capacity = 4usize;
    let ring_size = TensorRtBackend::required_ring_slots(downstream_capacity, job.max_batch as usize);

    // TensorRtBackend::new(model_path, ctx, device_id, ring_size, downstream_capacity).
    // Use with_precision to apply the precision policy and keep output slots batch-safe.
    let backend = TensorRtBackend::with_precision(
        std::path::PathBuf::from(&job.model_path),
        ctx.clone(),
        0, // device_id
        ring_size,
        downstream_capacity,
        precision_policy,
        BatchConfig {
            max_batch: job.max_batch as usize,
            ..BatchConfig::default()
        },
    );
    let backend = Arc::new(backend);
    backend
        .initialize()
        .await
        .map_err(|e| make_err("BACKEND_INIT", &classify_backend_init_error(&job.model_path, &e.to_string())))?;

    // ── Step 4: Compile kernels ───────────────────────────────────────────────
    let kernels = PreprocessKernels::compile(ctx.device())
        .map_err(|e| make_err("KERNEL_COMPILE", &format!("Kernel compile failed: {}", e)))?;
    let kernels = Arc::new(kernels);

    // ── Step 5: Create decoder with FFmpeg-streamed elementary source ────────
    tracing::info!(path = %job.input_path, codec = ?plan.source.codec, "Creating NVDEC decoder");
    let model_prec = match backend
        .metadata()
        .map_err(|e| make_err("BACKEND_INIT", &format!("Model metadata unavailable: {}", e)))?
        .input_format
    {
        videoforge_engine::core::types::PixelFormat::RgbPlanarF16 => ModelPrecision::F16,
        _ => ModelPrecision::F32,
    };
    let source = FfmpegBitstreamSource::spawn(&plan.ffmpeg_cmd, &job.input_path, plan.source.codec)
        .map_err(|e| make_err("SOURCE_OPEN", &format!("Cannot stream elementary input: {}", e)))?;
    let decoder = NvDecoder::new(ctx.clone(), Box::new(source), plan.source.codec)
        .map_err(|e| make_err("DECODER_INIT", &format!("NVDEC decoder init failed: {}", e)))?;

    // ── Step 6: Create encoder with streaming FFmpeg mux sink ────────────────
    tracing::info!(path = %plan.output_path, "Creating NVENC encoder and mux sink");

    let enc_config: NvEncConfig = plan.nvenc_config();
    let sink = StreamingMuxSink::new(
        &plan.ffmpeg_cmd,
        &plan.output_path,
        &job.input_path,
        job.preserve_audio,
        plan.mux_codec_hint.clone(),
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
        plan.ffmpeg_cmd.clone(),
        cuda_ctx,
        Box::new(sink),
        enc_config,
        PathBuf::from(&plan.output_path),
        plan.mux_codec_hint.clone(),
    )
    .map_err(|e| make_err("ENCODER_INIT", &format!("Encoder init failed: {}", e)))?;
    let encoder_mode = encoder.mode_handle();
    let encoder_detail = encoder.detail_handle();

    // ── Step 7: Run the pipeline ──────────────────────────────────────────────
    let config: PipelineConfig = plan.pipeline_config(model_prec, job.max_batch as usize);
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

    let metrics = pipeline.metrics();
    let frames = metrics
        .frames_encoded
        .load(std::sync::atomic::Ordering::Relaxed);
    let encoder_mode = encoder_mode.as_str().to_string();
    let encoder_detail = encoder_detail.get();
    let (vram_current, vram_peak) = ctx.vram_usage();
    let perf = build_direct_perf_report(
        job,
        &metrics,
        started.elapsed().as_millis() as u64,
        vram_current,
        vram_peak,
    );
    tracing::info!(
        frames_encoded = frames,
        encoder_mode = %encoder_mode,
        encoder_detail = encoder_detail.as_deref().unwrap_or("none"),
        "engine-v2 pipeline complete"
    );

    tracing::info!(output = %plan.output_path, "Native engine upscale complete");

    Ok(job.build_result(
        plan.output_path.clone(),
        "native_v2",
        encoder_mode,
        encoder_detail,
        perf,
        NativeExecutionRoute::direct(),
    ))
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
#[derive(Debug, Clone, Default)]
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
        output: &NativeVideoOutputProfile,
    ) -> videoforge_engine::error::Result<Self> {
        use std::process::{Command, Stdio};

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
            .arg(format!("{}x{}", output.width, output.height))
            .arg("-framerate")
            .arg(output.frame_rate_arg())
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
            width: output.width,
            height: output.height,
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
