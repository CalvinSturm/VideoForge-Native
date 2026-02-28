//! Native engine MVP — GPU-resident upscaling via engine-v2.
//!
//! This command is gated behind the `native_engine` Cargo feature.
//! When the feature is **disabled** (the default), the command returns a
//! structured `FEATURE_DISABLED` error immediately.
//!
//! When the feature is **enabled**, the pipeline is:
//! ```text
//! FFmpeg demux → Annex B elementary stream (.h264/.hevc temp file)
//!      ↓
//! engine-v2 NVDEC → TensorRT → NVENC → compressed output temp file
//!      ↓
//! FFmpeg mux: video_output + audio_from_original → final .mp4
//! ```
//!
//! # STOP CONDITION — dependency resolution
//!
//! `engine-v2` depends on `ort = "^2.0"` (ONNX Runtime).  As of 2026-02,
//! `ort 2.0.0` stable has **not** been published to crates.io; only release
//! candidates exist (`2.0.0-rc.11` is the latest).  Cargo refuses the path
//! dependency because rc versions do not satisfy the `^2.0` semver constraint.
//!
//! **Affected files**:
//! - `engine-v2/Cargo.toml` line 13: `ort = { version = "2.0", features = ["cuda", "tensorrt"] }`
//! - `src-tauri/Cargo.toml`: `videoforge-engine` path dep (commented out)
//!
//! **Reproduction**:
//! ```text
//! cd src-tauri && cargo check --features native_engine
//! # error: failed to select a version for the requirement `ort = "^2.0"`
//! # candidate versions found which didn't match: 2.0.0-rc.11, ...
//! ```
//!
//! **Fix before enabling the feature**:
//! 1. In `engine-v2/Cargo.toml` change `ort = { version = "2.0" }` to
//!    `ort = { version = "2.0.0-rc.11" }` (or current rc on crates.io).
//! 2. Uncomment the `videoforge-engine` dep in `src-tauri/Cargo.toml`.
//! 3. Change `native_engine = []` to `native_engine = ["dep:videoforge-engine"]`.
//!
//! The integration code in this file is complete and correct.
//! See `docs/NATIVE_ENGINE_MVP.md` for full instructions.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
fn configure_native_runtime_env() -> String {
    let ffmpeg_exe = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
    let ffprobe_exe = if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" };

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
        } else if let Some(nvinfer) = find_file_under(&root.join("third_party"), "nvinfer_10.dll", 4)
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
                        let Some(name) = src.file_name() else { continue };
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

    if let Some(bin) = ffmpeg_bin {
        return bin.join(ffmpeg_exe).to_string_lossy().to_string();
    }
    "ffmpeg".to_string()
}

/// Structured response returned by `upscale_request_native`.
#[derive(Debug, Serialize, Deserialize)]
pub struct NativeUpscaleResult {
    pub output_path: String,
    pub engine: String,
    pub frames_processed: u64,
    pub audio_preserved: bool,
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
        run_native_pipeline(
            input_path,
            output_path,
            model_path,
            scale,
            precision.unwrap_or_else(|| "fp32".to_string()),
            audio.unwrap_or(true),
            max_batch.unwrap_or(1),
        )
        .await
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
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let dir = Path::new(&input_path)
            .parent()
            .unwrap_or(Path::new("."))
            .display()
            .to_string();
        output_path = format!("{}/{}_{}_native_upscaled.mp4", dir, stem, ts);
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
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let elementary_stream_path = tmp_dir.join(format!("vf_native_input_{}.h264", ts));
    let native_output_path = tmp_dir.join(format!("vf_native_output_{}.h264", ts));

    tracing::info!(
        stream_path = %elementary_stream_path.display(),
        "Extracting H.264 elementary stream with FFmpeg"
    );

    let demux_status = Command::new(&ffmpeg_cmd)
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
        .stderr(Stdio::null())
        .status()
        .await
        .map_err(|e| make_err("FFMPEG_SPAWN", &format!("FFmpeg spawn failed: {}", e)))?;

    if !demux_status.success() {
        // Attempt HEVC fallback
        let hevc_path = tmp_dir.join(format!("vf_native_input_{}.hevc", ts));
        let hevc_status = Command::new(&ffmpeg_cmd)
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
            .stderr(Stdio::null())
            .status()
            .await
            .ok();

        if hevc_status.is_some_and(|s| s.success()) {
            tracing::info!("Falling back to HEVC elementary stream");
            return run_engine_pipeline(
                hevc_path,
                native_output_path,
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

        return Err(make_err(
            "DEMUX_FAILED",
            "FFmpeg could not extract an H.264 or HEVC elementary stream from the input. \
             Only H.264 and HEVC containers are supported by the native engine MVP.",
        ));
    }

    run_engine_pipeline(
        elementary_stream_path,
        native_output_path,
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

    // ── Step 1.5: Probe input for dimensions ──────────────────────────────────
    // encoder_nv12_pitch and encoder width/height must be set before
    // UpscalePipeline::new() — the pipeline asserts pitch > 0.
    tracing::info!(path = %original_input, "Probing input video dimensions");
    let (input_w, input_h, _duration, fps, _) = crate::video_pipeline::probe_video(&original_input)
        .map_err(|e| make_err("PROBE_FAILED", &format!("ffprobe probe failed: {}", e)))?;
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
    backend
        .initialize()
        .await
        .map_err(|e| make_err("BACKEND_INIT", &format!("TensorRT backend init failed: {}", e)))?;

    // ── Step 4: Compile kernels ───────────────────────────────────────────────
    let kernels = PreprocessKernels::compile(ctx.device())
        .map_err(|e| make_err("KERNEL_COMPILE", &format!("Kernel compile failed: {}", e)))?;
    let kernels = Arc::new(kernels);

    // ── Step 5: Create decoder with FileBitstreamSource ───────────────────────
    tracing::info!(path = %input_stream.display(), "Creating NVDEC decoder");
    let model_prec = if precision == "fp16" {
        ModelPrecision::F16
    } else {
        ModelPrecision::F32
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
    let encoder =
        videoforge_engine::codecs::nvenc::NvEncoder::new(cuda_ctx, Box::new(sink), enc_config)
            .map_err(|e| make_err("ENCODER_INIT", &format!("NVENC encoder init failed: {}", e)))?;

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
    tracing::info!(frames_encoded = frames, "engine-v2 pipeline complete");

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

    let mux_status = Command::new(&ffmpeg_cmd)
        .args(&mux_args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map_err(|e| make_err("FFMPEG_MUX_SPAWN", &e.to_string()))?;

    // Clean up temp files regardless of mux result.
    let _ = std::fs::remove_file(&input_stream);
    let _ = std::fs::remove_file(&engine_output);

    if !mux_status.success() {
        return Err(make_err(
            "MUX_FAILED",
            "FFmpeg mux step failed. The native engine output exists but was not muxed. \
             See docs/NATIVE_ENGINE_MVP.md for the 'audio_only_video' workaround.",
        ));
    }

    tracing::info!(output = %final_output, "Native engine upscale complete");

    Ok(NativeUpscaleResult {
        output_path: final_output,
        engine: "native_v2".to_string(),
        frames_processed: frames,
        audio_preserved: preserve_audio,
    })
}

// =============================================================================
// FileBitstreamSource — reads Annex B elementary stream from disk
// =============================================================================

#[cfg(feature = "native_engine")]
struct FileBitstreamSource {
    data: Vec<u8>,
    pos: usize,
    pts_counter: i64,
}

#[cfg(feature = "native_engine")]
impl FileBitstreamSource {
    fn new(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        Ok(Self {
            data,
            pos: 0,
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
        if self.pos >= self.data.len() {
            return Ok(None);
        }

        // Feed in 1 MiB chunks.  The NVDEC parser handles cross-chunk NAL
        // boundaries internally — no client-side start-code splitting needed.
        const CHUNK: usize = 1024 * 1024;
        let end = (self.pos + CHUNK).min(self.data.len());
        let chunk = self.data[self.pos..end].to_vec();

        // Heuristic keyframe detection: IDR NAL type 0x65 after a start code.
        let is_keyframe = chunk.windows(5).any(|w| {
            (w[0] == 0 && w[1] == 0 && w[2] == 0 && w[3] == 1 && (w[4] & 0x1F) == 5)
                || (w[0] == 0 && w[1] == 0 && w[2] == 1 && (w[3] & 0x1F) == 5)
        });

        let pts = self.pts_counter;
        self.pts_counter += 1;
        self.pos = end;

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
