// src-tauri/src/video_pipeline.rs

use anyhow::{Context, Result};
use std::process::Stdio;
use std::sync::OnceLock;
use tokio::io::AsyncReadExt;
use tokio::io::BufReader;
use tokio::process::{Child, Command};
use tokio::time::{timeout, Duration};

// -----------------------------------------------------------------------------
// NVDEC Hardware Decode Probe
// -----------------------------------------------------------------------------

/// Cached result of NVDEC availability check.
static NVDEC_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Check if FFmpeg supports CUDA/NVDEC hardware acceleration.
///
/// Runs `ffmpeg -hwaccels` once and caches the result.
/// Returns true if "cuda" appears in the output.
pub fn probe_nvdec() -> bool {
    *NVDEC_AVAILABLE.get_or_init(|| {
        let result = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-hwaccels"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output();

        match result {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let has_cuda = stdout.lines().any(|line| line.trim() == "cuda");
                eprintln!(
                    "NVDEC probe: {}",
                    if has_cuda {
                        "CUDA available"
                    } else {
                        "CUDA not found, using software decode"
                    }
                );
                has_cuda
            }
            Err(e) => {
                eprintln!("NVDEC probe failed (ffmpeg not found?): {}", e);
                false
            }
        }
    })
}

// -----------------------------------------------------------------------------
// Video Decoder (FFmpeg -> Raw RGB24 Frames)
// -----------------------------------------------------------------------------

#[allow(dead_code)]
pub struct VideoDecoder {
    child: Child,
    stdout: BufReader<tokio::process::ChildStdout>,
    pub total_frames: u32,
    pub using_hwaccel: bool,
}

impl VideoDecoder {
    /// Build the FFmpeg decoder argument list.
    ///
    /// When `hwaccel` is true, prepends `-hwaccel cuda` before the input.
    /// FFmpeg internally transfers NVDEC-decoded GPU frames to system memory
    /// when outputting to a rawvideo pipe (equivalent to `av_hwframe_transfer_data`).
    fn build_args(
        input: &str,
        start: f64,
        duration: f64,
        filter_str: &str,
        hwaccel: bool,
    ) -> Vec<String> {
        let mut args = vec![
            "-hide_banner".to_string(),
            "-loglevel".to_string(),
            "error".to_string(),
        ];

        // NVDEC hardware acceleration (must come before -i)
        if hwaccel {
            args.push("-hwaccel".to_string());
            args.push("cuda".to_string());
        }

        // Input seeking (fast seek before -i)
        if start > 0.0 {
            args.push("-ss".to_string());
            args.push(format!("{:.3}", start));
        }

        // Duration limit
        if duration > 0.0 {
            args.push("-t".to_string());
            args.push(format!("{:.3}", duration));
        }

        args.push("-i".to_string());
        args.push(input.to_string());

        // Apply filters if any
        if !filter_str.is_empty() {
            args.push("-vf".to_string());
            args.push(filter_str.to_string());
        }

        args.extend_from_slice(&[
            "-f".to_string(),
            "rawvideo".to_string(), // FFmpeg 7+/8+ broke image2pipe+rawvideo; use -f rawvideo directly
            "-pix_fmt".to_string(),
            "rgb24".to_string(), // Output raw RGB24 to Rust (no alpha)
            "-".to_string(),
        ]);

        args
    }

    /// Spawn FFmpeg with the given args, returning the child and stdout reader.
    fn spawn_decoder(args: &[String]) -> Result<(Child, BufReader<tokio::process::ChildStdout>)> {
        let mut child = Command::new("ffmpeg")
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn ffmpeg decoder")?;

        let stdout = child.stdout.take().context("Decoder stdout missing")?;
        let reader = BufReader::new(stdout);
        Ok((child, reader))
    }

    pub async fn new(
        input: &str,
        start: f64,
        duration: f64,
        filter_str: &str,
        use_hwaccel: bool,
    ) -> Result<Self> {
        // Try NVDEC if requested
        if use_hwaccel {
            eprintln!("FFmpeg Decoder Starting (NVDEC): {}", input);
            let args = Self::build_args(input, start, duration, filter_str, true);
            match Self::spawn_decoder(&args) {
                Ok((child, reader)) => {
                    return Ok(Self {
                        child,
                        stdout: reader,
                        total_frames: 0,
                        using_hwaccel: true,
                    });
                }
                Err(e) => {
                    eprintln!(
                        "NVDEC decoder spawn failed, falling back to software: {}",
                        e
                    );
                }
            }
        }

        // Software fallback
        eprintln!("FFmpeg Decoder Starting (software): {}", input);
        let args = Self::build_args(input, start, duration, filter_str, false);
        let (child, reader) = Self::spawn_decoder(&args)?;

        Ok(Self {
            child,
            stdout: reader,
            total_frames: 0,
            using_hwaccel: false,
        })
    }

    /// Reads exactly enough bytes to fill the buffer (one frame)
    pub async fn read_raw_frame_into(&mut self, buffer: &mut [u8]) -> Result<bool> {
        match self.stdout.read_exact(buffer).await {
            Ok(_) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
            Err(e) => Err(anyhow::anyhow!("Failed to read raw frame: {}", e)),
        }
    }
}

// -----------------------------------------------------------------------------
// Video Probe (Metadata Extraction)
// -----------------------------------------------------------------------------

/// Returns (Width, Height, Duration, FPS, TotalFrames)
/// Uses a single ffprobe call with JSON output for efficiency.
pub fn probe_video(path: &str) -> Result<(usize, usize, f64, f64, u64)> {
    let output = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("ffprobe failed: {}", stderr));
    }

    let json_str = String::from_utf8(output.stdout).context("Invalid UTF-8 in ffprobe output")?;
    let json: serde_json::Value =
        serde_json::from_str(&json_str).context("Failed to parse ffprobe JSON output")?;

    // Extract stream info
    let stream = json
        .get("streams")
        .and_then(|s| s.get(0))
        .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;

    let w = stream.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let h = stream.get("height").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

    // Parse frame rate (format: "30/1" or "30000/1001")
    let fps_str = stream
        .get("r_frame_rate")
        .and_then(|v| v.as_str())
        .unwrap_or("30/1");
    let fps_parts: Vec<&str> = fps_str.split('/').collect();
    let fps: f64 = if fps_parts.len() == 2 {
        let num: f64 = fps_parts[0].parse().unwrap_or(30.0);
        let den: f64 = fps_parts[1].parse().unwrap_or(1.0);
        if den == 0.0 {
            30.0
        } else {
            num / den
        }
    } else {
        fps_str.parse().unwrap_or(30.0)
    };

    // Extract format info (duration)
    let duration = json
        .get("format")
        .and_then(|f| f.get("duration"))
        .and_then(|d| d.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);

    let total_frames = (duration * fps).round() as u64;

    Ok((w, h, duration, fps, total_frames))
}

// -----------------------------------------------------------------------------
// NVENC Hardware Encode Probe
// -----------------------------------------------------------------------------

/// Cached result of NVENC encoder availability check.
static NVENC_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Check if FFmpeg supports NVENC hardware encoding.
///
/// Runs `ffmpeg -encoders` once and caches the result.
/// Returns true if "h264_nvenc" appears in the output.
pub fn probe_nvenc() -> bool {
    *NVENC_AVAILABLE.get_or_init(|| {
        let result = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-encoders"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output();

        match result {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let has_nvenc = stdout.lines().any(|line| line.contains("h264_nvenc"));
                eprintln!(
                    "NVENC probe: {}",
                    if has_nvenc {
                        "NVENC available"
                    } else {
                        "NVENC not found, using software encode"
                    }
                );
                has_nvenc
            }
            Err(e) => {
                eprintln!("NVENC probe failed (ffmpeg not found?): {}", e);
                false
            }
        }
    })
}

// -----------------------------------------------------------------------------
// Video Encoder (Raw RGB24 Frames -> High Quality MP4 with Interpolation)
// -----------------------------------------------------------------------------

#[allow(dead_code)]
pub struct VideoEncoder {
    child: Child,
    stdin: Option<tokio::process::ChildStdin>,
    pub using_nvenc: bool,
}

impl VideoEncoder {
    #[allow(dead_code)]
    pub async fn new(
        output: &str,
        source_fps: u32,
        target_fps: u32, // 0 = Keep Source FPS
        width: usize,
        height: usize,
        deterministic: bool,
    ) -> Result<Self> {
        Self::new_with_audio(
            output,
            source_fps,
            target_fps,
            width,
            height,
            None,
            0.0,
            0.0,
            deterministic,
        )
        .await
    }

    /// Build encoder argument list.
    ///
    /// `use_nvenc`: when true, uses h264_nvenc/hevc_nvenc with NVENC-specific
    /// preset/tune flags. When false, uses libx264/libx265 software encoding
    /// with equivalent quality settings.
    #[allow(clippy::too_many_arguments)] // TODO(clippy): keep flat args to avoid broad pipeline refactor in hygiene pass.
    fn build_encoder_args(
        output: &str,
        source_fps: u32,
        target_fps: u32,
        width: usize,
        height: usize,
        audio_source: Option<&str>,
        audio_start: f64,
        audio_duration: f64,
        use_nvenc: bool,
        deterministic: bool,
    ) -> Vec<String> {
        let resolution = format!("{}x{}", width, height);
        let is_large = width > 4096 || height > 4096;

        let mut args = vec![
            "-y".to_string(),
            "-hide_banner".to_string(),
            "-loglevel".to_string(),
            "error".to_string(),
            "-f".to_string(),
            "rawvideo".to_string(),
            "-pix_fmt".to_string(),
            "rgb24".to_string(),
            "-s".to_string(),
            resolution,
            "-r".to_string(),
            source_fps.to_string(),
            "-i".to_string(),
            "-".to_string(),
        ];

        // Add original file as second input for audio stream
        if let Some(audio_src) = audio_source {
            if audio_start > 0.0 {
                args.push("-ss".to_string());
                args.push(format!("{:.3}", audio_start));
            }
            if audio_duration > 0.0 {
                args.push("-t".to_string());
                args.push(format!("{:.3}", audio_duration));
            }
            args.push("-i".to_string());
            args.push(audio_src.to_string());
            args.push("-map".to_string());
            args.push("0:v".to_string());
            args.push("-map".to_string());
            args.push("1:a?".to_string());
        }

        // --- FILTER CHAIN CONSTRUCTION ---
        let mut filters = Vec::new();
        if target_fps > 0 && target_fps != source_fps {
            filters.push(format!(
                "minterpolate=fps={}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                target_fps
            ));
        }

        // Explicitly convert to YUV420P for encoder compatibility (NVENC/x264)
        // This prevents "Invalid argument" errors when feeding RGB24 to hardware encoders
        filters.push("format=yuv420p".to_string());

        if !filters.is_empty() {
            args.push("-vf".to_string());
            args.push(filters.join(","));
        }

        // --- ENCODING SETTINGS ---
        if use_nvenc {
            let codec = if is_large { "hevc_nvenc" } else { "h264_nvenc" };
            args.extend_from_slice(&[
                "-c:v".to_string(),
                codec.to_string(),
                "-pix_fmt".to_string(),
                "yuv420p".to_string(),
            ]);

            if deterministic {
                // Deterministic: Fixed GOP 60, Const QP 18
                args.extend_from_slice(&[
                    "-g".to_string(),
                    "60".to_string(),
                    "-rc".to_string(),
                    "constqp".to_string(),
                    "-qp".to_string(),
                    "18".to_string(),
                    "-preset".to_string(),
                    "p7".to_string(),
                ]);
            } else {
                // Standard: VBR 50M
                args.extend_from_slice(&[
                    "-preset".to_string(),
                    "p7".to_string(),
                    "-tune".to_string(),
                    "hq".to_string(),
                    "-b:v".to_string(),
                    "50M".to_string(),
                    "-maxrate".to_string(),
                    "100M".to_string(),
                    "-bufsize".to_string(),
                    "100M".to_string(),
                ]);
            }
        } else {
            // Software fallback
            let codec = if is_large { "libx265" } else { "libx264" };
            args.extend_from_slice(&[
                "-c:v".to_string(),
                codec.to_string(),
                "-pix_fmt".to_string(),
                "yuv420p".to_string(),
            ]);

            if deterministic {
                args.extend_from_slice(&[
                    "-g".to_string(),
                    "60".to_string(),
                    "-qp".to_string(),
                    "18".to_string(),
                    "-preset".to_string(),
                    "veryslow".to_string(),
                ]);
            } else {
                args.extend_from_slice(&[
                    "-preset".to_string(),
                    "slow".to_string(),
                    "-crf".to_string(),
                    "18".to_string(),
                    "-maxrate".to_string(),
                    "100M".to_string(),
                    "-bufsize".to_string(),
                    "100M".to_string(),
                ]);
            }
        }

        // Copy audio stream if present (no re-encode)
        if audio_source.is_some() {
            args.extend_from_slice(&["-c:a".to_string(), "copy".to_string()]);
        }

        // --- CONTAINER SETTINGS ---
        args.extend_from_slice(&[
            "-movflags".to_string(),
            "+faststart".to_string(),
            output.to_string(),
        ]);

        args
    }

    /// Spawn FFmpeg encoder, returning the child and stdin pipe.
    fn spawn_encoder(args: &[String]) -> Result<(Child, tokio::process::ChildStdin)> {
        let mut child = Command::new("ffmpeg")
            .args(args)
            .stdin(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn ffmpeg encoder")?;

        let stdin = child.stdin.take().context("Encoder stdin missing")?;
        Ok((child, stdin))
    }

    /// Create encoder with optional audio source from the original input file.
    /// Tries NVENC first (if available), falls back to libx264/libx265 software.
    #[allow(clippy::too_many_arguments)] // TODO(clippy): constructor mirrors encode pipeline inputs; refactor can be done separately.
    pub async fn new_with_audio(
        output: &str,
        source_fps: u32,
        target_fps: u32,
        width: usize,
        height: usize,
        audio_source: Option<&str>,
        audio_start: f64,
        audio_duration: f64,
        deterministic: bool,
    ) -> Result<Self> {
        let nvenc_available = probe_nvenc();

        // Try NVENC first
        if nvenc_available {
            let codec_label = if width > 4096 || height > 4096 {
                "hevc_nvenc"
            } else {
                "h264_nvenc"
            };
            eprintln!(
                "FFmpeg Encoder Starting (NVENC): {}x{} -> {} ({})",
                width, height, output, codec_label
            );
            let args = Self::build_encoder_args(
                output,
                source_fps,
                target_fps,
                width,
                height,
                audio_source,
                audio_start,
                audio_duration,
                true,
                deterministic,
            );
            match Self::spawn_encoder(&args) {
                Ok((child, stdin)) => {
                    return Ok(Self {
                        child,
                        stdin: Some(stdin),
                        using_nvenc: true,
                    });
                }
                Err(e) => {
                    eprintln!(
                        "NVENC encoder spawn failed, falling back to software: {}",
                        e
                    );
                }
            }
        }

        // Software fallback
        let codec_label = if width > 4096 || height > 4096 {
            "libx265"
        } else {
            "libx264"
        };
        eprintln!(
            "FFmpeg Encoder Starting (software): {}x{} -> {} ({})",
            width, height, output, codec_label
        );
        let args = Self::build_encoder_args(
            output,
            source_fps,
            target_fps,
            width,
            height,
            audio_source,
            audio_start,
            audio_duration,
            false,
            deterministic,
        );
        let (child, stdin) = Self::spawn_encoder(&args)?;

        Ok(Self {
            child,
            stdin: Some(stdin),
            using_nvenc: false,
        })
    }

    pub async fn write_raw_frame(&mut self, data: &[u8]) -> Result<()> {
        if let Some(stdin) = &mut self.stdin {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(data).await?;
        }
        Ok(())
    }

    pub async fn finish(&mut self) -> Result<()> {
        eprintln!("Encoder: Closing Stdin to signal EOF...");
        if let Some(mut stdin) = self.stdin.take() {
            tokio::io::AsyncWriteExt::shutdown(&mut stdin)
                .await
                .context("Failed to close encoder stdin")?;
            drop(stdin);
        }

        eprintln!("Encoder: Waiting for FFmpeg process to exit...");
        match timeout(Duration::from_secs(30), self.child.wait()).await {
            Ok(result) => {
                let status = result.context("Encoder wait failed")?;
                if !status.success() {
                    return Err(anyhow::anyhow!(
                        "Encoder exited with error code: {:?}",
                        status.code()
                    ));
                }
                eprintln!("Encoder: FFmpeg exited successfully.");
            }
            Err(_) => {
                eprintln!("Encoder: Timeout waiting for FFmpeg. Force killing...");
                let _ = self.child.start_kill();
                return Err(anyhow::anyhow!("Encoder timed out and was killed"));
            }
        }
        Ok(())
    }
}
