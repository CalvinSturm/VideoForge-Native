// src-tauri/src/video_pipeline.rs

use anyhow::{Context, Result};
use std::process::Stdio;
use tokio::io::AsyncReadExt;
use tokio::io::BufReader;
use tokio::process::{Child, Command};
use tokio::time::{timeout, Duration};

// -----------------------------------------------------------------------------
// Video Decoder (FFmpeg -> Raw RGBA Frames)
// -----------------------------------------------------------------------------

pub struct VideoDecoder {
    stdout: BufReader<tokio::process::ChildStdout>,
    pub total_frames: u32,
}

impl VideoDecoder {
    pub async fn new(
        input: &str,
        // Trim logic is handled via start/duration args
        start: f64,
        duration: f64,
        // Filters now passed in
        filter_str: &str,
    ) -> Result<Self> {
        // We don't use fps/width/height args for decoding setup directly,
        // but we assume caller has handled dimension calc for the SHM buffer.

        println!("FFmpeg Decoder Starting: {}", input);

        let mut args = vec![
            "-hide_banner".to_string(),
            "-loglevel".to_string(),
            "error".to_string(),
        ];

        // Input seeking (Fast seek before -i)
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
            "image2pipe".to_string(),
            "-pix_fmt".to_string(),
            "rgba".to_string(), // Output raw RGBA to Rust
            "-vcodec".to_string(),
            "rawvideo".to_string(),
            "-".to_string(),
        ]);

        let mut child = Command::new("ffmpeg")
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn ffmpeg decoder")?;

        let stdout = child.stdout.take().context("Decoder stdout missing")?;
        let reader = BufReader::new(stdout);

        Ok(Self {
            stdout: reader,
            total_frames: 0, // Logic for counting frames usually handled by probe or caller
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
        .args(&[
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

    let json_str = String::from_utf8(output.stdout)
        .context("Invalid UTF-8 in ffprobe output")?;
    let json: serde_json::Value = serde_json::from_str(&json_str)
        .context("Failed to parse ffprobe JSON output")?;

    // Extract stream info
    let stream = json
        .get("streams")
        .and_then(|s| s.get(0))
        .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;

    let w = stream
        .get("width")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let h = stream
        .get("height")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

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
// Video Encoder (Raw RGBA Frames -> High Quality MP4 with Interpolation)
// -----------------------------------------------------------------------------

pub struct VideoEncoder {
    child: Child,
    stdin: Option<tokio::process::ChildStdin>,
}

impl VideoEncoder {
    pub async fn new(
        output: &str,
        source_fps: u32,
        target_fps: u32, // 0 = Keep Source FPS
        width: usize,
        height: usize,
    ) -> Result<Self> {
        let resolution = format!("{}x{}", width, height);

        // Auto-switch codec: H.264 typically maxes at 4096px (4K).
        // For 8K or extreme wide, use HEVC.
        let codec = if width > 4096 || height > 4096 {
            "hevc_nvenc"
        } else {
            "h264_nvenc"
        };

        // Determine effective output FPS
        let final_fps = if target_fps > 0 {
            target_fps
        } else {
            source_fps
        };

        println!(
            "FFmpeg Encoder Starting: {} -> {} (Codec: {}, Source: {}fps, Target: {}fps)",
            resolution, output, codec, source_fps, final_fps
        );

        let mut args = vec![
            "-y".to_string(),
            "-hide_banner".to_string(),
            "-loglevel".to_string(),
            "error".to_string(),
            "-f".to_string(),
            "rawvideo".to_string(),
            "-pix_fmt".to_string(),
            "rgba".to_string(), // Input from Rust/SHM is RGBA
            "-s".to_string(),
            resolution.clone(),
            "-r".to_string(),
            source_fps.to_string(), // Input stream rate
            "-i".to_string(),
            "-".to_string(),
        ];

        // --- FILTER CHAIN CONSTRUCTION ---
        let mut filters = Vec::new();

        // 1. Frame Interpolation (Motion Smoothing)
        if target_fps > 0 && target_fps != source_fps {
            println!(
                "Applying Motion Interpolation: {} -> {}",
                source_fps, target_fps
            );
            // minterpolate options:
            // fps: Target frame rate
            // mi_mode=mci: Motion Compensated Interpolation (high quality)
            // mc_mode=aobmc: Adaptive Overlapped Block Motion Compensation (reduces artifacts)
            // me_mode=bidir: Bidirectional motion estimation
            // vsbmc=1: Variable-size block motion compensation
            filters.push(format!(
                "minterpolate=fps={}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                target_fps
            ));
        }

        if !filters.is_empty() {
            args.push("-vf".to_string());
            args.push(filters.join(","));
        }

        // --- ENCODING SETTINGS ---
        args.extend_from_slice(&[
            "-c:v".to_string(),
            codec.to_string(),
            "-pix_fmt".to_string(),
            "yuv420p".to_string(), // Standard pixel format for compatibility
            "-preset".to_string(),
            "p7".to_string(), // NVENC: Slowest/Best Quality
            "-tune".to_string(),
            "hq".to_string(), // NVENC: High Quality Tune
            "-b:v".to_string(),
            "50M".to_string(), // Target Bitrate: 50 Mbps
            "-maxrate".to_string(),
            "100M".to_string(), // Max Bitrate: 100 Mbps
            "-bufsize".to_string(),
            "100M".to_string(), // VBV Buffer
            // --- CONTAINER SETTINGS ---
            "-movflags".to_string(),
            "+faststart".to_string(), // Move metadata to start (faster muxing/playback)
            output.to_string(),
        ]);

        let mut child = Command::new("ffmpeg")
            .args(&args)
            .stdin(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn ffmpeg encoder")?;

        let stdin = child.stdin.take().context("Encoder stdin missing")?;
        Ok(Self {
            child,
            stdin: Some(stdin),
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
        println!("Encoder: Closing Stdin to signal EOF...");
        if let Some(mut stdin) = self.stdin.take() {
            use tokio::io::AsyncWriteExt;
            tokio::io::AsyncWriteExt::shutdown(&mut stdin)
                .await
                .context("Failed to close encoder stdin")?;
            drop(stdin); // Explicit drop to ensure pipe closes
        }

        println!("Encoder: Waiting for FFmpeg process to exit...");
        // Wait up to 30 seconds for FFmpeg (interpolation can add delay at end)
        match timeout(Duration::from_secs(30), self.child.wait()).await {
            Ok(result) => {
                let status = result.context("Encoder wait failed")?;
                if !status.success() {
                    return Err(anyhow::anyhow!(
                        "Encoder exited with error code: {:?}",
                        status.code()
                    ));
                }
                println!("Encoder: FFmpeg exited successfully.");
            }
            Err(_) => {
                println!("Encoder: Timeout waiting for FFmpeg. Force killing...");
                let _ = self.child.start_kill();
                return Err(anyhow::anyhow!("Encoder timed out and was killed"));
            }
        }
        Ok(())
    }
}
