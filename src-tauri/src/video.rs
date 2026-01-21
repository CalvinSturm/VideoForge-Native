use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine as _};
use tokio::{
    io::BufReader,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::watch,
};
use crate::inference::{InferenceBridge, Request, Response};

/* ─────────────────────────────────────────────
   Video Decoder / Encoder
───────────────────────────────────────────── */

pub struct VideoDecoder {
    child: Child,
    stdout: BufReader<ChildStdout>,
}

impl VideoDecoder {
    pub fn new(input: &str, start_time: f64, duration: f64) -> Result<Self> {
        let mut args = vec!["-i", input, "-f", "image2pipe", "-vcodec", "png", "-"];
        if start_time > 0.0 {
            args.splice(0..0, ["-ss", &format!("{:.3}", start_time)]);
        }
        if duration > 0.0 {
            args.splice(0..0, ["-t", &format!("{:.3}", duration)]);
        }

        let child = Command::new("ffmpeg")
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn ffmpeg decoder")?;

        let stdout = child.stdout.context("Decoder stdout missing")?;
        Ok(Self {
            child,
            stdout: BufReader::new(stdout),
        })
    }

    pub async fn read_frame(&mut self) -> Result<Option<Vec<u8>>> {
        let mut png = Vec::new();
        let mut signature = [0u8; 8];
        if self.stdout.read_exact(&mut signature).await.is_err() {
            return Ok(None);
        }
        png.extend_from_slice(&signature);

        let mut length_bytes = [0u8; 4];
        loop {
            if self.stdout.read_exact(&mut length_bytes).await.is_err() {
                break;
            }
            png.extend_from_slice(&length_bytes);
            let length = u32::from_be_bytes(length_bytes) as usize;

            let mut chunk_type = [0u8; 4];
            self.stdout.read_exact(&mut chunk_type).await?;
            png.extend_from_slice(&chunk_type);

            let mut chunk_data = vec![0u8; length];
            self.stdout.read_exact(&mut chunk_data).await?;
            png.extend_from_slice(&chunk_data);

            let mut crc = [0u8; 4];
            self.stdout.read_exact(&mut crc).await?;
            png.extend_from_slice(&crc);

            if &chunk_type == b"IEND" {
                break;
            }
        }

        Ok(Some(png))
    }

    pub async fn wait(&mut self) -> Result<std::process::ExitStatus> {
        self.child.wait().await.context("Waiting for decoder failed")
    }
}

pub struct VideoEncoder {
    child: Child,
    stdin: ChildStdin,
}

impl VideoEncoder {
    /// Create a new encoder with optional codec/pixel format (for HDR/alpha)
    pub fn new(
        output: &str,
        fps: u32,
        codec: &str,      // e.g., "prores_ks", "libx264", "libx265"
        pix_fmt: &str,    // e.g., "yuva444p10le", "yuv420p10le"
    ) -> Result<Self> {
        let mut child = Command::new("ffmpeg")
            .args(&[
                "-y",
                "-f",
                "image2pipe",
                "-r",
                &fps.to_string(),
                "-i",
                "-",
                "-c:v",
                codec,
                "-pix_fmt",
                pix_fmt,
                output,
            ])
            .stdin(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn ffmpeg encoder")?;

        let stdin = child.stdin.take().context("Encoder stdin missing")?;
        Ok(Self { child, stdin })
    }

    pub async fn write_frame(&mut self, png: &[u8]) -> Result<()> {
        self.stdin.write_all(png).await?;
        Ok(())
    }

    pub async fn finish(&mut self) -> Result<()> {
        self.stdin.flush().await?;
        drop(&self.stdin);
        Ok(())
    }

    pub async fn wait(&mut self) -> Result<std::process::ExitStatus> {
        self.child.wait().await.context("Waiting for encoder failed")
    }
}

/* ─────────────────────────────────────────────
   Video Upscale via InferenceBridge
───────────────────────────────────────────── */

pub async fn upscale_video(
    bridge: &mut InferenceBridge,
    input_path: &str,
    output_path: &str,
    start_time: f64,
    duration: f64,
    fps: u32,
    codec: &str,
    pix_fmt: &str,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let mut decoder = VideoDecoder::new(input_path, start_time, duration)?;
    let mut encoder = VideoEncoder::new(output_path, fps, codec, pix_fmt)?;

    let mut frame_idx = 0usize;

    while let Some(frame_png) = decoder.read_frame().await? {
        if *cancel_rx.borrow() {
            anyhow::bail!("Video upscale cancelled");
        }

        let req = Request {
            id: format!("frame-{}", frame_idx),
            command: "upscale_image_base64".to_string(),
            params: serde_json::json!({
                "image": base64::engine::general_purpose::STANDARD.encode(&frame_png),
            }),
        };

        let resp: Response = bridge
            .send(req, None::<fn(Response)>, cancel_rx.clone())
            .await?;

        if resp.status != "ok" {
            anyhow::bail!(
                "Frame {} failed: {:?}",
                frame_idx,
                resp.error.unwrap_or_else(|| serde_json::json!("unknown"))
            );
        }

        let upscaled_b64 = resp
            .result
            .as_ref()
            .and_then(|v| v.get("image"))
            .and_then(|v| v.as_str())
            .context("Missing 'image' in response")?;

        let upscaled_png = base64::engine::general_purpose::STANDARD.decode(upscaled_b64)?;
        encoder.write_frame(&upscaled_png).await?;

        frame_idx += 1;
    }

    encoder.finish().await?;
    encoder.wait().await?;
    decoder.wait().await?;

    Ok(())
}