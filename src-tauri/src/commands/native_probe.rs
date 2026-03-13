#[cfg(feature = "native_engine")]
use crate::commands::native_runtime::configure_native_probe_cmd;
#[cfg(feature = "native_engine")]
use crate::commands::native_routing::{NativeJobSpec, NativeOutputPathStyle};
#[cfg(feature = "native_engine")]
use crate::commands::native_streaming_io::StreamingCodecHint;

#[cfg(feature = "native_engine")]
use crate::commands::native_engine::NativeUpscaleError;

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

    let duration = json
        .get("format")
        .and_then(|f| f.get("duration"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let total_frames = (duration * fps).round().max(0.0) as u64;

    let codec_name = stream
        .get("codec_name")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let codec = match codec_name.as_str() {
        "h264" | "avc1" => videoforge_engine::codecs::sys::cudaVideoCodec::H264,
        "hevc" | "h265" | "hev1" | "hvc1" => videoforge_engine::codecs::sys::cudaVideoCodec::HEVC,
        other => {
            return Err(format!(
                "Unsupported video codec '{other}' for direct native pipeline"
            ))
        }
    };

    Ok((coded_width, coded_height, duration, fps, total_frames, codec))
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
pub(crate) struct NativeVideoSourceProfile {
    pub coded_width: usize,
    pub coded_height: usize,
    pub fps: f64,
    pub total_frames: u64,
    pub codec: videoforge_engine::codecs::sys::cudaVideoCodec,
}

#[cfg(feature = "native_engine")]
impl NativeVideoSourceProfile {
    pub(crate) fn probe(input_path: &str) -> Result<Self, String> {
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

    pub(crate) fn mux_codec_hint(&self) -> StreamingCodecHint {
        StreamingCodecHint::new(match self.codec {
            videoforge_engine::codecs::sys::cudaVideoCodec::H264 => Some("h264"),
            videoforge_engine::codecs::sys::cudaVideoCodec::HEVC => Some("hevc"),
            _ => None,
        })
    }

    pub(crate) fn scaled_output(&self, scale: u32) -> NativeVideoOutputProfile {
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
pub(crate) struct NativeVideoOutputProfile {
    pub width: usize,
    pub height: usize,
    pub nv12_pitch: usize,
    pub fps_num: u32,
    pub fps_den: u32,
}

#[cfg(feature = "native_engine")]
impl NativeVideoOutputProfile {
    pub(crate) fn frame_rate_arg(&self) -> String {
        format!("{}/{}", self.fps_num, self.fps_den.max(1))
    }

    pub(crate) fn nvenc_config(&self) -> videoforge_engine::codecs::nvenc::NvEncConfig {
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

    pub(crate) fn pipeline_config(
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
pub(crate) struct NativeDirectPlan {
    pub ffmpeg_cmd: String,
    pub output_path: String,
    pub source: NativeVideoSourceProfile,
    pub output: NativeVideoOutputProfile,
    pub mux_codec_hint: StreamingCodecHint,
}

#[cfg(feature = "native_engine")]
impl NativeDirectPlan {
    pub(crate) fn prepare(
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

    pub(crate) fn nvenc_config(&self) -> videoforge_engine::codecs::nvenc::NvEncConfig {
        self.output.nvenc_config()
    }

    pub(crate) fn pipeline_config(
        &self,
        model_precision: videoforge_engine::core::kernels::ModelPrecision,
        inference_max_batch: usize,
    ) -> videoforge_engine::engine::pipeline::PipelineConfig {
        self.output
            .pipeline_config(model_precision, inference_max_batch)
    }
}

#[cfg(all(test, feature = "native_engine"))]
mod tests {
    use super::{NativeVideoOutputProfile, NativeVideoSourceProfile};
    use videoforge_engine::codecs::sys::cudaVideoCodec;
    use videoforge_engine::core::kernels::ModelPrecision;

    #[test]
    fn source_profile_maps_codec_to_mux_hint() {
        let h264 = NativeVideoSourceProfile {
            coded_width: 1920,
            coded_height: 1080,
            fps: 30.0,
            total_frames: 300,
            codec: cudaVideoCodec::H264,
        };
        let hevc = NativeVideoSourceProfile {
            codec: cudaVideoCodec::HEVC,
            ..h264.clone()
        };

        assert_eq!(h264.mux_codec_hint().get(), Some("h264"));
        assert_eq!(hevc.mux_codec_hint().get(), Some("hevc"));
    }

    #[test]
    fn output_profile_builds_nvenc_and_pipeline_configs() {
        let output = NativeVideoOutputProfile {
            width: 3840,
            height: 2160,
            nv12_pitch: 4096,
            fps_num: 24_000,
            fps_den: 1001,
        };

        let nvenc = output.nvenc_config();
        assert_eq!(nvenc.width, 3840);
        assert_eq!(nvenc.height, 2160);
        assert_eq!(nvenc.nv12_pitch, 4096);
        assert_eq!(nvenc.fps_num, 24_000);
        assert_eq!(nvenc.fps_den, 1001);

        let pipeline = output.pipeline_config(ModelPrecision::F16, 6);
        assert_eq!(pipeline.encoder_nv12_pitch, 4096);
        assert_eq!(pipeline.inference_max_batch, 6);
        assert_eq!(pipeline.model_precision, ModelPrecision::F16);
    }
}
