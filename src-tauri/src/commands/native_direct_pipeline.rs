#[cfg(feature = "native_engine")]
use std::path::{Path, PathBuf};
#[cfg(feature = "native_engine")]
use std::sync::atomic::{AtomicU8, Ordering};
#[cfg(feature = "native_engine")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "native_engine")]
use crate::commands::native_probe::{NativeDirectPlan, NativeVideoOutputProfile};
#[cfg(feature = "native_engine")]
use crate::commands::native_routing::{
    build_direct_perf_report, build_native_observed_metrics, build_native_runtime_snapshot,
    NativeExecutionRoute, NativeJobSpec,
};
#[cfg(feature = "native_engine")]
use crate::commands::native_runtime::configure_native_runtime_env;
#[cfg(feature = "native_engine")]
use crate::commands::native_streaming_io::{
    FfmpegBitstreamSource, StreamingCodecHint, StreamingMuxSink,
};
#[cfg(feature = "native_engine")]
use crate::runtime_truth::{log_run_observed_metrics, RunStatus};

#[cfg(feature = "native_engine")]
use crate::commands::native_engine::{
    classify_backend_init_error, NativeUpscaleError, NativeUpscaleResult,
};

#[cfg(feature = "native_engine")]
pub(crate) async fn run_native_pipeline(
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

    run_engine_pipeline(job, &plan).await
}

#[cfg(feature = "native_engine")]
async fn run_engine_pipeline(
    job: &NativeJobSpec,
    plan: &NativeDirectPlan,
) -> Result<NativeUpscaleResult, String> {
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

    tracing::info!("Initialising GPU context (device 0)");
    let ctx = GpuContext::new(0)
        .map_err(|e| make_err("GPU_INIT", &format!("GPU context creation failed: {}", e)))?;

    tracing::info!(model = %job.model_path, "Loading TensorRT backend");
    let precision_policy = match job.precision.as_str() {
        "fp16" => PrecisionPolicy::Fp16,
        _ => PrecisionPolicy::Fp32,
    };
    let downstream_capacity = 4usize;
    let ring_size =
        TensorRtBackend::required_ring_slots(downstream_capacity, job.max_batch as usize);

    let backend = TensorRtBackend::with_precision(
        std::path::PathBuf::from(&job.model_path),
        ctx.clone(),
        0,
        ring_size,
        downstream_capacity,
        precision_policy,
        BatchConfig {
            max_batch: job.max_batch as usize,
            ..BatchConfig::default()
        },
    );
    let backend = Arc::new(backend);
    backend.initialize().await.map_err(|e| {
        make_err(
            "BACKEND_INIT",
            &classify_backend_init_error(&job.model_path, &e.to_string()),
        )
    })?;

    let kernels = PreprocessKernels::compile(ctx.device())
        .map_err(|e| make_err("KERNEL_COMPILE", &format!("Kernel compile failed: {}", e)))?;
    let kernels = Arc::new(kernels);

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
    let frames = metrics.frames_encoded.load(std::sync::atomic::Ordering::Relaxed);
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
    let route = NativeExecutionRoute::direct();
    let runtime_snapshot = build_native_runtime_snapshot(job, &route);
    let observed_metrics = build_native_observed_metrics(
        &job.run_id,
        "native_direct",
        RunStatus::Succeeded,
        Some(&perf),
        None,
    );
    let mut result = job.build_result(
        plan.output_path.clone(),
        "native_v2",
        encoder_mode,
        encoder_detail,
        perf,
        route,
    );
    result.runtime_snapshot = Some(runtime_snapshot);
    result.observed_metrics = Some(observed_metrics.clone());
    log_run_observed_metrics(&observed_metrics);

    Ok(result)
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
