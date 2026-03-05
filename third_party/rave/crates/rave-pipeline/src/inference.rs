//! End-to-end GPU-resident inference pipeline.
//!
//! Composes the preprocessing kernel chain (`PreprocessPipeline`) with the
//! TensorRT inference backend (`TensorRtBackend`) and the postprocess stage
//! to produce the full transform:
//!
//! ```text
//! NV12 (decoded) → preprocess → inference → postprocess → NV12 (encode-ready)
//! ```
//!
//! All data remains on the GPU.  No host staging, no implicit copies.
//! Stage-level metrics are exposed for latency and throughput monitoring.

use std::sync::Arc;

use cudarc::driver::CudaStream;
use tracing::{debug, info};

use rave_core::backend::UpscaleBackend;
use rave_core::context::GpuContext;
use rave_core::error::{EngineError, Result};
use rave_core::types::{FrameEnvelope, GpuTexture, PixelFormat};
use rave_cuda::kernels::{ModelPrecision, PreprocessKernels, PreprocessPipeline, StageMetrics};
use rave_tensorrt::tensorrt::TensorRtBackend;

/// End-to-end GPU-resident inference pipeline.
///
/// Orchestrates:
/// 1. **Preprocess** — NV12 → model-ready tensor (F32 or F16).
/// 2. **Inference** — TensorRT via ORT IO Binding.
/// 3. **Postprocess** — model output → NV12 for NVENC.
///
/// All operations are GPU-only.  Stage metrics are accumulated for
/// reporting at shutdown.
pub struct InferencePipeline {
    preprocess: PreprocessPipeline,
    backend: Arc<TensorRtBackend>,
    ctx: Arc<GpuContext>,

    /// Latency metrics for the NV12 → tensor preprocess stage.
    pub metrics_preprocess: StageMetrics,
    /// Latency metrics for the TensorRT inference stage.
    pub metrics_inference: StageMetrics,
    /// Latency metrics for the tensor → NV12 postprocess stage.
    pub metrics_postprocess: StageMetrics,
}

impl InferencePipeline {
    /// Create a new inference pipeline.
    ///
    /// `precision` selects F32 or F16 kernel path.  Must match the model's
    /// expected input tensor type.
    pub fn new(
        ctx: Arc<GpuContext>,
        backend: Arc<TensorRtBackend>,
        precision: ModelPrecision,
    ) -> Result<Self> {
        let kernels = PreprocessKernels::compile(ctx.device())?;
        let preprocess = PreprocessPipeline::new(kernels, precision);

        Ok(Self {
            preprocess,
            backend,
            ctx,

            metrics_preprocess: StageMetrics::default(),
            metrics_inference: StageMetrics::default(),
            metrics_postprocess: StageMetrics::default(),
        })
    }

    /// Process a single decoded NV12 frame through the full pipeline.
    ///
    /// Returns an NV12 `GpuTexture` ready for NVENC encoding.
    ///
    /// # Data flow (all GPU-resident)
    ///
    /// 1. NV12 → RGB [F32|F16] planar NCHW (preprocess kernel)
    /// 2. RGB planar → TensorRT inference (ORT IO Binding)
    /// 3. Inference output → NV12 (postprocess kernel)
    ///
    /// # Errors
    ///
    /// Propagates errors from any stage.  No partial results.
    pub async fn process_frame(
        &mut self,
        envelope: &FrameEnvelope,
        stream: &CudaStream,
    ) -> Result<GpuTexture> {
        let input = &envelope.texture;

        if input.format != PixelFormat::Nv12 {
            return Err(EngineError::FormatMismatch {
                expected: PixelFormat::Nv12,
                actual: input.format,
            });
        }



        // ── Stage 1: Preprocess (NV12 → model tensor) ──
        let model_input = self.preprocess.prepare(input, &self.ctx, stream)?;

        debug!(
            frame = envelope.frame_index,
            shape = ?model_input.shape,
            format = ?model_input.texture.format,
            "Preprocess complete"
        );

        // ── Stage 2: Inference ──
        let inference_output = self.backend.process(model_input.texture).await?;

        debug!(
            frame = envelope.frame_index,
            out_w = inference_output.width,
            out_h = inference_output.height,
            format = ?inference_output.format,
            "Inference complete"
        );

        // ── Stage 3: Postprocess (RGB → NV12) ──
        // Pitch must match the upscaled output width, NOT the decoded input width.
        // 256-byte alignment for NVENC compatibility.
        let nv12_pitch = (inference_output.width as usize + 255) & !255;

        let nv12_output =
            self.preprocess
                .postprocess(inference_output, nv12_pitch, &self.ctx, stream)?;

        debug!(
            frame = envelope.frame_index,
            pitch = nv12_output.pitch,
            "Postprocess complete — NV12 ready for encode"
        );

        Ok(nv12_output)
    }

    /// Report accumulated stage metrics.
    pub fn report_metrics(&self) {
        let snap = self.backend.inference_metrics.snapshot();
        info!(
            preprocess_avg_ms = format!("{:.3}", self.metrics_preprocess.avg_ms()),
            inference_avg_us = snap.avg_inference_us,
            inference_peak_us = snap.peak_inference_us,
            postprocess_avg_ms = format!("{:.3}", self.metrics_postprocess.avg_ms()),
            frames = snap.frames_inferred,
            "Pipeline stage metrics"
        );

        let (vram_current, vram_peak) = self.ctx.vram_usage();
        info!(
            current_mb = vram_current / (1024 * 1024),
            peak_mb = vram_peak / (1024 * 1024),
            "VRAM usage"
        );
    }
}
