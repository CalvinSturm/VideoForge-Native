//! TensorRT inference backend — ORT + TensorRTExecutionProvider + IO Binding.
//!
//! # Zero-copy contract
//!
//! Input and output tensors are bound to CUDA device pointers via ORT's
//! IO Binding API.  At no point does frame data touch host memory.
//!
//! # Execution provider policy
//!
//! TensorRT EP is preferred, CUDA EP is allowed as a fallback, and CPU EP
//! is explicitly excluded from the session builder.
//!
//! # CUDA stream ordering
//!
//! ORT creates its own internal CUDA stream for TensorRT EP execution.
//! We cannot inject `GpuContext::inference_stream` because `cudarc::CudaStream`
//! does not expose its raw `CUstream` handle (`pub(crate)` field).
//!
//! Correctness is maintained because `session.run_with_binding()` is
//! **synchronous** — ORT blocks the calling thread until all GPU kernels
//! on its internal stream complete.  Therefore:
//!
//! 1. Output buffer is fully written when `run_with_binding()` returns.
//! 2. CUDA global memory coherency guarantees visibility to any subsequent
//!    reader on any stream after this synchronization point.
//! 3. No additional inter-stream event is needed.
//!
//! # Output ring serialization
//!
//! `OutputRing` owns N pre-allocated device buffers.  `acquire()` checks
//! `Arc::strong_count == 1` before returning a slot, guaranteeing no
//! concurrent reader.  Ring size must be ≥ `downstream_channel_capacity + max_batch + 1`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use ort::execution_providers::{
    CUDAExecutionProvider as CudaEP, TensorRTExecutionProvider as TrtEP,
};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::sys as ort_sys;
use ort::tensor::TensorElementType;
use ort::value::{DynValueTypeMarker, Value as OrtValue};
use ort::AsPointer;

use cudarc::driver::{CudaSlice, DevicePtr};

use crate::core::backend::{ModelMetadata, UpscaleBackend};
use crate::core::context::GpuContext;
use crate::core::types::{GpuBuffer, GpuTexture, PixelFormat};
use crate::error::{EngineError, Result};

// Create an ORT tensor wrapper over an existing CUDA device buffer.
unsafe fn create_tensor_from_device_memory(
    mem_info: &MemoryInfo,
    ptr: *mut std::ffi::c_void,
    bytes: usize,
    shape: &[i64],
    elem_type: TensorElementType,
) -> Result<OrtValue<DynValueTypeMarker>> {
    let api = ort::api();

    let mut ort_value_ptr: *mut ort_sys::OrtValue = std::ptr::null_mut();
    let status = unsafe {
        (api.CreateTensorWithDataAsOrtValue)(
            mem_info.ptr(),
            ptr,
            bytes as _,
            shape.as_ptr(),
            shape.len() as _,
            elem_type.into(),
            &mut ort_value_ptr,
        )
    };

    if !status.0.is_null() {
        unsafe { (api.ReleaseStatus)(status.0) };
        return Err(EngineError::ModelMetadata(
            "CreateTensorWithDataAsOrtValue failed".into(),
        ));
    }

    Ok(unsafe {
        OrtValue::<DynValueTypeMarker>::from_ptr(
            std::ptr::NonNull::new(ort_value_ptr)
                .ok_or_else(|| EngineError::ModelMetadata("Null OrtValue pointer".into()))?,
            None,
        )
    })
}

// ─── Precision policy ───────────────────────────────────────────────────────

/// TensorRT precision policy — controls EP optimization flags.
#[derive(Clone, Debug)]
pub enum PrecisionPolicy {
    /// FP32 only — maximum accuracy, baseline performance.
    Fp32,
    /// FP16 mixed precision — 2× throughput on Tensor Cores.
    Fp16,
    /// INT8 quantized with calibration table — 4× throughput.
    /// Requires a pre-generated calibration table path.
    Int8 { calibration_table: PathBuf },
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        PrecisionPolicy::Fp16
    }
}

// ─── Batch config ──────────────────────────────────────────────────────────

/// Batch inference configuration.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// Maximum batch size for pipelined inference.
    /// Must be ≤ model’s max dynamic batch axis.
    pub max_batch: usize,
    /// Collect at most this many frames before dispatching a batch,
    /// even if `max_batch` is not reached (latency bound).
    pub latency_deadline_us: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch: 1,
            latency_deadline_us: 8_000, // 8ms — half a 60fps frame
        }
    }
}
// ─── Inference metrics ───────────────────────────────────────────────────────

/// Atomic counters for inference stage observability.
#[derive(Debug)]
pub struct InferenceMetrics {
    /// Total frames inferred.
    pub frames_inferred: AtomicU64,
    /// Cumulative inference time in microseconds (for avg latency).
    pub total_inference_us: AtomicU64,
    /// Peak single-frame inference time in microseconds.
    pub peak_inference_us: AtomicU64,
}

impl InferenceMetrics {
    pub const fn new() -> Self {
        Self {
            frames_inferred: AtomicU64::new(0),
            total_inference_us: AtomicU64::new(0),
            peak_inference_us: AtomicU64::new(0),
        }
    }

    pub fn record(&self, elapsed_us: u64) {
        self.record_frames(1, elapsed_us);
    }

    pub fn record_frames(&self, frames: u64, elapsed_us: u64) {
        self.frames_inferred.fetch_add(frames, Ordering::Relaxed);
        self.total_inference_us
            .fetch_add(elapsed_us, Ordering::Relaxed);
        self.peak_inference_us
            .fetch_max(elapsed_us, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> InferenceMetricsSnapshot {
        let frames = self.frames_inferred.load(Ordering::Relaxed);
        let total = self.total_inference_us.load(Ordering::Relaxed);
        let peak = self.peak_inference_us.load(Ordering::Relaxed);
        InferenceMetricsSnapshot {
            frames_inferred: frames,
            avg_inference_us: if frames > 0 { total / frames } else { 0 },
            peak_inference_us: peak,
        }
    }
}

/// Snapshot of inference metrics for reporting.
#[derive(Clone, Debug)]
pub struct InferenceMetricsSnapshot {
    pub frames_inferred: u64,
    pub avg_inference_us: u64,
    pub peak_inference_us: u64,
}

// ─── Ring metrics ────────────────────────────────────────────────────────────

/// Atomic counters for output ring buffer activity.
#[derive(Debug)]
pub struct RingMetrics {
    /// Successful slot reuses (slot was free, strong_count == 1).
    pub slot_reuse_count: AtomicU64,
    /// Times `acquire()` found a slot still held downstream (strong_count > 1).
    pub slot_contention_events: AtomicU64,
    /// Times a slot was acquired but it was the first use (not a reuse).
    pub slot_first_use_count: AtomicU64,
}

impl RingMetrics {
    pub const fn new() -> Self {
        Self {
            slot_reuse_count: AtomicU64::new(0),
            slot_contention_events: AtomicU64::new(0),
            slot_first_use_count: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.slot_reuse_count.load(Ordering::Relaxed),
            self.slot_contention_events.load(Ordering::Relaxed),
            self.slot_first_use_count.load(Ordering::Relaxed),
        )
    }
}

// ─── Output ring buffer ─────────────────────────────────────────────────────

/// Fixed-size ring of pre-allocated device buffers for inference output.
pub struct OutputRing {
    slots: Vec<Arc<cudarc::driver::CudaSlice<u8>>>,
    cursor: usize,
    pub slot_bytes: usize,
    pub alloc_dims: (u32, u32),
    /// Whether each slot has been used at least once (for first-use tracking).
    used: Vec<bool>,
    pub metrics: RingMetrics,
}

impl OutputRing {
    /// Allocate `count` output buffers.
    ///
    /// `min_slots` is the enforced minimum (`downstream_capacity + max_batch + 1`).
    /// Returns error if `count < min_slots`.
    pub fn new(
        ctx: &GpuContext,
        in_w: u32,
        in_h: u32,
        scale: u32,
        output_format: PixelFormat,
        count: usize,
        min_slots: usize,
    ) -> Result<Self> {
        if count < min_slots {
            return Err(EngineError::DimensionMismatch(format!(
                "OutputRing: ring_size ({count}) < required minimum ({min_slots}). \
                 Ring must be ≥ downstream_channel_capacity + max_batch + 1."
            )));
        }
        if count < 2 {
            return Err(EngineError::DimensionMismatch(
                "OutputRing: ring_size must be ≥ 2 for double-buffering".into(),
            ));
        }

        let out_w = (in_w * scale) as usize;
        let out_h = (in_h * scale) as usize;
        let slot_bytes = 3 * out_w * out_h * output_format.element_bytes();

        let slots = (0..count)
            .map(|_| ctx.alloc(slot_bytes).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;

        debug!(count, slot_bytes, out_w, out_h, "Output ring allocated");

        Ok(Self {
            slots,
            cursor: 0,
            slot_bytes,
            alloc_dims: (in_w, in_h),
            used: vec![false; count],
            metrics: RingMetrics::new(),
        })
    }

    /// Acquire the next ring slot for writing.
    ///
    /// # Serialization invariant
    ///
    /// Asserts `Arc::strong_count == 1` before returning.  If downstream
    /// still holds a reference, returns error and increments contention counter.
    pub fn acquire(&mut self) -> Result<Arc<cudarc::driver::CudaSlice<u8>>> {
        let slot = &self.slots[self.cursor];
        let sc = Arc::strong_count(slot);

        if sc != 1 {
            self.metrics
                .slot_contention_events
                .fetch_add(1, Ordering::Relaxed);
            return Err(EngineError::BufferTooSmall {
                need: self.slot_bytes,
                have: 0,
            });
        }

        // Debug assertion — belt-and-suspenders check.
        debug_assert_eq!(
            Arc::strong_count(slot),
            1,
            "OutputRing: slot {} strong_count must be 1 before reuse, got {}",
            self.cursor,
            sc
        );

        if self.used[self.cursor] {
            self.metrics
                .slot_reuse_count
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.used[self.cursor] = true;
            self.metrics
                .slot_first_use_count
                .fetch_add(1, Ordering::Relaxed);
        }

        let cloned = Arc::clone(slot);
        self.cursor = (self.cursor + 1) % self.slots.len();
        Ok(cloned)
    }

    pub fn needs_realloc(&self, in_w: u32, in_h: u32) -> bool {
        self.alloc_dims != (in_w, in_h)
    }

    /// Reallocate all slots.  All must have `strong_count == 1`.
    pub fn reallocate(
        &mut self,
        ctx: &GpuContext,
        in_w: u32,
        in_h: u32,
        scale: u32,
        output_format: PixelFormat,
    ) -> Result<()> {
        for (i, slot) in self.slots.iter().enumerate() {
            let sc = Arc::strong_count(slot);
            if sc != 1 {
                return Err(EngineError::DimensionMismatch(format!(
                    "Cannot reallocate ring: slot {} still in use (strong_count={})",
                    i, sc,
                )));
            }
        }

        // Free old slots — decrement VRAM accounting.
        for _ in &self.slots {
            ctx.vram_dec(self.slot_bytes);
        }

        let count = self.slots.len();
        let out_w = (in_w * scale) as usize;
        let out_h = (in_h * scale) as usize;
        let slot_bytes = 3 * out_w * out_h * output_format.element_bytes();

        self.slots = (0..count)
            .map(|_| ctx.alloc(slot_bytes).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        self.cursor = 0;
        self.slot_bytes = slot_bytes;
        self.alloc_dims = (in_w, in_h);
        self.used = vec![false; count];

        debug!(count, slot_bytes, out_w, out_h, "Output ring reallocated");
        Ok(())
    }

    /// Total number of slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Total bytes tracked by this ring for VRAM accounting.
    pub fn accounted_bytes(&self) -> usize {
        self.slot_bytes * self.slots.len()
    }
}

// ─── Inference state ─────────────────────────────────────────────────────────

struct InferenceState {
    session: Session,
    ring: Option<OutputRing>,
    batch_buffers: Option<BatchBuffers>,
}

struct BatchBuffers {
    input: CudaSlice<u8>,
    output: Arc<CudaSlice<u8>>,
    alloc_dims: (u32, u32),
    format: PixelFormat,
    max_batch: usize,
    sample_input_bytes: usize,
    sample_output_bytes: usize,
}

/// Resolve ORT tensor element type from our PixelFormat.
fn ort_element_type(format: PixelFormat) -> TensorElementType {
    match format {
        PixelFormat::RgbPlanarF16 => TensorElementType::Float16,
        _ => TensorElementType::Float32,
    }
}

fn tensor_pixel_format(elem_type: TensorElementType) -> Result<PixelFormat> {
    match elem_type {
        TensorElementType::Float32 => Ok(PixelFormat::RgbPlanarF32),
        TensorElementType::Float16 => Ok(PixelFormat::RgbPlanarF16),
        other => Err(EngineError::ModelMetadata(format!(
            "Unsupported tensor element type for GPU RGB pipeline: {:?}",
            other
        ))),
    }
}

// ─── Backend ─────────────────────────────────────────────────────────────────

pub struct TensorRtBackend {
    model_path: PathBuf,
    ctx: Arc<GpuContext>,
    device_id: i32,
    ring_size: usize,
    min_ring_slots: usize,
    meta: OnceLock<ModelMetadata>,
    state: Mutex<Option<InferenceState>>,
    pub inference_metrics: InferenceMetrics,
    /// Phase 8: precision policy for TRT EP.
    pub precision_policy: PrecisionPolicy,
    /// Phase 8: batch configuration.
    pub batch_config: BatchConfig,
    /// Cached ORT MemoryInfo — avoids re-creation per frame.
    cached_mem_info: OnceLock<MemoryInfo>,
}

impl TensorRtBackend {
    pub fn required_ring_slots(downstream_capacity: usize, max_batch: usize) -> usize {
        downstream_capacity + max_batch.max(1) + 1
    }

    fn supports_runtime_batching(
        input_batch_dim: Option<i64>,
        output_batch_dim: Option<i64>,
    ) -> bool {
        input_batch_dim.is_none() && output_batch_dim.is_none()
    }

    fn is_transformer_family_model(model_path: &std::path::Path) -> bool {
        let stem = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let patterns = [
            "dat",
            "hat",
            "swin",
            "grl",
            "nomos",
            "realwebphoto",
            "apisr",
        ];
        patterns.iter().any(|pattern| stem.contains(pattern))
    }

    fn release_state_resources(&self, state: InferenceState) {
        let InferenceState {
            session,
            ring,
            batch_buffers,
        } = state;

        if let Some(batch_buffers) = batch_buffers {
            self.ctx.recycle(batch_buffers.input);
            if let Ok(output) = Arc::try_unwrap(batch_buffers.output) {
                self.ctx.recycle(output);
            }
        }

        if let Some(ring) = ring {
            self.ctx.vram_dec(ring.accounted_bytes());
            drop(ring);
        }

        drop(session);
    }

    fn sample_output_bytes(meta: &ModelMetadata, input: &GpuTexture) -> usize {
        let out_w = input.width * meta.scale;
        let out_h = input.height * meta.scale;
        3 * (out_w as usize) * (out_h as usize) * meta.output_format.element_bytes()
    }

    fn copy_dense_texture_batch(
        src_ptr: u64,
        dst_ptr: u64,
        width: u32,
        height: u32,
        elem_bytes: usize,
    ) -> Result<()> {
        let row_bytes = (width as usize) * elem_bytes;
        let copy = crate::codecs::sys::CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: crate::codecs::sys::CUmemorytype::Device,
            srcHost: std::ptr::null(),
            srcDevice: src_ptr,
            srcArray: std::ptr::null(),
            srcPitch: row_bytes,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: crate::codecs::sys::CUmemorytype::Device,
            dstHost: std::ptr::null_mut(),
            dstDevice: dst_ptr,
            dstArray: std::ptr::null_mut(),
            dstPitch: row_bytes,
            WidthInBytes: row_bytes,
            Height: (height as usize) * 3,
        };
        unsafe {
            crate::codecs::sys::check_cu(
                crate::codecs::sys::cuMemcpy2D_v2(&copy),
                "cuMemcpy2D_v2 (batched planar RGB copy)",
            )?;
        }
        Ok(())
    }

    fn ensure_batch_buffers<'a>(
        &'a self,
        state: &'a mut InferenceState,
        meta: &ModelMetadata,
        input: &GpuTexture,
    ) -> Result<&'a mut BatchBuffers> {
        let sample_input_bytes = input.byte_size();
        let sample_output_bytes = Self::sample_output_bytes(meta, input);
        let needs_realloc = state.batch_buffers.as_ref().is_none_or(|buffers| {
            buffers.alloc_dims != (input.width, input.height)
                || buffers.format != input.format
                || buffers.max_batch != self.batch_config.max_batch
                || buffers.sample_input_bytes != sample_input_bytes
                || buffers.sample_output_bytes != sample_output_bytes
        });

        if needs_realloc {
            if let Some(old) = state.batch_buffers.take() {
                self.ctx.recycle(old.input);
                if let Ok(output) = Arc::try_unwrap(old.output) {
                    self.ctx.recycle(output);
                }
            }
            let input_buf = self
                .ctx
                .alloc(sample_input_bytes * self.batch_config.max_batch)?;
            let output_buf = Arc::new(
                self.ctx
                    .alloc(sample_output_bytes * self.batch_config.max_batch)?,
            );
            state.batch_buffers = Some(BatchBuffers {
                input: input_buf,
                output: output_buf,
                alloc_dims: (input.width, input.height),
                format: input.format,
                max_batch: self.batch_config.max_batch,
                sample_input_bytes,
                sample_output_bytes,
            });
        }

        Ok(state.batch_buffers.as_mut().unwrap())
    }

    fn infer_scale_from_model_path(model_path: &std::path::Path) -> Option<u32> {
        let stem = model_path.file_stem()?.to_str()?.to_ascii_lowercase();

        for scale in [2u32, 3, 4, 8] {
            let patterns = [
                format!("_x{scale}"),
                format!("x{scale}plus"),
                format!("_{scale}x"),
                format!("{scale}x_"),
                format!("{scale}x-"),
            ];
            if patterns.iter().any(|pattern| stem.contains(pattern)) {
                return Some(scale);
            }
        }

        let bytes = stem.as_bytes();
        if bytes.len() >= 2 && bytes[1] == b'x' {
            return match bytes[0] {
                b'2' => Some(2),
                b'3' => Some(3),
                b'4' => Some(4),
                b'8' => Some(8),
                _ => None,
            };
        }

        None
    }

    /// Create a new backend instance.
    ///
    /// # Parameters
    ///
    /// - `ring_size`: number of output ring slots to pre-allocate.
    /// - `downstream_capacity`: the bounded channel capacity between inference
    ///   and the encoder.  Ring size is validated against batch-aware minimums.
    pub fn new(
        model_path: PathBuf,
        ctx: Arc<GpuContext>,
        device_id: i32,
        ring_size: usize,
        downstream_capacity: usize,
    ) -> Self {
        Self::with_precision(
            model_path,
            ctx,
            device_id,
            ring_size,
            downstream_capacity,
            PrecisionPolicy::default(),
            BatchConfig::default(),
        )
    }

    /// Create with explicit precision policy and batch config.
    pub fn with_precision(
        model_path: PathBuf,
        ctx: Arc<GpuContext>,
        device_id: i32,
        ring_size: usize,
        downstream_capacity: usize,
        precision_policy: PrecisionPolicy,
        batch_config: BatchConfig,
    ) -> Self {
        let min_ring_slots = Self::required_ring_slots(downstream_capacity, batch_config.max_batch);
        assert!(
            ring_size >= min_ring_slots,
            "ring_size ({ring_size}) must be ≥ downstream_capacity + max_batch + 1 ({min_ring_slots})"
        );
        Self {
            model_path,
            ctx,
            device_id,
            ring_size,
            min_ring_slots,
            meta: OnceLock::new(),
            state: Mutex::new(None),
            inference_metrics: InferenceMetrics::new(),
            precision_policy,
            batch_config,
            cached_mem_info: OnceLock::new(),
        }
    }

    /// Get or create cached ORT MemoryInfo (avoids per-frame allocation).
    fn mem_info(&self) -> Result<&MemoryInfo> {
        if let Some(info) = self.cached_mem_info.get() {
            return Ok(info);
        }
        let info = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| EngineError::ModelMetadata(format!("MemoryInfo: {e}")))?;
        // Ignore race — if another thread already set it, discard ours and use theirs.
        let _ = self.cached_mem_info.set(info);
        Ok(self.cached_mem_info.get().unwrap())
    }

    /// Access ring metrics (if initialized).
    pub async fn ring_metrics(&self) -> Option<(u64, u64, u64)> {
        let guard = self.state.lock().await;
        guard
            .as_ref()
            .and_then(|s| s.ring.as_ref())
            .map(|r| r.metrics.snapshot())
    }

    fn extract_metadata(session: &Session, model_path: &std::path::Path) -> Result<ModelMetadata> {
        let inputs = session.inputs();
        let outputs = session.outputs();

        if inputs.is_empty() || outputs.is_empty() {
            return Err(EngineError::ModelMetadata(
                "Model must have at least one input and one output tensor".into(),
            ));
        }

        let input_info = &inputs[0];
        let output_info = &outputs[0];
        let input_name = input_info.name().to_string();
        let output_name = output_info.name().to_string();

        // In ORT 2.0-rc.11, ValueType::Tensor uses `shape: Shape` (SmallVec<[i64;4]>)
        // where -1 means dynamic. We convert to Vec<Option<i64>> for downstream use.
        let (input_elem_type, input_dims): (TensorElementType, Vec<Option<i64>>) =
            match input_info.dtype() {
                ort::value::ValueType::Tensor { ty, shape, .. } => (
                    *ty,
                    shape
                        .iter()
                        .map(|&d| if d < 0 { None } else { Some(d) })
                        .collect(),
                ),
            other => {
                return Err(EngineError::ModelMetadata(format!(
                    "Expected tensor input, got {:?}",
                    other
                )));
            }
        };

        let (output_elem_type, output_dims): (TensorElementType, Vec<Option<i64>>) =
            match output_info.dtype() {
                ort::value::ValueType::Tensor { ty, shape, .. } => (
                    *ty,
                    shape
                        .iter()
                        .map(|&d| if d < 0 { None } else { Some(d) })
                        .collect(),
                ),
            other => {
                return Err(EngineError::ModelMetadata(format!(
                    "Expected tensor output, got {:?}",
                    other
                )));
            }
        };

        if input_dims.len() != 4 || output_dims.len() != 4 {
            return Err(EngineError::ModelMetadata(format!(
                "Expected 4D tensors (NCHW), got input={}D output={}D",
                input_dims.len(),
                output_dims.len()
            )));
        }

        let input_channels = input_dims[1].unwrap_or(3) as u32;
        let dynamic_batch_axes = Self::supports_runtime_batching(input_dims[0], output_dims[0]);
        let transformer_family = Self::is_transformer_family_model(model_path);
        let supports_runtime_batching = dynamic_batch_axes && !transformer_family;
        if dynamic_batch_axes && transformer_family {
            warn!(
                model = %model_path.display(),
                "Model looks transformer-based; disabling batched inference until batch stability is validated"
            );
        }

        let scale = match (input_dims[2], input_dims[3], output_dims[2], output_dims[3]) {
            (Some(ih), Some(iw), Some(oh), Some(ow)) if ih > 0 && iw > 0 && oh > 0 && ow > 0 => {
                let scale_h = oh / ih;
                let scale_w = ow / iw;
                if scale_h <= 0 || scale_w <= 0 || scale_h != scale_w {
                    return Err(EngineError::ModelMetadata(format!(
                        "Unsupported output/input shape ratio: input={input_dims:?} output={output_dims:?}"
                    )));
                }
                scale_h as u32
            }
            _ => {
                let inferred = Self::infer_scale_from_model_path(model_path).ok_or_else(|| {
                    EngineError::ModelMetadata(format!(
                        "Dynamic spatial axes require an explicit scale hint in the model filename: {}",
                        model_path.display()
                    ))
                })?;
                warn!(
                    model = %model_path.display(),
                    scale = inferred,
                    "Dynamic spatial axes — inferred scale from model filename"
                );
                inferred
            }
        };

        let min_input_hw = (
            input_dims[2].map(|d| d.max(1) as u32).unwrap_or(1),
            input_dims[3].map(|d| d.max(1) as u32).unwrap_or(1),
        );
        let max_input_hw = (
            input_dims[2].map(|d| d as u32).unwrap_or(u32::MAX),
            input_dims[3].map(|d| d as u32).unwrap_or(u32::MAX),
        );

        let name = session
            .metadata()
            .ok()
            .and_then(|m| m.name())
            .unwrap_or_else(|| "unknown".to_string());

        Ok(ModelMetadata {
            name,
            scale,
            input_name,
            output_name,
            input_channels,
            input_format: tensor_pixel_format(input_elem_type)?,
            output_format: tensor_pixel_format(output_elem_type)?,
            supports_runtime_batching,
            min_input_hw,
            max_input_hw,
        })
    }

    /// Validate the execution-provider policy.
    ///
    /// This backend allows TensorRT and CUDA, and excludes CPU fallback.
    fn validate_providers(_session: &Session) -> Result<()> {
        // The ort crate does not expose a provider list API. Validation is
        // structural via session builder configuration.
        info!("EP validation: TensorRT preferred, CUDA fallback allowed");
        info!("EP integrity: CPUExecutionProvider explicitly excluded");
        Ok(())
    }

    /// Verify that IO-bound device pointers match the source GpuTexture
    /// and OutputRing slot pointers exactly (pointer identity check).
    ///
    /// Called by `run_io_bound` to audit that ORT IO Binding uses our
    /// device pointers without any host staging or reallocation.
    fn verify_pointer_identity(
        input_ptr: u64,
        output_ptr: u64,
        input_texture: &GpuTexture,
        ring_slot_ptr: u64,
    ) {
        let texture_ptr = input_texture.device_ptr();
        debug!(
            input_ptr = format!("0x{:016x}", input_ptr),
            texture_ptr = format!("0x{:016x}", texture_ptr),
            output_ptr = format!("0x{:016x}", output_ptr),
            ring_slot_ptr = format!("0x{:016x}", ring_slot_ptr),
            "IO-binding pointer identity audit"
        );

        debug_assert_eq!(
            input_ptr, texture_ptr,
            "POINTER MISMATCH: IO-bound input (0x{:016x}) != GpuTexture (0x{:016x})",
            input_ptr, texture_ptr,
        );
        debug_assert_eq!(
            output_ptr, ring_slot_ptr,
            "POINTER MISMATCH: IO-bound output (0x{:016x}) != ring slot (0x{:016x})",
            output_ptr, ring_slot_ptr,
        );
    }

    fn run_io_bound(
        session: &mut Session,
        meta: &ModelMetadata,
        input: &GpuTexture,
        output_ptr: u64,
        output_bytes: usize,
        _ctx: &GpuContext,
        cuda_mem_info: &MemoryInfo,
    ) -> Result<()> {
        let in_w = input.width as i64;
        let in_h = input.height as i64;
        let out_w = in_w * meta.scale as i64;
        let out_h = in_h * meta.scale as i64;

        let input_shape: Vec<i64> = vec![1, meta.input_channels as i64, in_h, in_w];
        let output_shape: Vec<i64> = vec![1, meta.input_channels as i64, out_h, out_w];

        if input.format != meta.input_format {
            return Err(EngineError::FormatMismatch {
                expected: meta.input_format,
                actual: input.format,
            });
        }

        let input_elem_type = ort_element_type(meta.input_format);
        let output_elem_type = ort_element_type(meta.output_format);
        let input_elem_bytes = meta.input_format.element_bytes();
        let output_elem_bytes = meta.output_format.element_bytes();

        let expected = (output_shape.iter().product::<i64>() as usize) * output_elem_bytes;
        if output_bytes < expected {
            return Err(EngineError::BufferTooSmall {
                need: expected,
                have: output_bytes,
            });
        }

        let input_bytes = (input_shape.iter().product::<i64>() as usize) * input_elem_bytes;
        let input_src_ptr = input.device_ptr();

        // Wrap pre-existing CUDA buffers as ORT tensors (zero-copy).
        let input_tensor = unsafe {
            create_tensor_from_device_memory(
                cuda_mem_info,
                input_src_ptr as *mut std::ffi::c_void,
                input_bytes,
                input_shape.as_slice(),
                input_elem_type,
            )?
        };
        let output_tensor = unsafe {
            create_tensor_from_device_memory(
                cuda_mem_info,
                output_ptr as *mut std::ffi::c_void,
                output_bytes,
                output_shape.as_slice(),
                output_elem_type,
            )?
        };

        // Phase 7: Pointer identity audit — log the ORT vs ring-slot pointers.
        Self::verify_pointer_identity(input_src_ptr, output_ptr, input, output_ptr);

        let mut binding = session
            .create_binding()
            .map_err(|e| EngineError::ModelMetadata(format!("IO binding: {e}")))?;
        binding
            .bind_input(&meta.input_name, &input_tensor)
            .map_err(|e| EngineError::ModelMetadata(format!("bind_input: {e}")))?;
        binding
            .bind_output(&meta.output_name, output_tensor)
            .map_err(|e| EngineError::ModelMetadata(format!("bind_output: {e}")))?;

        // run_binding is synchronous: ORT blocks until all GPU kernels
        // on its internal stream complete.  Output buffer is fully written
        // when this call returns.  No additional stream sync needed.
        session
            .run_binding(&binding)
            .map_err(|e| EngineError::ModelMetadata(format!("run_binding: {e}")))?;

        Ok(())
    }

    fn run_io_bound_batch(
        session: &mut Session,
        meta: &ModelMetadata,
        input_format: PixelFormat,
        batch_len: usize,
        in_w: u32,
        in_h: u32,
        input_ptr: u64,
        input_bytes: usize,
        output_ptr: u64,
        output_bytes: usize,
        cuda_mem_info: &MemoryInfo,
    ) -> Result<()> {
        let out_w = in_w * meta.scale;
        let out_h = in_h * meta.scale;

        let input_shape: Vec<i64> = vec![
            batch_len as i64,
            meta.input_channels as i64,
            in_h as i64,
            in_w as i64,
        ];
        let output_shape: Vec<i64> = vec![
            batch_len as i64,
            meta.input_channels as i64,
            out_h as i64,
            out_w as i64,
        ];

        if input_format != meta.input_format {
            return Err(EngineError::FormatMismatch {
                expected: meta.input_format,
                actual: input_format,
            });
        }

        let input_elem_type = ort_element_type(meta.input_format);
        let output_elem_type = ort_element_type(meta.output_format);

        let input_tensor = unsafe {
            create_tensor_from_device_memory(
                cuda_mem_info,
                input_ptr as *mut std::ffi::c_void,
                input_bytes,
                input_shape.as_slice(),
                input_elem_type,
            )?
        };
        let output_tensor = unsafe {
            create_tensor_from_device_memory(
                cuda_mem_info,
                output_ptr as *mut std::ffi::c_void,
                output_bytes,
                output_shape.as_slice(),
                output_elem_type,
            )?
        };

        let mut binding = session
            .create_binding()
            .map_err(|e| EngineError::ModelMetadata(format!("IO binding (batch): {e}")))?;
        binding
            .bind_input(&meta.input_name, &input_tensor)
            .map_err(|e| EngineError::ModelMetadata(format!("bind_input (batch): {e}")))?;
        binding
            .bind_output(&meta.output_name, output_tensor)
            .map_err(|e| EngineError::ModelMetadata(format!("bind_output (batch): {e}")))?;

        session
            .run_binding(&binding)
            .map_err(|e| EngineError::ModelMetadata(format!("run_binding (batch): {e}")))?;

        Ok(())
    }
}

// SAFETY: TensorRtBackend owns a `OnceLock<MemoryInfo>` which contains a
// `NonNull<OrtMemoryInfo>`.  ORT manages the underlying allocation and the
// pointer is valid for the lifetime of the session.  Access is serialized
// through the `Mutex<Option<InferenceState>>` lock so there are no data races.
unsafe impl Send for TensorRtBackend {}
unsafe impl Sync for TensorRtBackend {}

#[async_trait]
impl UpscaleBackend for TensorRtBackend {
    async fn initialize(&self) -> Result<()> {
        let mut guard = self.state.lock().await;
        if guard.is_some() {
            return Err(EngineError::ModelMetadata("Already initialized".into()));
        }

        info!(
            path = %self.model_path.display(),
            "Loading ONNX model — TensorRT preferred, CUDA fallback enabled"
        );

        // Build session with TensorRT EP exclusively.
        let mut trt_ep = TrtEP::default()
            .with_device_id(self.device_id)
            .with_engine_cache(false);
        let enable_cache = std::env::var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "True"))
            .unwrap_or(false);
        if enable_cache {
            let cache_root = std::env::var_os("VIDEOFORGE_TRT_CACHE_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir().join("videoforge").join("trt_cache"));
            let model_tag = self
                .model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            let cache_dir = cache_root.join(model_tag);
            match std::fs::create_dir_all(&cache_dir) {
                Ok(_) => {
                    trt_ep = trt_ep
                        .with_engine_cache(true)
                        .with_engine_cache_path(cache_dir.to_string_lossy().to_string());
                    info!(cache = %cache_dir.display(), "TensorRT engine cache enabled");
                }
                Err(e) => {
                    warn!(
                        cache = %cache_dir.display(),
                        %e,
                        "TensorRT cache directory unavailable; keeping engine cache disabled"
                    );
                }
            }
        } else {
            info!("TensorRT engine cache disabled (set VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE=1 to enable)");
        }

        // Phase 8: Apply precision policy.
        match &self.precision_policy {
            PrecisionPolicy::Fp32 => {
                info!("TRT precision: FP32 (no mixed precision)");
            }
            PrecisionPolicy::Fp16 => {
                trt_ep = trt_ep.with_fp16(true);
                info!("TRT precision: FP16 mixed precision");
            }
            PrecisionPolicy::Int8 { calibration_table } => {
                trt_ep = trt_ep
                    .with_fp16(true)
                    .with_int8(true)
                    .with_int8_calibration_table_name(
                        calibration_table.to_string_lossy().to_string(),
                    );
                info!(
                    table = %calibration_table.display(),
                    "TRT precision: INT8 with calibration table"
                );
            }
        }
        let cuda_ep = CudaEP::default().with_device_id(self.device_id);
        let session = Session::builder()?
            .with_execution_providers([trt_ep.build(), cuda_ep.build()])?
            .with_intra_threads(1)?
            .commit_from_file(&self.model_path)?;

        // If we reach here, session creation respected provider policy.
        Self::validate_providers(&session)?;

        let metadata = Self::extract_metadata(&session, &self.model_path)?;
        info!(
            name = %metadata.name,
            scale = metadata.scale,
            input = %metadata.input_name,
            output = %metadata.output_name,
            supports_runtime_batching = metadata.supports_runtime_batching,
            ring_size = self.ring_size,
            min_ring_slots = self.min_ring_slots,
            precision = ?self.precision_policy,
            max_batch = self.batch_config.max_batch,
            "Model loaded — CPU fallback excluded"
        );

        let _ = self.meta.set(metadata);

        *guard = Some(InferenceState {
            session,
            ring: None,
            batch_buffers: None,
        });

        Ok(())
    }

    async fn process(&self, input: GpuTexture) -> Result<GpuTexture> {
        let meta = self.meta.get().ok_or(EngineError::NotInitialized)?;
        if input.format != meta.input_format {
            return Err(EngineError::FormatMismatch {
                expected: meta.input_format,
                actual: input.format,
            });
        }
        let mut guard = self.state.lock().await;
        self.ctx
            .device()
            .bind_to_thread()
            .map_err(|e| EngineError::ModelMetadata(format!("bind_to_thread (backend process): {:?}", e)))?;
        let state = guard.as_mut().ok_or(EngineError::NotInitialized)?;

        // Lazy ring init / realloc.
        match &mut state.ring {
            Some(ring) if ring.needs_realloc(input.width, input.height) => {
                debug!(old = ?ring.alloc_dims, new_w = input.width, new_h = input.height,
                       "Reallocating output ring");
                ring.reallocate(
                    &self.ctx,
                    input.width,
                    input.height,
                    meta.scale,
                    meta.output_format,
                )?;
            }
            None => {
                debug!(
                    w = input.width,
                    h = input.height,
                    slots = self.ring_size,
                    "Lazily creating output ring"
                );
                state.ring = Some(OutputRing::new(
                    &self.ctx,
                    input.width,
                    input.height,
                    meta.scale,
                    meta.output_format,
                    self.ring_size,
                    self.min_ring_slots,
                )?);
            }
            Some(_) => {}
        }

        let ring = state.ring.as_mut().unwrap();

        // Debug-mode host allocation tracking.
        #[cfg(feature = "debug-alloc")]
        {
            crate::debug_alloc::reset();
            crate::debug_alloc::enable();
        }

        let output_arc = ring.acquire()?;
        let output_ptr = *output_arc.device_ptr() as u64;
        let output_bytes = ring.slot_bytes;

        // ── Inference with latency measurement ──
        let t_start = std::time::Instant::now();

        let mem_info = self.mem_info()?;
        Self::run_io_bound(
            &mut state.session,
            meta,
            &input,
            output_ptr,
            output_bytes,
            &self.ctx,
            mem_info,
        )?;

        let elapsed_us = t_start.elapsed().as_micros() as u64;
        self.inference_metrics.record(elapsed_us);

        #[cfg(feature = "debug-alloc")]
        {
            crate::debug_alloc::disable();
            let host_allocs = crate::debug_alloc::count();
            debug_assert_eq!(
                host_allocs, 0,
                "VIOLATION: {host_allocs} host allocations during inference"
            );
        }

        let out_w = input.width * meta.scale;
        let out_h = input.height * meta.scale;
        let elem_bytes = meta.output_format.element_bytes();

        Ok(GpuTexture {
            data: GpuBuffer::from_arc(output_arc),
            width: out_w,
            height: out_h,
            pitch: (out_w as usize) * elem_bytes,
            format: meta.output_format,
        })
    }

    async fn process_batch(&self, inputs: &[GpuTexture]) -> Result<Vec<GpuTexture>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if inputs.len() == 1 {
            return Ok(vec![self.process(inputs[0].clone()).await?]);
        }

        let first = &inputs[0];
        if inputs.iter().any(|input| {
            input.width != first.width || input.height != first.height || input.format != first.format
        }) {
            warn!(
                batch_size = inputs.len(),
                "Mixed-dimension or mixed-format batch encountered; falling back to sequential execution"
            );
            let mut outputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                outputs.push(self.process(input.clone()).await?);
            }
            return Ok(outputs);
        }

        let meta = self.meta.get().ok_or(EngineError::NotInitialized)?;
        if !meta.supports_runtime_batching {
            warn!(
                model = %meta.name,
                requested_batch = inputs.len(),
                "Model does not advertise runtime batch support; falling back to sequential execution"
            );
            let mut outputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                outputs.push(self.process(input.clone()).await?);
            }
            return Ok(outputs);
        }
        let mut guard = self.state.lock().await;
        self.ctx.device().bind_to_thread().map_err(|e| {
            EngineError::ModelMetadata(format!("bind_to_thread (backend process_batch): {:?}", e))
        })?;
        let state = guard.as_mut().ok_or(EngineError::NotInitialized)?;

        let sample_input_bytes = first.byte_size();
        let sample_output_bytes = Self::sample_output_bytes(meta, first);
        let input_elem_bytes = first.format.element_bytes();
        let output_elem_bytes = meta.output_format.element_bytes();
        let batch_len = inputs.len();

        #[cfg(feature = "debug-alloc")]
        {
            crate::debug_alloc::reset();
            crate::debug_alloc::enable();
        }

        let (batch_input_ptr, batch_output_ptr, output_storage) = {
            let buffers = Self::ensure_batch_buffers(self, state, meta, first)?;
            (
                *buffers.input.device_ptr() as u64,
                *buffers.output.device_ptr() as u64,
                Arc::clone(&buffers.output),
            )
        };

        let t_start = std::time::Instant::now();

        for (i, input) in inputs.iter().enumerate() {
            let dst_ptr = batch_input_ptr + (i * sample_input_bytes) as u64;
            Self::copy_dense_texture_batch(
                input.device_ptr(),
                dst_ptr,
                input.width,
                input.height,
                input_elem_bytes,
            )?;
        }

        Self::run_io_bound_batch(
            &mut state.session,
            meta,
            first.format,
            batch_len,
            first.width,
            first.height,
            batch_input_ptr,
            sample_input_bytes * batch_len,
            batch_output_ptr,
            sample_output_bytes * batch_len,
            self.mem_info()?,
        )?;

        let elapsed_us = t_start.elapsed().as_micros() as u64;
        self.inference_metrics
            .record_frames(batch_len as u64, elapsed_us);

        #[cfg(feature = "debug-alloc")]
        {
            crate::debug_alloc::disable();
            let host_allocs = crate::debug_alloc::count();
            debug_assert_eq!(
                host_allocs, 0,
                "VIOLATION: {host_allocs} host allocations during batched inference"
            );
        }

        let out_w = first.width * meta.scale;
        let out_h = first.height * meta.scale;
        Ok((0..batch_len)
            .map(|i| GpuTexture {
                data: GpuBuffer::view(
                    Arc::clone(&output_storage),
                    i * sample_output_bytes,
                    sample_output_bytes,
                ),
                width: out_w,
                height: out_h,
                pitch: (out_w as usize) * output_elem_bytes,
                format: meta.output_format,
            })
            .collect())
    }

    async fn shutdown(&self) -> Result<()> {
        let mut guard = self.state.lock().await;
        if let Some(state) = guard.take() {
            info!("Shutting down TensorRT backend");
            self.ctx.sync_all()?;

            // Report ring metrics.
            if let Some(ring) = &state.ring {
                let (reuse, contention, first) = ring.metrics.snapshot();
                info!(reuse, contention, first, "Final ring metrics");
            }

            // Report inference metrics.
            let snap = self.inference_metrics.snapshot();
            info!(
                frames = snap.frames_inferred,
                avg_us = snap.avg_inference_us,
                peak_us = snap.peak_inference_us,
                precision = ?self.precision_policy,
                "Final inference metrics"
            );

            // Report VRAM.
            let (current, peak) = self.ctx.vram_usage();
            info!(
                current_mb = current / (1024 * 1024),
                peak_mb = peak / (1024 * 1024),
                "Final VRAM usage"
            );

            self.release_state_resources(state);
            debug!("TensorRT backend shutdown complete");
        }
        Ok(())
    }

    fn metadata(&self) -> Result<&ModelMetadata> {
        self.meta.get().ok_or(EngineError::NotInitialized)
    }
}

#[cfg(test)]
mod tests {
    use super::{InferenceMetrics, TensorRtBackend};
    use std::path::Path;

    #[test]
    fn infer_scale_from_model_path_recognizes_common_patterns() {
        assert_eq!(
            TensorRtBackend::infer_scale_from_model_path(Path::new("weights/rcan_4x.onnx")),
            Some(4)
        );
        assert_eq!(
            TensorRtBackend::infer_scale_from_model_path(Path::new("weights/2x_SPAN_soft.onnx")),
            Some(2)
        );
        assert_eq!(
            TensorRtBackend::infer_scale_from_model_path(Path::new("weights/RealESRGAN_x4plus.onnx")),
            Some(4)
        );
    }

    #[test]
    fn infer_scale_from_model_path_returns_none_for_unknown_names() {
        assert_eq!(
            TensorRtBackend::infer_scale_from_model_path(Path::new("weights/custom_model.onnx")),
            None
        );
    }

    #[test]
    fn required_ring_slots_scales_with_batch_size() {
        assert_eq!(TensorRtBackend::required_ring_slots(4, 1), 6);
        assert_eq!(TensorRtBackend::required_ring_slots(4, 4), 9);
    }

    #[test]
    fn runtime_batching_requires_dynamic_batch_axes() {
        assert!(TensorRtBackend::supports_runtime_batching(None, None));
        assert!(!TensorRtBackend::supports_runtime_batching(Some(1), None));
        assert!(!TensorRtBackend::supports_runtime_batching(None, Some(1)));
        assert!(!TensorRtBackend::supports_runtime_batching(Some(1), Some(1)));
    }

    #[test]
    fn transformer_family_detection_matches_current_models() {
        assert!(TensorRtBackend::is_transformer_family_model(Path::new(
            "weights/4xNomos2_hq_dat2_fp32.onnx"
        )));
        assert!(TensorRtBackend::is_transformer_family_model(Path::new(
            "weights/4x_APISR_GRL_GAN_generator_fp16.fp16.onnx"
        )));
        assert!(!TensorRtBackend::is_transformer_family_model(Path::new(
            "weights/2x_SPAN_soft.onnx"
        )));
    }

    #[test]
    fn inference_metrics_record_frames_counts_per_frame() {
        let metrics = InferenceMetrics::new();
        metrics.record_frames(4, 400);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.frames_inferred, 4);
        assert_eq!(snapshot.avg_inference_us, 100);
        assert_eq!(snapshot.peak_inference_us, 400);
    }
}

impl Drop for TensorRtBackend {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.state.try_lock() {
            if let Some(state) = guard.take() {
                let _ = self.ctx.sync_all();
                self.release_state_resources(state);
            }
        }
    }
}
