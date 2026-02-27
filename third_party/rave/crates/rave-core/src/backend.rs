//! Upscale backend trait — the GPU-only inference contract.

use async_trait::async_trait;

use crate::error::Result;
use crate::types::GpuTexture;

/// Metadata extracted from an ONNX model's input/output tensor descriptors.
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    /// Model identifier string from the ONNX graph name field.
    pub name: String,
    /// Spatial upscale factor (e.g. `2` for 2×, `4` for 4×).
    pub scale: u32,
    /// Name of the model's input tensor node.
    pub input_name: String,
    /// Name of the model's output tensor node.
    pub output_name: String,
    /// Number of input channels (typically `3` for RGB planar).
    pub input_channels: u32,
    /// Minimum supported input resolution as `(height, width)`.
    pub min_input_hw: (u32, u32),
    /// Maximum supported input resolution as `(height, width)`.
    pub max_input_hw: (u32, u32),
}

/// GPU-only super-resolution inference backend.
///
/// All methods operate entirely on the GPU — no host staging, no implicit
/// device synchronization outside of shutdown.
#[async_trait]
pub trait UpscaleBackend: Send + Sync {
    /// Warm up the backend: load the model, allocate buffers, build engine plans.
    async fn initialize(&self) -> Result<()>;
    /// Run a single upscale pass on the given GPU texture, returning the upscaled output.
    async fn process(&self, input: GpuTexture) -> Result<GpuTexture>;
    /// Run a batched upscale pass on multiple GPU textures.
    ///
    /// The default implementation calls [`process`](Self::process) sequentially.
    /// Backends with native batch support (e.g. TensorRT) should override this
    /// to issue a single batched inference call for better throughput.
    ///
    /// All inputs **must** share the same `(width, height, format)`.
    async fn process_batch(&self, inputs: Vec<GpuTexture>) -> Result<Vec<GpuTexture>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.process(input).await?);
        }
        Ok(outputs)
    }
    /// Flush any pending work, synchronize streams, and release GPU resources.
    async fn shutdown(&self) -> Result<()>;
    /// Return the model metadata extracted during [`initialize`](Self::initialize).
    fn metadata(&self) -> Result<&ModelMetadata>;
}
