//! Upscale backend trait — the GPU-only inference contract.

use async_trait::async_trait;

use crate::error::Result;
use crate::types::GpuTexture;

/// Metadata extracted from an ONNX model's input/output tensor descriptors.
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub name: String,
    pub scale: u32,
    pub input_name: String,
    pub output_name: String,
    pub input_channels: u32,
    pub min_input_hw: (u32, u32),
    pub max_input_hw: (u32, u32),
}

/// GPU-only super-resolution inference backend.
#[async_trait]
pub trait UpscaleBackend: Send + Sync {
    async fn initialize(&self) -> Result<()>;
    async fn process(&self, input: GpuTexture) -> Result<GpuTexture>;
    async fn shutdown(&self) -> Result<()>;
    fn metadata(&self) -> Result<&ModelMetadata>;
}
