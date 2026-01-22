use anyhow::{anyhow, Result};
use memmap2::MmapMut;
use std::fs::OpenOptions;
use thiserror::Error;

pub const RING_SIZE: usize = 3;

#[derive(Error, Debug)]
pub enum ShmError {
    #[error("SHM slot index {index} out of bounds (max {max})")]
    IndexOutOfBounds { index: usize, max: usize },
}

pub struct VideoShm {
    // We hold the MmapMut which keeps the mapping alive
    pub mmap: MmapMut,
    pub width: usize,
    pub height: usize,
    pub scale: usize,

    slot_input_size: usize,
    slot_output_size: usize,
    slot_total_size: usize,
}

// MmapMut is Send/Sync safe for our use case (Mutex protected in lib.rs)
unsafe impl Send for VideoShm {}
unsafe impl Sync for VideoShm {}

impl VideoShm {
    /// Open a Raw File Mapping created by Python
    pub fn open(file_path: &str, width: usize, height: usize, scale: usize) -> Result<Self> {
        let slot_input_size = width * height * 4;
        let slot_output_size = (width * scale) * (height * scale) * 4;
        let slot_total_size = slot_input_size + slot_output_size;
        let total_size = slot_total_size * RING_SIZE;

        println!(
            "Rust: Opening SHM File: {} (Expect {} bytes)",
            file_path, total_size
        );

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(file_path)
            .map_err(|e| anyhow!("Failed to open SHM file: {}", e))?;

        // Verify size
        let meta = file.metadata()?;
        if meta.len() < total_size as u64 {
            return Err(anyhow!(
                "SHM File too small. Expected {}, got {}",
                total_size,
                meta.len()
            ));
        }

        // Map the file
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(Self {
            mmap,
            width,
            height,
            scale,
            slot_input_size,
            slot_output_size,
            slot_total_size,
        })
    }

    pub fn input_slot_mut(&mut self, index: usize) -> Result<&mut [u8], ShmError> {
        if index >= RING_SIZE {
            return Err(ShmError::IndexOutOfBounds {
                index,
                max: RING_SIZE - 1,
            });
        }
        let offset = index * self.slot_total_size;
        let end = offset + self.slot_input_size;
        Ok(&mut self.mmap[offset..end])
    }

    pub fn output_slot(&self, index: usize) -> Result<&[u8], ShmError> {
        if index >= RING_SIZE {
            return Err(ShmError::IndexOutOfBounds {
                index,
                max: RING_SIZE - 1,
            });
        }
        let offset = (index * self.slot_total_size) + self.slot_input_size;
        let end = offset + self.slot_output_size;
        Ok(&self.mmap[offset..end])
    }
}
