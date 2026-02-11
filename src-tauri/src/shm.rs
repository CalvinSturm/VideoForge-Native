use anyhow::{anyhow, Result};
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU32, Ordering};
use thiserror::Error;

pub const RING_SIZE: usize = 3;

// =============================================================================
// SLOT STATE MACHINE
// =============================================================================

/// Slot is free — decoder may write.
pub const SLOT_EMPTY: u32 = 0;
/// Rust decoder is writing a frame into this slot.
pub const SLOT_RUST_WRITING: u32 = 1;
/// Frame is ready for Python AI inference.
pub const SLOT_READY_FOR_AI: u32 = 2;
/// Python is processing this slot.
pub const SLOT_AI_PROCESSING: u32 = 3;
/// AI output is ready — encoder may read.
pub const SLOT_READY_FOR_ENCODE: u32 = 4;
/// Rust encoder is reading from this slot.
pub const SLOT_ENCODING: u32 = 5;

// =============================================================================
// SLOT HEADER LAYOUT (per slot, 16 bytes)
// =============================================================================
//
// Offset 0:  u32  write_index   — frame counter (set by Rust decoder)
// Offset 4:  u32  read_index    — frame counter (set by Rust encoder)
// Offset 8:  u32  state         — atomic state machine value
// Offset 12: u32  frame_bytes   — actual frame data size in bytes
//
// All fields use SeqCst ordering for cross-process visibility.

const SLOT_HEADER_SIZE: usize = 16; // 4 × u32
/// Total bytes reserved for all slot headers at the start of the SHM file.
pub const HEADER_REGION_SIZE: usize = SLOT_HEADER_SIZE * RING_SIZE;

// Byte offsets within a single SlotHeader
const STATE_OFFSET: usize = 8;
const FRAME_BYTES_OFFSET: usize = 12;
const WRITE_INDEX_OFFSET: usize = 0;

// =============================================================================
// SHM ERROR
// =============================================================================

#[derive(Error, Debug)]
pub enum ShmError {
    #[error("SHM slot index {index} out of bounds (max {max})")]
    IndexOutOfBounds { index: usize, max: usize },
}

// =============================================================================
// VIDEO SHM
// =============================================================================

pub struct VideoShm {
    pub mmap: MmapMut,
    pub width: usize,
    pub height: usize,
    pub scale: usize,

    slot_input_size: usize,
    slot_output_size: usize,
    slot_total_size: usize,
}

// MmapMut is Send/Sync safe for our use case (synchronization via atomic states)
unsafe impl Send for VideoShm {}
unsafe impl Sync for VideoShm {}

impl VideoShm {
    /// Open a Raw File Mapping created by Python.
    ///
    /// Layout:
    /// ```text
    /// [ SlotHeader × RING_SIZE (48 bytes) ]
    /// [ Slot 0: input (W×H×3) | output (sW×sH×3) ]
    /// [ Slot 1: input | output ]
    /// [ Slot 2: input | output ]
    /// ```
    pub fn open(file_path: &str, width: usize, height: usize, scale: usize) -> Result<Self> {
        let slot_input_size = width * height * 3;
        let slot_output_size = (width * scale) * (height * scale) * 3;
        let slot_total_size = slot_input_size + slot_output_size;
        let total_size = HEADER_REGION_SIZE + slot_total_size * RING_SIZE;

        println!(
            "Rust: Opening SHM File: {} (Expect {} bytes, header={})",
            file_path, total_size, HEADER_REGION_SIZE
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

    // -------------------------------------------------------------------------
    // SLOT STATE ACCESSORS (atomic, cross-process safe)
    // -------------------------------------------------------------------------

    /// Byte offset of a field within the header for slot `index`.
    #[inline]
    fn header_field_ptr(&self, index: usize, field_offset: usize) -> *const AtomicU32 {
        let byte_offset = index * SLOT_HEADER_SIZE + field_offset;
        // SAFETY: header region is at the start of the mmap, properly aligned
        // (16-byte slot headers on 4-byte boundaries). Both Rust and Python
        // access these via atomic u32 operations.
        unsafe { self.mmap.as_ptr().add(byte_offset) as *const AtomicU32 }
    }

    /// Read the atomic state of slot `index`.
    pub fn slot_state(&self, index: usize) -> u32 {
        assert!(index < RING_SIZE, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, STATE_OFFSET) };
        atom.load(Ordering::SeqCst)
    }

    /// Set the atomic state of slot `index`.
    pub fn set_slot_state(&self, index: usize, state: u32) {
        assert!(index < RING_SIZE, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, STATE_OFFSET) };
        atom.store(state, Ordering::SeqCst);
    }

    /// Compare-and-swap the slot state. Returns true if the swap succeeded.
    pub fn cas_slot_state(&self, index: usize, expected: u32, new: u32) -> bool {
        assert!(index < RING_SIZE, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, STATE_OFFSET) };
        atom.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Read the frame_bytes field for slot `index`.
    pub fn slot_frame_bytes(&self, index: usize) -> u32 {
        assert!(index < RING_SIZE, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, FRAME_BYTES_OFFSET) };
        atom.load(Ordering::SeqCst)
    }

    /// Set the frame_bytes field for slot `index`.
    pub fn set_slot_frame_bytes(&self, index: usize, bytes: u32) {
        assert!(index < RING_SIZE, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, FRAME_BYTES_OFFSET) };
        atom.store(bytes, Ordering::SeqCst);
    }

    /// Set the write_index (frame counter) for slot `index`.
    pub fn set_slot_write_index(&self, index: usize, frame_id: u32) {
        assert!(index < RING_SIZE, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, WRITE_INDEX_OFFSET) };
        atom.store(frame_id, Ordering::SeqCst);
    }

    /// Reset all slot headers to EMPTY state.
    pub fn reset_all_slots(&self) {
        for i in 0..RING_SIZE {
            self.set_slot_state(i, SLOT_EMPTY);
            self.set_slot_frame_bytes(i, 0);
            self.set_slot_write_index(i, 0);
        }
    }

    // -------------------------------------------------------------------------
    // DATA REGION ACCESSORS (offset by HEADER_REGION_SIZE)
    // -------------------------------------------------------------------------

    pub fn input_slot_mut(&mut self, index: usize) -> Result<&mut [u8], ShmError> {
        if index >= RING_SIZE {
            return Err(ShmError::IndexOutOfBounds {
                index,
                max: RING_SIZE - 1,
            });
        }
        let offset = HEADER_REGION_SIZE + index * self.slot_total_size;
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
        let offset = HEADER_REGION_SIZE + (index * self.slot_total_size) + self.slot_input_size;
        let end = offset + self.slot_output_size;
        Ok(&self.mmap[offset..end])
    }
}
