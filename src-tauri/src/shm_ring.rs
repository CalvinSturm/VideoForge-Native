//! Lock-Free SHM Ring Buffer for VideoForge Research Pipeline
//!
//! Provides a zero-copy, lock-free ring buffer over memory-mapped files.
//! The Rust side (producer) writes decoded frames; the Python side (consumer)
//! reads them for AI inference.
//!
//! Memory Layout (file):
//! ```text
//! [ RingBufferState (32 bytes) ][ Slot 0 ][ Slot 1 ]...[ Slot N-1 ]
//! ```
//!
//! Each slot:
//! ```text
//! [ input_frame (W*H*4 RGBA) ][ output_frame (sW*sH*4 RGBA) ]
//! ```
//!
//! Python worker must never see half-written indices.
//! Acquire/Release memory ordering ensures visibility.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, Result};
use memmap2::MmapMut;

// =============================================================================
// FFI-SAFE RING BUFFER STATE
// =============================================================================

/// Shared state header at the start of the memory-mapped file.
/// Both Rust and Python read/write these fields through raw memory.
/// Python side reads with struct.unpack("<QQQQ") — must match this layout exactly.
#[repr(C)]
pub struct RingBufferState {
    /// Producer (Rust) increments after writing a frame into the slot.
    /// Consumer (Python) reads this to detect available frames.
    pub write_cursor: AtomicU64,

    /// Consumer (Python) increments after processing a frame.
    /// Producer reads this to detect free slots.
    pub read_cursor: AtomicU64,

    /// Number of slots in the ring (immutable after creation).
    pub buffer_size: u64,

    /// Monotonically increasing frame counter.
    /// Updated by the producer with each new frame written.
    pub frame_id: AtomicU64,
}

const HEADER_SIZE: usize = std::mem::size_of::<RingBufferState>();
// Static assert: header must be exactly 32 bytes for Python struct.unpack compatibility
const _: () = assert!(HEADER_SIZE == 32);

impl RingBufferState {
    fn new(buffer_size: u64) -> Self {
        Self {
            write_cursor: AtomicU64::new(0),
            read_cursor: AtomicU64::new(0),
            buffer_size,
            frame_id: AtomicU64::new(0),
        }
    }
}

// =============================================================================
// SLOT METADATA
// =============================================================================

/// Describes the geometry of a single ring buffer slot.
#[derive(Debug, Clone, Copy)]
pub struct SlotLayout {
    pub width: usize,
    pub height: usize,
    pub scale: usize,
    /// Input frame size in bytes (W * H * 4 for RGBA)
    pub input_size: usize,
    /// Output frame size in bytes (sW * sH * 4 for RGBA)
    pub output_size: usize,
    /// Total slot size = input_size + output_size
    pub total_size: usize,
}

impl SlotLayout {
    pub fn new(width: usize, height: usize, scale: usize) -> Self {
        let input_size = width * height * 4;
        let output_size = (width * scale) * (height * scale) * 4;
        Self {
            width,
            height,
            scale,
            input_size,
            output_size,
            total_size: input_size + output_size,
        }
    }
}

// =============================================================================
// RING BUFFER
// =============================================================================

pub struct ShmRingBuffer {
    mmap: MmapMut,
    slot_layout: SlotLayout,
    slot_count: usize,
    pub shm_path: PathBuf,
}

// MmapMut is Send/Sync safe for our use case (we coordinate via atomics, not Mutex)
unsafe impl Send for ShmRingBuffer {}
unsafe impl Sync for ShmRingBuffer {}

impl ShmRingBuffer {
    /// Create a new ring buffer backed by a temporary file.
    pub fn create(
        width: usize,
        height: usize,
        scale: usize,
        slot_count: usize,
    ) -> Result<Self> {
        let layout = SlotLayout::new(width, height, scale);
        let total_size = HEADER_SIZE + layout.total_size * slot_count;

        // Create temp file
        let dir = std::env::temp_dir();
        let filename = format!(
            "vf_ring_{}.bin",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let shm_path = dir.join(filename);

        let mut file = File::create(&shm_path)?;
        // Zero-fill the entire region
        let zeros = vec![0u8; total_size];
        file.write_all(&zeros)?;
        file.flush()?;
        drop(file);

        // Memory-map it
        let file = OpenOptions::new().read(true).write(true).open(&shm_path)?;
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Initialize header
        let state = RingBufferState::new(slot_count as u64);
        let header_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &state as *const RingBufferState as *const u8,
                HEADER_SIZE,
            )
        };
        mmap[..HEADER_SIZE].copy_from_slice(header_bytes);
        mmap.flush()?;

        println!(
            "ShmRing: Created {} slots ({}x{} x{}) at {:?} ({} bytes)",
            slot_count, width, height, scale, shm_path, total_size
        );

        Ok(Self {
            mmap,
            slot_layout: layout,
            slot_count,
            shm_path,
        })
    }

    /// Open an existing ring buffer file (e.g., created by Python).
    pub fn open(
        path: &str,
        width: usize,
        height: usize,
        scale: usize,
        slot_count: usize,
    ) -> Result<Self> {
        let layout = SlotLayout::new(width, height, scale);
        let expected_size = HEADER_SIZE + layout.total_size * slot_count;
        let shm_path = PathBuf::from(path);

        let file = OpenOptions::new().read(true).write(true).open(&shm_path)?;
        let meta = file.metadata()?;
        if (meta.len() as usize) < expected_size {
            return Err(anyhow!(
                "SHM ring file too small: expected {}, got {}",
                expected_size,
                meta.len()
            ));
        }

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(Self {
            mmap,
            slot_layout: layout,
            slot_count,
            shm_path,
        })
    }

    // -------------------------------------------------------------------------
    // HEADER ACCESS (ATOMIC)
    // -------------------------------------------------------------------------

    fn state(&self) -> &RingBufferState {
        unsafe { &*(self.mmap.as_ptr() as *const RingBufferState) }
    }

    pub fn write_cursor(&self) -> u64 {
        self.state().write_cursor.load(Ordering::Acquire)
    }

    pub fn read_cursor(&self) -> u64 {
        self.state().read_cursor.load(Ordering::Acquire)
    }

    pub fn frame_id(&self) -> u64 {
        self.state().frame_id.load(Ordering::Acquire)
    }

    /// Number of frames currently in the buffer (unread by consumer).
    pub fn available(&self) -> u64 {
        let w = self.write_cursor();
        let r = self.read_cursor();
        w.wrapping_sub(r)
    }

    /// True if the buffer is full (producer must wait for consumer to catch up).
    pub fn is_full(&self) -> bool {
        self.available() >= self.slot_count as u64
    }

    /// True if the buffer is empty (consumer has nothing to read).
    pub fn is_empty(&self) -> bool {
        self.available() == 0
    }

    // -------------------------------------------------------------------------
    // PRODUCER API (RUST WRITES FRAMES)
    // -------------------------------------------------------------------------

    /// Try to acquire the next writable input slot. Returns None if ring is full.
    /// The returned slice is the input portion of the slot (W*H*4 bytes).
    pub fn try_write_input(&mut self) -> Option<(usize, &mut [u8])> {
        if self.is_full() {
            return None;
        }

        let w = self.write_cursor();
        let slot_idx = (w % self.slot_count as u64) as usize;
        let offset = HEADER_SIZE + slot_idx * self.slot_layout.total_size;
        let end = offset + self.slot_layout.input_size;

        Some((slot_idx, &mut self.mmap[offset..end]))
    }

    /// Commit the written frame: advance write_cursor and increment frame_id.
    /// Must be called AFTER the frame data is fully written to the slot.
    /// Uses Release ordering so Python sees the data before the cursor update.
    pub fn commit_write(&self) {
        let state = self.state();
        state.frame_id.fetch_add(1, Ordering::Release);
        // Release: all prior writes (frame data) are visible before cursor advances
        state.write_cursor.fetch_add(1, Ordering::Release);
    }

    // -------------------------------------------------------------------------
    // CONSUMER API (READ OUTPUT AFTER PYTHON PROCESSING)
    // -------------------------------------------------------------------------

    /// Read the output portion of a processed slot (sW*sH*4 bytes).
    /// Returns None if the ring is empty.
    pub fn try_read_output(&self) -> Option<(usize, &[u8])> {
        if self.is_empty() {
            return None;
        }

        let r = self.read_cursor();
        let slot_idx = (r % self.slot_count as u64) as usize;
        let base = HEADER_SIZE + slot_idx * self.slot_layout.total_size;
        let offset = base + self.slot_layout.input_size;
        let end = offset + self.slot_layout.output_size;

        Some((slot_idx, &self.mmap[offset..end]))
    }

    /// Advance the read cursor after consuming a processed frame.
    /// Uses Release ordering so the producer sees the freed slot.
    pub fn commit_read(&self) {
        self.state().read_cursor.fetch_add(1, Ordering::Release);
    }

    // -------------------------------------------------------------------------
    // DIRECT SLOT ACCESS (for compatibility with existing VideoShm pattern)
    // -------------------------------------------------------------------------

    pub fn input_slot_mut(&mut self, index: usize) -> Result<&mut [u8]> {
        if index >= self.slot_count {
            return Err(anyhow!("Slot index {} out of bounds (max {})", index, self.slot_count - 1));
        }
        let offset = HEADER_SIZE + index * self.slot_layout.total_size;
        let end = offset + self.slot_layout.input_size;
        Ok(&mut self.mmap[offset..end])
    }

    pub fn output_slot(&self, index: usize) -> Result<&[u8]> {
        if index >= self.slot_count {
            return Err(anyhow!("Slot index {} out of bounds (max {})", index, self.slot_count - 1));
        }
        let base = HEADER_SIZE + index * self.slot_layout.total_size;
        let offset = base + self.slot_layout.input_size;
        let end = offset + self.slot_layout.output_size;
        Ok(&self.mmap[offset..end])
    }

    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    pub fn slot_layout(&self) -> &SlotLayout {
        &self.slot_layout
    }
}

impl Drop for ShmRingBuffer {
    fn drop(&mut self) {
        if let Err(e) = self.mmap.flush() {
            eprintln!("ShmRing: flush error on drop: {}", e);
        }
        // Don't delete the file here — the consumer (Python) may still be using it.
        // Cleanup is handled by the pipeline coordinator.
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        assert_eq!(HEADER_SIZE, 32);
    }

    #[test]
    fn test_create_and_basic_ops() {
        let mut ring = ShmRingBuffer::create(64, 64, 4, 3).unwrap();

        assert!(ring.is_empty());
        assert!(!ring.is_full());
        assert_eq!(ring.available(), 0);

        // Write a frame
        let (slot, input) = ring.try_write_input().unwrap();
        assert_eq!(slot, 0);
        assert_eq!(input.len(), 64 * 64 * 4);
        // Fill with test pattern
        input[0] = 0xFF;
        input[1] = 0xAB;
        ring.commit_write();

        assert!(!ring.is_empty());
        assert_eq!(ring.available(), 1);
        assert_eq!(ring.frame_id(), 1);

        // Read output (would be zero since Python hasn't processed it)
        let (rslot, output) = ring.try_read_output().unwrap();
        assert_eq!(rslot, 0);
        assert_eq!(output.len(), (64 * 4) * (64 * 4) * 4);
        ring.commit_read();

        assert!(ring.is_empty());

        // Cleanup
        let _ = std::fs::remove_file(&ring.shm_path);
    }

    #[test]
    fn test_full_ring() {
        let mut ring = ShmRingBuffer::create(8, 8, 2, 2).unwrap();

        // Fill both slots
        ring.try_write_input().unwrap();
        ring.commit_write();
        ring.try_write_input().unwrap();
        ring.commit_write();

        assert!(ring.is_full());
        assert!(ring.try_write_input().is_none());

        // Free one
        ring.try_read_output().unwrap();
        ring.commit_read();

        assert!(!ring.is_full());
        assert!(ring.try_write_input().is_some());

        let _ = std::fs::remove_file(&ring.shm_path);
    }

    #[test]
    fn test_slot_layout() {
        let layout = SlotLayout::new(1920, 1080, 4);
        assert_eq!(layout.input_size, 1920 * 1080 * 4);
        assert_eq!(layout.output_size, (1920 * 4) * (1080 * 4) * 4);
        assert_eq!(layout.total_size, layout.input_size + layout.output_size);
    }

    #[test]
    fn test_direct_slot_access() {
        let mut ring = ShmRingBuffer::create(16, 16, 2, 3).unwrap();

        // Direct write
        let input = ring.input_slot_mut(1).unwrap();
        input[0] = 42;

        // Direct read
        let output = ring.output_slot(1).unwrap();
        assert_eq!(output[0], 0); // Output is zero-initialized

        let _ = std::fs::remove_file(&ring.shm_path);
    }

    #[test]
    fn test_wrap_around() {
        let mut ring = ShmRingBuffer::create(4, 4, 1, 2).unwrap();

        // Write 4 frames (wraps around a 2-slot ring)
        for i in 0..4u8 {
            // Must read before writing if full
            if ring.is_full() {
                ring.try_read_output().unwrap();
                ring.commit_read();
            }
            let (_, input) = ring.try_write_input().unwrap();
            input[0] = i;
            ring.commit_write();
        }

        assert_eq!(ring.frame_id(), 4);

        let _ = std::fs::remove_file(&ring.shm_path);
    }
}
