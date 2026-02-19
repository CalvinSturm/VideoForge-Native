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
// GLOBAL HEADER LAYOUT (36 bytes at offset 0)
// =============================================================================
//
// Offset  0: u8[8]  magic        = b"VFSHM001"
// Offset  8: u32    version      = SHM_VERSION (1)
// Offset 12: u32    header_size  = HEADER_REGION_SIZE (84)
// Offset 16: u32    slot_count   = RING_SIZE (3)
// Offset 20: u32    width        = frame width in pixels
// Offset 24: u32    height       = frame height in pixels
// Offset 28: u32    scale        = upscale factor
// Offset 32: u32    pixel_format = PIXEL_FORMAT_RGB24 (1)
//                                  ── 36 bytes ──
//
// =============================================================================
// SLOT HEADER LAYOUT (per slot, 16 bytes each — starts at offset 36)
// =============================================================================
//
// Offset 0:  u32  write_index   — frame counter (set by Rust decoder)
// Offset 4:  u32  read_index    — frame counter (set by Rust encoder)
// Offset 8:  u32  state         — atomic state machine value
// Offset 12: u32  frame_bytes   — actual frame data size in bytes
//
// All fields use SeqCst ordering for cross-process visibility.

const GLOBAL_HEADER_SIZE: usize = 36;
const MAGIC: &[u8; 8] = b"VFSHM001";
pub const SHM_VERSION: u32 = 1;
/// Pixel format identifier for RGB24 (3 bytes/pixel, interleaved).
pub const PIXEL_FORMAT_RGB24: u32 = 1;

const SLOT_HEADER_SIZE: usize = 16; // 4 × u32
const SLOT_HEADER_REGION: usize = SLOT_HEADER_SIZE * RING_SIZE; // 48

/// Total bytes reserved for all headers (global + slot headers) at the
/// start of the SHM file.  Data region begins at this offset.
pub const HEADER_REGION_SIZE: usize = GLOBAL_HEADER_SIZE + SLOT_HEADER_REGION; // 84

// Byte offsets within a single slot header (relative to that slot's base).
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
    /// Open a SHM file created by Python and validate its global header.
    ///
    /// Layout:
    /// ```text
    /// [ Global Header  36 bytes (magic, version, dimensions) ]
    /// [ SlotHeader × RING_SIZE  48 bytes ]
    /// [ Slot 0: input (W×H×3) | output (sW×sH×3) ]
    /// [ Slot 1: input | output ]
    /// [ Slot 2: input | output ]
    /// ```
    pub fn open(file_path: &str, width: usize, height: usize, scale: usize) -> Result<Self> {
        let slot_input_size = width * height * 3;
        let slot_output_size = (width * scale) * (height * scale) * 3;
        let slot_total_size = slot_input_size + slot_output_size;
        let total_size = HEADER_REGION_SIZE + slot_total_size * RING_SIZE;

        tracing::info!(
            path = %file_path,
            expected_bytes = total_size,
            header_bytes = HEADER_REGION_SIZE,
            width,
            height,
            scale,
            "Opening SHM file"
        );

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(file_path)
            .map_err(|e| anyhow!("Failed to open SHM file '{}': {}", file_path, e))?;

        let meta = file.metadata()?;
        if meta.len() < total_size as u64 {
            return Err(anyhow!(
                "SHM file too small: expected {} bytes, got {}",
                total_size,
                meta.len()
            ));
        }

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        // ── Validate global header ────────────────────────────────────────────
        if mmap.len() < GLOBAL_HEADER_SIZE {
            return Err(anyhow!(
                "SHM file too small for global header ({} bytes minimum)",
                GLOBAL_HEADER_SIZE
            ));
        }

        let actual_magic = &mmap[0..8];
        if actual_magic != MAGIC {
            return Err(anyhow!(
                "SHM magic mismatch: expected {:?}, got {:?}",
                std::str::from_utf8(MAGIC).unwrap_or("<invalid>"),
                std::str::from_utf8(actual_magic).unwrap_or("<non-utf8>")
            ));
        }

        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if version != SHM_VERSION {
            return Err(anyhow!(
                "SHM version mismatch: expected {}, got {}",
                SHM_VERSION,
                version
            ));
        }

        let header_size = u32::from_le_bytes(mmap[12..16].try_into().unwrap());
        if header_size != HEADER_REGION_SIZE as u32 {
            return Err(anyhow!(
                "SHM header_size mismatch: expected {}, got {}",
                HEADER_REGION_SIZE,
                header_size
            ));
        }

        tracing::info!(
            shm_version = version,
            header_size,
            "SHM header validated"
        );

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

    /// Byte offset of a field within the per-slot header for slot `index`.
    /// Accounts for the global header at the start of the file.
    #[inline]
    fn header_field_ptr(&self, index: usize, field_offset: usize) -> *const AtomicU32 {
        let byte_offset = GLOBAL_HEADER_SIZE + index * SLOT_HEADER_SIZE + field_offset;
        // SAFETY: slot header region starts at GLOBAL_HEADER_SIZE, all u32
        // fields are 4-byte aligned within their 16-byte slot header.
        // Both Rust and Python access these via atomic u32 operations.
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
        tracing::debug!("All SHM slots reset to EMPTY");
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

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal valid SHM file in `data` with the given parameters.
    fn make_shm_bytes(
        magic: &[u8; 8],
        version: u32,
        header_size: u32,
        slot_count: u32,
        width: u32,
        height: u32,
        scale: u32,
        pixel_format: u32,
    ) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let s = scale as usize;
        let slot_in = w * h * 3;
        let slot_out = (w * s) * (h * s) * 3;
        let total = HEADER_REGION_SIZE + (slot_in + slot_out) * RING_SIZE;
        let mut data = vec![0u8; total];

        data[0..8].copy_from_slice(magic);
        data[8..12].copy_from_slice(&version.to_le_bytes());
        data[12..16].copy_from_slice(&header_size.to_le_bytes());
        data[16..20].copy_from_slice(&slot_count.to_le_bytes());
        data[20..24].copy_from_slice(&width.to_le_bytes());
        data[24..28].copy_from_slice(&height.to_le_bytes());
        data[28..32].copy_from_slice(&scale.to_le_bytes());
        data[32..36].copy_from_slice(&pixel_format.to_le_bytes());
        data
    }

    fn write_temp_shm(name: &str, data: &[u8]) -> String {
        let path = format!("{}/{}", std::env::temp_dir().display(), name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(data).unwrap();
        path
    }

    #[test]
    fn open_valid_header_succeeds() {
        let data = make_shm_bytes(
            b"VFSHM001", SHM_VERSION, HEADER_REGION_SIZE as u32,
            RING_SIZE as u32, 64, 64, 2, PIXEL_FORMAT_RGB24,
        );
        let path = write_temp_shm("vf_test_valid.bin", &data);
        let result = VideoShm::open(&path, 64, 64, 2);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_ok(), "Valid header must open without error");
    }

    #[test]
    fn open_bad_magic_is_rejected() {
        let data = make_shm_bytes(
            b"INVALID!", SHM_VERSION, HEADER_REGION_SIZE as u32,
            RING_SIZE as u32, 64, 64, 2, PIXEL_FORMAT_RGB24,
        );
        let path = write_temp_shm("vf_test_magic.bin", &data);
        let result = VideoShm::open(&path, 64, 64, 2);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "Bad magic must be rejected");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("magic"),
            "Error must mention 'magic': {}", msg
        );
    }

    #[test]
    fn open_wrong_version_is_rejected() {
        let data = make_shm_bytes(
            b"VFSHM001", 99, HEADER_REGION_SIZE as u32,
            RING_SIZE as u32, 64, 64, 2, PIXEL_FORMAT_RGB24,
        );
        let path = write_temp_shm("vf_test_ver.bin", &data);
        let result = VideoShm::open(&path, 64, 64, 2);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "Wrong version must be rejected");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("version"),
            "Error must mention 'version': {}", msg
        );
    }

    #[test]
    fn open_wrong_header_size_is_rejected() {
        let data = make_shm_bytes(
            b"VFSHM001", SHM_VERSION, 48, // old header size — now invalid
            RING_SIZE as u32, 64, 64, 2, PIXEL_FORMAT_RGB24,
        );
        let path = write_temp_shm("vf_test_hdrsize.bin", &data);
        let result = VideoShm::open(&path, 64, 64, 2);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "Wrong header_size must be rejected");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("header_size"),
            "Error must mention 'header_size': {}", msg
        );
    }

    #[test]
    fn header_region_size_constant_is_correct() {
        // 36 global + 48 slot headers = 84
        assert_eq!(GLOBAL_HEADER_SIZE, 36);
        assert_eq!(SLOT_HEADER_REGION, 48);
        assert_eq!(HEADER_REGION_SIZE, 84);
    }

    #[test]
    fn slot_state_roundtrip() {
        let data = make_shm_bytes(
            b"VFSHM001", SHM_VERSION, HEADER_REGION_SIZE as u32,
            RING_SIZE as u32, 16, 16, 2, PIXEL_FORMAT_RGB24,
        );
        let path = write_temp_shm("vf_test_state.bin", &data);
        let shm = VideoShm::open(&path, 16, 16, 2).unwrap();
        shm.reset_all_slots();
        for i in 0..RING_SIZE {
            assert_eq!(shm.slot_state(i), SLOT_EMPTY);
        }
        shm.set_slot_state(1, SLOT_READY_FOR_AI);
        assert_eq!(shm.slot_state(1), SLOT_READY_FOR_AI);
        // Other slots must be unaffected.
        assert_eq!(shm.slot_state(0), SLOT_EMPTY);
        assert_eq!(shm.slot_state(2), SLOT_EMPTY);
        let _ = std::fs::remove_file(&path);
    }
}
