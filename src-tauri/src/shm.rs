use anyhow::{anyhow, Result};
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU32, Ordering};
use thiserror::Error;

include!(concat!(env!("OUT_DIR"), "/shm_constants.rs"));

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
    /// [ SlotHeader × RING_SIZE  96 bytes ]
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

        tracing::info!(shm_version = version, header_size, "SHM header validated");

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

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SlotState {
        Empty,
        RustWriting,
        ReadyForAi,
        AiProcessing,
        ReadyForEncode,
        Encoding,
    }

    impl SlotState {
        fn as_u32(self) -> u32 {
            match self {
                Self::Empty => SLOT_EMPTY,
                Self::RustWriting => SLOT_RUST_WRITING,
                Self::ReadyForAi => SLOT_READY_FOR_AI,
                Self::AiProcessing => SLOT_AI_PROCESSING,
                Self::ReadyForEncode => SLOT_READY_FOR_ENCODE,
                Self::Encoding => SLOT_ENCODING,
            }
        }

        fn from_u32(v: u32) -> Option<Self> {
            match v {
                SLOT_EMPTY => Some(Self::Empty),
                SLOT_RUST_WRITING => Some(Self::RustWriting),
                SLOT_READY_FOR_AI => Some(Self::ReadyForAi),
                SLOT_AI_PROCESSING => Some(Self::AiProcessing),
                SLOT_READY_FOR_ENCODE => Some(Self::ReadyForEncode),
                SLOT_ENCODING => Some(Self::Encoding),
                _ => None,
            }
        }
    }

    fn is_valid_transition(from: SlotState, to: SlotState) -> bool {
        matches!(
            (from, to),
            (SlotState::Empty, SlotState::RustWriting)
                | (SlotState::RustWriting, SlotState::ReadyForAi)
                | (SlotState::ReadyForAi, SlotState::AiProcessing)
                | (SlotState::AiProcessing, SlotState::ReadyForEncode)
                | (SlotState::ReadyForEncode, SlotState::Encoding)
                | (SlotState::Encoding, SlotState::Empty)
        )
    }

    fn apply_transition(slot_state: &mut SlotState, to: SlotState) -> Result<(), &'static str> {
        if is_valid_transition(*slot_state, to) {
            *slot_state = to;
            Ok(())
        } else {
            Err("invalid slot transition")
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct DecodedGlobalHeader {
        magic: [u8; 8],
        version: u32,
        header_size: u32,
        slot_count: u32,
        width: u32,
        height: u32,
        scale: u32,
        pixel_format: u32,
    }

    fn decode_global_header(bytes: &[u8]) -> Result<DecodedGlobalHeader, String> {
        if bytes.len() < GLOBAL_HEADER_SIZE {
            return Err(format!(
                "header too small: got {}, need {}",
                bytes.len(),
                GLOBAL_HEADER_SIZE
            ));
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);

        Ok(DecodedGlobalHeader {
            magic,
            version: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            header_size: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            slot_count: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            width: u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
            height: u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            scale: u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
            pixel_format: u32::from_le_bytes(bytes[32..36].try_into().unwrap()),
        })
    }

    /// Build a minimal valid SHM file in `data` with the given parameters.
    #[allow(clippy::too_many_arguments)] // TODO(clippy): test helper keeps explicit header fields for readability.
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
            b"VFSHM001",
            SHM_VERSION,
            HEADER_REGION_SIZE as u32,
            RING_SIZE as u32,
            64,
            64,
            2,
            PIXEL_FORMAT_RGB24,
        );
        let path = write_temp_shm("vf_test_valid.bin", &data);
        let result = VideoShm::open(&path, 64, 64, 2);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_ok(), "Valid header must open without error");
    }

    #[test]
    fn open_bad_magic_is_rejected() {
        let data = make_shm_bytes(
            b"INVALID!",
            SHM_VERSION,
            HEADER_REGION_SIZE as u32,
            RING_SIZE as u32,
            64,
            64,
            2,
            PIXEL_FORMAT_RGB24,
        );
        let path = write_temp_shm("vf_test_magic.bin", &data);
        let result = VideoShm::open(&path, 64, 64, 2);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "Bad magic must be rejected");
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("magic"), "Error must mention 'magic': {}", msg);
    }

    #[test]
    fn open_wrong_version_is_rejected() {
        let data = make_shm_bytes(
            b"VFSHM001",
            99,
            HEADER_REGION_SIZE as u32,
            RING_SIZE as u32,
            64,
            64,
            2,
            PIXEL_FORMAT_RGB24,
        );
        let path = write_temp_shm("vf_test_ver.bin", &data);
        let result = VideoShm::open(&path, 64, 64, 2);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "Wrong version must be rejected");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("version"),
            "Error must mention 'version': {}",
            msg
        );
    }

    #[test]
    fn open_wrong_header_size_is_rejected() {
        let data = make_shm_bytes(
            b"VFSHM001",
            SHM_VERSION,
            48, // old header size — now invalid
            RING_SIZE as u32,
            64,
            64,
            2,
            PIXEL_FORMAT_RGB24,
        );
        let path = write_temp_shm("vf_test_hdrsize.bin", &data);
        let result = VideoShm::open(&path, 64, 64, 2);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "Wrong header_size must be rejected");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("header_size"),
            "Error must mention 'header_size': {}",
            msg
        );
    }

    #[test]
    fn header_region_size_constant_is_correct() {
        // 36 global + 96 slot headers (16 × 6) = 132
        assert_eq!(GLOBAL_HEADER_SIZE, 36);
        assert_eq!(SLOT_HEADER_REGION, 96);
        assert_eq!(HEADER_REGION_SIZE, 132);
    }

    #[test]
    fn slot_state_roundtrip() {
        let data = make_shm_bytes(
            b"VFSHM001",
            SHM_VERSION,
            HEADER_REGION_SIZE as u32,
            RING_SIZE as u32,
            16,
            16,
            2,
            PIXEL_FORMAT_RGB24,
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

    #[test]
    fn global_header_roundtrip_fields_are_correct() {
        let width = 1920;
        let height = 1080;
        let scale = 4;
        let slot_count = RING_SIZE as u32;
        let data = make_shm_bytes(
            MAGIC,
            SHM_VERSION,
            HEADER_REGION_SIZE as u32,
            slot_count,
            width,
            height,
            scale,
            PIXEL_FORMAT_RGB24,
        );
        let header = decode_global_header(&data).expect("decode header");

        assert_eq!(header.magic, *MAGIC);
        assert_eq!(header.version, SHM_VERSION);
        assert_eq!(header.header_size, HEADER_REGION_SIZE as u32);
        assert_eq!(header.slot_count, slot_count);
        assert_eq!(header.width, width);
        assert_eq!(header.height, height);
        assert_eq!(header.scale, scale);
        assert_eq!(header.pixel_format, PIXEL_FORMAT_RGB24);
    }

    #[test]
    fn open_rejects_wrong_magic_version_and_header_size() {
        let bad_magic = make_shm_bytes(
            b"BADMAGIC",
            SHM_VERSION,
            HEADER_REGION_SIZE as u32,
            RING_SIZE as u32,
            64,
            64,
            2,
            PIXEL_FORMAT_RGB24,
        );
        let bad_magic_path = write_temp_shm("vf_test_invalid_magic.bin", &bad_magic);
        let bad_magic_result = VideoShm::open(&bad_magic_path, 64, 64, 2);
        let _ = std::fs::remove_file(&bad_magic_path);
        assert!(bad_magic_result.is_err());
        assert!(
            bad_magic_result
                .err()
                .unwrap()
                .to_string()
                .contains("magic"),
            "wrong magic must fail deterministically"
        );

        let bad_version = make_shm_bytes(
            MAGIC,
            SHM_VERSION + 1,
            HEADER_REGION_SIZE as u32,
            RING_SIZE as u32,
            64,
            64,
            2,
            PIXEL_FORMAT_RGB24,
        );
        let bad_version_path = write_temp_shm("vf_test_invalid_version.bin", &bad_version);
        let bad_version_result = VideoShm::open(&bad_version_path, 64, 64, 2);
        let _ = std::fs::remove_file(&bad_version_path);
        assert!(bad_version_result.is_err());
        assert!(
            bad_version_result
                .err()
                .unwrap()
                .to_string()
                .contains("version"),
            "wrong version must fail deterministically"
        );

        let bad_header_size = make_shm_bytes(
            MAGIC,
            SHM_VERSION,
            (HEADER_REGION_SIZE as u32) + 4,
            RING_SIZE as u32,
            64,
            64,
            2,
            PIXEL_FORMAT_RGB24,
        );
        let bad_header_size_path =
            write_temp_shm("vf_test_invalid_header_size.bin", &bad_header_size);
        let bad_header_size_result = VideoShm::open(&bad_header_size_path, 64, 64, 2);
        let _ = std::fs::remove_file(&bad_header_size_path);
        assert!(bad_header_size_result.is_err());
        assert!(
            bad_header_size_result
                .err()
                .unwrap()
                .to_string()
                .contains("header_size"),
            "wrong header_size must fail deterministically"
        );
    }

    #[test]
    fn slot_fsm_allows_expected_transitions() {
        let mut slot = SlotState::Empty;
        apply_transition(&mut slot, SlotState::RustWriting).expect("EMPTY -> RUST_WRITING");
        apply_transition(&mut slot, SlotState::ReadyForAi).expect("RUST_WRITING -> READY_FOR_AI");
        apply_transition(&mut slot, SlotState::AiProcessing)
            .expect("READY_FOR_AI -> AI_PROCESSING");
        apply_transition(&mut slot, SlotState::ReadyForEncode)
            .expect("AI_PROCESSING -> READY_FOR_ENCODE");
        apply_transition(&mut slot, SlotState::Encoding).expect("READY_FOR_ENCODE -> ENCODING");
        apply_transition(&mut slot, SlotState::Empty).expect("ENCODING -> EMPTY");
    }

    #[test]
    fn slot_fsm_rejects_forbidden_transitions() {
        let mut slot = SlotState::Empty;
        assert!(apply_transition(&mut slot, SlotState::ReadyForEncode).is_err());
        assert!(apply_transition(&mut slot, SlotState::ReadyForAi).is_err());
        assert_eq!(slot, SlotState::Empty);

        slot = SlotState::ReadyForAi;
        assert!(apply_transition(&mut slot, SlotState::Empty).is_err());
        assert_eq!(slot, SlotState::ReadyForAi);
    }

    #[test]
    fn simulated_frame_loop_makes_deterministic_forward_progress() {
        const TOTAL_FRAMES: u32 = 64;
        const MAX_ITERS: usize = 10_000;

        let mut slot_states = [SlotState::Empty; RING_SIZE];
        let mut slot_write_index = [0u32; RING_SIZE];
        let mut next_frame_id = 1u32;
        let mut completed = 0u32;
        let mut iterations = 0usize;

        while completed < TOTAL_FRAMES && iterations < MAX_ITERS {
            iterations += 1;

            if next_frame_id <= TOTAL_FRAMES {
                if let Some(i) = slot_states.iter().position(|s| *s == SlotState::Empty) {
                    apply_transition(&mut slot_states[i], SlotState::RustWriting)
                        .expect("producer EMPTY -> RUST_WRITING");
                    assert!(
                        next_frame_id > slot_write_index[i],
                        "write index must be monotonic per slot"
                    );
                    slot_write_index[i] = next_frame_id;
                    next_frame_id += 1;
                    apply_transition(&mut slot_states[i], SlotState::ReadyForAi)
                        .expect("producer RUST_WRITING -> READY_FOR_AI");
                }
            }

            if let Some(i) = slot_states.iter().position(|s| *s == SlotState::ReadyForAi) {
                apply_transition(&mut slot_states[i], SlotState::AiProcessing)
                    .expect("ai READY_FOR_AI -> AI_PROCESSING");
                apply_transition(&mut slot_states[i], SlotState::ReadyForEncode)
                    .expect("ai AI_PROCESSING -> READY_FOR_ENCODE");
            }

            if let Some(i) = slot_states
                .iter()
                .position(|s| *s == SlotState::ReadyForEncode)
            {
                apply_transition(&mut slot_states[i], SlotState::Encoding)
                    .expect("encoder READY_FOR_ENCODE -> ENCODING");
                apply_transition(&mut slot_states[i], SlotState::Empty)
                    .expect("encoder ENCODING -> EMPTY");
                completed += 1;
            }

            if iterations.is_multiple_of(3) {
                std::thread::yield_now();
            }
        }

        assert_eq!(
            completed, TOTAL_FRAMES,
            "all frames should complete without deadlock"
        );
        assert!(
            iterations < MAX_ITERS,
            "simulation must terminate within bounded iterations"
        );
        assert!(
            next_frame_id > TOTAL_FRAMES,
            "producer should submit all frames"
        );
        assert!(slot_states
            .iter()
            .all(|s| SlotState::from_u32(s.as_u32()).is_some()));
    }
}
