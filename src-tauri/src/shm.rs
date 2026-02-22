use anyhow::{anyhow, Result};
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU32, Ordering};
use thiserror::Error;

include!(concat!(env!("OUT_DIR"), "/shm_constants.rs"));

/// Extended SHM header version (opt-in).
///
/// Legacy/default writers use `SHM_VERSION` from `shm_protocol.json`.
pub const SHM_VERSION_V2: u32 = SHM_VERSION + 1;
/// Numeric protocol identity used only by extended SHM headers.
pub const SHM_PROTOCOL_VERSION: u32 = 1;
const SHM_PROTOCOL_VERSION_OFFSET: usize = GLOBAL_HEADER_SIZE;
#[cfg(test)]
const HEADER_REGION_SIZE_V2: usize = HEADER_REGION_SIZE + 4;
const SHM_PROTOCOL_MISMATCH_CODE: &str = "SHM_PROTOCOL_VERSION_MISMATCH";
const SHM_RING_SIZE_MISMATCH_CODE: &str = "SHM_RING_SIZE_MISMATCH";

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
    ring_size: usize,
    global_header_size: usize,
    header_region_size: usize,

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
        Self::open_with_expected_ring_size(file_path, width, height, scale, None)
    }

    pub fn open_with_expected_ring_size(
        file_path: &str,
        width: usize,
        height: usize,
        scale: usize,
        expected_ring_size: Option<usize>,
    ) -> Result<Self> {
        let slot_input_size = width * height * 3;
        let slot_output_size = (width * scale) * (height * scale) * 3;
        let slot_total_size = slot_input_size + slot_output_size;

        tracing::info!(
            path = %file_path,
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
        if meta.len() < GLOBAL_HEADER_SIZE as u64 {
            return Err(anyhow!(
                "SHM file too small: expected at least {} bytes, got {}",
                GLOBAL_HEADER_SIZE,
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
        let header_size = u32::from_le_bytes(mmap[12..16].try_into().unwrap());
        let slot_count = u32::from_le_bytes(mmap[16..20].try_into().unwrap()) as usize;
        if slot_count == 0 {
            return Err(anyhow!("SHM slot_count must be > 0"));
        }

        match version {
            SHM_VERSION => {
                if header_size != HEADER_REGION_SIZE as u32 {
                    return Err(anyhow!(
                        "SHM header_size mismatch: expected {}, got {}",
                        HEADER_REGION_SIZE,
                        header_size
                    ));
                }
                if slot_count != RING_SIZE {
                    return Err(anyhow!(
                        "{code}: expected={}, found={}. Legacy SHM header requires default ring size.",
                        RING_SIZE,
                        slot_count,
                        code = SHM_RING_SIZE_MISMATCH_CODE
                    ));
                }
                if let Some(expected) = expected_ring_size {
                    if slot_count != expected {
                        return Err(anyhow!(
                            "{code}: expected={expected}, found={found}. Ring override requires SHM v2 mode.",
                            code = SHM_RING_SIZE_MISMATCH_CODE,
                            expected = expected,
                            found = slot_count
                        ));
                    }
                }
            }
            SHM_VERSION_V2 => {
                let expected_header_size_v2 =
                    (GLOBAL_HEADER_SIZE + 4 + SLOT_HEADER_SIZE * slot_count) as u32;
                if header_size != expected_header_size_v2 {
                    return Err(anyhow!(
                        "SHM v2 header_size mismatch: expected {}, got {}",
                        expected_header_size_v2,
                        header_size
                    ));
                }
                if mmap.len() < GLOBAL_HEADER_SIZE + 4 {
                    return Err(anyhow!(
                        "SHM v2 header too small for protocol_version field"
                    ));
                }
                let found_protocol = u32::from_le_bytes(
                    mmap[SHM_PROTOCOL_VERSION_OFFSET..SHM_PROTOCOL_VERSION_OFFSET + 4]
                        .try_into()
                        .unwrap(),
                );
                if found_protocol != SHM_PROTOCOL_VERSION {
                    return Err(anyhow!(
                        "{code}: expected={expected}, found={found}. Update Python worker/engine or disable SHM v2 mode.",
                        code = SHM_PROTOCOL_MISMATCH_CODE,
                        expected = SHM_PROTOCOL_VERSION,
                        found = found_protocol
                    ));
                }
                if let Some(expected) = expected_ring_size {
                    if slot_count != expected {
                        return Err(anyhow!(
                            "{code}: expected={expected}, found={found}. Disable ring override or update worker/engine.",
                            code = SHM_RING_SIZE_MISMATCH_CODE,
                            expected = expected,
                            found = slot_count
                        ));
                    }
                }
            }
            found => {
                return Err(anyhow!(
                    "SHM version mismatch: expected {} or {}, got {}",
                    SHM_VERSION,
                    SHM_VERSION_V2,
                    found
                ));
            }
        }

        let header_region_size = header_size as usize;
        let slot_header_region = slot_count * SLOT_HEADER_SIZE;
        if header_region_size < slot_header_region {
            return Err(anyhow!(
                "SHM header_size too small for slot headers: header_size={}, slot_headers={}",
                header_region_size,
                slot_header_region
            ));
        }
        let total_size = header_region_size + slot_total_size * slot_count;
        if meta.len() < total_size as u64 {
            return Err(anyhow!(
                "SHM file too small: expected {} bytes, got {}",
                total_size,
                meta.len()
            ));
        }
        let global_header_size = header_region_size - slot_header_region;

        tracing::info!(
            shm_version = version,
            header_size,
            slot_count,
            "SHM header validated"
        );

        Ok(Self {
            mmap,
            width,
            height,
            scale,
            ring_size: slot_count,
            global_header_size,
            header_region_size,
            slot_input_size,
            slot_output_size,
            slot_total_size,
        })
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }

    // -------------------------------------------------------------------------
    // SLOT STATE ACCESSORS (atomic, cross-process safe)
    // -------------------------------------------------------------------------

    /// Byte offset of a field within the per-slot header for slot `index`.
    /// Accounts for the global header at the start of the file.
    #[inline]
    fn header_field_ptr(&self, index: usize, field_offset: usize) -> *const AtomicU32 {
        let byte_offset = self.global_header_size + index * SLOT_HEADER_SIZE + field_offset;
        // SAFETY: slot header region starts at GLOBAL_HEADER_SIZE, all u32
        // fields are 4-byte aligned within their 16-byte slot header.
        // Both Rust and Python access these via atomic u32 operations.
        unsafe { self.mmap.as_ptr().add(byte_offset) as *const AtomicU32 }
    }

    /// Read the atomic state of slot `index`.
    pub fn slot_state(&self, index: usize) -> u32 {
        assert!(index < self.ring_size, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, STATE_OFFSET) };
        atom.load(Ordering::SeqCst)
    }

    /// Set the atomic state of slot `index`.
    pub fn set_slot_state(&self, index: usize, state: u32) {
        assert!(index < self.ring_size, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, STATE_OFFSET) };
        atom.store(state, Ordering::SeqCst);
    }

    /// Compare-and-swap the slot state. Returns true if the swap succeeded.
    pub fn cas_slot_state(&self, index: usize, expected: u32, new: u32) -> bool {
        assert!(index < self.ring_size, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, STATE_OFFSET) };
        atom.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Read the frame_bytes field for slot `index`.
    pub fn slot_frame_bytes(&self, index: usize) -> u32 {
        assert!(index < self.ring_size, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, FRAME_BYTES_OFFSET) };
        atom.load(Ordering::SeqCst)
    }

    /// Set the frame_bytes field for slot `index`.
    pub fn set_slot_frame_bytes(&self, index: usize, bytes: u32) {
        assert!(index < self.ring_size, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, FRAME_BYTES_OFFSET) };
        atom.store(bytes, Ordering::SeqCst);
    }

    /// Set the write_index (frame counter) for slot `index`.
    pub fn set_slot_write_index(&self, index: usize, frame_id: u32) {
        assert!(index < self.ring_size, "slot index out of bounds");
        let atom = unsafe { &*self.header_field_ptr(index, WRITE_INDEX_OFFSET) };
        atom.store(frame_id, Ordering::SeqCst);
    }

    /// Reset all slot headers to EMPTY state.
    pub fn reset_all_slots(&self) {
        for i in 0..self.ring_size {
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
        if index >= self.ring_size {
            return Err(ShmError::IndexOutOfBounds {
                index,
                max: self.ring_size - 1,
            });
        }
        let offset = self.header_region_size + index * self.slot_total_size;
        let end = offset + self.slot_input_size;
        Ok(&mut self.mmap[offset..end])
    }

    pub fn output_slot(&self, index: usize) -> Result<&[u8], ShmError> {
        if index >= self.ring_size {
            return Err(ShmError::IndexOutOfBounds {
                index,
                max: self.ring_size - 1,
            });
        }
        let offset =
            self.header_region_size + (index * self.slot_total_size) + self.slot_input_size;
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
        let slots = slot_count as usize;
        let slot_in = w * h * 3;
        let slot_out = (w * s) * (h * s) * 3;
        let total = (header_size as usize) + (slot_in + slot_out) * slots;
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

    #[allow(clippy::too_many_arguments)] // TODO(clippy): test helper keeps explicit header fields for readability.
    fn make_shm_bytes_v2(
        magic: &[u8; 8],
        version: u32,
        header_size: u32,
        slot_count: u32,
        width: u32,
        height: u32,
        scale: u32,
        pixel_format: u32,
        protocol_version: u32,
    ) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let s = scale as usize;
        let slots = slot_count as usize;
        let slot_in = w * h * 3;
        let slot_out = (w * s) * (h * s) * 3;
        let total = (header_size as usize) + (slot_in + slot_out) * slots;
        let mut data = vec![0u8; total];

        data[0..8].copy_from_slice(magic);
        data[8..12].copy_from_slice(&version.to_le_bytes());
        data[12..16].copy_from_slice(&header_size.to_le_bytes());
        data[16..20].copy_from_slice(&slot_count.to_le_bytes());
        data[20..24].copy_from_slice(&width.to_le_bytes());
        data[24..28].copy_from_slice(&height.to_le_bytes());
        data[28..32].copy_from_slice(&scale.to_le_bytes());
        data[32..36].copy_from_slice(&pixel_format.to_le_bytes());
        data[36..40].copy_from_slice(&protocol_version.to_le_bytes());
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
    fn shm_open_v1_header_still_works_without_protocol_version() {
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
        let path = write_temp_shm("vf_test_v1_header.bin", &data);
        let result = VideoShm::open(&path, 64, 64, 2);
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_ok(),
            "v1 header path must remain backward compatible"
        );
    }

    #[test]
    fn shm_open_v2_header_match_ok() {
        let data = make_shm_bytes_v2(
            b"VFSHM001",
            SHM_VERSION_V2,
            HEADER_REGION_SIZE_V2 as u32,
            RING_SIZE as u32,
            64,
            64,
            2,
            PIXEL_FORMAT_RGB24,
            SHM_PROTOCOL_VERSION,
        );
        let path = write_temp_shm("vf_test_v2_header_match.bin", &data);
        let result = VideoShm::open(&path, 64, 64, 2);
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_ok(),
            "v2 header with matching protocol_version must open"
        );
    }

    #[test]
    fn shm_open_v2_header_mismatch_returns_clear_error() {
        let data = make_shm_bytes_v2(
            b"VFSHM001",
            SHM_VERSION_V2,
            HEADER_REGION_SIZE_V2 as u32,
            RING_SIZE as u32,
            64,
            64,
            2,
            PIXEL_FORMAT_RGB24,
            SHM_PROTOCOL_VERSION + 1,
        );
        let path = write_temp_shm("vf_test_v2_header_mismatch.bin", &data);
        let result = VideoShm::open(&path, 64, 64, 2);
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_err(),
            "protocol mismatch must fail deterministically"
        );
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("SHM_PROTOCOL_VERSION_MISMATCH"));
        assert!(msg.contains("expected="));
        assert!(msg.contains("found="));
    }

    #[test]
    fn v2_shm_open_validates_ring_size_match_ok() {
        let data = make_shm_bytes_v2(
            b"VFSHM001",
            SHM_VERSION_V2,
            (GLOBAL_HEADER_SIZE + SLOT_HEADER_SIZE * 8 + 4) as u32,
            8,
            64,
            64,
            2,
            PIXEL_FORMAT_RGB24,
            SHM_PROTOCOL_VERSION,
        );
        let path = write_temp_shm("vf_test_v2_ring8_match.bin", &data);
        let result = VideoShm::open_with_expected_ring_size(&path, 64, 64, 2, Some(8));
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_ok(),
            "expected ring size 8 should open when header slot_count=8"
        );
    }

    #[test]
    fn v2_shm_open_ring_size_mismatch_errors() {
        let data = make_shm_bytes_v2(
            b"VFSHM001",
            SHM_VERSION_V2,
            HEADER_REGION_SIZE_V2 as u32,
            6,
            64,
            64,
            2,
            PIXEL_FORMAT_RGB24,
            SHM_PROTOCOL_VERSION,
        );
        let path = write_temp_shm("vf_test_v2_ring_mismatch.bin", &data);
        let result = VideoShm::open_with_expected_ring_size(&path, 64, 64, 2, Some(8));
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "ring mismatch must fail deterministically");
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("SHM_RING_SIZE_MISMATCH"));
        assert!(msg.contains("expected=8"));
        assert!(msg.contains("found=6"));
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
            SHM_VERSION_V2 + 1,
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
    fn shm_protocol_schema_and_ring_metadata_are_valid() {
        let raw = include_str!("../../ipc/shm_protocol.json");
        let parsed: serde_json::Value =
            serde_json::from_str(raw).expect("protocol json should parse");

        assert_eq!(
            parsed["schema_version"].as_str(),
            Some("videoforge.shm_protocol.v1")
        );

        let ring_size_default = parsed["ring_size_default"]
            .as_u64()
            .expect("ring_size_default must exist");
        let ring_size_max = parsed["ring_size_max"]
            .as_u64()
            .expect("ring_size_max must exist");

        assert_eq!(ring_size_default, 6);
        assert!(
            ring_size_max >= ring_size_default,
            "ring_size_max must be >= ring_size_default"
        );
    }

    #[test]
    fn effective_ring_size_remains_unchanged_with_schema_metadata() {
        let raw = include_str!("../../ipc/shm_protocol.json");
        let parsed: serde_json::Value =
            serde_json::from_str(raw).expect("protocol json should parse");

        let ring_size = parsed["ring_size"].as_u64().expect("ring_size must exist") as usize;
        let ring_size_default = parsed["ring_size_default"]
            .as_u64()
            .expect("ring_size_default must exist") as usize;

        assert_eq!(ring_size, RING_SIZE);
        assert_eq!(
            ring_size_default, RING_SIZE,
            "default ring metadata must not change effective ring size"
        );
        // Header layout must continue to use the effective ring size constant.
        assert_eq!(SLOT_HEADER_REGION, SLOT_HEADER_SIZE * RING_SIZE);
        assert_eq!(HEADER_REGION_SIZE, GLOBAL_HEADER_SIZE + SLOT_HEADER_REGION);
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

    #[derive(Debug, Clone, Copy)]
    struct SimConfig {
        total_frames: u32,
        ring_size: usize,
        step_cap: usize,
        drop_per_1000: u16,
        spurious_per_1000: u16,
        poll_fallback_ticks: u8,
    }

    #[derive(Debug, Clone, Copy)]
    struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        fn seeded(seed: u64) -> Self {
            // xorshift requires non-zero state.
            let state = if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            };
            Self { state }
        }

        fn next_u32(&mut self) -> u32 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            (x >> 32) as u32
        }

        fn chance_per_1000(&mut self, per_1000: u16) -> bool {
            (self.next_u32() % 1000) < per_1000 as u32
        }

        fn choose_role(&mut self) -> u8 {
            (self.next_u32() % 3) as u8
        }
    }

    #[derive(Debug)]
    struct EventHintSim {
        cfg: SimConfig,
        rng: XorShift64,
        slots: Vec<SlotState>,
        slot_frame_index: Vec<u32>,
        slot_last_assigned: Vec<u32>,
        completed_seen: Vec<bool>,
        produced: u32,
        completed: u32,
        steps: usize,
        input_ready_hint: bool,
        output_ready_hint: bool,
        ai_hintless_ticks: u8,
        enc_hintless_ticks: u8,
    }

    impl EventHintSim {
        fn new(cfg: SimConfig, seed: u64) -> Self {
            assert!(cfg.ring_size > 0 && cfg.ring_size <= RING_SIZE);
            assert!(cfg.total_frames > 0);
            assert!(cfg.poll_fallback_ticks > 0);
            Self {
                cfg,
                rng: XorShift64::seeded(seed),
                slots: vec![SlotState::Empty; cfg.ring_size],
                slot_frame_index: vec![0; cfg.ring_size],
                slot_last_assigned: vec![0; cfg.ring_size],
                completed_seen: vec![false; cfg.total_frames as usize + 1],
                produced: 0,
                completed: 0,
                steps: 0,
                input_ready_hint: false,
                output_ready_hint: false,
                ai_hintless_ticks: 0,
                enc_hintless_ticks: 0,
            }
        }

        fn run_to_completion(&mut self) {
            while self.completed < self.cfg.total_frames && self.steps < self.cfg.step_cap {
                self.steps += 1;

                if self.rng.chance_per_1000(self.cfg.spurious_per_1000) {
                    self.input_ready_hint = true;
                }
                if self.rng.chance_per_1000(self.cfg.spurious_per_1000) {
                    self.output_ready_hint = true;
                }

                match self.rng.choose_role() {
                    0 => self.producer_step(),
                    1 => self.ai_step(),
                    _ => self.encoder_step(),
                }
            }
        }

        fn producer_step(&mut self) {
            if self.produced >= self.cfg.total_frames {
                return;
            }
            let Some(slot_idx) = self.slots.iter().position(|s| *s == SlotState::Empty) else {
                return;
            };

            apply_transition(&mut self.slots[slot_idx], SlotState::RustWriting)
                .expect("producer EMPTY -> RUST_WRITING must be valid");

            let frame_idx = self.produced + 1;
            assert!(
                frame_idx > self.slot_last_assigned[slot_idx],
                "frame index must be strictly monotonic per slot"
            );
            self.slot_last_assigned[slot_idx] = frame_idx;
            self.slot_frame_index[slot_idx] = frame_idx;
            self.produced += 1;

            apply_transition(&mut self.slots[slot_idx], SlotState::ReadyForAi)
                .expect("producer RUST_WRITING -> READY_FOR_AI must be valid");

            // Event hints are unreliable: signals can be dropped.
            if !self.rng.chance_per_1000(self.cfg.drop_per_1000) {
                self.input_ready_hint = true;
            }
        }

        fn ai_step(&mut self) {
            let should_scan = if self.input_ready_hint {
                self.input_ready_hint = false;
                self.ai_hintless_ticks = 0;
                true
            } else {
                self.ai_hintless_ticks = self.ai_hintless_ticks.saturating_add(1);
                if self.ai_hintless_ticks >= self.cfg.poll_fallback_ticks {
                    self.ai_hintless_ticks = 0;
                    true
                } else {
                    false
                }
            };

            if !should_scan {
                return;
            }

            let Some(slot_idx) = self.slots.iter().position(|s| *s == SlotState::ReadyForAi) else {
                return;
            };

            apply_transition(&mut self.slots[slot_idx], SlotState::AiProcessing)
                .expect("ai READY_FOR_AI -> AI_PROCESSING must be valid");
            apply_transition(&mut self.slots[slot_idx], SlotState::ReadyForEncode)
                .expect("ai AI_PROCESSING -> READY_FOR_ENCODE must be valid");

            if !self.rng.chance_per_1000(self.cfg.drop_per_1000) {
                self.output_ready_hint = true;
            }
        }

        fn encoder_step(&mut self) {
            let should_scan = if self.output_ready_hint {
                self.output_ready_hint = false;
                self.enc_hintless_ticks = 0;
                true
            } else {
                self.enc_hintless_ticks = self.enc_hintless_ticks.saturating_add(1);
                if self.enc_hintless_ticks >= self.cfg.poll_fallback_ticks {
                    self.enc_hintless_ticks = 0;
                    true
                } else {
                    false
                }
            };

            if !should_scan {
                return;
            }

            let Some(slot_idx) = self
                .slots
                .iter()
                .position(|s| *s == SlotState::ReadyForEncode)
            else {
                return;
            };

            apply_transition(&mut self.slots[slot_idx], SlotState::Encoding)
                .expect("encoder READY_FOR_ENCODE -> ENCODING must be valid");

            let frame_idx = self.slot_frame_index[slot_idx];
            assert!(frame_idx >= 1 && frame_idx <= self.cfg.total_frames);
            let seen = &mut self.completed_seen[frame_idx as usize];
            assert!(!*seen, "frame must not be completed more than once");
            *seen = true;
            self.completed += 1;

            apply_transition(&mut self.slots[slot_idx], SlotState::Empty)
                .expect("encoder ENCODING -> EMPTY must be valid");
            self.slot_frame_index[slot_idx] = 0;
        }

        fn assert_invariants(&self) {
            assert_eq!(
                self.completed, self.cfg.total_frames,
                "all frames must complete (no deadlock)"
            );
            assert!(
                self.steps < self.cfg.step_cap,
                "simulation must terminate within bounded step cap"
            );
            assert_eq!(
                self.produced, self.cfg.total_frames,
                "producer must submit all frames exactly once"
            );
            for frame_idx in 1..=self.cfg.total_frames as usize {
                assert!(
                    self.completed_seen[frame_idx],
                    "frame {} was never completed",
                    frame_idx
                );
            }
            assert!(
                self.slots.iter().all(|s| *s == SlotState::Empty),
                "all slots should be EMPTY at completion"
            );
            assert!(
                self.slots
                    .iter()
                    .all(|s| SlotState::from_u32(s.as_u32()).is_some()),
                "all slots must remain in known FSM states"
            );
        }
    }

    fn run_event_hint_sim(drop_per_1000: u16, spurious_per_1000: u16) {
        let cfg = SimConfig {
            total_frames: 200,
            ring_size: RING_SIZE,
            step_cap: 200_000,
            drop_per_1000,
            spurious_per_1000,
            poll_fallback_ticks: 3,
        };

        let mut sim = EventHintSim::new(cfg, 0x00C0_FFEE_u64);
        sim.run_to_completion();
        sim.assert_invariants();
    }

    #[test]
    fn event_hint_sim_completes_with_no_drops() {
        run_event_hint_sim(0, 0);
    }

    #[test]
    fn event_hint_sim_completes_with_drops() {
        run_event_hint_sim(200, 0);
    }

    #[test]
    fn event_hint_sim_completes_with_spurious_signals() {
        run_event_hint_sim(0, 100);
    }

    #[test]
    fn event_hint_sim_completes_with_drops_and_spurious() {
        run_event_hint_sim(200, 100);
    }
}
