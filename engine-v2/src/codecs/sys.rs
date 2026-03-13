//! Raw FFI bindings to NVIDIA Video Codec SDK (nvcuvid + nvEncodeAPI).
//!
//! Covers the minimal subset required by [`NvDecoder`](super::nvdec) and
//! [`NvEncoder`](super::nvenc).  Matches Video Codec SDK v12.x headers.
//!
//! # Linking
//!
//! `build.rs` emits `-l nvcuvid` and `-l nvencodeapi` (Windows: `nvEncodeAPI64`).
//! Libraries are located via `CUDA_PATH` env var.
//!
//! # Safety
//!
//! All functions in this module are `unsafe extern "C"`.  The safe wrappers
//! in `nvdec.rs` and `nvenc.rs` enforce invariants documented below.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;
use std::os::raw::{c_int, c_short, c_uint, c_ulong, c_ulonglong};

// ═══════════════════════════════════════════════════════════════════════════
//  COMMON TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// CUDA result code.
pub type CUresult = c_int;
pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_NOT_READY: CUresult = 600;

/// CUDA device pointer (64-bit).
pub type CUdeviceptr = c_ulonglong;

/// CUDA stream handle.
pub type CUstream = *mut c_void;

/// CUDA context handle.
pub type CUcontext = *mut c_void;

// ═══════════════════════════════════════════════════════════════════════════
//  NVDEC — cuviddec.h / nvcuvid.h
// ═══════════════════════════════════════════════════════════════════════════

/// Opaque decoder handle.
pub type CUvideodecoder = *mut c_void;

/// Opaque parser handle.
pub type CUvideoparser = *mut c_void;

// ─── Enums ───────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cudaVideoCodec {
    MPEG1 = 0,
    MPEG2 = 1,
    MPEG4 = 2,
    VC1 = 3,
    H264 = 4,
    JPEG = 5,
    H264_SVC = 6,
    H264_MVC = 7,
    HEVC = 8,
    VP8 = 9,
    VP9 = 10,
    AV1 = 11,
    NumCodecs = 12,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cudaVideoSurfaceFormat {
    NV12 = 0,
    P016 = 1,
    YUV444 = 2,
    YUV444_16Bit = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cudaVideoChromaFormat {
    Monochrome = 0,
    _420 = 1,
    _422 = 2,
    _444 = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cudaVideoDeinterlaceMode {
    Weave = 0,
    Bob = 1,
    Adaptive = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cudaVideoCreateFlags {
    /// Default (no flags).
    PreferCUVID = 0,
    /// Use dedicated hardware decoder (NVDEC).
    PreferDXVA = 1,
    /// Use CUDA-based decoder.
    PreferCUDA = 2,
}

// ─── Decoder creation params ─────────────────────────────────────────────

/// Cropping rectangle for decode output.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CUVIDDECODECREATEINFO_display_area {
    pub left: c_short,
    pub top: c_short,
    pub right: c_short,
    pub bottom: c_short,
}

/// Target output rectangle (scaling).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CUVIDDECODECREATEINFO_target_rect {
    pub left: c_short,
    pub top: c_short,
    pub right: c_short,
    pub bottom: c_short,
}

/// Parameters for `cuvidCreateDecoder`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDDECODECREATEINFO {
    pub ulWidth: c_ulong,
    pub ulHeight: c_ulong,
    pub ulNumDecodeSurfaces: c_ulong,
    pub CodecType: cudaVideoCodec,
    pub ChromaFormat: cudaVideoChromaFormat,
    pub ulCreationFlags: c_ulong,
    pub bitDepthMinus8: c_ulong,
    pub ulIntraDecodeOnly: c_ulong,
    pub ulMaxWidth: c_ulong,
    pub ulMaxHeight: c_ulong,
    pub Reserved1: c_ulong,
    pub display_area: CUVIDDECODECREATEINFO_display_area,
    pub OutputFormat: cudaVideoSurfaceFormat,
    pub DeinterlaceMode: cudaVideoDeinterlaceMode,
    pub ulTargetWidth: c_ulong,
    pub ulTargetHeight: c_ulong,
    pub ulNumOutputSurfaces: c_ulong,
    pub vidLock: *mut c_void,
    pub target_rect: CUVIDDECODECREATEINFO_target_rect,
    pub enableHistogram: c_ulong,
    pub Reserved2: [c_ulong; 4],
}

// ─── Picture params (decode a single frame) ──────────────────────────────

/// Simplified picture params — full struct is codec-union, we use opaque bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUVIDPICPARAMS {
    pub PicWidthInMbs: c_int,
    pub FrameHeightInMbs: c_int,
    pub CurrPicIdx: c_int,
    pub field_pic_flag: c_int,
    pub bottom_field_flag: c_int,
    pub second_field: c_int,
    pub nBitstreamDataLen: c_uint,
    pub pBitstreamData: *const u8,
    pub nNumSlices: c_uint,
    pub pSliceDataOffsets: *const c_uint,
    pub ref_pic_flag: c_int,
    pub intra_pic_flag: c_int,
    pub Reserved: [c_uint; 30],
    /// Codec-specific packed data (H.264/HEVC/VP9/AV1 union).
    pub CodecSpecific: [u8; 1024],
}

// ─── Parser callback types ───────────────────────────────────────────────

/// Video format information emitted by the parser.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDEOFORMAT {
    pub codec: cudaVideoCodec,
    pub frame_rate: CUVIDEOFORMAT_frame_rate,
    pub progressive_sequence: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub min_num_decode_surfaces: u8,
    pub coded_width: c_uint,
    pub coded_height: c_uint,
    pub display_area: CUVIDEOFORMAT_display_area,
    pub chroma_format: cudaVideoChromaFormat,
    pub bitrate: c_uint,
    pub display_aspect_ratio: CUVIDEOFORMAT_display_aspect_ratio,
    pub video_signal_description: CUVIDEOFORMAT_video_signal_description,
    pub seqhdr_data_length: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDEOFORMAT_frame_rate {
    pub numerator: c_uint,
    pub denominator: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDEOFORMAT_display_area {
    pub left: c_int,
    pub top: c_int,
    pub right: c_int,
    pub bottom: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDEOFORMAT_display_aspect_ratio {
    pub x: c_int,
    pub y: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDEOFORMAT_video_signal_description {
    /// Bitfield-packed flags from `nvcuvid.h`.
    /// bits 0..=2 video_format, bit 3 full_range, bits 4..=7 reserved.
    pub flags: u8,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

/// Parser dispatch info for a decoded picture.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDPARSERDISPINFO {
    pub picture_index: c_int,
    pub progressive_frame: c_int,
    pub top_field_first: c_int,
    pub repeat_first_field: c_int,
    pub timestamp: c_ulonglong,
}

/// Callback: sequence header parsed (reports format).
pub type PFNVIDSEQUENCECALLBACK =
    unsafe extern "system" fn(user_data: *mut c_void, format: *mut CUVIDEOFORMAT) -> c_int;

/// Callback: a picture has been decoded.
pub type PFNVIDDECODECALLBACK =
    unsafe extern "system" fn(user_data: *mut c_void, pic_params: *mut CUVIDPICPARAMS) -> c_int;

/// Callback: a decoded picture is ready for display.
pub type PFNVIDDISPLAYCALLBACK =
    unsafe extern "system" fn(user_data: *mut c_void, disp_info: *mut CUVIDPARSERDISPINFO) -> c_int;

/// Opaque AV1 operating-point info passed to parser callback.
#[repr(C)]
pub struct CUVIDOPERATINGPOINTINFO {
    _private: [u8; 0],
}

/// Opaque SEI message info passed to parser callback.
#[repr(C)]
pub struct CUVIDSEIMESSAGEINFO {
    _private: [u8; 0],
}

/// Opaque extended video format info.
#[repr(C)]
pub struct CUVIDEOFORMATEX {
    _private: [u8; 0],
}

/// Callback: parser requests AV1 operating point selection.
pub type PFNVIDOPPOINTCALLBACK = unsafe extern "system" fn(
    user_data: *mut c_void,
    op_info: *mut CUVIDOPERATINGPOINTINFO,
) -> c_int;

/// Callback: parser reports parsed SEI message batch.
pub type PFNVIDSEIMSGCALLBACK =
    unsafe extern "system" fn(user_data: *mut c_void, sei_info: *mut CUVIDSEIMESSAGEINFO) -> c_int;

/// Parser creation params.
#[repr(C)]
pub struct CUVIDPARSERPARAMS {
    pub CodecType: cudaVideoCodec,
    pub ulMaxNumDecodeSurfaces: c_uint,
    pub ulClockRate: c_uint,
    pub ulErrorThreshold: c_uint,
    pub ulMaxDisplayDelay: c_uint,
    /// Bitfield-packed parser flags from `nvcuvid.h`:
    /// bit 0 = bAnnexb, bit 1 = bMemoryOptimize, bits 2..31 reserved.
    pub ulParserFlags: c_uint,
    pub uReserved1: [c_uint; 4],
    pub pUserData: *mut c_void,
    pub pfnSequenceCallback: Option<PFNVIDSEQUENCECALLBACK>,
    pub pfnDecodePicture: Option<PFNVIDDECODECALLBACK>,
    pub pfnDisplayPicture: Option<PFNVIDDISPLAYCALLBACK>,
    pub pfnGetOperatingPoint: Option<PFNVIDOPPOINTCALLBACK>,
    pub pfnGetSEIMsg: Option<PFNVIDSEIMSGCALLBACK>,
    pub pvReserved2: [*mut c_void; 5],
    pub pExtVideoInfo: *mut CUVIDEOFORMATEX,
}

/// Bitstream packet fed to the parser.
#[repr(C)]
pub struct CUVIDSOURCEDATAPACKET {
    pub flags: c_ulong,
    pub payload_size: c_ulong,
    pub payload: *const u8,
    pub timestamp: c_ulonglong,
}

/// Parser input flags.
pub const CUVID_PKT_ENDOFSTREAM: c_ulong = 0x01;
pub const CUVID_PKT_TIMESTAMP: c_ulong = 0x02;
pub const CUVID_PKT_DISCONTINUITY: c_ulong = 0x04;

/// Processing params for `cuvidMapVideoFrame64`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDPROCPARAMS {
    pub progressive_frame: c_int,
    pub second_field: c_int,
    pub top_field_first: c_int,
    pub unpaired_field: c_int,
    pub reserved_flags: c_uint,
    pub reserved_zero: c_uint,
    pub raw_input_dptr: c_ulonglong,
    pub raw_input_pitch: c_uint,
    pub raw_input_format: c_uint,
    pub raw_output_dptr: c_ulonglong,
    pub raw_output_pitch: c_uint,
    pub Reserved1: c_uint,
    pub output_stream: CUstream,
    pub Reserved: [c_uint; 46],
    pub histogram_dptr: *mut c_ulonglong,
    pub Reserved2: [*mut c_void; 1],
}

// ─── NVDEC functions ─────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn cuvidCreateVideoParser(
        parser: *mut CUvideoparser,
        params: *mut CUVIDPARSERPARAMS,
    ) -> CUresult;

    pub fn cuvidParseVideoData(
        parser: CUvideoparser,
        packet: *mut CUVIDSOURCEDATAPACKET,
    ) -> CUresult;

    pub fn cuvidDestroyVideoParser(parser: CUvideoparser) -> CUresult;

    pub fn cuvidCreateDecoder(
        decoder: *mut CUvideodecoder,
        params: *mut CUVIDDECODECREATEINFO,
    ) -> CUresult;

    pub fn cuvidDecodePicture(decoder: CUvideodecoder, pic_params: *mut CUVIDPICPARAMS)
    -> CUresult;

    pub fn cuvidMapVideoFrame64(
        decoder: CUvideodecoder,
        pic_idx: c_int,
        dev_ptr: *mut CUdeviceptr,
        pitch: *mut c_uint,
        params: *mut CUVIDPROCPARAMS,
    ) -> CUresult;

    pub fn cuvidUnmapVideoFrame64(decoder: CUvideodecoder, dev_ptr: CUdeviceptr) -> CUresult;

    pub fn cuvidDestroyDecoder(decoder: CUvideodecoder) -> CUresult;
}

// ═══════════════════════════════════════════════════════════════════════════
//  NVENC — nvEncodeAPI.h (bindgen-generated)
// ═══════════════════════════════════════════════════════════════════════════

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_imports,
    dead_code,
    clippy::all
)]
mod nvenc_generated {
    include!("nvenc_bindings_generated.rs");
}

pub use nvenc_generated::{
    GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_BUFFER_USAGE, NV_ENC_CODEC_CONFIG, NV_ENC_CONFIG,
    NV_ENC_CONFIG_HEVC, NV_ENC_CREATE_BITSTREAM_BUFFER, NV_ENC_CREATE_INPUT_BUFFER,
    NV_ENC_DEVICE_TYPE, NV_ENC_INITIALIZE_PARAMS, NV_ENC_INPUT_RESOURCE_TYPE,
    NV_ENC_LOCK_BITSTREAM, NV_ENC_LOCK_INPUT_BUFFER, NV_ENC_MAP_INPUT_RESOURCE,
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS, NV_ENC_PARAMS_RC_MODE, NV_ENC_PIC_PARAMS,
    NV_ENC_PIC_STRUCT, NV_ENC_PIC_TYPE, NV_ENC_PRESET_CONFIG, NV_ENC_RC_PARAMS,
    NV_ENC_REGISTER_RESOURCE, NV_ENC_TUNING_INFO, NV_ENCODE_API_FUNCTION_LIST,
    NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE, NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION,
    NVENCAPI_VERSION, NVENCSTATUS, NvEncodeAPICreateInstance, NvEncodeAPIGetMaxSupportedVersion,
};

pub const NV_ENC_SUCCESS: NVENCSTATUS = nvenc_generated::_NVENCSTATUS_NV_ENC_SUCCESS;
pub const NV_ENC_ERR_NO_ENCODE_DEVICE: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_NO_ENCODE_DEVICE;
pub const NV_ENC_ERR_UNSUPPORTED_DEVICE: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_UNSUPPORTED_DEVICE;
pub const NV_ENC_ERR_INVALID_ENCODERDEVICE: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_INVALID_ENCODERDEVICE;
pub const NV_ENC_ERR_INVALID_DEVICE: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_INVALID_DEVICE;
pub const NV_ENC_ERR_DEVICE_NOT_EXIST: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_DEVICE_NOT_EXIST;
pub const NV_ENC_ERR_INVALID_PTR: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_INVALID_PTR;
pub const NV_ENC_ERR_INVALID_EVENT: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_INVALID_EVENT;
pub const NV_ENC_ERR_INVALID_PARAM: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_INVALID_PARAM;
pub const NV_ENC_ERR_INVALID_CALL: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_INVALID_CALL;
pub const NV_ENC_ERR_OUT_OF_MEMORY: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_OUT_OF_MEMORY;
pub const NV_ENC_ERR_ENCODER_NOT_INITIALIZED: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_ENCODER_NOT_INITIALIZED;
pub const NV_ENC_ERR_UNSUPPORTED_PARAM: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_UNSUPPORTED_PARAM;
pub const NV_ENC_ERR_LOCK_BUSY: NVENCSTATUS = nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_LOCK_BUSY;
pub const NV_ENC_ERR_NOT_ENOUGH_BUFFER: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_NOT_ENOUGH_BUFFER;
pub const NV_ENC_ERR_INVALID_VERSION: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_INVALID_VERSION;
pub const NV_ENC_ERR_MAP_FAILED: NVENCSTATUS = nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_MAP_FAILED;
pub const NV_ENC_ERR_NEED_MORE_INPUT: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_NEED_MORE_INPUT;
pub const NV_ENC_ERR_ENCODER_BUSY: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_ENCODER_BUSY;
pub const NV_ENC_ERR_EVENT_NOT_REGISTERD: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_EVENT_NOT_REGISTERD;
pub const NV_ENC_ERR_GENERIC: NVENCSTATUS = nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_GENERIC;
pub const NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY;
pub const NV_ENC_ERR_UNIMPLEMENTED: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_UNIMPLEMENTED;
pub const NV_ENC_ERR_RESOURCE_REGISTER_FAILED: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_RESOURCE_REGISTER_FAILED;
pub const NV_ENC_ERR_RESOURCE_NOT_REGISTERED: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_RESOURCE_NOT_REGISTERED;
pub const NV_ENC_ERR_RESOURCE_NOT_MAPPED: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_RESOURCE_NOT_MAPPED;
pub const NV_ENC_ERR_NEED_MORE_OUTPUT: NVENCSTATUS =
    nvenc_generated::_NVENCSTATUS_NV_ENC_ERR_NEED_MORE_OUTPUT;

pub const NV_ENC_DEVICE_TYPE_DIRECTX: NV_ENC_DEVICE_TYPE =
    nvenc_generated::_NV_ENC_DEVICE_TYPE_NV_ENC_DEVICE_TYPE_DIRECTX;
pub const NV_ENC_DEVICE_TYPE_CUDA: NV_ENC_DEVICE_TYPE =
    nvenc_generated::_NV_ENC_DEVICE_TYPE_NV_ENC_DEVICE_TYPE_CUDA;
pub const NV_ENC_DEVICE_TYPE_OPENGL: NV_ENC_DEVICE_TYPE =
    nvenc_generated::_NV_ENC_DEVICE_TYPE_NV_ENC_DEVICE_TYPE_OPENGL;

pub const NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX: NV_ENC_INPUT_RESOURCE_TYPE =
    nvenc_generated::_NV_ENC_INPUT_RESOURCE_TYPE_NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
pub const NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR: NV_ENC_INPUT_RESOURCE_TYPE =
    nvenc_generated::_NV_ENC_INPUT_RESOURCE_TYPE_NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
pub const NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY: NV_ENC_INPUT_RESOURCE_TYPE =
    nvenc_generated::_NV_ENC_INPUT_RESOURCE_TYPE_NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY;
pub const NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX: NV_ENC_INPUT_RESOURCE_TYPE =
    nvenc_generated::_NV_ENC_INPUT_RESOURCE_TYPE_NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX;

pub const NV_ENC_BUFFER_FORMAT_NV12: NV_ENC_BUFFER_FORMAT =
    nvenc_generated::_NV_ENC_BUFFER_FORMAT_NV_ENC_BUFFER_FORMAT_NV12;
pub const NV_ENC_INPUT_IMAGE: NV_ENC_BUFFER_USAGE =
    nvenc_generated::_NV_ENC_BUFFER_USAGE_NV_ENC_INPUT_IMAGE;

pub const NV_ENC_PIC_TYPE_IDR: NV_ENC_PIC_TYPE =
    nvenc_generated::_NV_ENC_PIC_TYPE_NV_ENC_PIC_TYPE_IDR;
pub const NV_ENC_PIC_STRUCT_FRAME: NV_ENC_PIC_STRUCT =
    nvenc_generated::_NV_ENC_PIC_STRUCT_NV_ENC_PIC_STRUCT_FRAME;

pub const NV_ENC_TUNING_INFO_UNDEFINED: NV_ENC_TUNING_INFO =
    nvenc_generated::NV_ENC_TUNING_INFO_NV_ENC_TUNING_INFO_UNDEFINED;
pub const NV_ENC_TUNING_INFO_HIGH_QUALITY: NV_ENC_TUNING_INFO =
    nvenc_generated::NV_ENC_TUNING_INFO_NV_ENC_TUNING_INFO_HIGH_QUALITY;
pub const NV_ENC_TUNING_INFO_LOW_LATENCY: NV_ENC_TUNING_INFO =
    nvenc_generated::NV_ENC_TUNING_INFO_NV_ENC_TUNING_INFO_LOW_LATENCY;

pub const NV_ENC_PARAMS_RC_VBR: NV_ENC_PARAMS_RC_MODE =
    nvenc_generated::_NV_ENC_PARAMS_RC_MODE_NV_ENC_PARAMS_RC_VBR;

// Well-known GUIDs used by engine codepaths.
pub const NV_ENC_CODEC_H264_GUID: GUID = GUID {
    Data1: 0x6BC82762,
    Data2: 0x4E63,
    Data3: 0x4CA4,
    Data4: [0xAA, 0x85, 0x1A, 0x4F, 0x6A, 0x21, 0xF5, 0x07],
};

pub const NV_ENC_CODEC_HEVC_GUID: GUID = GUID {
    Data1: 0x790CDC88,
    Data2: 0x4522,
    Data3: 0x4D7B,
    Data4: [0x94, 0x25, 0xBD, 0xA9, 0x97, 0x5F, 0x76, 0x03],
};

pub const NV_ENC_PRESET_P7_GUID: GUID = GUID {
    Data1: 0x84848C12,
    Data2: 0x6F71,
    Data3: 0x4C13,
    Data4: [0x93, 0x1B, 0x53, 0xE5, 0x6F, 0x78, 0x84, 0x3B],
};

pub const NV_ENC_PRESET_P4_GUID: GUID = GUID {
    Data1: 0x90A7B826,
    Data2: 0xDF06,
    Data3: 0x4862,
    Data4: [0xB9, 0xD2, 0xCD, 0x6D, 0x73, 0xA0, 0x86, 0x81],
};

pub const NV_ENC_HEVC_PROFILE_MAIN_GUID: GUID = GUID {
    Data1: 0xB514C39A,
    Data2: 0xB55B,
    Data3: 0x40FA,
    Data4: [0x87, 0x87, 0x67, 0xED, 0x5E, 0x28, 0x49, 0x4D],
};

pub const NV_ENC_HEVC_PROFILE_MAIN10_GUID: GUID = GUID {
    Data1: 0xFA4D2B6C,
    Data2: 0x3A5B,
    Data3: 0x411A,
    Data4: [0x80, 0x18, 0x0A, 0x3F, 0x5E, 0x3C, 0x9B, 0x44],
};

/// Compute a struct version field: `NVENCAPI_VERSION | (struct_ver << 16) | (0x7 << 28)`.
#[inline]
pub const fn nvenc_struct_version(struct_ver: u32) -> u32 {
    NVENCAPI_VERSION | (struct_ver << 16) | (0x7 << 28)
}

/// Compute a struct version field using a runtime-negotiated NVENC API version.
#[inline]
pub const fn nvenc_struct_version_with_api(api_version: u32, struct_ver: u32) -> u32 {
    api_version | (struct_ver << 16) | (0x7 << 28)
}

/// End-of-stream / keyframe flags for NV_ENC_PIC_PARAMS::encodePicFlags.
pub const NV_ENC_PIC_FLAG_EOS: u32 = 0x08;
pub const NV_ENC_PIC_FLAG_FORCEIDR: u32 = 0x02;
// ═══════════════════════════════════════════════════════════════════════════
//  CUDA DRIVER — event functions (not in cudarc)
// ═══════════════════════════════════════════════════════════════════════════

/// CUDA event handle.
pub type CUevent = *mut c_void;

pub const CU_EVENT_DISABLE_TIMING: c_uint = 0x02;

unsafe extern "C" {
    pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuEventCreate(phEvent: *mut CUevent, Flags: c_uint) -> CUresult;
    pub fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;
    pub fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;
    pub fn cuEventQuery(hEvent: CUevent) -> CUresult;
    pub fn cuEventSynchronize(hEvent: CUevent) -> CUresult;
    pub fn cuStreamSynchronize(hStream: CUstream) -> CUresult;
    pub fn cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, Flags: c_uint) -> CUresult;
}

// ═══════════════════════════════════════════════════════════════════════════
//  CUDA DRIVER — async memcpy (for D2D copy of decoded surfaces)
// ═══════════════════════════════════════════════════════════════════════════

unsafe extern "C" {
    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpyDtoH_v2(
        dstHost: *mut c_void,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult;
    pub fn cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;
    pub fn cuMemcpy2DAsync_v2(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) -> CUresult;
}

/// 2D memory copy descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUDA_MEMCPY2D {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: *const c_void,
    pub srcPitch: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: *mut c_void,
    pub dstPitch: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CUmemorytype {
    Host = 0x01,
    Device = 0x02,
    Array = 0x03,
    Unified = 0x04,
}

// ═══════════════════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a CUDA result to an engine Result.
#[inline]
pub fn check_cu(result: CUresult, context: &str) -> crate::error::Result<()> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(crate::error::EngineError::Decode(format!(
            "{context}: CUDA error code {result}"
        )))
    }
}

/// Convert a CUDA result to an encode-context engine Result.
#[inline]
pub fn check_cu_encode(result: CUresult, context: &str) -> crate::error::Result<()> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(crate::error::EngineError::Encode(format!(
            "{context}: CUDA error code {result}"
        )))
    }
}

/// Convert an NVENC status to an engine Result.
#[inline]
pub fn check_nvenc(status: NVENCSTATUS, context: &str) -> crate::error::Result<()> {
    if status == NV_ENC_SUCCESS {
        Ok(())
    } else {
        let name = match status {
            NV_ENC_ERR_NO_ENCODE_DEVICE => "NV_ENC_ERR_NO_ENCODE_DEVICE",
            NV_ENC_ERR_UNSUPPORTED_DEVICE => "NV_ENC_ERR_UNSUPPORTED_DEVICE",
            NV_ENC_ERR_INVALID_ENCODERDEVICE => "NV_ENC_ERR_INVALID_ENCODERDEVICE",
            NV_ENC_ERR_INVALID_DEVICE => "NV_ENC_ERR_INVALID_DEVICE",
            NV_ENC_ERR_DEVICE_NOT_EXIST => "NV_ENC_ERR_DEVICE_NOT_EXIST",
            NV_ENC_ERR_INVALID_PTR => "NV_ENC_ERR_INVALID_PTR",
            NV_ENC_ERR_INVALID_EVENT => "NV_ENC_ERR_INVALID_EVENT",
            NV_ENC_ERR_INVALID_PARAM => "NV_ENC_ERR_INVALID_PARAM",
            NV_ENC_ERR_INVALID_CALL => "NV_ENC_ERR_INVALID_CALL",
            NV_ENC_ERR_OUT_OF_MEMORY => "NV_ENC_ERR_OUT_OF_MEMORY",
            NV_ENC_ERR_ENCODER_NOT_INITIALIZED => "NV_ENC_ERR_ENCODER_NOT_INITIALIZED",
            NV_ENC_ERR_UNSUPPORTED_PARAM => "NV_ENC_ERR_UNSUPPORTED_PARAM",
            NV_ENC_ERR_LOCK_BUSY => "NV_ENC_ERR_LOCK_BUSY",
            NV_ENC_ERR_NOT_ENOUGH_BUFFER => "NV_ENC_ERR_NOT_ENOUGH_BUFFER",
            NV_ENC_ERR_INVALID_VERSION => "NV_ENC_ERR_INVALID_VERSION",
            NV_ENC_ERR_MAP_FAILED => "NV_ENC_ERR_MAP_FAILED",
            NV_ENC_ERR_NEED_MORE_INPUT => "NV_ENC_ERR_NEED_MORE_INPUT",
            NV_ENC_ERR_ENCODER_BUSY => "NV_ENC_ERR_ENCODER_BUSY",
            NV_ENC_ERR_EVENT_NOT_REGISTERD => "NV_ENC_ERR_EVENT_NOT_REGISTERD",
            NV_ENC_ERR_GENERIC => "NV_ENC_ERR_GENERIC",
            NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY => "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY",
            NV_ENC_ERR_UNIMPLEMENTED => "NV_ENC_ERR_UNIMPLEMENTED",
            NV_ENC_ERR_RESOURCE_REGISTER_FAILED => "NV_ENC_ERR_RESOURCE_REGISTER_FAILED",
            NV_ENC_ERR_RESOURCE_NOT_REGISTERED => "NV_ENC_ERR_RESOURCE_NOT_REGISTERED",
            NV_ENC_ERR_RESOURCE_NOT_MAPPED => "NV_ENC_ERR_RESOURCE_NOT_MAPPED",
            NV_ENC_ERR_NEED_MORE_OUTPUT => "NV_ENC_ERR_NEED_MORE_OUTPUT",
            _ => "NV_ENC_ERR_UNKNOWN",
        };
        Err(crate::error::EngineError::Encode(format!(
            "{context}: NVENC error {name} ({})",
            status as i32,
        )))
    }
}
