//! Shared CUDA and Video Codec SDK FFI type aliases.
//!
//! These are pure type definitions with **no extern functions**.
//! Used by `rave-cuda`, `rave-nvcodec`, `rave-pipeline`, etc.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;
use std::os::raw::{c_int, c_ulonglong};

/// CUDA result code.
pub type CUresult = c_int;
pub const CUDA_SUCCESS: CUresult = 0;

/// CUDA device pointer (64-bit).
pub type CUdeviceptr = c_ulonglong;

/// CUDA stream handle.
pub type CUstream = *mut c_void;

/// CUDA context handle.
pub type CUcontext = *mut c_void;

/// CUDA event handle.
pub type CUevent = *mut c_void;

/// CUDA event flag: disable timing for lightweight sync-only events.
pub const CU_EVENT_DISABLE_TIMING: u32 = 0x02;

/// CUDA memory type for 2D copy descriptors.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CUmemorytype {
    Host = 0x01,
    Device = 0x02,
    Array = 0x03,
    Unified = 0x04,
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

/// NVDEC video codec enum.
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
