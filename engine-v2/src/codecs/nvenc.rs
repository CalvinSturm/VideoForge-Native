//! NVENC hardware encoder for GPU-resident NV12 input.
//!
//! # Architecture
//!
//! ```text
//! GpuTexture { NV12 } ──(device ptr)──▸ nvEncRegisterResource(CUDA)
//!                                              │
//!                                    nvEncMapInputResource
//!                                              │
//!                                     nvEncEncodePicture
//!                                              │
//!                                    nvEncLockBitstream
//!                                              │
//!                                    BitstreamSink.write_packet()
//!                                              │
//!                                    nvEncUnlockBitstream
//!                                    nvEncUnmapInputResource
//! ```
//!
//! # Input strategy
//!
//! The preferred path registers the NV12 `GpuTexture` device pointer directly
//! as an NVENC input resource via `nvEncRegisterResource(CUDADEVICEPTR)`.
//! Some runtime and driver combinations reject that path, so the encoder can
//! fall back to a legacy CUDA staging surface and register that surface
//! instead.
//!
//! # Resource registration strategy
//!
//! Each unique device pointer must be registered before use.  Since the
//! pipeline reuses a bounded set of buffers (OutputRing or recycled pool),
//! we cache registrations keyed by device pointer.  A registration is valid
//! as long as the device pointer remains valid (guaranteed by `Arc<CudaSlice>`
//! lifetime in the `FrameEnvelope`).

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

use tracing::{debug, info, warn};

use crate::codecs::sys::*;
use crate::core::types::{FrameEnvelope, PixelFormat};
use crate::engine::pipeline::FrameEncoder;
use crate::error::{EngineError, Result};

// ─── Bitstream sink trait ────────────────────────────────────────────────

/// Receives encoded bitstream output.
///
/// Implementations: file writer, muxer, network sender, etc.
pub trait BitstreamSink: Send + 'static {
    fn write_packet(&mut self, data: &[u8], pts: i64, is_keyframe: bool) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
}

// ─── NVENC configuration ─────────────────────────────────────────────────

/// Encoder configuration parameters.
#[derive(Clone, Debug)]
pub struct NvEncConfig {
    /// Target width.
    pub width: u32,
    /// Target height.
    pub height: u32,
    /// Framerate numerator.
    pub fps_num: u32,
    /// Framerate denominator.
    pub fps_den: u32,
    /// Average bitrate in bits/sec (0 = CQP mode).
    pub bitrate: u32,
    /// Max bitrate in bits/sec (VBR mode).
    pub max_bitrate: u32,
    /// GOP length (frames between IDR).
    pub gop_length: u32,
    /// B-frame interval (0 = no B-frames).
    pub b_frames: u32,
    /// NV12 row pitch (must match incoming frame pitch).
    pub nv12_pitch: u32,
}

// ─── Registration cache ──────────────────────────────────────────────────

/// Caches NVENC resource registrations keyed by device pointer.
///
/// NVENC requires `nvEncRegisterResource` before a device pointer can be
/// used as input.  Since the pipeline reuses a bounded set of buffers,
/// caching registrations avoids per-frame registration overhead.
struct RegistrationCache {
    /// Map from device pointer → registered resource handle.
    entries: HashMap<u64, *mut c_void>,
}

impl RegistrationCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn get(&self, dev_ptr: u64) -> Option<*mut c_void> {
        self.entries.get(&dev_ptr).copied()
    }

    fn insert(&mut self, dev_ptr: u64, handle: *mut c_void) {
        self.entries.insert(dev_ptr, handle);
    }

    fn handles(&self) -> impl Iterator<Item = *mut c_void> + '_ {
        self.entries.values().copied()
    }
}

struct CudaContextGuard {
    prev_ctx: CUcontext,
    restore: bool,
}

impl CudaContextGuard {
    fn make_current(target_ctx: CUcontext) -> Result<Self> {
        let mut prev_ctx: CUcontext = ptr::null_mut();
        unsafe {
            check_cu_encode(cuCtxGetCurrent(&mut prev_ctx), "cuCtxGetCurrent (nvenc)")?;
        }
        if prev_ctx != target_ctx {
            unsafe {
                check_cu_encode(cuCtxSetCurrent(target_ctx), "cuCtxSetCurrent (nvenc)")?;
            }
        }
        Ok(Self {
            prev_ctx,
            restore: prev_ctx != target_ctx,
        })
    }
}

impl Drop for CudaContextGuard {
    fn drop(&mut self) {
        if self.restore {
            let rc = unsafe { cuCtxSetCurrent(self.prev_ctx) };
            if rc != CUDA_SUCCESS {
                warn!(rc, "cuCtxSetCurrent restore failed (nvenc)");
            }
        }
    }
}

// ─── NvEncoder ───────────────────────────────────────────────────────────

/// NVENC hardware encoder consuming GPU-resident NV12 `FrameEnvelope`s.
///
/// Implements [`FrameEncoder`].
pub struct NvEncoder {
    /// NVENC encoder session handle.
    encoder: *mut c_void,
    /// NVENC function pointer table.
    fns: NV_ENCODE_API_FUNCTION_LIST,
    /// Bitstream output buffer (NVENC-allocated).
    bitstream_buf: *mut c_void,
    /// Output sink for encoded data.
    sink: Box<dyn BitstreamSink>,
    /// Cached resource registrations.
    reg_cache: RegistrationCache,
    /// Encoder configuration.
    config: NvEncConfig,
    /// CUDA context handle used for all NVENC API calls.
    cuda_context: CUcontext,
    /// Runtime-negotiated NVENC API version used to build struct version fields.
    nvenc_api_version: u32,
    /// Legacy (cuMemAlloc) staging NV12 surface for NVENC compatibility fallback.
    legacy_staging_ptr: CUdeviceptr,
    /// Whether to force legacy staging for all frames.
    use_legacy_staging: bool,
    /// Frame counter for encode ordering.
    frame_idx: u32,
}

// SAFETY: NvEncoder is only used from the encode stage (single blocking thread).
// The NVENC API is thread-safe for a single session from one thread.
unsafe impl Send for NvEncoder {}

fn ptr_hex(ptr: *mut c_void) -> String {
    format!("{ptr:p}")
}

fn guid_short(guid: GUID) -> String {
    format!("{:08x}-{:04x}-{:04x}", guid.Data1, guid.Data2, guid.Data3)
}

#[derive(Clone, Copy)]
struct PresetAttempt {
    preset: GUID,
    tuning: NV_ENC_TUNING_INFO,
}

#[derive(Clone, Copy)]
struct CodecAttempt {
    guid: GUID,
    label: &'static str,
}

fn codec_attempts() -> [CodecAttempt; 2] {
    [
        CodecAttempt {
            guid: NV_ENC_CODEC_HEVC_GUID,
            label: "hevc",
        },
        CodecAttempt {
            guid: NV_ENC_CODEC_H264_GUID,
            label: "h264",
        },
    ]
}

fn guid_eq(a: GUID, b: GUID) -> bool {
    a.Data1 == b.Data1 && a.Data2 == b.Data2 && a.Data3 == b.Data3 && a.Data4 == b.Data4
}

fn preset_attempts() -> [PresetAttempt; 6] {
    [
        PresetAttempt {
            preset: NV_ENC_PRESET_P7_GUID,
            tuning: NV_ENC_TUNING_INFO_HIGH_QUALITY,
        },
        PresetAttempt {
            preset: NV_ENC_PRESET_P7_GUID,
            tuning: NV_ENC_TUNING_INFO_LOW_LATENCY,
        },
        PresetAttempt {
            preset: NV_ENC_PRESET_P7_GUID,
            tuning: NV_ENC_TUNING_INFO_UNDEFINED,
        },
        PresetAttempt {
            preset: NV_ENC_PRESET_P4_GUID,
            tuning: NV_ENC_TUNING_INFO_HIGH_QUALITY,
        },
        PresetAttempt {
            preset: NV_ENC_PRESET_P4_GUID,
            tuning: NV_ENC_TUNING_INFO_LOW_LATENCY,
        },
        PresetAttempt {
            preset: NV_ENC_PRESET_P4_GUID,
            tuning: NV_ENC_TUNING_INFO_UNDEFINED,
        },
    ]
}

impl NvEncoder {
    #[inline]
    fn struct_version(&self, ver: u32) -> u32 {
        nvenc_struct_version_with_api(self.nvenc_api_version, ver)
    }

    #[inline]
    fn struct_version_ext(&self, ver: u32) -> u32 {
        self.struct_version(ver) | (1 << 31)
    }

    /// Create and initialize an NVENC encoder session.
    ///
    /// `cuda_context` is the raw CUcontext handle.  On cudarc, this can
    /// be obtained from the device.
    pub fn new(
        cuda_context: *mut c_void,
        sink: Box<dyn BitstreamSink>,
        config: NvEncConfig,
    ) -> Result<Self> {
        let target_cuda_ctx = cuda_context as CUcontext;
        let mut api_version = NVENCAPI_VERSION;
        let mut max_supported_version = 0u32;
        unsafe {
            check_nvenc(
                NvEncodeAPIGetMaxSupportedVersion(&mut max_supported_version),
                "NvEncodeAPIGetMaxSupportedVersion",
            )?;
        }
        if api_version > max_supported_version {
            info!(
                requested_api_version = api_version,
                max_supported_version,
                "NVENC API version exceeds driver support; downgrading"
            );
            api_version = max_supported_version;
        }

        let struct_version = |ver: u32| nvenc_struct_version_with_api(api_version, ver);

        // ── Get function table ──
        let mut fns: NV_ENCODE_API_FUNCTION_LIST = unsafe { std::mem::zeroed() };
        fns.version = struct_version(2);

        // SAFETY: fns is zeroed with version set.
        // NvEncodeAPICreateInstance fills the function pointers.
        unsafe {
            check_nvenc(
                NvEncodeAPICreateInstance(&mut fns),
                "NvEncodeAPICreateInstance",
            )?;
        }

        // ── Open session ──
        let mut open_params: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS = unsafe { std::mem::zeroed() };
        open_params.version = struct_version(1);
        open_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        open_params.device = cuda_context;
        open_params.apiVersion = api_version;

        let mut encoder: *mut c_void = ptr::null_mut();
        let open_fn = fns
            .nvEncOpenEncodeSessionEx
            .ok_or_else(|| EngineError::Encode("nvEncOpenEncodeSessionEx not found".into()))?;

        let current_ctx_ptr = match cudarc::driver::result::ctx::get_current() {
            Ok(Some(ctx)) => ctx as *mut c_void,
            Ok(None) => ptr::null_mut(),
            Err(_) => ptr::null_mut(),
        };
        info!(
            open_params_version = format_args!("{:#010x}", open_params.version),
            open_params_api_version = open_params.apiVersion,
            device_type_raw = open_params.deviceType as u32,
            open_device_ptr = %ptr_hex(open_params.device),
            current_cuda_ctx = %ptr_hex(current_ctx_ptr),
            target_cuda_ctx = %ptr_hex(cuda_context),
            "NVENC open-session diagnostics"
        );
        if current_ctx_ptr.is_null() {
            warn!(
                "No current CUDA context bound on this thread before nvEncOpenEncodeSessionEx"
            );
        }
        let _ctx_guard = CudaContextGuard::make_current(target_cuda_ctx)?;

        // SAFETY: open_params is fully initialized.
        let open_status = unsafe { open_fn(&mut open_params, &mut encoder) };
        if open_status != NV_ENC_SUCCESS {
            warn!(
                status = open_status as i32,
                "nvEncOpenEncodeSessionEx failed"
            );
            return Err(EngineError::Encode(format!(
                "nvEncOpenEncodeSessionEx: NVENC error code {} (current_ctx={}, target_ctx={})",
                open_status as i32,
                ptr_hex(current_ctx_ptr),
                ptr_hex(cuda_context),
            )));
        }

        info!(
            width = config.width,
            height = config.height,
            fps = format!("{}/{}", config.fps_num, config.fps_den),
            bitrate = config.bitrate,
            "NVENC session opened"
        );

        // ── Capability discovery (codecs / presets / input formats) ──
        let get_encode_guid_count = fns
            .nvEncGetEncodeGUIDCount
            .ok_or_else(|| EngineError::Encode("nvEncGetEncodeGUIDCount not found".into()))?;
        let get_encode_guids = fns
            .nvEncGetEncodeGUIDs
            .ok_or_else(|| EngineError::Encode("nvEncGetEncodeGUIDs not found".into()))?;
        let get_input_format_count = fns
            .nvEncGetInputFormatCount
            .ok_or_else(|| EngineError::Encode("nvEncGetInputFormatCount not found".into()))?;
        let get_input_formats = fns
            .nvEncGetInputFormats
            .ok_or_else(|| EngineError::Encode("nvEncGetInputFormats not found".into()))?;
        let get_preset_guid_count = fns
            .nvEncGetEncodePresetCount
            .ok_or_else(|| EngineError::Encode("nvEncGetEncodePresetCount not found".into()))?;
        let get_preset_guids = fns
            .nvEncGetEncodePresetGUIDs
            .ok_or_else(|| EngineError::Encode("nvEncGetEncodePresetGUIDs not found".into()))?;

        let mut codec_guid_count: u32 = 0;
        unsafe {
            check_nvenc(
                get_encode_guid_count(encoder, &mut codec_guid_count),
                "nvEncGetEncodeGUIDCount",
            )?;
        }
        let mut codec_guids = vec![
            GUID {
                Data1: 0,
                Data2: 0,
                Data3: 0,
                Data4: [0; 8],
            };
            codec_guid_count as usize
        ];
        let mut returned_codec_guid_count: u32 = 0;
        unsafe {
            check_nvenc(
                get_encode_guids(
                    encoder,
                    codec_guids.as_mut_ptr(),
                    codec_guids.len() as u32,
                    &mut returned_codec_guid_count,
                ),
                "nvEncGetEncodeGUIDs",
            )?;
        }
        codec_guids.truncate(returned_codec_guid_count as usize);

        let mut capability_codecs: Vec<(CodecAttempt, Vec<PresetAttempt>)> = Vec::new();
        for codec in codec_attempts() {
            if !codec_guids.iter().copied().any(|g| guid_eq(g, codec.guid)) {
                info!(codec = codec.label, "Skipping unsupported NVENC codec");
                continue;
            }

            let mut input_fmt_count: u32 = 0;
            unsafe {
                check_nvenc(
                    get_input_format_count(encoder, codec.guid, &mut input_fmt_count),
                    "nvEncGetInputFormatCount",
                )?;
            }
            let mut input_fmts = vec![0i32; input_fmt_count as usize];
            let mut returned_input_fmt_count: u32 = 0;
            unsafe {
                check_nvenc(
                    get_input_formats(
                        encoder,
                        codec.guid,
                        input_fmts.as_mut_ptr(),
                        input_fmts.len() as u32,
                        &mut returned_input_fmt_count,
                    ),
                    "nvEncGetInputFormats",
                )?;
            }
            input_fmts.truncate(returned_input_fmt_count as usize);
            if !input_fmts
                .iter()
                .copied()
                .any(|fmt| fmt == NV_ENC_BUFFER_FORMAT_NV12)
            {
                info!(codec = codec.label, "Skipping codec without NV12 input format support");
                continue;
            }

            let mut preset_guid_count: u32 = 0;
            unsafe {
                check_nvenc(
                    get_preset_guid_count(encoder, codec.guid, &mut preset_guid_count),
                    "nvEncGetEncodePresetCount",
                )?;
            }
            let mut preset_guids = vec![
                GUID {
                    Data1: 0,
                    Data2: 0,
                    Data3: 0,
                    Data4: [0; 8],
                };
                preset_guid_count as usize
            ];
            let mut returned_preset_guid_count: u32 = 0;
            unsafe {
                check_nvenc(
                    get_preset_guids(
                        encoder,
                        codec.guid,
                        preset_guids.as_mut_ptr(),
                        preset_guids.len() as u32,
                        &mut returned_preset_guid_count,
                    ),
                    "nvEncGetEncodePresetGUIDs",
                )?;
            }
            preset_guids.truncate(returned_preset_guid_count as usize);
            let runtime_preset_attempts: Vec<PresetAttempt> = preset_attempts()
                .into_iter()
                .filter(|attempt| preset_guids.iter().copied().any(|g| guid_eq(g, attempt.preset)))
                .collect();

            if runtime_preset_attempts.is_empty() {
                info!(codec = codec.label, "Skipping codec with no supported P7/P4 presets");
                continue;
            }

            info!(
                codec = codec.label,
                preset_count = runtime_preset_attempts.len(),
                "NVENC capability-selected codec"
            );
            capability_codecs.push((codec, runtime_preset_attempts));
        }
        if capability_codecs.is_empty() {
            return Err(EngineError::Encode(
                "No NVENC codec supports required preset + NV12 input in this runtime".into(),
            ));
        }

        // ── Get preset config ──
        let get_preset_ex_fn = fns
            .nvEncGetEncodePresetConfigEx
            .ok_or_else(|| EngineError::Encode("nvEncGetEncodePresetConfigEx not found".into()))?;

        let mut preset_config: NV_ENC_PRESET_CONFIG = unsafe { std::mem::zeroed() };
        let preset_version_candidates = [5, 4, 3, 2, 1];
        let mut got_preset = false;
        let mut selected_codec = NV_ENC_CODEC_HEVC_GUID;
        let mut selected_codec_name = "hevc";
        let mut selected_preset_guid = NV_ENC_PRESET_P7_GUID;
        let mut selected_tuning = NV_ENC_TUNING_INFO_HIGH_QUALITY;
        let mut last_preset_status: Option<NVENCSTATUS> = None;

        'outer_preset: for (codec, runtime_preset_attempts) in &capability_codecs {
            for attempt in runtime_preset_attempts.iter().copied() {
                for ver in preset_version_candidates {
                    preset_config = unsafe { std::mem::zeroed() };
                    preset_config.version = struct_version(ver) | (1 << 31);
                    preset_config.presetCfg.version = struct_version(9) | (1 << 31);
                    let status = unsafe {
                        get_preset_ex_fn(
                            encoder,
                            codec.guid,
                            attempt.preset,
                            attempt.tuning,
                            &mut preset_config,
                        )
                    };
                    last_preset_status = Some(status);
                    if status == NV_ENC_SUCCESS {
                        got_preset = true;
                        selected_codec = codec.guid;
                        selected_codec_name = codec.label;
                        selected_preset_guid = attempt.preset;
                        selected_tuning = attempt.tuning;
                        info!(
                            codec = selected_codec_name,
                            struct_version = format_args!("{:#010x}", preset_config.version),
                            preset_guid = %guid_short(selected_preset_guid),
                            tuning = ?selected_tuning,
                            "Loaded NVENC preset via nvEncGetEncodePresetConfigEx"
                        );
                        break 'outer_preset;
                    }
                    if status == NV_ENC_ERR_INVALID_VERSION
                        || status == NV_ENC_ERR_UNSUPPORTED_PARAM
                        || status == NV_ENC_ERR_INVALID_PARAM
                    {
                        info!(
                            status = status as i32,
                            codec = codec.label,
                            preset_guid = %guid_short(attempt.preset),
                            tuning = ?attempt.tuning,
                            preset_struct_version = format_args!("{:#010x}", preset_config.version),
                            "nvEncGetEncodePresetConfigEx retryable status"
                        );
                        continue;
                    }
                    warn!(
                        status = status as i32,
                        codec = codec.label,
                        preset_guid = %guid_short(attempt.preset),
                        tuning = ?attempt.tuning,
                        "nvEncGetEncodePresetConfigEx returned non-retryable status; trying fallback attempts"
                    );
                    break;
                }
            }
        }

        // ── Configure encoder ──
        let mut enc_config = if got_preset {
            preset_config.presetCfg
        } else {
            warn!(
                last_status = last_preset_status.map(|s| s as i32).unwrap_or(-1),
                "Could not query NVENC preset config; using NVENC internal defaults (encodeConfig=null)"
            );
            unsafe { std::mem::zeroed() }
        };
        let use_custom_enc_config = false;
        if use_custom_enc_config {
            if enc_config.version == 0 {
                enc_config.version = struct_version(9) | (1 << 31);
            }
            if selected_codec.Data1 == NV_ENC_CODEC_HEVC_GUID.Data1
                && selected_codec.Data2 == NV_ENC_CODEC_HEVC_GUID.Data2
                && selected_codec.Data3 == NV_ENC_CODEC_HEVC_GUID.Data3
                && selected_codec.Data4 == NV_ENC_CODEC_HEVC_GUID.Data4
            {
                enc_config.profileGUID = NV_ENC_HEVC_PROFILE_MAIN_GUID;
            }
            enc_config.gopLength = config.gop_length;
            enc_config.frameIntervalP = (config.b_frames + 1) as i32;

            if config.bitrate > 0 {
                // VBR mode.
                enc_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
                enc_config.rcParams.averageBitRate = config.bitrate;
                enc_config.rcParams.maxBitRate = if config.max_bitrate > 0 {
                    config.max_bitrate
                } else {
                    config.bitrate * 3 / 2
                };
            }
            // If bitrate == 0, the preset default (typically CQP) is used.
        }

        let mut init_params: NV_ENC_INITIALIZE_PARAMS = unsafe { std::mem::zeroed() };
        init_params.version = struct_version(7) | (1 << 31);
        init_params.encodeGUID = selected_codec;
        init_params.presetGUID = selected_preset_guid;
        init_params.encodeWidth = config.width;
        init_params.encodeHeight = config.height;
        init_params.darWidth = config.width;
        init_params.darHeight = config.height;
        init_params.frameRateNum = config.fps_num;
        init_params.frameRateDen = config.fps_den;
        init_params.enablePTD = 1; // Enable picture-type decision.
        init_params.encodeConfig = if use_custom_enc_config {
            &mut enc_config
        } else {
            ptr::null_mut()
        };
        init_params.tuningInfo = selected_tuning;
        init_params.maxEncodeWidth = config.width;
        init_params.maxEncodeHeight = config.height;

        let init_fn = fns
            .nvEncInitializeEncoder
            .ok_or_else(|| EngineError::Encode("nvEncInitializeEncoder not found".into()))?;

        let mut init_ok = false;
        let mut last_init_status: Option<NVENCSTATUS> = None;
        let init_version_candidates = [7, 6, 5, 4, 3, 2, 1];
        'outer_init: for (codec, runtime_preset_attempts) in &capability_codecs {
            init_params.encodeGUID = codec.guid;
            for attempt in runtime_preset_attempts.iter().copied() {
                init_params.presetGUID = attempt.preset;
                init_params.tuningInfo = attempt.tuning;
                for ver in init_version_candidates {
                    init_params.version = struct_version(ver) | (1 << 31);
                    let status = unsafe { init_fn(encoder, &mut init_params) };
                    last_init_status = Some(status);
                    if status == NV_ENC_SUCCESS {
                        init_ok = true;
                        selected_codec_name = codec.label;
                        selected_preset_guid = attempt.preset;
                        selected_tuning = attempt.tuning;
                        break 'outer_init;
                    }
                    if status == NV_ENC_ERR_INVALID_VERSION
                        || status == NV_ENC_ERR_UNSUPPORTED_PARAM
                        || status == NV_ENC_ERR_INVALID_PARAM
                    {
                        info!(
                            status = status as i32,
                            codec = codec.label,
                            preset_guid = %guid_short(attempt.preset),
                            tuning = ?attempt.tuning,
                            init_version = format_args!("{:#010x}", init_params.version),
                            "nvEncInitializeEncoder retryable status"
                        );
                        continue;
                    }
                    warn!(
                        status = status as i32,
                        codec = codec.label,
                        preset_guid = %guid_short(attempt.preset),
                        tuning = ?attempt.tuning,
                        struct_version = format_args!("{:#010x}", init_params.version),
                        "nvEncInitializeEncoder non-retryable status"
                    );
                    check_nvenc(status, "nvEncInitializeEncoder")?;
                }
            }
        }
        if !init_ok {
            return Err(EngineError::Encode(format!(
                "nvEncInitializeEncoder: all codec/preset/tuning/version retries failed (last_status={})",
                last_init_status.map(|s| s as i32).unwrap_or(-1)
            )));
        }

        info!(
            codec = selected_codec_name,
            preset_guid = %guid_short(selected_preset_guid),
            tuning = ?selected_tuning,
            "NVENC encoder initialized"
        );

        // ── Create bitstream buffer ──
        let create_bs_fn = fns
            .nvEncCreateBitstreamBuffer
            .ok_or_else(|| EngineError::Encode("nvEncCreateBitstreamBuffer not found".into()))?;

        let mut bs_params: NV_ENC_CREATE_BITSTREAM_BUFFER = unsafe { std::mem::zeroed() };
        bs_params.version = struct_version(1);

        // SAFETY: bs_params is initialized. NVENC allocates the buffer.
        unsafe {
            check_nvenc(
                create_bs_fn(encoder, &mut bs_params),
                "nvEncCreateBitstreamBuffer",
            )?;
        }

        let bitstream_buf = bs_params.bitstreamBuffer;
        debug!("NVENC bitstream buffer created");

        Ok(Self {
            encoder,
            fns,
            bitstream_buf,
            sink,
            reg_cache: RegistrationCache::new(),
            config,
            cuda_context: target_cuda_ctx,
            nvenc_api_version: api_version,
            legacy_staging_ptr: 0,
            use_legacy_staging: false,
            frame_idx: 0,
        })
    }

    fn ensure_legacy_staging(&mut self) -> Result<u64> {
        if self.legacy_staging_ptr != 0 {
            return Ok(self.legacy_staging_ptr as u64);
        }
        let bytes = PixelFormat::Nv12.byte_size(
            self.config.width,
            self.config.height,
            self.config.nv12_pitch as usize,
        );
        let mut ptr: CUdeviceptr = 0;
        unsafe {
            check_cu_encode(
                cuMemAlloc_v2(&mut ptr as *mut CUdeviceptr, bytes),
                "cuMemAlloc_v2 (nvenc legacy staging)",
            )?;
        }
        self.legacy_staging_ptr = ptr;
        info!(
            ptr = %format_args!("{:#x}", ptr),
            bytes,
            pitch = self.config.nv12_pitch,
            width = self.config.width,
            height = self.config.height,
            "NVENC legacy staging allocated"
        );
        Ok(ptr as u64)
    }

    fn copy_to_legacy_staging(
        &mut self,
        src_dev_ptr: u64,
        src_pitch: u32,
        width: u32,
        height: u32,
    ) -> Result<u64> {
        let dst_dev_ptr = self.ensure_legacy_staging()?;
        let dst_pitch = self.config.nv12_pitch;
        let y_bytes = width as usize;
        let uv_height = (height / 2) as usize;
        let y_height = height as usize;

        let y_copy = CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: CUmemorytype::Device,
            srcHost: std::ptr::null(),
            srcDevice: src_dev_ptr as CUdeviceptr,
            srcArray: std::ptr::null_mut(),
            srcPitch: src_pitch as usize,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CUmemorytype::Device,
            dstHost: std::ptr::null_mut(),
            dstDevice: dst_dev_ptr as CUdeviceptr,
            dstArray: std::ptr::null_mut(),
            dstPitch: dst_pitch as usize,
            WidthInBytes: y_bytes,
            Height: y_height,
        };

        let uv_copy = CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: CUmemorytype::Device,
            srcHost: std::ptr::null(),
            srcDevice: (src_dev_ptr + (src_pitch as u64 * height as u64)) as CUdeviceptr,
            srcArray: std::ptr::null_mut(),
            srcPitch: src_pitch as usize,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CUmemorytype::Device,
            dstHost: std::ptr::null_mut(),
            dstDevice: (dst_dev_ptr + (dst_pitch as u64 * height as u64)) as CUdeviceptr,
            dstArray: std::ptr::null_mut(),
            dstPitch: dst_pitch as usize,
            WidthInBytes: y_bytes,
            Height: uv_height,
        };

        unsafe {
            check_cu_encode(cuMemcpy2D_v2(&y_copy as *const _), "cuMemcpy2D_v2 Y (nvenc)")?;
            check_cu_encode(cuMemcpy2D_v2(&uv_copy as *const _), "cuMemcpy2D_v2 UV (nvenc)")?;
        }
        Ok(dst_dev_ptr)
    }

    /// Register a CUDA device pointer as an NVENC input resource.
    fn register_resource(
        &mut self,
        dev_ptr: u64,
        pitch: u32,
        width: u32,
        height: u32,
    ) -> Result<*mut c_void> {
        let _ctx_guard = CudaContextGuard::make_current(self.cuda_context)?;
        if width == 0 || height == 0 {
            return Err(EngineError::Encode(format!(
                "nvEncRegisterResource preflight failed: invalid dimensions {}x{}",
                width, height
            )));
        }
        if pitch < width {
            return Err(EngineError::Encode(format!(
                "nvEncRegisterResource preflight failed: pitch {} < width {}",
                pitch, width
            )));
        }
        if pitch != self.config.nv12_pitch {
            warn!(
                configured_pitch = self.config.nv12_pitch,
                actual_pitch = pitch,
                "NVENC register resource pitch differs from encoder config"
            );
        }

        let reg_fn = self
            .fns
            .nvEncRegisterResource
            .ok_or_else(|| EngineError::Encode("nvEncRegisterResource not found".into()))?;

        let mut reg: NV_ENC_REGISTER_RESOURCE = unsafe { std::mem::zeroed() };
        reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        reg.width = width;
        reg.height = height;
        reg.pitch = pitch;
        reg.resourceToRegister = dev_ptr as *mut c_void;
        reg.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
        reg.version = self.struct_version(1);

        // SAFETY: reg is fully initialized. NVENC validates the device pointer.
        unsafe {
            check_nvenc(reg_fn(self.encoder, &mut reg), "nvEncRegisterResource")?;
        }

        let handle = reg.registeredResource;
        self.reg_cache.insert(dev_ptr, handle);
        debug!(dev_ptr, "NVENC resource registered");
        Ok(handle)
    }

    /// Encode a single frame from a registered CUDA resource.
    fn encode_frame(&mut self, frame: &FrameEnvelope) -> Result<()> {
        let _ctx_guard = CudaContextGuard::make_current(self.cuda_context)?;
        let input_dev_ptr = frame.texture.device_ptr();
        let input_pitch = frame.texture.pitch as u32;
        let width = frame.texture.width;
        let height = frame.texture.height;

        let mut dev_ptr = input_dev_ptr;
        let mut pitch = input_pitch;
        let encode_input_buffer: *mut c_void;

        // Get or create registration. If direct registration is rejected, fall back
        // to a legacy CUDA staging surface and register that surface instead.
        let reg_handle = if self.use_legacy_staging {
            dev_ptr = self.copy_to_legacy_staging(input_dev_ptr, input_pitch, width, height)?;
            pitch = self.config.nv12_pitch;
            match self.reg_cache.get(dev_ptr) {
                Some(h) => h,
                None => self.register_resource(dev_ptr, pitch, width, height)?,
            }
        } else {
            match self.reg_cache.get(dev_ptr) {
                Some(h) => h,
                None => match self.register_resource(dev_ptr, pitch, width, height) {
                    Ok(h) => h,
                    Err(EngineError::Encode(msg))
                        if msg.contains("NV_ENC_ERR_INVALID_PARAM")
                            || msg.contains("last status 8")
                            || msg.contains("all probed struct versions failed") =>
                    {
                        warn!(
                            dev_ptr = %format_args!("{:#x}", dev_ptr),
                            pitch,
                            width,
                            height,
                            err = %msg,
                            "NVENC direct resource registration rejected; enabling legacy-staging fallback"
                        );
                        self.use_legacy_staging = true;
                        dev_ptr = self.copy_to_legacy_staging(input_dev_ptr, input_pitch, width, height)?;
                        pitch = self.config.nv12_pitch;
                        match self.reg_cache.get(dev_ptr) {
                            Some(h) => h,
                            None => self.register_resource(dev_ptr, pitch, width, height)?,
                        }
                    }
                    Err(e) => return Err(e),
                },
            }
        };

        let map_fn = self
            .fns
            .nvEncMapInputResource
            .ok_or_else(|| EngineError::Encode("nvEncMapInputResource not found".into()))?;

        let mut map_params: NV_ENC_MAP_INPUT_RESOURCE = unsafe { std::mem::zeroed() };
        map_params.version = self.struct_version(1);
        map_params.subResourceIndex = 0;
        map_params.registeredResource = reg_handle;

        unsafe {
            check_nvenc(
                map_fn(self.encoder, &mut map_params),
                "nvEncMapInputResource",
            )?;
        }
        let mapped_resource = map_params.mappedResource;
        encode_input_buffer = mapped_resource;

        // Encode.
        let encode_fn = self
            .fns
            .nvEncEncodePicture
            .ok_or_else(|| EngineError::Encode("nvEncEncodePicture not found".into()))?;

        let mut pic_params: NV_ENC_PIC_PARAMS = unsafe { std::mem::zeroed() };
        pic_params.version = self.struct_version_ext(7);
        pic_params.inputWidth = self.config.width;
        pic_params.inputHeight = self.config.height;
        pic_params.inputPitch = pitch;
        pic_params.inputBuffer = encode_input_buffer;
        pic_params.outputBitstream = self.bitstream_buf;
        pic_params.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
        pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
        pic_params.frameIdx = self.frame_idx;
        pic_params.inputTimeStamp = frame.pts as u64;

        // Force IDR on keyframes from the pipeline.
        if frame.is_keyframe {
            pic_params.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
        }

        // SAFETY: All pointers are valid NVENC-owned handles.
        let encode_status = unsafe { encode_fn(self.encoder, &mut pic_params) };
        if encode_status != NV_ENC_SUCCESS && encode_status != NV_ENC_ERR_NEED_MORE_INPUT {
            check_nvenc(encode_status, "nvEncEncodePicture")?;
        }

        self.frame_idx += 1;

        if encode_status == NV_ENC_ERR_NEED_MORE_INPUT {
            info!(
                frame_idx = self.frame_idx,
                "NVENC encode returned NEED_MORE_INPUT; waiting for additional frames"
            );
            let unmap_fn = self
                .fns
                .nvEncUnmapInputResource
                .ok_or_else(|| EngineError::Encode("nvEncUnmapInputResource not found".into()))?;
            unsafe {
                check_nvenc(
                    unmap_fn(self.encoder, mapped_resource),
                    "nvEncUnmapInputResource",
                )?;
            }
            return Ok(());
        }

        // Lock bitstream output.
        let lock_fn = self
            .fns
            .nvEncLockBitstream
            .ok_or_else(|| EngineError::Encode("nvEncLockBitstream not found".into()))?;
        let unlock_fn = self
            .fns
            .nvEncUnlockBitstream
            .ok_or_else(|| EngineError::Encode("nvEncUnlockBitstream not found".into()))?;

        let mut lock_params: NV_ENC_LOCK_BITSTREAM = unsafe { std::mem::zeroed() };
        lock_params.version = self.struct_version_ext(2);
        lock_params.outputBitstream = self.bitstream_buf;

        // SAFETY: bitstream_buf is valid (from nvEncCreateBitstreamBuffer).
        unsafe {
            check_nvenc(
                lock_fn(self.encoder, &mut lock_params),
                "nvEncLockBitstream",
            )?;
        }

        // Copy encoded data to sink.
        let is_idr = lock_params.pictureType == NV_ENC_PIC_TYPE_IDR;
        let data = unsafe {
            // SAFETY: bitstreamBufferPtr is valid for bitstreamSizeInBytes.
            std::slice::from_raw_parts(
                lock_params.bitstreamBufferPtr as *const u8,
                lock_params.bitstreamSizeInBytes as usize,
            )
        };

        let sink_result = self.sink.write_packet(data, frame.pts, is_idr);

        // Unlock regardless of sink result.
        // SAFETY: bitstream_buf was locked above.
        unsafe {
            check_nvenc(
                unlock_fn(self.encoder, self.bitstream_buf),
                "nvEncUnlockBitstream",
            )?;
        }

        // Unmap input resource.
        let unmap_fn = self
            .fns
            .nvEncUnmapInputResource
            .ok_or_else(|| EngineError::Encode("nvEncUnmapInputResource not found".into()))?;

        // SAFETY: mapped_resource is valid (from nvEncMapInputResource).
        unsafe {
            check_nvenc(
                unmap_fn(self.encoder, mapped_resource),
                "nvEncUnmapInputResource",
            )?;
        }

        sink_result
    }

    /// Send EOS to flush the encoder.
    fn send_eos(&mut self) -> Result<()> {
        let _ctx_guard = CudaContextGuard::make_current(self.cuda_context)?;
        let encode_fn = self
            .fns
            .nvEncEncodePicture
            .ok_or_else(|| EngineError::Encode("nvEncEncodePicture not found".into()))?;

        let mut eos_params: NV_ENC_PIC_PARAMS = unsafe { std::mem::zeroed() };
        eos_params.version = self.struct_version_ext(7);
        eos_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;

        // SAFETY: EOS params with no input buffer — signals end of encode.
        unsafe {
            check_nvenc(
                encode_fn(self.encoder, &mut eos_params),
                "nvEncEncodePicture (EOS)",
            )?;
        }

        Ok(())
    }
}

impl FrameEncoder for NvEncoder {
    fn encode(&mut self, frame: FrameEnvelope) -> Result<()> {
        if frame.texture.format != PixelFormat::Nv12 {
            return Err(EngineError::FormatMismatch {
                expected: PixelFormat::Nv12,
                actual: frame.texture.format,
            });
        }
        self.encode_frame(&frame)
    }

    fn flush(&mut self) -> Result<()> {
        self.send_eos()?;
        self.sink.flush()?;
        info!(frames = self.frame_idx, "NVENC encoder flushed");
        Ok(())
    }
}

impl Drop for NvEncoder {
    fn drop(&mut self) {
        let _ctx_guard = CudaContextGuard::make_current(self.cuda_context).ok();
        // Unregister all cached resources.
        if let Some(unreg_fn) = self.fns.nvEncUnregisterResource {
            for handle in self.reg_cache.handles().collect::<Vec<_>>() {
                // SAFETY: handle was registered via nvEncRegisterResource.
                unsafe {
                    unreg_fn(self.encoder, handle);
                }
            }
        }

        // Destroy bitstream buffer.
        if !self.bitstream_buf.is_null() {
            if let Some(destroy_fn) = self.fns.nvEncDestroyBitstreamBuffer {
                // SAFETY: bitstream_buf was created via nvEncCreateBitstreamBuffer.
                unsafe {
                    destroy_fn(self.encoder, self.bitstream_buf);
                }
            }
        }

        if self.legacy_staging_ptr != 0 {
            unsafe {
                let _ = cuMemFree_v2(self.legacy_staging_ptr);
            }
            self.legacy_staging_ptr = 0;
        }

        // Destroy encoder session.
        if !self.encoder.is_null() {
            if let Some(destroy_fn) = self.fns.nvEncDestroyEncoder {
                // SAFETY: encoder was opened via nvEncOpenEncodeSessionEx.
                unsafe {
                    destroy_fn(self.encoder);
                }
            }
        }
    }
}
