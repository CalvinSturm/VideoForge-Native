//! NVDEC hardware decoder — zero-copy GPU-resident NV12 output.
//!
//! # Architecture
//!
//! ```text
//! BitstreamSource ──(host NAL)──▸ cuvidParseVideoData
//!                                       │
//!                        ┌───callback────┘
//!                        ▼
//!              cuvidDecodePicture (on NVDEC HW)
//!                        │
//!              cuvidMapVideoFrame64
//!                        │
//!           ┌── NVDEC surface (device ptr + pitch) ──┐
//!           │                                        │
//!           │  cuMemcpy2DAsync (D2D on decode_stream)│
//!           │                                        │
//!           └── our CudaSlice buffer ────────────────┘
//!                        │
//!              cuvidUnmapVideoFrame64
//!                        │
//!              cuEventRecord(decode_done, decode_stream)
//!                        │
//!              GpuTexture { NV12, pitch-aligned }
//! ```
//!
//! # Why D2D copy?
//!
//! NVDEC surfaces are a finite pool (typically 8-16).  They must be
//! returned quickly via `cuvidUnmapVideoFrame64` to avoid stalling the
//! hardware.  Copying to our own buffer (~24 µs for 4K NV12 at 500 GB/s)
//! decouples decoder surface lifetime from pipeline frame lifetime.
//!
//! # Cross-stream synchronization
//!
//! After the D2D copy completes on `decode_stream`, a CUDA event is
//! recorded.  The preprocess stage must call
//! `cuStreamWaitEvent(preprocess_stream, event)` before reading the buffer.
//! This ensures ordering without CPU-blocking sync.

use std::collections::VecDeque;
use std::ffi::{c_int, c_short, c_uint, c_ulong, c_ulonglong, c_void};
use std::ptr;
use std::sync::Arc;

use cudarc::driver::DevicePtr;

use tracing::{debug, info};

use crate::codecs::sys::*;
use crate::core::context::GpuContext;
use crate::core::types::{FrameEnvelope, GpuTexture, PixelFormat};
use crate::engine::pipeline::{DecodedFrameEnvelope, FrameDecoder, StreamReadyEvent};
use crate::error::{EngineError, Result};

// ─── Bitstream source trait ──────────────────────────────────────────────

/// Demuxed compressed bitstream packets (host-side, NOT raw pixels).
///
/// Implementations: file reader, network receiver, FFmpeg demuxer, etc.
pub trait BitstreamSource: Send + 'static {
    fn read_packet(&mut self) -> Result<Option<BitstreamPacket>>;
}

/// A single demuxed NAL unit or access unit.
pub struct BitstreamPacket {
    /// Compressed bitstream data (Annex B or length-prefixed).
    /// Host memory is acceptable — this is codec-compressed (~10 KB/frame).
    pub data: Vec<u8>,
    /// Presentation timestamp in stream time base.
    pub pts: i64,
    /// Whether this packet encodes an IDR/keyframe.
    pub is_keyframe: bool,
}

// ─── Decoded frame event ─────────────────────────────────────────────────

/// A decoded NV12 frame with its associated sync event.
///
/// The preprocess stage MUST wait on `decode_event` before reading
/// `texture.data` to ensure the D2D copy has completed on `decode_stream`.
pub struct DecodedFrame {
    pub envelope: FrameEnvelope,
    /// CUDA event recorded on `decode_stream` after D2D copy.
    /// Downstream calls `cuStreamWaitEvent(preprocess_stream, event, 0)`.
    pub decode_event: StreamReadyEvent,
}

// ─── Per-frame event pool ────────────────────────────────────────────────

// ─── Decoder callback state ─────────────────────────────────────────────

/// Shared state between parser callbacks and the main decoder.
///
/// Parser callbacks push decoded frame info here; `decode_next()` drains it.
struct CallbackState {
    decoder: CUvideodecoder,
    format: Option<CUVIDEOFORMAT>,
    pending_display: VecDeque<CUVIDPARSERDISPINFO>,
    decoder_created: bool,
    max_decode_surfaces: u32,
    codec: cudaVideoCodec,
}

struct PendingUnmap {
    src_ptr: CUdeviceptr,
    decode_ready: StreamReadyEvent,
}

// ─── NvDecoder ───────────────────────────────────────────────────────────

/// NVDEC hardware decoder producing GPU-resident NV12 `FrameEnvelope`s.
///
/// Implements [`FrameDecoder`].  Each call to `decode_next()` returns one
/// frame with a CUDA event for cross-stream synchronization.
pub struct NvDecoder {
    parser: CUvideoparser,
    ctx: Arc<GpuContext>,
    source: Box<dyn BitstreamSource>,
    cb_state: Box<CallbackState>,
    pending_unmaps: VecDeque<PendingUnmap>,
    frame_index: u64,
    eos_sent: bool,
}

impl NvDecoder {
    fn init_parser_if_needed(&mut self) -> Result<()> {
        if !self.parser.is_null() {
            return Ok(());
        }
        let cb_state_ptr: *mut CallbackState = &mut *self.cb_state;
        let codec = self.cb_state.codec;

        let mut parser_params: CUVIDPARSERPARAMS = unsafe { std::mem::zeroed() };
        parser_params.CodecType = codec;
        parser_params.ulMaxNumDecodeSurfaces = 8;
        parser_params.ulMaxDisplayDelay = 4;
        // nvcuvid.h bitfield: bAnnexb=bit0, bMemoryOptimize=bit1.
        // bAnnexb only applies to AV1 Annex-B streams.
        parser_params.ulParserFlags = if matches!(codec, cudaVideoCodec::AV1) {
            0x1
        } else {
            0
        };
        parser_params.pUserData = cb_state_ptr as *mut c_void;
        parser_params.pfnSequenceCallback = Some(sequence_callback);
        parser_params.pfnDecodePicture = Some(decode_callback);
        parser_params.pfnDisplayPicture = Some(display_callback);

        let mut parser: CUvideoparser = ptr::null_mut();
        unsafe {
            check_cu(
                cuvidCreateVideoParser(&mut parser, &mut parser_params),
                "cuvidCreateVideoParser",
            )?;
        }
        self.parser = parser;
        info!(
            ?codec,
            thread = ?std::thread::current().id(),
            parser = format_args!("{:p}", parser),
            "NVDEC parser created"
        );
        Ok(())
    }

    /// Create a new hardware decoder.
    ///
    /// The parser is created immediately.  The actual hardware decoder is
    /// created lazily when the first sequence header is parsed (parser
    /// callback determines resolution, codec profile, etc.).
    pub fn new(
        ctx: Arc<GpuContext>,
        source: Box<dyn BitstreamSource>,
        codec: cudaVideoCodec,
    ) -> Result<Self> {
        let cb_state = Box::new(CallbackState {
            decoder: ptr::null_mut(),
            format: None,
            pending_display: VecDeque::new(),
            decoder_created: false,
            max_decode_surfaces: 8,
            codec,
        });

        Ok(Self {
            parser: ptr::null_mut(),
            ctx,
            source,
            cb_state,
            pending_unmaps: VecDeque::new(),
            frame_index: 0,
            eos_sent: false,
        })
    }

    fn max_pending_unmaps(&self) -> usize {
        self.cb_state.max_decode_surfaces.saturating_sub(2).max(1) as usize
    }

    fn unmap_surface(&self, src_ptr: CUdeviceptr) -> Result<()> {
        let decoder = self.cb_state.decoder;
        if decoder.is_null() {
            return Ok(());
        }
        unsafe {
            check_cu(
                cuvidUnmapVideoFrame64(decoder, src_ptr),
                "cuvidUnmapVideoFrame64",
            )?;
        }
        Ok(())
    }

    fn release_completed_unmaps(&mut self) -> Result<()> {
        while let Some(front) = self.pending_unmaps.front() {
            if !front.decode_ready.query_complete()? {
                break;
            }
            let pending = self.pending_unmaps.pop_front().unwrap();
            self.unmap_surface(pending.src_ptr)?;
        }
        Ok(())
    }

    fn enforce_unmap_budget(&mut self) -> Result<()> {
        while self.pending_unmaps.len() >= self.max_pending_unmaps() {
            let pending = self.pending_unmaps.pop_front().unwrap();
            pending.decode_ready.synchronize()?;
            self.unmap_surface(pending.src_ptr)?;
        }
        Ok(())
    }

    fn release_all_pending_unmaps(&mut self) -> Result<()> {
        while let Some(pending) = self.pending_unmaps.pop_front() {
            pending.decode_ready.synchronize()?;
            self.unmap_surface(pending.src_ptr)?;
        }
        Ok(())
    }

    /// Feed one bitstream packet to the parser.
    fn feed_packet(&mut self, packet: &BitstreamPacket) -> Result<()> {
        let mut pkt = CUVIDSOURCEDATAPACKET {
            flags: CUVID_PKT_TIMESTAMP,
            payload_size: packet.data.len() as c_ulong,
            payload: packet.data.as_ptr(),
            timestamp: packet.pts as c_ulonglong,
        };

        // SAFETY: pkt.payload points to valid host memory (packet.data).
        // Parser copies the data internally before returning.
        debug!(
            thread = ?std::thread::current().id(),
            parser = format_args!("{:p}", self.parser),
            payload_size = packet.data.len(),
            pts = packet.pts,
            "NVDEC feed_packet enter cuvidParseVideoData"
        );
        unsafe {
            check_cu(
                cuvidParseVideoData(self.parser, &mut pkt),
                "cuvidParseVideoData",
            )?;
        }
        debug!(
            thread = ?std::thread::current().id(),
            parser = format_args!("{:p}", self.parser),
            "NVDEC feed_packet exit cuvidParseVideoData"
        );
        Ok(())
    }

    /// Send EOS to the parser to flush remaining frames.
    fn send_eos(&mut self) -> Result<()> {
        let mut pkt = CUVIDSOURCEDATAPACKET {
            flags: CUVID_PKT_ENDOFSTREAM,
            payload_size: 0,
            payload: ptr::null(),
            timestamp: 0,
        };

        // SAFETY: EOS packet has no payload.
        unsafe {
            check_cu(
                cuvidParseVideoData(self.parser, &mut pkt),
                "cuvidParseVideoData (EOS)",
            )?;
        }
        self.eos_sent = true;
        Ok(())
    }

    /// Map a decoded surface, D2D copy to our buffer, unmap, record event.
    fn map_and_copy(&mut self, disp: &CUVIDPARSERDISPINFO) -> Result<DecodedFrame> {
        let decoder = self.cb_state.decoder;
        if decoder.is_null() {
            return Err(EngineError::Decode("Decoder not created yet".into()));
        }
        info!(
            thread = ?std::thread::current().id(),
            decoder = format_args!("{:p}", decoder),
            pic_idx = disp.picture_index,
            pts = disp.timestamp,
            queue_len = self.cb_state.pending_display.len(),
            "NVDEC map_and_copy start"
        );
        debug!(
            thread = ?std::thread::current().id(),
            decoder = format_args!("{:p}", decoder),
            pic_idx = disp.picture_index,
            pts = disp.timestamp,
            "NVDEC map_and_copy enter"
        );

        let format = self
            .cb_state
            .format
            .as_ref()
            .ok_or_else(|| EngineError::Decode("No format received".into()))?;

        let width = format.coded_width;
        let height = format.coded_height;

        // Map the decoded surface to a device pointer.
        let mut src_ptr: CUdeviceptr = 0;
        let mut src_pitch: c_uint = 0;
        let mut proc_params: CUVIDPROCPARAMS = unsafe { std::mem::zeroed() };
        proc_params.progressive_frame = disp.progressive_frame;
        proc_params.top_field_first = disp.top_field_first;
        proc_params.second_field = 0;

        // SAFETY: decoder is valid (created in sequence_callback).
        // src_ptr and src_pitch are outputs.
        unsafe {
            debug!(
                thread = ?std::thread::current().id(),
                decoder = format_args!("{:p}", decoder),
                pic_idx = disp.picture_index,
                "NVDEC map_and_copy enter cuvidMapVideoFrame64"
            );
            check_cu(
                cuvidMapVideoFrame64(
                    decoder,
                    disp.picture_index,
                    &mut src_ptr,
                    &mut src_pitch,
                    &mut proc_params,
                ),
                "cuvidMapVideoFrame64",
            )?;
            debug!(
                thread = ?std::thread::current().id(),
                src_ptr = format_args!("0x{:x}", src_ptr),
                src_pitch = src_pitch,
                "NVDEC map_and_copy exit cuvidMapVideoFrame64"
            );
        }
        info!(
            src_ptr = format_args!("0x{:x}", src_ptr),
            src_pitch = src_pitch,
            width,
            height,
            "NVDEC map_and_copy mapped surface"
        );

        let src_pitch_usize = src_pitch as usize;

        // Allocate our destination buffer with the same pitch for NV12.
        // NV12 total: pitch * height * 3 / 2
        let dst_size = PixelFormat::Nv12.byte_size(width, height, src_pitch_usize);
        let dst_buf = self.ctx.alloc(dst_size)?;
        let dst_ptr = *dst_buf.device_ptr() as CUdeviceptr;
        info!(
            dst_ptr = format_args!("0x{:x}", dst_ptr),
            dst_size,
            "NVDEC map_and_copy allocated destination buffer"
        );

        // ── D2D copy: Y plane ──
        let y_copy = CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: CUmemorytype::Device,
            srcHost: ptr::null(),
            srcDevice: src_ptr,
            srcArray: ptr::null(),
            srcPitch: src_pitch_usize,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CUmemorytype::Device,
            dstHost: ptr::null_mut(),
            dstDevice: dst_ptr,
            dstArray: ptr::null_mut(),
            dstPitch: src_pitch_usize,
            WidthInBytes: width as usize,
            Height: height as usize,
        };

        // ── D2D copy: UV plane ──
        let uv_src_offset = src_ptr + (src_pitch_usize * height as usize) as CUdeviceptr;
        let uv_dst_offset = dst_ptr + (src_pitch_usize * height as usize) as CUdeviceptr;

        let uv_copy = CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: CUmemorytype::Device,
            srcHost: ptr::null(),
            srcDevice: uv_src_offset,
            srcArray: ptr::null(),
            srcPitch: src_pitch_usize,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CUmemorytype::Device,
            dstHost: ptr::null_mut(),
            dstDevice: uv_dst_offset,
            dstArray: ptr::null_mut(),
            dstPitch: src_pitch_usize,
            WidthInBytes: width as usize,
            Height: (height / 2) as usize,
        };

        // Direct-path validation is currently unstable with raw CUstream use
        // here on Windows, so keep the NVDEC copy path synchronous for now.
        unsafe {
            check_cu(
                cuMemcpy2D_v2(&y_copy),
                "cuMemcpy2D_v2 (Y plane)",
            )?;
            check_cu(
                cuMemcpy2D_v2(&uv_copy),
                "cuMemcpy2D_v2 (UV plane)",
            )?;
        }
        info!("NVDEC map_and_copy completed synchronous D2D copies");

        unsafe {
            check_cu(
                cuvidUnmapVideoFrame64(decoder, src_ptr),
                "cuvidUnmapVideoFrame64",
            )?;
        }
        info!("NVDEC map_and_copy unmapped surface");

        // The copy is complete when control returns, but keep the envelope
        // contract uniform by recording an already-satisfied event.
        let mut event: CUevent = ptr::null_mut();
        unsafe {
            check_cu(
                cuEventCreate(&mut event, CU_EVENT_DISABLE_TIMING),
                "cuEventCreate (decode_done)",
            )?;
            check_cu(
                cuEventRecord(event, ptr::null_mut()),
                "cuEventRecord (decode_done)",
            )?;
        }
        info!(
            event = format_args!("{:p}", event),
            "NVDEC map_and_copy recorded decode event"
        );

        let decode_ready = StreamReadyEvent::from_raw(event);

        let texture = GpuTexture {
            data: Arc::new(dst_buf),
            width,
            height,
            pitch: src_pitch_usize,
            format: PixelFormat::Nv12,
        };

        let envelope = FrameEnvelope {
            texture,
            frame_index: self.frame_index,
            pts: disp.timestamp as i64,
            is_keyframe: false, // Parser doesn't directly expose this.
        };

        self.frame_index += 1;
        info!(
            frame_index = envelope.frame_index,
            pts = envelope.pts,
            pending_unmaps = self.pending_unmaps.len(),
            "NVDEC map_and_copy complete"
        );

        Ok(DecodedFrame {
            envelope,
            decode_event: decode_ready,
        })
    }
}

// SAFETY: NvDecoder owns a `CUvideoparser` (opaque raw pointer) that is
// created and destroyed exclusively from a single thread at a time.  The
// CUVID parser API is thread-safe for distinct parser handles.  No data races
// can occur because `NvDecoder` is consumed by `spawn_blocking` which
// executes on exactly one thread.
unsafe impl Send for NvDecoder {}

impl FrameDecoder for NvDecoder {
    /// Decode the next frame.
    ///
    /// Feeds bitstream packets to the parser until a decoded frame is
    /// available, then maps/copies it and returns the `FrameEnvelope`.
    ///
    /// Returns `None` at EOS.
    fn decode_next(&mut self) -> Result<Option<DecodedFrameEnvelope>> {
        self.ctx
            .device()
            .bind_to_thread()
            .map_err(|e| EngineError::Decode(format!("bind_to_thread (nvdec): {:?}", e)))?;
        self.init_parser_if_needed()?;
        debug!(
            thread = ?std::thread::current().id(),
            parser = format_args!("{:p}", self.parser),
            "NVDEC decode_next enter"
        );
        loop {
            self.release_completed_unmaps()?;
            self.enforce_unmap_budget()?;

            // Check if we have a pending decoded frame.
            if let Some(disp) = self.cb_state.pending_display.pop_front() {
                let decoded = self.map_and_copy(&disp)?;
                return Ok(Some(DecodedFrameEnvelope::new(
                    decoded.envelope,
                    Some(decoded.decode_event),
                )));
            }

            // No pending frame — feed more bitstream.
            if self.eos_sent {
                self.release_all_pending_unmaps()?;
                return Ok(None);
            }

            match self.source.read_packet()? {
                Some(packet) => {
                    info!(
                        thread = ?std::thread::current().id(),
                        bytes = packet.data.len(),
                        pts = packet.pts,
                        key = packet.is_keyframe,
                        "NVDEC decode_next got packet from source"
                    );
                    debug!(
                        thread = ?std::thread::current().id(),
                        bytes = packet.data.len(),
                        pts = packet.pts,
                        key = packet.is_keyframe,
                        "NVDEC decode_next read packet"
                    );
                    self.feed_packet(&packet)?;
                }
                None => {
                    self.send_eos()?;
                    // After EOS, pending_display may have flushed frames.
                    // Loop around to check.
                }
            }
        }
    }
}

impl Drop for NvDecoder {
    fn drop(&mut self) {
        let _ = self.release_all_pending_unmaps();

        // Destroy parser first (stops callbacks).
        if !self.parser.is_null() {
            // SAFETY: parser was created by cuvidCreateVideoParser.
            unsafe {
                cuvidDestroyVideoParser(self.parser);
            }
        }

        // Destroy decoder.
        if self.cb_state.decoder_created && !self.cb_state.decoder.is_null() {
            // SAFETY: decoder was created by cuvidCreateDecoder.
            unsafe {
                cuvidDestroyDecoder(self.cb_state.decoder);
            }
        }
        debug!("NVDEC decoder destroyed");
    }
}

// ─── Parser callbacks ────────────────────────────────────────────────────
//
// These are called by cuvidParseVideoData on the calling thread.
// They must not block and must not call Rust allocator-heavy operations.

/// Called when a sequence header is parsed — creates/recreates the decoder.
unsafe extern "system" fn sequence_callback(
    user_data: *mut c_void,
    format: *mut CUVIDEOFORMAT,
) -> c_int {
    unsafe {
        info!(
            user_data = format_args!("{:p}", user_data),
            format_ptr = format_args!("{:p}", format),
            "NVDEC sequence_callback enter"
        );
        let state = &mut *(user_data as *mut CallbackState);
        let fmt = &*format;

        state.format = Some(*fmt);

        // Determine required decode surfaces.
        let num_surfaces = (fmt.min_num_decode_surfaces as u32).max(8);
        state.max_decode_surfaces = num_surfaces;

        // Destroy existing decoder if resolution changed.
        if state.decoder_created && !state.decoder.is_null() {
            cuvidDestroyDecoder(state.decoder);
            state.decoder = ptr::null_mut();
            state.decoder_created = false;
        }

        // Create decoder.
        let mut create_info: CUVIDDECODECREATEINFO = std::mem::zeroed();
        create_info.ulWidth = fmt.coded_width as c_ulong;
        create_info.ulHeight = fmt.coded_height as c_ulong;
        create_info.ulNumDecodeSurfaces = num_surfaces as c_ulong;
        create_info.CodecType = state.codec;
        create_info.ChromaFormat = fmt.chroma_format;
        create_info.ulCreationFlags = cudaVideoCreateFlags::PreferCUVID as c_ulong;
        create_info.bitDepthMinus8 = fmt.bit_depth_luma_minus8 as c_ulong;
        create_info.ulIntraDecodeOnly = 0;
        create_info.ulMaxWidth = fmt.coded_width as c_ulong;
        create_info.ulMaxHeight = fmt.coded_height as c_ulong;
        create_info.display_area = CUVIDDECODECREATEINFO_display_area {
            left: 0,
            top: 0,
            right: fmt.coded_width as c_short,
            bottom: fmt.coded_height as c_short,
        };
        create_info.OutputFormat = cudaVideoSurfaceFormat::NV12;
        create_info.DeinterlaceMode = cudaVideoDeinterlaceMode::Adaptive;
        create_info.ulTargetWidth = fmt.coded_width as c_ulong;
        create_info.ulTargetHeight = fmt.coded_height as c_ulong;
        create_info.ulNumOutputSurfaces = 2;

        let result = cuvidCreateDecoder(&mut state.decoder, &mut create_info);
        if result != CUDA_SUCCESS {
            info!(result, "NVDEC sequence_callback decoder creation failed");
            // Return 0 to signal failure to the parser.
            return 0;
        }

        state.decoder_created = true;
        info!(
            decoder = format_args!("{:p}", state.decoder),
            width = fmt.coded_width,
            height = fmt.coded_height,
            num_surfaces,
            "NVDEC sequence_callback decoder created"
        );

        // Return the number of decode surfaces to indicate success.
        num_surfaces as c_int
    }
}

/// Called when a picture has been decoded — enqueue for GPU processing.
unsafe extern "system" fn decode_callback(
    user_data: *mut c_void,
    pic_params: *mut CUVIDPICPARAMS,
) -> c_int {
    unsafe {
        info!(
            user_data = format_args!("{:p}", user_data),
            pic_params = format_args!("{:p}", pic_params),
            "NVDEC decode_callback enter"
        );
        let state = &mut *(user_data as *mut CallbackState);

        if !state.decoder_created || state.decoder.is_null() {
            info!("NVDEC decode_callback skipped: decoder not ready");
            return 0;
        }

        let result = cuvidDecodePicture(state.decoder, pic_params);
        if result != CUDA_SUCCESS {
            info!(result, "NVDEC decode_callback cuvidDecodePicture failed");
            return 0;
        }

        info!("NVDEC decode_callback success");
        1 // Success.
    }
}

/// Called when a decoded picture is ready for display (reordered).
unsafe extern "system" fn display_callback(
    user_data: *mut c_void,
    disp_info: *mut CUVIDPARSERDISPINFO,
) -> c_int {
    unsafe {
        info!(
            user_data = format_args!("{:p}", user_data),
            disp_info = format_args!("{:p}", disp_info),
            "NVDEC display_callback enter"
        );
        let state = &mut *(user_data as *mut CallbackState);

        if disp_info.is_null() {
            // Null means EOS from parser.
            info!("NVDEC display_callback EOS");
            return 1;
        }

        state.pending_display.push_back(*disp_info);
        info!(
            queue_len = state.pending_display.len(),
            picture_index = (*disp_info).picture_index,
            timestamp = (*disp_info).timestamp,
            "NVDEC display_callback queued frame"
        );

        1 // Success.
    }
}

// ─── Raw stream handle extraction ────────────────────────────────────────

/// Extract raw CUstream handle from cudarc's CudaStream.
///
/// # Safety
///
pub fn get_raw_stream(stream: &cudarc::driver::CudaStream) -> CUstream {
    stream.stream.cast()
}

// ─── Stream wait helper (public for pipeline use) ────────────────────────

/// Make `target_stream` wait for `event` without blocking the CPU.
///
/// This is the cross-stream synchronization primitive.  Call this in the
/// preprocess stage before reading a decoded frame's texture data.
///
/// ```rust,ignore
/// // In preprocess stage:
/// wait_for_event(&ctx.preprocess_stream, decoded_frame.decode_event)?;
/// // Now safe to read decoded_frame.envelope.texture on preprocess_stream.
/// ```
pub fn wait_for_event(target_stream: &cudarc::driver::CudaStream, event: CUevent) -> Result<()> {
    let raw_stream = get_raw_stream(target_stream);
    // SAFETY: raw_stream and event are valid handles.
    // Flags = 0 is the only defined value.
    unsafe {
        check_cu(cuStreamWaitEvent(raw_stream, event, 0), "cuStreamWaitEvent")?;
    }
    Ok(())
}
