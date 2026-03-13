#[cfg(feature = "native_engine")]
use std::ffi::CString;
#[cfg(feature = "native_engine")]
use std::fmt::{Display, Formatter};

#[cfg(feature = "native_engine")]
unsafe extern "C" {
    pub fn videoforge_avformat_stream(
        ctx: *mut ffmpeg_sys_next::AVFormatContext,
        index: std::ffi::c_int,
    ) -> *mut ffmpeg_sys_next::AVStream;
    pub fn videoforge_avformat_oformat(
        ctx: *const ffmpeg_sys_next::AVFormatContext,
    ) -> *const ffmpeg_sys_next::AVOutputFormat;
    pub fn videoforge_avformat_pb(
        ctx: *mut ffmpeg_sys_next::AVFormatContext,
    ) -> *mut *mut ffmpeg_sys_next::AVIOContext;
}

#[cfg(feature = "native_engine")]
#[repr(C)]
pub struct AVBitStreamFilter {
    _opaque: [u8; 0],
}

#[cfg(feature = "native_engine")]
#[repr(C)]
pub struct AVBSFContext {
    pub av_class: *const std::ffi::c_void,
    pub filter: *const AVBitStreamFilter,
    pub priv_data: *mut std::ffi::c_void,
    pub par_in: *mut ffmpeg_sys_next::AVCodecParameters,
    pub par_out: *mut ffmpeg_sys_next::AVCodecParameters,
    pub time_base_in: ffmpeg_sys_next::AVRational,
    pub time_base_out: ffmpeg_sys_next::AVRational,
}

#[cfg(feature = "native_engine")]
unsafe extern "C" {
    pub fn av_bsf_get_by_name(name: *const std::ffi::c_char) -> *const AVBitStreamFilter;
    pub fn av_bsf_alloc(
        filter: *const AVBitStreamFilter,
        ctx: *mut *mut AVBSFContext,
    ) -> std::ffi::c_int;
    pub fn av_bsf_init(ctx: *mut AVBSFContext) -> std::ffi::c_int;
    pub fn av_bsf_send_packet(
        ctx: *mut AVBSFContext,
        pkt: *const ffmpeg_sys_next::AVPacket,
    ) -> std::ffi::c_int;
    pub fn av_bsf_receive_packet(
        ctx: *mut AVBSFContext,
        pkt: *mut ffmpeg_sys_next::AVPacket,
    ) -> std::ffi::c_int;
    pub fn av_bsf_free(ctx: *mut *mut AVBSFContext);
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
pub struct FfmpegErrorDetail {
    pub context: String,
    pub code: i32,
    pub message: String,
}

#[cfg(feature = "native_engine")]
impl Display for FfmpegErrorDetail {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} (code {})", self.context, self.message, self.code)
    }
}

#[cfg(feature = "native_engine")]
pub fn check_ffmpeg(ret: i32, context: &str) -> std::result::Result<(), FfmpegErrorDetail> {
    if ret >= 0 {
        return Ok(());
    }

    let mut buf = [0 as std::ffi::c_char; 256];
    unsafe {
        ffmpeg_sys_next::av_strerror(ret, buf.as_mut_ptr(), buf.len());
    }
    let msg = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
        .to_str()
        .unwrap_or("unknown error")
        .to_string();

    Err(FfmpegErrorDetail {
        context: context.to_string(),
        code: ret,
        message: msg,
    })
}

#[cfg(feature = "native_engine")]
pub fn to_cstring(s: &str) -> std::result::Result<CString, String> {
    CString::new(s).map_err(|e| format!("Invalid path string: {e}"))
}
