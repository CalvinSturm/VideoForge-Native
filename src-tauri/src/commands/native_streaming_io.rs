#[cfg(feature = "native_engine")]
use std::collections::VecDeque;
#[cfg(feature = "native_engine")]
use std::path::{Path, PathBuf};
#[cfg(feature = "native_engine")]
use std::ptr;
#[cfg(feature = "native_engine")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "native_engine")]
use crate::native_ffmpeg_sys::{
    av_bsf_alloc, av_bsf_free, av_bsf_get_by_name, av_bsf_init, av_bsf_receive_packet,
    av_bsf_send_packet, check_ffmpeg, to_cstring, videoforge_avformat_oformat,
    videoforge_avformat_pb, videoforge_avformat_stream, AVBSFContext,
};
#[cfg(feature = "native_engine")]
use ffmpeg_sys_next::*;

#[cfg(feature = "native_engine")]
const EAGAIN: i32 = 11;

#[cfg(feature = "native_engine")]
pub(crate) struct FfmpegBitstreamSource {
    fmt_ctx: *mut AVFormatContext,
    bsf_ctx: *mut AVBSFContext,
    video_stream_index: i32,
    pkt_read: *mut AVPacket,
    pkt_filtered: *mut AVPacket,
    pkt_pending: *mut AVPacket,
    time_base: AVRational,
    eof: bool,
    bsf_machine: BsfMachine<PacketSlot>,
}

#[cfg(feature = "native_engine")]
unsafe impl Send for FfmpegBitstreamSource {}

#[cfg(feature = "native_engine")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PacketSlot {
    Read,
    Pending,
}

#[cfg(feature = "native_engine")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SendInput<P> {
    Packet(P),
    Flush,
}

#[cfg(feature = "native_engine")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SendOutcome<P> {
    Accepted,
    Again(SendInput<P>),
}

#[cfg(feature = "native_engine")]
enum RecvOutcome {
    Packet(videoforge_engine::codecs::nvdec::BitstreamPacket),
    Again,
    Eof,
}

#[cfg(feature = "native_engine")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadOutcome<P> {
    Packet(P),
    Eof,
}

#[cfg(feature = "native_engine")]
trait BsfIo {
    type Packet: Copy + Eq;

    fn recv_filtered(&mut self) -> videoforge_engine::error::Result<RecvOutcome>;
    fn send_input(
        &mut self,
        input: SendInput<Self::Packet>,
    ) -> videoforge_engine::error::Result<SendOutcome<Self::Packet>>;
    fn read_next_video_packet(
        &mut self,
    ) -> videoforge_engine::error::Result<ReadOutcome<Self::Packet>>;
}

#[cfg(feature = "native_engine")]
#[derive(Debug)]
struct BsfMachine<P> {
    pending: Option<SendInput<P>>,
    flushing: bool,
    flush_sent: bool,
    terminal_eos: bool,
    idle_loops: usize,
}

#[cfg(feature = "native_engine")]
impl<P> Default for BsfMachine<P> {
    fn default() -> Self {
        Self {
            pending: None,
            flushing: false,
            flush_sent: false,
            terminal_eos: false,
            idle_loops: 0,
        }
    }
}

#[cfg(feature = "native_engine")]
impl<P: Copy + Eq> BsfMachine<P> {
    fn poll<IO>(
        &mut self,
        io: &mut IO,
    ) -> videoforge_engine::error::Result<Option<videoforge_engine::codecs::nvdec::BitstreamPacket>>
    where
        IO: BsfIo<Packet = P>,
    {
        const MAX_IDLE_LOOPS: usize = 1024;

        if self.terminal_eos {
            return Ok(None);
        }

        loop {
            let mut progressed = false;

            match io.recv_filtered()? {
                RecvOutcome::Packet(pkt) => return Ok(Some(pkt)),
                RecvOutcome::Again => {}
                RecvOutcome::Eof => {
                    if self.flush_sent {
                        self.terminal_eos = true;
                        return Ok(None);
                    }
                    if !self.flushing {
                        return Err(videoforge_engine::error::EngineError::Decode(
                            "Bitstream filter reached EOF before flush was initiated".into(),
                        ));
                    }
                }
            }

            if let Some(input) = self.pending.take() {
                match io.send_input(input)? {
                    SendOutcome::Accepted => progressed = true,
                    SendOutcome::Again(input) => self.pending = Some(input),
                }
            } else if self.flushing {
                if !self.flush_sent {
                    match io.send_input(SendInput::Flush)? {
                        SendOutcome::Accepted => {
                            self.flush_sent = true;
                            progressed = true;
                        }
                        SendOutcome::Again(input) => self.pending = Some(input),
                    }
                }
            } else {
                match io.read_next_video_packet()? {
                    ReadOutcome::Packet(pkt) => {
                        progressed = true;
                        match io.send_input(SendInput::Packet(pkt))? {
                            SendOutcome::Accepted => {}
                            SendOutcome::Again(input) => self.pending = Some(input),
                        }
                    }
                    ReadOutcome::Eof => {
                        self.flushing = true;
                        progressed = true;
                    }
                }
            }

            if progressed {
                self.idle_loops = 0;
            } else {
                self.idle_loops += 1;
                if self.idle_loops > MAX_IDLE_LOOPS {
                    return Err(videoforge_engine::error::EngineError::Decode(
                        "Bitstream filter state machine stalled (no forward progress)".into(),
                    ));
                }
            }
        }
    }
}

#[cfg(feature = "native_engine")]
#[derive(Clone, Default)]
struct SharedStderrTail(Arc<Mutex<VecDeque<String>>>);

#[cfg(feature = "native_engine")]
impl SharedStderrTail {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(VecDeque::with_capacity(16))))
    }

    fn spawn_reader<R>(&self, reader: R)
    where
        R: std::io::Read + Send + 'static,
    {
        let tail = self.clone();
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};

            let buf = BufReader::new(reader);
            for line in buf.lines().map_while(Result::ok) {
                tail.push(line);
            }
        });
    }

    fn push(&self, line: String) {
        let mut guard = self.0.lock().expect("stderr tail mutex poisoned");
        if guard.len() >= 12 {
            guard.pop_front();
        }
        guard.push_back(line);
    }

    fn snapshot(&self) -> String {
        self.0
            .lock()
            .expect("stderr tail mutex poisoned")
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(feature = "native_engine")]
fn native_mux_debug_enabled() -> bool {
    std::env::var_os("VIDEOFORGE_NATIVE_MUX_DEBUG").as_deref() == Some("1".as_ref())
}

#[cfg(feature = "native_engine")]
fn native_mux_debug_dir() -> PathBuf {
    if let Some(override_dir) = std::env::var_os("VIDEOFORGE_NVDEC_DEBUG_DUMP_DIR") {
        return PathBuf::from(override_dir);
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("artifacts")
        .join("nvdec_debug")
}

#[cfg(feature = "native_engine")]
#[derive(Clone, Default)]
struct NativeMuxDebugLog(Arc<Mutex<Option<std::fs::File>>>);

#[cfg(feature = "native_engine")]
impl NativeMuxDebugLog {
    fn maybe_create(output_path: &str) -> Self {
        if !native_mux_debug_enabled() {
            return Self::default();
        }

        let file = (|| -> std::io::Result<std::fs::File> {
            let dir = native_mux_debug_dir();
            std::fs::create_dir_all(&dir)?;
            let log_path = dir.join("native_mux_debug.log");
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)?;
            use std::io::Write;
            writeln!(file, "=== native mux session start ===")?;
            writeln!(file, "output_path={output_path}")?;
            Ok(file)
        })()
        .map_err(|e| {
            tracing::warn!(error = %e, "Failed to create native mux debug log");
            e
        })
        .ok();

        Self(Arc::new(Mutex::new(file)))
    }

    fn is_enabled(&self) -> bool {
        self.0.lock().expect("mux debug mutex poisoned").is_some()
    }

    fn write_line(&self, line: impl AsRef<str>) {
        if let Some(file) = self.0.lock().expect("mux debug mutex poisoned").as_mut() {
            use std::io::Write;
            if let Err(e) = writeln!(file, "{}", line.as_ref()) {
                tracing::warn!(error = %e, "Failed to write native mux debug line");
            }
        }
    }
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone, Default)]
pub(crate) struct StreamingCodecHint(Arc<Mutex<Option<&'static str>>>);

#[cfg(feature = "native_engine")]
impl StreamingCodecHint {
    pub(crate) fn new(initial: Option<&'static str>) -> Self {
        Self(Arc::new(Mutex::new(initial)))
    }

    pub(crate) fn set(&self, format: &'static str) {
        let mut guard = self.0.lock().expect("codec hint mutex poisoned");
        *guard = Some(format);
    }

    pub(crate) fn get(&self) -> Option<&'static str> {
        *self.0.lock().expect("codec hint mutex poisoned")
    }
}

#[cfg(feature = "native_engine")]
impl FfmpegBitstreamSource {
    pub(crate) fn spawn(
        _ffmpeg_cmd: &str,
        input_path: &str,
        codec: videoforge_engine::codecs::sys::cudaVideoCodec,
    ) -> std::io::Result<Self> {
        let path = Path::new(input_path);
        let path_str = path.to_str().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Non-UTF8 input path")
        })?;
        let c_path = to_cstring(path_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        let mut fmt_ctx: *mut AVFormatContext = ptr::null_mut();
        let ret = unsafe {
            avformat_open_input(&mut fmt_ctx, c_path.as_ptr(), ptr::null(), ptr::null_mut())
        };
        check_ffmpeg(ret, "avformat_open_input")
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        let ret = unsafe { avformat_find_stream_info(fmt_ctx, ptr::null_mut()) };
        if ret < 0 {
            unsafe { avformat_close_input(&mut fmt_ctx) };
            check_ffmpeg(ret, "avformat_find_stream_info")
                .map_err(|e| std::io::Error::other(e.to_string()))?;
        }

        let stream_index = unsafe {
            av_find_best_stream(
                fmt_ctx,
                AVMediaType::AVMEDIA_TYPE_VIDEO,
                -1,
                -1,
                ptr::null_mut(),
                0,
            )
        };
        if stream_index < 0 {
            unsafe { avformat_close_input(&mut fmt_ctx) };
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No video stream found in container",
            ));
        }

        let stream_ptr = unsafe { videoforge_avformat_stream(fmt_ctx, stream_index) };
        if stream_ptr.is_null() {
            unsafe { avformat_close_input(&mut fmt_ctx) };
            return Err(std::io::Error::other(format!(
                "Failed to resolve stream index {stream_index}"
            )));
        }
        let time_base = unsafe { (*stream_ptr).time_base };

        let bsf_name = match codec {
            videoforge_engine::codecs::sys::cudaVideoCodec::H264 => Some(c"h264_mp4toannexb"),
            videoforge_engine::codecs::sys::cudaVideoCodec::HEVC => Some(c"hevc_mp4toannexb"),
            _ => None,
        };

        let mut bsf_ctx: *mut AVBSFContext = ptr::null_mut();
        if let Some(bsf_name) = bsf_name {
            let bsf = unsafe { av_bsf_get_by_name(bsf_name.as_ptr()) };
            if bsf.is_null() {
                unsafe { avformat_close_input(&mut fmt_ctx) };
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "Required FFmpeg bitstream filter not available",
                ));
            }

            let ret = unsafe { av_bsf_alloc(bsf, &mut bsf_ctx) };
            if ret < 0 {
                unsafe { avformat_close_input(&mut fmt_ctx) };
                check_ffmpeg(ret, "av_bsf_alloc")
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
            }

            let ret = unsafe { avcodec_parameters_copy((*bsf_ctx).par_in, (*stream_ptr).codecpar) };
            if ret < 0 {
                unsafe {
                    av_bsf_free(&mut bsf_ctx);
                    avformat_close_input(&mut fmt_ctx);
                }
                check_ffmpeg(ret, "avcodec_parameters_copy")
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
            }

            let ret = unsafe { av_bsf_init(bsf_ctx) };
            if ret < 0 {
                unsafe {
                    av_bsf_free(&mut bsf_ctx);
                    avformat_close_input(&mut fmt_ctx);
                }
                check_ffmpeg(ret, "av_bsf_init")
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
            }
        }

        let mut pkt_read = unsafe { av_packet_alloc() };
        let mut pkt_filtered = unsafe { av_packet_alloc() };
        let mut pkt_pending = unsafe { av_packet_alloc() };
        if pkt_read.is_null() || pkt_filtered.is_null() || pkt_pending.is_null() {
            unsafe {
                if !pkt_pending.is_null() {
                    av_packet_free(&mut pkt_pending);
                }
                if !pkt_filtered.is_null() {
                    av_packet_free(&mut pkt_filtered);
                }
                if !pkt_read.is_null() {
                    av_packet_free(&mut pkt_read);
                }
                if !bsf_ctx.is_null() {
                    av_bsf_free(&mut bsf_ctx);
                }
                avformat_close_input(&mut fmt_ctx);
            }
            return Err(std::io::Error::other("Failed to allocate FFmpeg packets"));
        }

        Ok(Self {
            fmt_ctx,
            bsf_ctx,
            video_stream_index: stream_index,
            pkt_read,
            pkt_filtered,
            pkt_pending,
            time_base,
            eof: false,
            bsf_machine: BsfMachine::default(),
        })
    }

    fn rescale_pts(&self, pts: i64) -> i64 {
        if pts == AV_NOPTS_VALUE {
            return 0;
        }
        let us_tb = AVRational {
            num: 1,
            den: 1_000_000,
        };
        unsafe { av_rescale_q(pts, self.time_base, us_tb) }
    }

    fn copy_packet_data(pkt: &AVPacket) -> videoforge_engine::error::Result<Vec<u8>> {
        if pkt.size <= 0 {
            return Ok(Vec::new());
        }
        if pkt.data.is_null() {
            return Err(videoforge_engine::error::EngineError::Decode(
                "FFmpeg produced packet with null data pointer".into(),
            ));
        }
        Ok(unsafe { std::slice::from_raw_parts(pkt.data, pkt.size as usize) }.to_vec())
    }

    fn bsf_recv_filtered(&mut self) -> videoforge_engine::error::Result<RecvOutcome> {
        let ret = unsafe { av_bsf_receive_packet(self.bsf_ctx, self.pkt_filtered) };
        if ret == 0 {
            let pkt = unsafe { &*self.pkt_filtered };
            let data = Self::copy_packet_data(pkt)?;
            let pts = self.rescale_pts(pkt.pts);
            let is_keyframe = (pkt.flags & AV_PKT_FLAG_KEY) != 0;
            unsafe { av_packet_unref(self.pkt_filtered) };
            if data.is_empty() {
                return Ok(RecvOutcome::Again);
            }
            return Ok(RecvOutcome::Packet(
                videoforge_engine::codecs::nvdec::BitstreamPacket {
                    data,
                    pts,
                    is_keyframe,
                },
            ));
        }
        if ret == AVERROR(EAGAIN) {
            return Ok(RecvOutcome::Again);
        }
        if ret == AVERROR_EOF {
            return Ok(RecvOutcome::Eof);
        }
        check_ffmpeg(ret, "av_bsf_receive_packet")
            .map_err(|e| videoforge_engine::error::EngineError::Decode(e.to_string()))?;
        Err(videoforge_engine::error::EngineError::Decode(
            "unreachable: av_bsf_receive_packet error should have returned".into(),
        ))
    }

    fn bsf_send_input(
        &mut self,
        input: SendInput<PacketSlot>,
    ) -> videoforge_engine::error::Result<SendOutcome<PacketSlot>> {
        match input {
            SendInput::Packet(PacketSlot::Read) => {
                let ret = unsafe { av_bsf_send_packet(self.bsf_ctx, self.pkt_read) };
                if ret == 0 {
                    unsafe { av_packet_unref(self.pkt_read) };
                    return Ok(SendOutcome::Accepted);
                }
                if ret == AVERROR(EAGAIN) {
                    unsafe { av_packet_move_ref(self.pkt_pending, self.pkt_read) };
                    return Ok(SendOutcome::Again(SendInput::Packet(PacketSlot::Pending)));
                }
                unsafe { av_packet_unref(self.pkt_read) };
                check_ffmpeg(ret, "av_bsf_send_packet")
                    .map_err(|e| videoforge_engine::error::EngineError::Decode(e.to_string()))?;
                Err(videoforge_engine::error::EngineError::Decode(
                    "unreachable: av_bsf_send_packet error should have returned".into(),
                ))
            }
            SendInput::Packet(PacketSlot::Pending) => {
                let ret = unsafe { av_bsf_send_packet(self.bsf_ctx, self.pkt_pending) };
                if ret == 0 {
                    unsafe { av_packet_unref(self.pkt_pending) };
                    return Ok(SendOutcome::Accepted);
                }
                if ret == AVERROR(EAGAIN) {
                    return Ok(SendOutcome::Again(SendInput::Packet(PacketSlot::Pending)));
                }
                unsafe { av_packet_unref(self.pkt_pending) };
                check_ffmpeg(ret, "av_bsf_send_packet (pending)")
                    .map_err(|e| videoforge_engine::error::EngineError::Decode(e.to_string()))?;
                Err(videoforge_engine::error::EngineError::Decode(
                    "unreachable: av_bsf_send_packet pending error should have returned".into(),
                ))
            }
            SendInput::Flush => {
                let ret = unsafe { av_bsf_send_packet(self.bsf_ctx, ptr::null()) };
                if ret == 0 {
                    return Ok(SendOutcome::Accepted);
                }
                if ret == AVERROR(EAGAIN) {
                    return Ok(SendOutcome::Again(SendInput::Flush));
                }
                check_ffmpeg(ret, "av_bsf_send_packet (flush)")
                    .map_err(|e| videoforge_engine::error::EngineError::Decode(e.to_string()))?;
                Err(videoforge_engine::error::EngineError::Decode(
                    "unreachable: av_bsf_send_packet flush error should have returned".into(),
                ))
            }
        }
    }

    fn bsf_read_next_video_packet(
        &mut self,
    ) -> videoforge_engine::error::Result<ReadOutcome<PacketSlot>> {
        loop {
            let ret = unsafe { av_read_frame(self.fmt_ctx, self.pkt_read) };
            if ret < 0 {
                if ret == AVERROR_EOF {
                    return Ok(ReadOutcome::Eof);
                }
                check_ffmpeg(ret, "av_read_frame")
                    .map_err(|e| videoforge_engine::error::EngineError::Decode(e.to_string()))?;
                return Err(videoforge_engine::error::EngineError::Decode(
                    "unreachable: av_read_frame error should have returned".into(),
                ));
            }

            let pkt = unsafe { &*self.pkt_read };
            if pkt.stream_index != self.video_stream_index {
                unsafe { av_packet_unref(self.pkt_read) };
                continue;
            }
            return Ok(ReadOutcome::Packet(PacketSlot::Read));
        }
    }

    fn read_packet_passthrough(
        &mut self,
    ) -> videoforge_engine::error::Result<Option<videoforge_engine::codecs::nvdec::BitstreamPacket>>
    {
        loop {
            let ret = unsafe { av_read_frame(self.fmt_ctx, self.pkt_read) };
            if ret < 0 {
                if ret == AVERROR_EOF {
                    self.eof = true;
                    return Ok(None);
                }
                check_ffmpeg(ret, "av_read_frame")
                    .map_err(|e| videoforge_engine::error::EngineError::Decode(e.to_string()))?;
                return Err(videoforge_engine::error::EngineError::Decode(
                    "unreachable: av_read_frame error should have returned".into(),
                ));
            }

            let pkt = unsafe { &*self.pkt_read };
            if pkt.stream_index != self.video_stream_index {
                unsafe { av_packet_unref(self.pkt_read) };
                continue;
            }

            let data = Self::copy_packet_data(pkt)?;
            let pts = self.rescale_pts(pkt.pts);
            let is_keyframe = (pkt.flags & AV_PKT_FLAG_KEY) != 0;
            unsafe { av_packet_unref(self.pkt_read) };
            if data.is_empty() {
                continue;
            }
            return Ok(Some(videoforge_engine::codecs::nvdec::BitstreamPacket {
                data,
                pts,
                is_keyframe,
            }));
        }
    }
}

#[cfg(feature = "native_engine")]
impl videoforge_engine::codecs::nvdec::BitstreamSource for FfmpegBitstreamSource {
    fn read_packet(
        &mut self,
    ) -> videoforge_engine::error::Result<Option<videoforge_engine::codecs::nvdec::BitstreamPacket>>
    {
        if self.eof {
            return Ok(None);
        }

        if self.bsf_ctx.is_null() {
            return self.read_packet_passthrough();
        }

        let mut machine = std::mem::take(&mut self.bsf_machine);
        let out = {
            let mut io = FfmpegBsfIo { source: self };
            machine.poll(&mut io)
        };
        self.bsf_machine = machine;
        if matches!(out, Ok(None)) {
            self.eof = true;
        }
        out
    }
}

#[cfg(feature = "native_engine")]
struct FfmpegBsfIo<'a> {
    source: &'a mut FfmpegBitstreamSource,
}

#[cfg(feature = "native_engine")]
impl BsfIo for FfmpegBsfIo<'_> {
    type Packet = PacketSlot;

    fn recv_filtered(&mut self) -> videoforge_engine::error::Result<RecvOutcome> {
        self.source.bsf_recv_filtered()
    }

    fn send_input(
        &mut self,
        input: SendInput<Self::Packet>,
    ) -> videoforge_engine::error::Result<SendOutcome<Self::Packet>> {
        self.source.bsf_send_input(input)
    }

    fn read_next_video_packet(
        &mut self,
    ) -> videoforge_engine::error::Result<ReadOutcome<Self::Packet>> {
        self.source.bsf_read_next_video_packet()
    }
}

#[cfg(feature = "native_engine")]
impl Drop for FfmpegBitstreamSource {
    fn drop(&mut self) {
        unsafe {
            av_packet_free(&mut self.pkt_pending);
            av_packet_free(&mut self.pkt_filtered);
            av_packet_free(&mut self.pkt_read);
            if !self.bsf_ctx.is_null() {
                av_bsf_free(&mut self.bsf_ctx);
            }
            if !self.fmt_ctx.is_null() {
                avformat_close_input(&mut self.fmt_ctx);
            }
        }
    }
}

#[cfg(feature = "native_engine")]
pub(crate) struct StreamingMuxSink {
    ffmpeg_cmd: String,
    output_path: String,
    original_input: String,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    b_frames: u32,
    preserve_audio: bool,
    codec_hint: StreamingCodecHint,
    stderr_tail: SharedStderrTail,
    debug_log: NativeMuxDebugLog,
    packet_write_index: u64,
    muxer: Option<PacketAwareMuxer>,
    mux_path: Option<PathBuf>,
    bitstream_format: Option<&'static str>,
}

#[cfg(feature = "native_engine")]
struct PacketAwareMuxer {
    fmt_ctx: *mut AVFormatContext,
    stream: *mut AVStream,
    pkt: *mut AVPacket,
    time_base: AVRational,
    us_tb: AVRational,
    packet_counter: u64,
    header_written: bool,
}

#[cfg(feature = "native_engine")]
unsafe impl Send for PacketAwareMuxer {}

#[cfg(feature = "native_engine")]
impl PacketAwareMuxer {
    fn new(
        path: &Path,
        width: u32,
        height: u32,
        fps_num: u32,
        fps_den: u32,
        bitstream_format: &'static str,
    ) -> videoforge_engine::error::Result<Self> {
        let codec_id = match bitstream_format {
            "h264" => AVCodecID::AV_CODEC_ID_H264,
            _ => AVCodecID::AV_CODEC_ID_HEVC,
        };
        let path_str = path.to_str().ok_or_else(|| {
            videoforge_engine::error::EngineError::Encode(format!(
                "Non-UTF8 mux path: {}",
                path.display()
            ))
        })?;
        let c_path = to_cstring(path_str).map_err(videoforge_engine::error::EngineError::Encode)?;

        let mut fmt_ctx: *mut AVFormatContext = ptr::null_mut();
        let ret = unsafe {
            avformat_alloc_output_context2(&mut fmt_ctx, ptr::null(), ptr::null(), c_path.as_ptr())
        };
        if ret < 0 || fmt_ctx.is_null() {
            return Err(videoforge_engine::error::EngineError::Encode(format!(
                "Failed to create output context for {}",
                path.display()
            )));
        }

        let stream = unsafe { avformat_new_stream(fmt_ctx, ptr::null()) };
        if stream.is_null() {
            unsafe { avformat_free_context(fmt_ctx) };
            return Err(videoforge_engine::error::EngineError::Encode(
                "Failed to create output stream".into(),
            ));
        }

        unsafe {
            let par = (*stream).codecpar;
            (*par).codec_type = AVMediaType::AVMEDIA_TYPE_VIDEO;
            (*par).codec_id = codec_id;
            (*par).width = width as i32;
            (*par).height = height as i32;
            (*stream).time_base = AVRational {
                num: fps_den.max(1) as i32,
                den: fps_num.max(1) as i32,
            };
        }

        let oformat = unsafe { videoforge_avformat_oformat(fmt_ctx) };
        if oformat.is_null() {
            unsafe { avformat_free_context(fmt_ctx) };
            return Err(videoforge_engine::error::EngineError::Encode(
                "Output format context missing AVOutputFormat".into(),
            ));
        }
        let needs_file = unsafe { (*oformat).flags & AVFMT_NOFILE == 0 };
        if needs_file {
            let pb = unsafe { videoforge_avformat_pb(fmt_ctx) };
            if pb.is_null() {
                unsafe { avformat_free_context(fmt_ctx) };
                return Err(videoforge_engine::error::EngineError::Encode(
                    "Output format context missing AVIOContext pointer".into(),
                ));
            }
            let ret = unsafe { avio_open(pb, c_path.as_ptr(), AVIO_FLAG_WRITE) };
            if ret < 0 {
                unsafe { avformat_free_context(fmt_ctx) };
                check_ffmpeg(ret, "avio_open")
                    .map_err(|e| videoforge_engine::error::EngineError::Encode(e.to_string()))?;
            }
        }

        let pkt = unsafe { av_packet_alloc() };
        if pkt.is_null() {
            unsafe {
                if needs_file {
                    if let Some(pb) = videoforge_avformat_pb(fmt_ctx).as_mut() {
                        avio_closep(pb);
                    }
                }
                avformat_free_context(fmt_ctx);
            }
            return Err(videoforge_engine::error::EngineError::Encode(
                "Failed to allocate AVPacket".into(),
            ));
        }

        Ok(Self {
            fmt_ctx,
            stream,
            pkt,
            time_base: AVRational { num: 0, den: 1 },
            us_tb: AVRational {
                num: 1,
                den: 1_000_000,
            },
            packet_counter: 0,
            header_written: false,
        })
    }

    fn write_header_if_needed(&mut self) -> videoforge_engine::error::Result<()> {
        if self.header_written {
            return Ok(());
        }

        let ret = unsafe { avformat_write_header(self.fmt_ctx, ptr::null_mut()) };
        check_ffmpeg(ret, "avformat_write_header")
            .map_err(|e| videoforge_engine::error::EngineError::Encode(e.to_string()))?;
        self.time_base = unsafe { (*self.stream).time_base };
        self.header_written = true;
        Ok(())
    }

    fn write_packet(
        &mut self,
        data: &[u8],
        pts: i64,
        dts: i64,
        is_keyframe: bool,
    ) -> videoforge_engine::error::Result<()> {
        self.write_header_if_needed()?;

        unsafe {
            let ret = av_new_packet(self.pkt, data.len() as i32);
            if ret < 0 {
                check_ffmpeg(ret, "av_new_packet")
                    .map_err(|e| videoforge_engine::error::EngineError::Encode(e.to_string()))?;
            }
            ptr::copy_nonoverlapping(data.as_ptr(), (*self.pkt).data, data.len());
            (*self.pkt).pts = av_rescale_q(pts, self.us_tb, self.time_base);
            (*self.pkt).dts = av_rescale_q(dts, self.us_tb, self.time_base);
            (*self.pkt).stream_index = 0;
            (*self.pkt).duration = 1;
            (*self.pkt).flags = 0;
            if is_keyframe {
                (*self.pkt).flags |= AV_PKT_FLAG_KEY;
            }

            let ret = av_interleaved_write_frame(self.fmt_ctx, self.pkt);
            if ret < 0 {
                av_packet_unref(self.pkt);
                check_ffmpeg(ret, "av_interleaved_write_frame")
                    .map_err(|e| videoforge_engine::error::EngineError::Encode(e.to_string()))?;
            }
        }

        self.packet_counter += 1;
        Ok(())
    }

    fn flush(&mut self) -> videoforge_engine::error::Result<()> {
        if !self.header_written {
            return Ok(());
        }

        let ret = unsafe { av_write_trailer(self.fmt_ctx) };
        check_ffmpeg(ret, "av_write_trailer")
            .map_err(|e| videoforge_engine::error::EngineError::Encode(e.to_string()))?;
        Ok(())
    }
}

#[cfg(feature = "native_engine")]
impl Drop for PacketAwareMuxer {
    fn drop(&mut self) {
        unsafe {
            av_packet_free(&mut self.pkt);

            let oformat = videoforge_avformat_oformat(self.fmt_ctx);
            let pb = videoforge_avformat_pb(self.fmt_ctx);
            if !oformat.is_null() && !pb.is_null() && (*oformat).flags & AVFMT_NOFILE == 0 {
                avio_closep(pb);
            }

            avformat_free_context(self.fmt_ctx);
            self.fmt_ctx = ptr::null_mut();
        }
    }
}

#[cfg(feature = "native_engine")]
impl StreamingMuxSink {
    pub(crate) fn new(
        ffmpeg_cmd: &str,
        output_path: &str,
        original_input: &str,
        width: u32,
        height: u32,
        fps_num: u32,
        fps_den: u32,
        b_frames: u32,
        preserve_audio: bool,
        codec_hint: StreamingCodecHint,
    ) -> std::io::Result<Self> {
        Ok(Self {
            ffmpeg_cmd: ffmpeg_cmd.to_string(),
            output_path: output_path.to_string(),
            original_input: original_input.to_string(),
            width,
            height,
            fps_num,
            fps_den,
            b_frames,
            preserve_audio,
            codec_hint,
            stderr_tail: SharedStderrTail::new(),
            debug_log: NativeMuxDebugLog::maybe_create(output_path),
            packet_write_index: 0,
            muxer: None,
            mux_path: None,
            bitstream_format: None,
        })
    }

    fn ensure_started(
        &mut self,
        bitstream_format: &'static str,
    ) -> videoforge_engine::error::Result<&mut PacketAwareMuxer> {
        if self.muxer.is_none() {
            let mux_path = if self.preserve_audio {
                unique_native_mux_container_path(&self.output_path)
            } else {
                PathBuf::from(&self.output_path)
            };
            let muxer = PacketAwareMuxer::new(
                &mux_path,
                self.width,
                self.height,
                self.fps_num,
                self.fps_den,
                bitstream_format,
            )?;

            if self.debug_log.is_enabled() {
                self.debug_log.write_line(format!(
                    "mux_path={} bitstream_format={} fps={}/{} preserve_audio={} output_path={}",
                    mux_path.display(),
                    bitstream_format,
                    self.fps_num,
                    self.fps_den,
                    self.preserve_audio,
                    self.output_path
                ));
                self.debug_log
                    .write_line(format!("mux_codec_hint={:?}", self.codec_hint.get()));
                self.debug_log.write_line("mux_mode=packet_aware");
            }

            self.bitstream_format = Some(bitstream_format);
            self.mux_path = Some(mux_path);
            self.muxer = Some(muxer);
        }

        self.muxer.as_mut().ok_or_else(|| {
            videoforge_engine::error::EngineError::Encode("Packet muxer unavailable".into())
        })
    }
}

#[cfg(feature = "native_engine")]
fn unique_native_mux_container_path(output_path: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let ext = Path::new(output_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .filter(|ext| !ext.is_empty())
        .unwrap_or("mp4");
    std::env::temp_dir().join(format!(
        "videoforge_native_mux_{}_{}.{}",
        std::process::id(),
        nanos,
        ext
    ))
}

#[cfg(feature = "native_engine")]
fn infer_annex_b_bitstream_format(data: &[u8]) -> Option<&'static str> {
    let mut i = 0usize;
    while i + 4 < data.len() {
        let (nal_start, header_index) = if data[i..].starts_with(&[0, 0, 1]) {
            (i, i + 3)
        } else if data[i..].starts_with(&[0, 0, 0, 1]) {
            (i, i + 4)
        } else {
            i += 1;
            continue;
        };

        if header_index >= data.len() {
            break;
        }

        let header = data[header_index];
        let h264_type = header & 0x1F;
        if matches!(h264_type, 1 | 5 | 7 | 8) {
            return Some("h264");
        }

        if header_index + 1 < data.len() {
            let hevc_type = (header >> 1) & 0x3F;
            if matches!(hevc_type, 1 | 19 | 20 | 32 | 33 | 34) {
                return Some("hevc");
            }
        }

        i = nal_start + 3;
    }

    None
}

#[cfg(feature = "native_engine")]
fn pick_streaming_mux_format(data: &[u8]) -> &'static str {
    match infer_annex_b_bitstream_format(data) {
        Some(format) => format,
        None => {
            tracing::warn!(
                bytes = data.len(),
                "Could not infer Annex B bitstream format from packet; defaulting streaming mux to HEVC"
            );
            "hevc"
        }
    }
}

#[cfg(feature = "native_engine")]
impl videoforge_engine::codecs::nvenc::BitstreamSink for StreamingMuxSink {
    fn write_packet(
        &mut self,
        data: &[u8],
        pts: i64,
        dts: i64,
        is_keyframe: bool,
    ) -> videoforge_engine::error::Result<()> {
        let bitstream_format = self
            .codec_hint
            .get()
            .unwrap_or_else(|| pick_streaming_mux_format(data));

        if self.debug_log.is_enabled() {
            let mux_pts = if self.b_frames == 0 { dts } else { pts };
            self.debug_log.write_line(format!(
                "packet_write_index={} pts={} dts={} mux_pts={} is_keyframe={} bytes={} bitstream_format={} codec_hint={:?}",
                self.packet_write_index,
                pts,
                dts,
                mux_pts,
                is_keyframe,
                data.len(),
                bitstream_format,
                self.codec_hint.get()
            ));
        }

        let mux_pts = if self.b_frames == 0 { dts } else { pts };
        let muxer = self.ensure_started(bitstream_format)?;
        muxer.write_packet(data, mux_pts, dts, is_keyframe)?;
        self.packet_write_index += 1;
        Ok(())
    }

    fn flush(&mut self) -> videoforge_engine::error::Result<()> {
        if self.muxer.is_none() {
            return Err(videoforge_engine::error::EngineError::Encode(
                "Streaming mux never received any packets".into(),
            ));
        }
        if self.debug_log.is_enabled() {
            self.debug_log
                .write_line(format!("flush packet_count={}", self.packet_write_index));
        }
        let mut muxer = self.muxer.take().ok_or_else(|| {
            videoforge_engine::error::EngineError::Encode("Packet muxer missing".into())
        })?;
        muxer.flush()?;
        drop(muxer);

        if self.preserve_audio {
            use std::process::{Command, Stdio};

            let mux_path = self.mux_path.clone().ok_or_else(|| {
                videoforge_engine::error::EngineError::Encode(
                    "Packet mux output path missing".into(),
                )
            })?;
            let bitstream_format = self.bitstream_format.ok_or_else(|| {
                videoforge_engine::error::EngineError::Encode(
                    "Packet mux bitstream format missing".into(),
                )
            })?;
            let mut cmd = Command::new(&self.ffmpeg_cmd);
            cmd.arg("-y")
                .arg("-hide_banner")
                .arg("-loglevel")
                .arg("warning")
                .arg("-i")
                .arg(&mux_path)
                .arg("-i")
                .arg(&self.original_input)
                .arg("-c:v")
                .arg("copy")
                .arg("-c:a")
                .arg("copy")
                .arg("-map")
                .arg("0:v:0")
                .arg("-map")
                .arg("1:a?")
                .arg("-movflags")
                .arg("+faststart")
                .arg(&self.output_path)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());

            if self.debug_log.is_enabled() {
                self.debug_log.write_line(format!(
                    "spawn ffmpeg_cmd={} merge_bitstream_format={} staged_video_input={} output_path={}",
                    self.ffmpeg_cmd,
                    bitstream_format,
                    mux_path.display(),
                    self.output_path
                ));
            }

            let mut child = cmd.spawn().map_err(|e| {
                videoforge_engine::error::EngineError::Encode(format!(
                    "FFmpeg audio merge spawn failed: {e}"
                ))
            })?;
            if let Some(stderr) = child.stderr.take() {
                self.stderr_tail.spawn_reader(stderr);
            }
            let status = child.wait().map_err(|e| {
                videoforge_engine::error::EngineError::Encode(format!(
                    "FFmpeg audio merge wait failed: {e}"
                ))
            })?;
            if self.debug_log.is_enabled() {
                self.debug_log
                    .write_line(format!("ffmpeg_exit_status={}", status));
                let stderr_snapshot = self.stderr_tail.snapshot();
                if !stderr_snapshot.is_empty() {
                    self.debug_log.write_line("ffmpeg_stderr_begin");
                    for line in stderr_snapshot.lines() {
                        self.debug_log.write_line(line);
                    }
                    self.debug_log.write_line("ffmpeg_stderr_end");
                }
            }
            let _ = std::fs::remove_file(&mux_path);
            if !status.success() {
                return Err(videoforge_engine::error::EngineError::Encode(format!(
                    "FFmpeg audio merge failed with {}: {}",
                    status,
                    self.stderr_tail.snapshot()
                )));
            }
        }

        if self.debug_log.is_enabled() {
            self.debug_log.write_line("=== native mux session end ===");
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "native_engine"))]
mod tests {
    use super::{
        native_mux_debug_enabled, pick_streaming_mux_format, SharedStderrTail, StreamingCodecHint,
    };

    #[test]
    fn codec_hint_roundtrips_latest_value() {
        let hint = StreamingCodecHint::new(None);
        assert_eq!(hint.get(), None);

        hint.set("h264");
        assert_eq!(hint.get(), Some("h264"));

        hint.set("hevc");
        assert_eq!(hint.get(), Some("hevc"));
    }

    #[test]
    fn picks_h264_and_hevc_from_annex_b_packets() {
        let h264_packet = [0x00, 0x00, 0x00, 0x01, 0x65, 0x88];
        let hevc_packet = [0x00, 0x00, 0x00, 0x01, 0x40, 0x01];

        assert_eq!(pick_streaming_mux_format(&h264_packet), "h264");
        assert_eq!(pick_streaming_mux_format(&hevc_packet), "hevc");
    }

    #[test]
    fn stderr_tail_keeps_latest_lines_bounded() {
        let tail = SharedStderrTail::new();
        for i in 0..20 {
            tail.push(format!("line-{i}"));
        }

        let snapshot = tail.snapshot();
        assert!(!snapshot.contains("line-0"));
        assert!(snapshot.contains("line-8"));
        assert!(snapshot.contains("line-19"));
        assert_eq!(snapshot.lines().count(), 12);
    }

    #[test]
    fn native_mux_debug_env_matches_expected_flags() {
        let original_mux = std::env::var_os("VIDEOFORGE_NATIVE_MUX_DEBUG");

        std::env::remove_var("VIDEOFORGE_NATIVE_MUX_DEBUG");
        assert!(!native_mux_debug_enabled());

        std::env::set_var("VIDEOFORGE_NATIVE_MUX_DEBUG", "1");
        assert!(native_mux_debug_enabled());

        std::env::remove_var("VIDEOFORGE_NATIVE_MUX_DEBUG");
        assert!(!native_mux_debug_enabled());

        match original_mux {
            Some(value) => std::env::set_var("VIDEOFORGE_NATIVE_MUX_DEBUG", value),
            None => std::env::remove_var("VIDEOFORGE_NATIVE_MUX_DEBUG"),
        }
    }
}
