#[cfg(feature = "native_engine")]
use std::collections::VecDeque;
#[cfg(feature = "native_engine")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "native_engine")]
pub(crate) struct FfmpegBitstreamSource {
    child: std::process::Child,
    stdout: std::process::ChildStdout,
    stderr_tail: SharedStderrTail,
    eof: bool,
    pts_counter: i64,
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
        ffmpeg_cmd: &str,
        input_path: &str,
        codec: videoforge_engine::codecs::sys::cudaVideoCodec,
    ) -> std::io::Result<Self> {
        use std::process::{Command, Stdio};

        let bitstream_filter = match codec {
            videoforge_engine::codecs::sys::cudaVideoCodec::H264 => "h264_mp4toannexb",
            videoforge_engine::codecs::sys::cudaVideoCodec::HEVC => "hevc_mp4toannexb",
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Unsupported streamed demux codec: {codec:?}"),
                ));
            }
        };
        let output_format = match codec {
            videoforge_engine::codecs::sys::cudaVideoCodec::H264 => "h264",
            videoforge_engine::codecs::sys::cudaVideoCodec::HEVC => "hevc",
            _ => unreachable!(),
        };

        let mut child = Command::new(ffmpeg_cmd)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-vcodec",
                "copy",
                "-an",
                "-bsf:v",
                bitstream_filter,
                "-f",
                output_format,
                "-",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdout = child.stdout.take().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "FFmpeg demux stdout unavailable",
            )
        })?;
        let stderr_tail = SharedStderrTail::new();
        if let Some(stderr) = child.stderr.take() {
            stderr_tail.spawn_reader(stderr);
        }

        Ok(Self {
            child,
            stdout,
            stderr_tail,
            eof: false,
            pts_counter: 0,
        })
    }
}

#[cfg(feature = "native_engine")]
impl videoforge_engine::codecs::nvdec::BitstreamSource for FfmpegBitstreamSource {
    fn read_packet(
        &mut self,
    ) -> videoforge_engine::error::Result<Option<videoforge_engine::codecs::nvdec::BitstreamPacket>>
    {
        use std::io::Read;

        if self.eof {
            return Ok(None);
        }

        const CHUNK: usize = 1024 * 1024;
        let mut chunk = vec![0_u8; CHUNK];
        let bytes_read = self.stdout.read(&mut chunk).map_err(|e| {
            videoforge_engine::error::EngineError::Decode(format!("read FFmpeg demux stream: {e}"))
        })?;

        if bytes_read == 0 {
            self.eof = true;
            let status = self.child.wait().map_err(|e| {
                videoforge_engine::error::EngineError::Decode(format!(
                    "wait for FFmpeg demux process: {e}"
                ))
            })?;
            if !status.success() {
                return Err(videoforge_engine::error::EngineError::Decode(format!(
                    "FFmpeg demux failed with {status}: {}",
                    self.stderr_tail.snapshot()
                )));
            }
            return Ok(None);
        }

        chunk.truncate(bytes_read);
        let is_keyframe = chunk.windows(5).any(|w| {
            (w[0] == 0 && w[1] == 0 && w[2] == 0 && w[3] == 1 && (w[4] & 0x1F) == 5)
                || (w[0] == 0 && w[1] == 0 && w[2] == 1 && (w[3] & 0x1F) == 5)
        });

        let pts = self.pts_counter;
        self.pts_counter += 1;

        Ok(Some(videoforge_engine::codecs::nvdec::BitstreamPacket {
            data: chunk,
            pts,
            is_keyframe,
        }))
    }
}

#[cfg(feature = "native_engine")]
pub(crate) struct StreamingMuxSink {
    ffmpeg_cmd: String,
    output_path: String,
    original_input: String,
    preserve_audio: bool,
    codec_hint: StreamingCodecHint,
    stderr_tail: SharedStderrTail,
    child: Option<std::process::Child>,
    stdin: Option<std::process::ChildStdin>,
}

#[cfg(feature = "native_engine")]
impl StreamingMuxSink {
    pub(crate) fn new(
        ffmpeg_cmd: &str,
        output_path: &str,
        original_input: &str,
        preserve_audio: bool,
        codec_hint: StreamingCodecHint,
    ) -> std::io::Result<Self> {
        Ok(Self {
            ffmpeg_cmd: ffmpeg_cmd.to_string(),
            output_path: output_path.to_string(),
            original_input: original_input.to_string(),
            preserve_audio,
            codec_hint,
            stderr_tail: SharedStderrTail::new(),
            child: None,
            stdin: None,
        })
    }

    fn ensure_started(
        &mut self,
        bitstream_format: &'static str,
    ) -> videoforge_engine::error::Result<&mut std::process::ChildStdin> {
        use std::process::{Command, Stdio};

        if self.stdin.is_none() {
            let mut cmd = Command::new(&self.ffmpeg_cmd);
            cmd.arg("-y")
                .arg("-hide_banner")
                .arg("-loglevel")
                .arg("warning")
                .arg("-f")
                .arg(bitstream_format)
                .arg("-i")
                .arg("-");

            if self.preserve_audio {
                cmd.arg("-i").arg(&self.original_input);
            }

            cmd.arg("-c:v").arg("copy");

            if self.preserve_audio {
                cmd.arg("-c:a")
                    .arg("copy")
                    .arg("-map")
                    .arg("0:v:0")
                    .arg("-map")
                    .arg("1:a?");
            } else {
                cmd.arg("-an");
            }

            cmd.arg("-movflags")
                .arg("+faststart")
                .arg(&self.output_path)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());

            let mut child = cmd.spawn().map_err(|e| {
                videoforge_engine::error::EngineError::Encode(format!(
                    "FFmpeg mux spawn failed for {bitstream_format}: {e}"
                ))
            })?;
            if let Some(stderr) = child.stderr.take() {
                self.stderr_tail.spawn_reader(stderr);
            }
            let stdin = child.stdin.take().ok_or_else(|| {
                videoforge_engine::error::EngineError::Encode("Mux stdin unavailable".into())
            })?;
            self.stdin = Some(stdin);
            self.child = Some(child);
        }

        self.stdin
            .as_mut()
            .ok_or_else(|| videoforge_engine::error::EngineError::Encode("Mux stdin closed".into()))
    }
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
        _pts: i64,
        _is_keyframe: bool,
    ) -> videoforge_engine::error::Result<()> {
        use std::io::Write;

        let bitstream_format = self
            .codec_hint
            .get()
            .unwrap_or_else(|| pick_streaming_mux_format(data));
        let stdin = self.ensure_started(bitstream_format)?;
        stdin
            .write_all(data)
            .map_err(|e| videoforge_engine::error::EngineError::Encode(e.to_string()))
    }

    fn flush(&mut self) -> videoforge_engine::error::Result<()> {
        if self.child.is_none() {
            return Err(videoforge_engine::error::EngineError::Encode(
                "Streaming mux never received any packets".into(),
            ));
        }
        drop(self.stdin.take());
        let child = self.child.take().ok_or_else(|| {
            videoforge_engine::error::EngineError::Encode("Mux process missing".into())
        })?;
        let output = child.wait_with_output().map_err(|e| {
            videoforge_engine::error::EngineError::Encode(format!("FFmpeg mux wait failed: {e}"))
        })?;
        if !output.status.success() {
            return Err(videoforge_engine::error::EngineError::Encode(format!(
                "FFmpeg mux failed with {}: {}",
                output.status,
                self.stderr_tail.snapshot()
            )));
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "native_engine"))]
mod tests {
    use super::{pick_streaming_mux_format, SharedStderrTail, StreamingCodecHint};

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
}
