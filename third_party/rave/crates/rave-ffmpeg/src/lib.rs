#![doc = include_str!("../README.md")]

pub mod ffmpeg_demuxer;
pub mod ffmpeg_muxer;
pub mod ffmpeg_sys;
pub mod file_sink;
pub mod file_source;
pub mod probe;

pub use probe::{ContainerMetadata, probe_container};

#[cfg(test)]
mod tests {
    #[test]
    fn demux_and_mux_stay_at_packet_boundary() {
        let demux = include_str!("ffmpeg_demuxer.rs");
        let mux = include_str!("ffmpeg_muxer.rs");

        for forbidden in [
            "avcodec_send_packet",
            "avcodec_receive_frame",
            "AVFrame",
            "sws_scale",
            "rawvideo",
        ] {
            assert!(
                !demux.contains(forbidden),
                "demux path should not include raw frame API `{forbidden}`"
            );
            assert!(
                !mux.contains(forbidden),
                "mux path should not include raw frame API `{forbidden}`"
            );
        }
    }
}
