# Post-Consolidation Validation (2026-02-18)

Exact working directory:
```text
/home/calvin/src/rave
```

## git rev-parse --show-toplevel
```text
/home/calvin/src/rave
[exit_code=0]
```

## pwd
```text
/home/calvin/src/rave
[exit_code=0]
```

## ls -la
```text
total 136
drwxrwxrwx 12 calvin calvin  4096 Feb 18 15:45 .
drwxr-xr-x  3 calvin calvin  4096 Feb 17 04:50 ..
drwxrwxrwx  8 calvin calvin  4096 Feb 18 15:41 .git
-rw-r--r--  1 calvin calvin   536 Feb 17 17:25 .gitignore
-rw-r--r--  1 calvin calvin 11714 Feb 17 17:25 ARCHITECTURE.md
-rw-r--r--  1 calvin calvin 38726 Feb 18 15:45 Cargo.lock
-rw-r--r--  1 calvin calvin   200 Feb 18 15:26 Cargo.toml
-rw-r--r--  1 calvin calvin  1069 Feb 17 17:25 LICENSE
-rw-r--r--  1 calvin calvin  5149 Feb 18 15:04 README.md
drwxr-xr-x  8 calvin calvin  4096 Feb 18 15:04 crates
drwxrwxrwx  3 calvin calvin  4096 Feb 18 15:04 docs
drwxr-xr-x  2 calvin calvin  4096 Feb 18 15:04 examples
-rw-r--r--  1 calvin calvin  2583 Feb 17 17:25 file_map.txt
drwxr-xr-x  3 calvin calvin  4096 Feb 18 15:04 legacy
-rw-r--r--  1 calvin calvin  8341 Feb 17 17:25 plan.md
drwxr-xr-x  4 calvin calvin  4096 Feb 18 15:43 rave-cli
drwxr-xr-x  2 calvin calvin  4096 Feb 18 15:27 reports
drwxr-xr-x  3 calvin calvin  4096 Feb 18 15:04 scripts
drwxrwxrwx  6 calvin calvin  4096 Feb 17 15:06 target
drwxrwxrwx  4 calvin calvin  4096 Feb 15 06:00 third_party
[exit_code=0]
```

## git ls-tree --name-only HEAD | rg '^Cargo\.toml$'
```text
[exit_code=1]
```

## find crates rave-cli -maxdepth 3 -name Cargo.toml -type f | sort
```text
crates/rave-core/Cargo.toml
crates/rave-cuda/Cargo.toml
crates/rave-ffmpeg/Cargo.toml
crates/rave-nvcodec/Cargo.toml
crates/rave-pipeline/Cargo.toml
crates/rave-tensorrt/Cargo.toml
rave-cli/Cargo.toml
[exit_code=0]
```

## git restore --source=HEAD -- crates rave-cli
```text
[exit_code=0]
```

## git restore --source=HEAD -- crates/*/Cargo.toml rave-cli/Cargo.toml
```text
error: pathspec 'crates/rave-core/Cargo.toml' did not match any file(s) known to git
error: pathspec 'crates/rave-cuda/Cargo.toml' did not match any file(s) known to git
error: pathspec 'crates/rave-ffmpeg/Cargo.toml' did not match any file(s) known to git
error: pathspec 'crates/rave-nvcodec/Cargo.toml' did not match any file(s) known to git
error: pathspec 'crates/rave-pipeline/Cargo.toml' did not match any file(s) known to git
error: pathspec 'crates/rave-tensorrt/Cargo.toml' did not match any file(s) known to git
error: pathspec 'rave-cli/Cargo.toml' did not match any file(s) known to git
[exit_code=1]
```

## Root Workspace Manifest
```toml
[workspace]
members = [
  "crates/rave-core",
  "crates/rave-cuda",
  "crates/rave-ffmpeg",
  "crates/rave-nvcodec",
  "crates/rave-pipeline",
  "crates/rave-tensorrt",
  "rave-cli",
]
resolver = "2"
```

## git cat-file -t 51d2f1a
```text
commit
[exit_code=0]
```

## git cat-file -t 63eb0b8
```text
commit
[exit_code=0]
```

## git cat-file -t 6fcb9fa
```text
commit
[exit_code=0]
```

## git cat-file -t 4bfdda5
```text
commit
[exit_code=0]
```

## git branch --contains 51d2f1a
```text
* main
  phase10-clippy-dwarnings
  phase10-closeout-wsl-legacy-docs
  phase11-cli-pr1-help-json
  phase11-cli-pr2-devices-probe
  phase11-cli-pr3-progress-ux
  phase11-cli-pr4-consistency-sweep
  phase11-cli-pr5-json-schema-contract
[exit_code=0]
```

## git branch --contains 63eb0b8
```text
* main
  phase10-clippy-dwarnings
  phase11-cli-pr1-help-json
  phase11-cli-pr2-devices-probe
  phase11-cli-pr3-progress-ux
  phase11-cli-pr4-consistency-sweep
  phase11-cli-pr5-json-schema-contract
[exit_code=0]
```

## git branch --contains 6fcb9fa
```text
* main
  phase11-cli-pr1-help-json
  phase11-cli-pr2-devices-probe
  phase11-cli-pr3-progress-ux
  phase11-cli-pr4-consistency-sweep
  phase11-cli-pr5-json-schema-contract
[exit_code=0]
```

## git branch --contains 4bfdda5
```text
* main
  phase11-cli-pr2-devices-probe
  phase11-cli-pr3-progress-ux
  phase11-cli-pr4-consistency-sweep
  phase11-cli-pr5-json-schema-contract
[exit_code=0]
```

## cargo fmt --check
```text
Diff in /home/calvin/src/rave/legacy/engine-v2/src/core/kernels.rs:36:
 
 use std::sync::Arc;
 
[31m-use cudarc::driver::{
(B[m[31m-    CudaDevice, CudaFunction, CudaStream, DevicePtr, LaunchAsync, LaunchConfig,
(B[m[31m-};
(B[m[32m+use cudarc::driver::{CudaDevice, CudaFunction, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
(B[m use tracing::info;
 
 use crate::codecs::sys::{self, CUevent};
Diff in /home/calvin/src/rave/legacy/engine-v2/src/engine/inference.rs:20:
 use crate::core::backend::UpscaleBackend;
 use crate::core::context::GpuContext;
 use crate::core::kernels::{ModelPrecision, PreprocessKernels, PreprocessPipeline, StageMetrics};
[31m-use crate::core::types::{GpuTexture, FrameEnvelope, PixelFormat};
(B[m[32m+use crate::core::types::{FrameEnvelope, GpuTexture, PixelFormat};
(B[m use crate::error::{EngineError, Result};
 
 /// End-to-end GPU-resident inference pipeline.
Diff in /home/calvin/src/rave/legacy/engine-v2/src/engine/inference.rs:126:
             (inference_output.width as usize + 255) & !255, // 256-byte aligned
         );
 
[31m-        let nv12_output = self.preprocess.postprocess(
(B[m[31m-            inference_output,
(B[m[31m-            nv12_pitch,
(B[m[31m-            &self.ctx,
(B[m[31m-            stream,
(B[m[31m-        )?;
(B[m[32m+        let nv12_output =
(B[m[32m+            self.preprocess
(B[m[32m+                .postprocess(inference_output, nv12_pitch, &self.ctx, stream)?;
(B[m 
         debug!(
             frame = envelope.frame_index,
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_demuxer.rs:15:
 use crate::codecs::sys::cudaVideoCodec;
 use crate::error::{EngineError, Result};
 use crate::io::ffmpeg_sys::{
[31m-    check_ffmpeg, to_cstring,
(B[m     // BSF FFI (missing from ffmpeg-sys-next v8)
[31m-    AVBSFContext, av_bsf_alloc, av_bsf_free, av_bsf_get_by_name,
(B[m[31m-    av_bsf_init, av_bsf_receive_packet, av_bsf_send_packet,
(B[m[32m+    AVBSFContext,
(B[m[32m+    av_bsf_alloc,
(B[m[32m+    av_bsf_free,
(B[m[32m+    av_bsf_get_by_name,
(B[m[32m+    av_bsf_init,
(B[m[32m+    av_bsf_receive_packet,
(B[m[32m+    av_bsf_send_packet,
(B[m[32m+    check_ffmpeg,
(B[m[32m+    to_cstring,
(B[m };
 
 /// Demuxes a container file and produces Annex B bitstream packets.
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_demuxer.rs:50:
         // ── Open container ──
         let mut fmt_ctx: *mut AVFormatContext = ptr::null_mut();
         let ret = unsafe {
[31m-            avformat_open_input(
(B[m[31m-                &mut fmt_ctx,
(B[m[31m-                c_path.as_ptr(),
(B[m[31m-                ptr::null(),
(B[m[31m-                ptr::null_mut(),
(B[m[31m-            )
(B[m[32m+            avformat_open_input(&mut fmt_ctx, c_path.as_ptr(), ptr::null(), ptr::null_mut())
(B[m         };
         check_ffmpeg(ret, "avformat_open_input")?;
 
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_demuxer.rs:135:
         }
 
         // Copy codec parameters from the stream to the BSF.
[31m-        let ret = unsafe {
(B[m[31m-            avcodec_parameters_copy((*bsf_ctx).par_in, stream.codecpar)
(B[m[31m-        };
(B[m[32m+        let ret = unsafe { avcodec_parameters_copy((*bsf_ctx).par_in, stream.codecpar) };
(B[m         if ret < 0 {
             unsafe {
                 av_bsf_free(&mut bsf_ctx);
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_demuxer.rs:189:
         if pts == AV_NOPTS_VALUE {
             return 0;
         }
[31m-        let us_tb = AVRational { num: 1, den: 1_000_000 };
(B[m[32m+        let us_tb = AVRational {
(B[m[32m+            num: 1,
(B[m[32m+            den: 1_000_000,
(B[m[32m+        };
(B[m         unsafe { av_rescale_q(pts, self.time_base, us_tb) }
     }
 }
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_demuxer.rs:206:
                 let ret = unsafe { av_bsf_receive_packet(self.bsf_ctx, self.pkt_filtered) };
                 if ret == 0 {
                     let pkt = unsafe { &*self.pkt_filtered };
[31m-                    let data = unsafe {
(B[m[31m-                        std::slice::from_raw_parts(pkt.data, pkt.size as usize)
(B[m[31m-                    }.to_vec();
(B[m[32m+                    let data =
(B[m[32m+                        unsafe { std::slice::from_raw_parts(pkt.data, pkt.size as usize) }.to_vec();
(B[m                     let pts = self.rescale_pts(pkt.pts);
                     let is_keyframe = (pkt.flags & AV_PKT_FLAG_KEY) != 0;
                     unsafe { av_packet_unref(self.pkt_filtered) };
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_demuxer.rs:232:
                     if !self.bsf_ctx.is_null() {
                         unsafe { av_bsf_send_packet(self.bsf_ctx, ptr::null()) };
                         // Try to receive remaining packets.
[31m-                        let ret2 = unsafe {
(B[m[31m-                            av_bsf_receive_packet(self.bsf_ctx, self.pkt_filtered)
(B[m[31m-                        };
(B[m[32m+                        let ret2 =
(B[m[32m+                            unsafe { av_bsf_receive_packet(self.bsf_ctx, self.pkt_filtered) };
(B[m                         if ret2 == 0 {
                             let pkt = unsafe { &*self.pkt_filtered };
[31m-                            let data = unsafe {
(B[m[31m-                                std::slice::from_raw_parts(pkt.data, pkt.size as usize)
(B[m[31m-                            }.to_vec();
(B[m[32m+                            let data =
(B[m[32m+                                unsafe { std::slice::from_raw_parts(pkt.data, pkt.size as usize) }
(B[m[32m+                                    .to_vec();
(B[m                             let pts = self.rescale_pts(pkt.pts);
                             let is_keyframe = (pkt.flags & AV_PKT_FLAG_KEY) != 0;
                             unsafe { av_packet_unref(self.pkt_filtered) };
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_demuxer.rs:265:
 
             if self.bsf_ctx.is_null() {
                 // No BSF — return packet directly.
[31m-                let data = unsafe {
(B[m[31m-                    std::slice::from_raw_parts(pkt.data, pkt.size as usize)
(B[m[31m-                }.to_vec();
(B[m[32m+                let data =
(B[m[32m+                    unsafe { std::slice::from_raw_parts(pkt.data, pkt.size as usize) }.to_vec();
(B[m                 let pts = self.rescale_pts(pkt.pts);
                 let is_keyframe = (pkt.flags & AV_PKT_FLAG_KEY) != 0;
                 unsafe { av_packet_unref(self.pkt_read) };
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_muxer.rs:47:
         // ── Create output format context ──
         let mut fmt_ctx: *mut AVFormatContext = ptr::null_mut();
         let ret = unsafe {
[31m-            avformat_alloc_output_context2(
(B[m[31m-                &mut fmt_ctx,
(B[m[31m-                ptr::null(),
(B[m[31m-                ptr::null(),
(B[m[31m-                c_path.as_ptr(),
(B[m[31m-            )
(B[m[32m+            avformat_alloc_output_context2(&mut fmt_ctx, ptr::null(), ptr::null(), c_path.as_ptr())
(B[m         };
         if ret < 0 || fmt_ctx.is_null() {
             return Err(EngineError::Mux(format!(
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_muxer.rs:89:
             let ret = unsafe { avio_open(&mut (*fmt_ctx).pb, c_path.as_ptr(), AVIO_FLAG_WRITE) };
             if ret < 0 {
                 unsafe { avformat_free_context(fmt_ctx) };
[31m-                check_ffmpeg(ret, "avio_open")
(B[m[31m-                    .map_err(|e| EngineError::Mux(format!("{e}")))?;
(B[m[32m+                check_ffmpeg(ret, "avio_open").map_err(|e| EngineError::Mux(format!("{e}")))?;
(B[m             }
         }
 
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_muxer.rs:119:
             stream,
             pkt,
             time_base: AVRational { num: 0, den: 1 },
[31m-            us_tb: AVRational { num: 1, den: 1_000_000 },
(B[m[32m+            us_tb: AVRational {
(B[m[32m+                num: 1,
(B[m[32m+                den: 1_000_000,
(B[m[32m+            },
(B[m             packet_counter: 0,
             header_written: false,
         })
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_muxer.rs:132:
         }
 
         let ret = unsafe { avformat_write_header(self.fmt_ctx, ptr::null_mut()) };
[31m-        check_ffmpeg(ret, "avformat_write_header")
(B[m[31m-            .map_err(|e| EngineError::Mux(format!("{e}")))?;
(B[m[32m+        check_ffmpeg(ret, "avformat_write_header").map_err(|e| EngineError::Mux(format!("{e}")))?;
(B[m 
         // Capture the actual time_base after muxer initialization
         // (the muxer may adjust it).
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_muxer.rs:158:
             // Copy data into the AVPacket.
             let ret = av_new_packet(self.pkt, data.len() as i32);
             if ret < 0 {
[31m-                check_ffmpeg(ret, "av_new_packet")
(B[m[31m-                    .map_err(|e| EngineError::Mux(format!("{e}")))?;
(B[m[32m+                check_ffmpeg(ret, "av_new_packet").map_err(|e| EngineError::Mux(format!("{e}")))?;
(B[m             }
             ptr::copy_nonoverlapping(data.as_ptr(), (*self.pkt).data, data.len());
 
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/ffmpeg_muxer.rs:198:
         }
 
         let ret = unsafe { av_write_trailer(self.fmt_ctx) };
[31m-        check_ffmpeg(ret, "av_write_trailer")
(B[m[31m-            .map_err(|e| EngineError::Mux(format!("{e}")))?;
(B[m[32m+        check_ffmpeg(ret, "av_write_trailer").map_err(|e| EngineError::Mux(format!("{e}")))?;
(B[m 
         tracing::info!(
             packets = self.packet_counter,
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/file_sink.rs:49:
 }
 
 impl BitstreamSink for FileBitstreamSink {
[31m-    fn write_packet(&mut self, data: &[u8], _pts: i64, _dts: i64, _is_keyframe: bool) -> Result<()> {
(B[m[32m+    fn write_packet(
(B[m[32m+        &mut self,
(B[m[32m+        data: &[u8],
(B[m[32m+        _pts: i64,
(B[m[32m+        _dts: i64,
(B[m[32m+        _is_keyframe: bool,
(B[m[32m+    ) -> Result<()> {
(B[m         self.writer.write_all(data).map_err(|e| {
             EngineError::Encode(format!("Failed to write to {}: {}", self.path.display(), e))
         })?;
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/mod.rs:13:
 pub mod file_source;
 pub mod probe;
 
[31m-pub use probe::{probe_container, ContainerMetadata};
(B[m[32m+pub use probe::{ContainerMetadata, probe_container};
(B[m 
Diff in /home/calvin/src/rave/legacy/engine-v2/src/io/probe.rs:58:
 
     // Open the container.
     // SAFETY: c_path is a valid null-terminated C string. fmt_ctx is an output.
[31m-    let ret = unsafe {
(B[m[31m-        avformat_open_input(
(B[m[31m-            &mut fmt_ctx,
(B[m[31m-            c_path.as_ptr(),
(B[m[31m-            ptr::null(),
(B[m[31m-            ptr::null_mut(),
(B[m[31m-        )
(B[m[31m-    };
(B[m[31m-    check_ffmpeg(ret, "avformat_open_input")
(B[m[31m-        .map_err(|e| EngineError::Probe(format!("{e}")))?;
(B[m[32m+    let ret =
(B[m[32m+        unsafe { avformat_open_input(&mut fmt_ctx, c_path.as_ptr(), ptr::null(), ptr::null_mut()) };
(B[m[32m+    check_ffmpeg(ret, "avformat_open_input").map_err(|e| EngineError::Probe(format!("{e}")))?;
(B[m 
     let guard = FormatGuard { ctx: fmt_ctx };
 
Diff in /home/calvin/src/rave/legacy/engine-v2/src/main.rs:15:
 use clap::Parser;
 
 use rave_engine::backends::tensorrt::TensorRtBackend;
[31m-use rave_engine::core::backend::UpscaleBackend;
(B[m use rave_engine::codecs::nvdec::NvDecoder;
 use rave_engine::codecs::nvenc::{NvEncConfig, NvEncoder};
 use rave_engine::codecs::sys::cudaVideoCodec;
Diff in /home/calvin/src/rave/legacy/engine-v2/src/main.rs:22:
[32m+use rave_engine::core::backend::UpscaleBackend;
(B[m use rave_engine::core::context::GpuContext;
 use rave_engine::core::kernels::{ModelPrecision, PreprocessKernels};
 use rave_engine::engine::pipeline::{PipelineConfig, UpscalePipeline};
Diff in /home/calvin/src/rave/legacy/engine-v2/src/main.rs:25:
 use rave_engine::error::Result;
[31m-use rave_engine::io::file_sink::FileBitstreamSink;
(B[m[31m-use rave_engine::io::file_source::FileBitstreamSource;
(B[m use rave_engine::io::ffmpeg_demuxer::FfmpegDemuxer;
 use rave_engine::io::ffmpeg_muxer::FfmpegMuxer;
[32m+use rave_engine::io::file_sink::FileBitstreamSink;
(B[m[32m+use rave_engine::io::file_source::FileBitstreamSource;
(B[m use rave_engine::io::probe_container;
 
 // ─── CLI argument definition ─────────────────────────────────────────────────
[exit_code=1]
```

## cargo clippy --workspace --all-targets -- -D warnings
```text
warning: rave-engine@2.0.0: Video Codec SDK libs not found in /home/calvin/src/rave/legacy/third_party/nvcodec. Falling back to CUDA lib dir.
    Checking rave-engine v2.0.0 (/home/calvin/src/rave/legacy/engine-v2)
error: this `impl` can be derived
   --> legacy/engine-v2/src/backends/tensorrt.rs:137:1
    |
137 | / impl Default for PrecisionPolicy {
138 | |     fn default() -> Self {
139 | |         PrecisionPolicy::Fp16
140 | |     }
141 | | }
    | |_^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#derivable_impls
    = note: `-D clippy::derivable-impls` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::derivable_impls)]`
help: replace the manual implementation with a derive attribute and mark the default variant
    |
127 + #[derive(Default)]
128 | pub enum PrecisionPolicy {
129 |     /// FP32 only — maximum accuracy, baseline performance.
130 |     Fp32,
131 |     /// FP16 mixed precision — 2× throughput on Tensor Cores.
132 ~     #[default]
133 ~     Fp16,
    |

error: you should consider adding a `Default` implementation for `InferenceMetrics`
   --> legacy/engine-v2/src/backends/tensorrt.rs:178:5
    |
178 | /     pub const fn new() -> Self {
179 | |         Self {
180 | |             frames_inferred: AtomicU64::new(0),
181 | |             total_inference_us: AtomicU64::new(0),
...   |
184 | |     }
    | |_____^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#new_without_default
    = note: `-D clippy::new-without-default` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::new_without_default)]`
help: try adding this
    |
177 + impl Default for InferenceMetrics {
178 +     fn default() -> Self {
179 +         Self::new()
180 +     }
181 + }
    |

error: you should consider adding a `Default` implementation for `RingMetrics`
   --> legacy/engine-v2/src/backends/tensorrt.rs:228:5
    |
228 | /     pub const fn new() -> Self {
229 | |         Self {
230 | |             slot_reuse_count: AtomicU64::new(0),
231 | |             slot_contention_events: AtomicU64::new(0),
...   |
234 | |     }
    | |_____^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#new_without_default
help: try adding this
    |
227 + impl Default for RingMetrics {
228 +     fn default() -> Self {
229 +         Self::new()
230 +     }
231 + }
    |

error: struct `OutputRing` has a public `len` method, but no `is_empty` method
   --> legacy/engine-v2/src/backends/tensorrt.rs:387:5
    |
387 |     pub fn len(&self) -> usize {
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#len_without_is_empty
    = note: `-D clippy::len-without-is-empty` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::len_without_is_empty)]`

error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> legacy/engine-v2/src/backends/tensorrt.rs:847:26
    |
847 |         let output_ptr = *(*output_arc).device_ptr() as u64;
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*(*output_arc).device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast
    = note: `-D clippy::unnecessary-cast` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::unnecessary_cast)]`

error: this `if` statement can be collapsed
   --> legacy/engine-v2/src/backends/tensorrt.rs:934:9
    |
934 | /         if let Ok(mut guard) = self.state.try_lock() {
935 | |             if let Some(state) = guard.take() {
936 | |                 let _ = self.ctx.sync_all();
937 | |                 drop(state);
938 | |             }
939 | |         }
    | |_________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
    = note: `-D clippy::collapsible-if` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::collapsible_if)]`
help: collapse nested if block
    |
934 ~         if let Ok(mut guard) = self.state.try_lock()
935 ~             && let Some(state) = guard.take() {
936 |                 let _ = self.ctx.sync_all();
937 |                 drop(state);
938 ~             }
    |

error: this public function might dereference a raw pointer but is not marked `unsafe`
   --> legacy/engine-v2/src/codecs/nvdec.rs:600:48
    |
600 |         check_cu(cuStreamWaitEvent(raw_stream, event, 0), "cuStreamWaitEvent")?;
    |                                                ^^^^^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#not_unsafe_ptr_arg_deref
    = note: `#[deny(clippy::not_unsafe_ptr_arg_deref)]` on by default

error: this `if` statement can be collapsed
   --> legacy/engine-v2/src/codecs/nvenc.rs:506:9
    |
506 | /         if !self.bitstream_buf.is_null() {
507 | |             if let Some(destroy_fn) = self.fns.nvEncDestroyBitstreamBuffer {
508 | |                 // SAFETY: bitstream_buf was created via nvEncCreateBitstreamBuffer.
509 | |                 unsafe {
...   |
513 | |         }
    | |_________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
help: collapse nested if block
    |
506 ~         if !self.bitstream_buf.is_null()
507 ~             && let Some(destroy_fn) = self.fns.nvEncDestroyBitstreamBuffer {
508 |                 // SAFETY: bitstream_buf was created via nvEncCreateBitstreamBuffer.
...
511 |                 }
512 ~             }
    |

error: this `if` statement can be collapsed
   --> legacy/engine-v2/src/codecs/nvenc.rs:516:9
    |
516 | /         if !self.encoder.is_null() {
517 | |             if let Some(destroy_fn) = self.fns.nvEncDestroyEncoder {
518 | |                 // SAFETY: encoder was opened via nvEncOpenEncodeSessionEx.
519 | |                 unsafe {
...   |
523 | |         }
    | |_________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
help: collapse nested if block
    |
516 ~         if !self.encoder.is_null()
517 ~             && let Some(destroy_fn) = self.fns.nvEncDestroyEncoder {
518 |                 // SAFETY: encoder was opened via nvEncOpenEncodeSessionEx.
...
521 |                 }
522 ~             }
    |

error: you should consider adding a `Default` implementation for `QueueDepthTracker`
   --> legacy/engine-v2/src/core/context.rs:811:5
    |
811 | /     pub fn new() -> Self {
812 | |         Self {
813 | |             decode: AtomicUsize::new(0),
814 | |             preprocess: AtomicUsize::new(0),
...   |
817 | |     }
    | |_____^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#new_without_default
help: try adding this
    |
810 + impl Default for QueueDepthTracker {
811 +     fn default() -> Self {
812 +         Self::new()
813 +     }
814 + }
    |

error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> legacy/engine-v2/src/core/kernels.rs:351:23
    |
351 |         let out_ptr = *output_buf.device_ptr() as u64;
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*output_buf.device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> legacy/engine-v2/src/core/kernels.rs:417:23
    |
417 |         let out_ptr = *output_buf.device_ptr() as u64;
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*output_buf.device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> legacy/engine-v2/src/core/kernels.rs:472:23
    |
472 |         let out_ptr = *output_buf.device_ptr() as u64;
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*output_buf.device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> legacy/engine-v2/src/core/kernels.rs:518:23
    |
518 |         let out_ptr = *output_buf.device_ptr() as u64;
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*output_buf.device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> legacy/engine-v2/src/core/kernels.rs:567:21
    |
567 |         let y_ptr = *output_buf.device_ptr() as u64;
    |                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*output_buf.device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: manually reimplementing `div_ceil`
   --> legacy/engine-v2/src/core/kernels.rs:850:9
    |
850 |         (width + block.0 - 1) / block.0,
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: consider using `.div_ceil()`: `width.div_ceil(block.0)`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_div_ceil
    = note: `-D clippy::manual-div-ceil` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::manual_div_ceil)]`

error: manually reimplementing `div_ceil`
   --> legacy/engine-v2/src/core/kernels.rs:851:9
    |
851 |         (height + block.1 - 1) / block.1,
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: consider using `.div_ceil()`: `height.div_ceil(block.1)`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_div_ceil

error: manually reimplementing `div_ceil`
   --> legacy/engine-v2/src/core/kernels.rs:867:17
    |
867 |     let grid = ((count as u32 + block - 1) / block, 1, 1);
    |                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: consider using `.div_ceil()`: `(count as u32).div_ceil(block)`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_div_ceil

error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> legacy/engine-v2/src/core/types.rs:158:9
    |
158 |         *self.data.device_ptr() as u64
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*self.data.device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: manually reimplementing `div_ceil`
   --> legacy/engine-v2/src/engine/pipeline.rs:456:26
    |
456 |         let nv12_pitch = ((width as usize + 255) / 256) * 256;
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: consider using `.div_ceil()`: `(width as usize).div_ceil(256)`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_div_ceil

error: this block may be rewritten with the `?` operator
   --> legacy/engine-v2/src/engine/pipeline.rs:547:9
    |
547 | /         if let Err(e) = run_ok {
548 | |             return Err(e);
549 | |         }
    | |_________^ help: replace it with: `run_ok?;`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#question_mark
    = note: `-D clippy::question-mark` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::question_mark)]`

error: manually reimplementing `div_ceil`
   --> legacy/engine-v2/src/engine/pipeline.rs:659:26
    |
659 |         let nv12_pitch = ((width as usize + 255) / 256) * 256;
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: consider using `.div_ceil()`: `(width as usize).div_ceil(256)`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_div_ceil

error: manual absolute difference pattern without using `abs_diff`
   --> legacy/engine-v2/src/engine/pipeline.rs:744:26
    |
744 |           let vram_delta = if vram_end > vram_start {
    |  __________________________^
745 | |             vram_end - vram_start
746 | |         } else {
747 | |             vram_start - vram_end
748 | |         };
    | |_________^ help: replace with `abs_diff`: `vram_end.abs_diff(vram_start)`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_abs_diff
    = note: `-D clippy::manual-abs-diff` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::manual_abs_diff)]`

error: this `if` statement can be collapsed
   --> legacy/engine-v2/src/engine/pipeline.rs:809:13
    |
809 | /             if !report.host_alloc_check.is_pass() {
810 | |                 if let AuditResult::Fail(msg) = &report.host_alloc_check {
811 | |                     failures.push(msg.clone());
812 | |                 }
813 | |             }
    | |_____________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
help: collapse nested if block
    |
809 ~             if !report.host_alloc_check.is_pass()
810 ~                 && let AuditResult::Fail(msg) = &report.host_alloc_check {
811 |                     failures.push(msg.clone());
812 ~                 }
    |

error: this `if` statement can be collapsed
   --> legacy/engine-v2/src/engine/pipeline.rs:814:13
    |
814 | /             if !report.vram_leak_check.is_pass() {
815 | |                 if let AuditResult::Fail(msg) = &report.vram_leak_check {
816 | |                     failures.push(msg.clone());
817 | |                 }
818 | |             }
    | |_____________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
help: collapse nested if block
    |
814 ~             if !report.vram_leak_check.is_pass()
815 ~                 && let AuditResult::Fail(msg) = &report.vram_leak_check {
816 |                     failures.push(msg.clone());
817 ~                 }
    |

error: this `if` statement can be collapsed
   --> legacy/engine-v2/src/engine/pipeline.rs:819:13
    |
819 | /             if !report.pool_hit_rate_check.is_pass() {
820 | |                 if let AuditResult::Fail(msg) = &report.pool_hit_rate_check {
821 | |                     failures.push(msg.clone());
822 | |                 }
823 | |             }
    | |_____________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
help: collapse nested if block
    |
819 ~             if !report.pool_hit_rate_check.is_pass()
820 ~                 && let AuditResult::Fail(msg) = &report.pool_hit_rate_check {
821 |                     failures.push(msg.clone());
822 ~                 }
    |

error: this `if` statement can be collapsed
   --> legacy/engine-v2/src/engine/pipeline.rs:824:13
    |
824 | /             if !report.stream_overlap_check.is_pass() {
825 | |                 if let AuditResult::Fail(msg) = &report.stream_overlap_check {
826 | |                     failures.push(msg.clone());
827 | |                 }
828 | |             }
    | |_____________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
help: collapse nested if block
    |
824 ~             if !report.stream_overlap_check.is_pass()
825 ~                 && let AuditResult::Fail(msg) = &report.stream_overlap_check {
826 |                     failures.push(msg.clone());
827 ~                 }
    |

error: this function has too many arguments (8/7)
   --> legacy/engine-v2/src/engine/pipeline.rs:887:1
    |
887 | / async fn preprocess_stage(
888 | |     mut rx: mpsc::Receiver<DecodedFrame>,
889 | |     tx: &mpsc::Sender<FrameEnvelope>,
890 | |     _kernels: &PreprocessKernels,
...   |
895 | |     profiler_ctx: Option<&GpuContext>,
896 | | ) -> Result<()> {
    | |_______________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#too_many_arguments
    = note: `-D clippy::too-many-arguments` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::too_many_arguments)]`

error: this function has too many arguments (10/7)
   --> legacy/engine-v2/src/engine/pipeline.rs:983:1
    |
983 | / async fn inference_stage<B: UpscaleBackend>(
984 | |     mut rx: mpsc::Receiver<FrameEnvelope>,
985 | |     tx: &mpsc::Sender<FrameEnvelope>,
986 | |     backend: &B,
...   |
993 | |     profiler_ctx: Option<&GpuContext>,
994 | | ) -> Result<()> {
    | |_______________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#too_many_arguments

error: manual implementation of `.is_multiple_of()`
    --> legacy/engine-v2/src/engine/pipeline.rs:1179:26
     |
1179 |             is_keyframe: self.idx % 30 == 0,
     |                          ^^^^^^^^^^^^^^^^^^ help: replace with: `self.idx.is_multiple_of(30)`
     |
     = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_is_multiple_of
     = note: `-D clippy::manual-is-multiple-of` implied by `-D warnings`
     = help: to override `-D warnings` add `#[allow(clippy::manual_is_multiple_of)]`

error: manual implementation of `.is_multiple_of()`
   --> legacy/engine-v2/src/io/ffmpeg_muxer.rs:187:12
    |
187 |         if self.packet_counter % 100 == 0 {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: replace with: `self.packet_counter.is_multiple_of(100)`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_is_multiple_of

error: manual implementation of `.is_multiple_of()`
  --> legacy/engine-v2/src/io/file_sink.rs:60:12
   |
60 |         if self.packets_written % 100 == 0 {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: replace with: `self.packets_written.is_multiple_of(100)`
   |
   = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_is_multiple_of

warning: rave-engine@2.0.0: Video Codec SDK libs not found in /home/calvin/src/rave/legacy/third_party/nvcodec. Falling back to CUDA lib dir.
error: could not compile `rave-engine` (lib) due to 32 previous errors
warning: build failed, waiting for other jobs to finish...
warning: rave-engine@2.0.0: Video Codec SDK libs not found in /home/calvin/src/rave/legacy/third_party/nvcodec. Falling back to CUDA lib dir.
error: could not compile `rave-engine` (lib test) due to 32 previous errors
[exit_code=101]
```

## cargo test --workspace
```text
warning: rave-engine@2.0.0: Video Codec SDK libs not found in /home/calvin/src/rave/legacy/third_party/nvcodec. Falling back to CUDA lib dir.
   Compiling rave-engine v2.0.0 (/home/calvin/src/rave/legacy/engine-v2)
   Compiling rave-core v0.1.0 (/home/calvin/src/rave/crates/rave-core)
   Compiling rave-tensorrt v0.1.0 (/home/calvin/src/rave/crates/rave-tensorrt)
   Compiling rave-cuda v0.1.0 (/home/calvin/src/rave/crates/rave-cuda)
   Compiling rave-pipeline v0.1.0 (/home/calvin/src/rave/crates/rave-pipeline)
   Compiling rave-ffmpeg v0.1.0 (/home/calvin/src/rave/crates/rave-ffmpeg)
   Compiling rave-nvcodec v0.1.0 (/home/calvin/src/rave/crates/rave-nvcodec)
   Compiling rave-cli v0.1.0 (/home/calvin/src/rave/rave-cli)
error[E0433]: failed to resolve: use of unresolved module or unlinked crate `ort`
 --> crates/rave-tensorrt/tests/provider_bridge_smoke.rs:8:5
  |
8 | use ort::execution_providers::TensorRTExecutionProvider;
  |     ^^^ use of unresolved module or unlinked crate `ort`
  |
  = help: if you wanted to use a crate named `ort`, use `cargo add ort` to add it to your `Cargo.toml`

error[E0433]: failed to resolve: use of unresolved module or unlinked crate `ort`
 --> crates/rave-tensorrt/tests/provider_bridge_smoke.rs:9:5
  |
9 | use ort::session::Session;
  |     ^^^ use of unresolved module or unlinked crate `ort`
  |
  = help: if you wanted to use a crate named `ort`, use `cargo add ort` to add it to your `Cargo.toml`

For more information about this error, try `rustc --explain E0433`.
error: could not compile `rave-tensorrt` (test "provider_bridge_smoke") due to 2 previous errors
warning: build failed, waiting for other jobs to finish...
error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-m64" "/tmp/rustcmi4VVO/symbols.o" "<6 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libtest-*,libgetopts-*,librustc_std_workspace_std-*}.rlib" "/home/calvin/src/rave/target/debug/deps/{librave_engine-e2a827911d1bd822,libthiserror-c31e995d4d2456ab,libffmpeg_sys_next-d0d03443a27bde17,libtokio_util-495535f88a74692e,libfutures_util-d653fb3bb168b6e9,libslab-bf54227bd3b10ac1,libfutures_task-9b426e5331f0de27,libbytes-51256f3823e5c3ab,libfutures_core-e7fa090f587e0d16,libfutures_sink-606fc6487acc0e7d,libort-9f40a4dae8e96b14,libort_sys-2c30b9a5507bc559,libndarray-541c3e7fe60a0595,libmatrixmultiply-82f75d05f86948e0,libnum_complex-a48b17186caed1f0,libnum_integer-8e22011dfd2f00f7,libnum_traits-c9bd09bea10dfdf2,librawpointer-632d42a28d05c689,libsmallvec-e1aea5a53633896e,libtracing-27bcbd6d575c4bb3,libtracing_core-1f29259c68130f65,libonce_cell-f3864f3383bfa6d4,libtokio-6a40c18244dcad06,libsignal_hook_registry-c402b8ceae4a9dbb,liberrno-1f0976c6c32f6634,libmio-8a8a169e6a68339d,liblibc-0c2f154968cd63cf,libpin_project_lite-492733ea6c61e226,libcudarc-9f668e51870e9d72,liblibloading-a16efcae9348f444,libcfg_if-59c7dc747326a75e}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lcuda" "-lnvcuvid" "-lnvidia-encode" "-lavutil" "-lavformat" "-lavfilter" "-lavdevice" "-lswscale" "-lswresample" "-lavcodec" "-lstdc++" "-ldl" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustcmi4VVO/raw-dylibs" "-B<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld" "-fuse-ld=lld" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/usr/local/cuda/lib64" "-L" "/usr/lib/x86_64-linux-gnu" "-L" "/home/calvin/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/d3c01924b801c77ff17d300b24e6dcd46d378348a921a48d96f115f87074fbb1" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/home/calvin/src/rave/target/debug/deps/rave_cuda-22a1b6c3450dfd4d" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: rust-lld: error: unable to find library -lcuda
          rust-lld: error: unable to find library -lnvcuvid
          rust-lld: error: unable to find library -lnvidia-encode
          collect2: error: ld returned 1 exit status
          

error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-m64" "/tmp/rustcmbDtON/symbols.o" "<6 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libtest-*,libgetopts-*,librustc_std_workspace_std-*}.rlib" "/home/calvin/src/rave/target/debug/deps/{librave_engine-e2a827911d1bd822,libthiserror-c31e995d4d2456ab,libffmpeg_sys_next-d0d03443a27bde17,libtokio_util-495535f88a74692e,libfutures_util-d653fb3bb168b6e9,libslab-bf54227bd3b10ac1,libfutures_task-9b426e5331f0de27,libbytes-51256f3823e5c3ab,libfutures_core-e7fa090f587e0d16,libfutures_sink-606fc6487acc0e7d,libort-9f40a4dae8e96b14,libort_sys-2c30b9a5507bc559,libndarray-541c3e7fe60a0595,libmatrixmultiply-82f75d05f86948e0,libnum_complex-a48b17186caed1f0,libnum_integer-8e22011dfd2f00f7,libnum_traits-c9bd09bea10dfdf2,librawpointer-632d42a28d05c689,libsmallvec-e1aea5a53633896e,libtracing-27bcbd6d575c4bb3,libtracing_core-1f29259c68130f65,libonce_cell-f3864f3383bfa6d4,libtokio-6a40c18244dcad06,libsignal_hook_registry-c402b8ceae4a9dbb,liberrno-1f0976c6c32f6634,libmio-8a8a169e6a68339d,liblibc-0c2f154968cd63cf,libpin_project_lite-492733ea6c61e226,libcudarc-9f668e51870e9d72,liblibloading-a16efcae9348f444,libcfg_if-59c7dc747326a75e}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lcuda" "-lnvcuvid" "-lnvidia-encode" "-lavutil" "-lavformat" "-lavfilter" "-lavdevice" "-lswscale" "-lswresample" "-lavcodec" "-lstdc++" "-ldl" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustcmbDtON/raw-dylibs" "-B<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld" "-fuse-ld=lld" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/usr/local/cuda/lib64" "-L" "/usr/lib/x86_64-linux-gnu" "-L" "/home/calvin/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/d3c01924b801c77ff17d300b24e6dcd46d378348a921a48d96f115f87074fbb1" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/home/calvin/src/rave/target/debug/deps/rave_tensorrt-f21e61e3794309a8" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: rust-lld: error: unable to find library -lcuda
          rust-lld: error: unable to find library -lnvcuvid
          rust-lld: error: unable to find library -lnvidia-encode
          collect2: error: ld returned 1 exit status
          

error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-m64" "/tmp/rustcv4rffq/symbols.o" "<6 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libtest-*,libgetopts-*,librustc_std_workspace_std-*}.rlib" "/home/calvin/src/rave/target/debug/deps/{librave_engine-e2a827911d1bd822,libthiserror-c31e995d4d2456ab,libffmpeg_sys_next-d0d03443a27bde17,libtokio_util-495535f88a74692e,libfutures_util-d653fb3bb168b6e9,libslab-bf54227bd3b10ac1,libfutures_task-9b426e5331f0de27,libbytes-51256f3823e5c3ab,libfutures_core-e7fa090f587e0d16,libfutures_sink-606fc6487acc0e7d,libort-9f40a4dae8e96b14,libort_sys-2c30b9a5507bc559,libndarray-541c3e7fe60a0595,libmatrixmultiply-82f75d05f86948e0,libnum_complex-a48b17186caed1f0,libnum_integer-8e22011dfd2f00f7,libnum_traits-c9bd09bea10dfdf2,librawpointer-632d42a28d05c689,libsmallvec-e1aea5a53633896e,libtracing-27bcbd6d575c4bb3,libtracing_core-1f29259c68130f65,libonce_cell-f3864f3383bfa6d4,libtokio-6a40c18244dcad06,libsignal_hook_registry-c402b8ceae4a9dbb,liberrno-1f0976c6c32f6634,libmio-8a8a169e6a68339d,liblibc-0c2f154968cd63cf,libpin_project_lite-492733ea6c61e226,libcudarc-9f668e51870e9d72,liblibloading-a16efcae9348f444,libcfg_if-59c7dc747326a75e}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lcuda" "-lnvcuvid" "-lnvidia-encode" "-lavutil" "-lavformat" "-lavfilter" "-lavdevice" "-lswscale" "-lswresample" "-lavcodec" "-lstdc++" "-ldl" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustcv4rffq/raw-dylibs" "-B<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld" "-fuse-ld=lld" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/usr/local/cuda/lib64" "-L" "/usr/lib/x86_64-linux-gnu" "-L" "/home/calvin/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/d3c01924b801c77ff17d300b24e6dcd46d378348a921a48d96f115f87074fbb1" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/home/calvin/src/rave/target/debug/deps/rave_core-32dd47fe8bf16e0b" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: rust-lld: error: unable to find library -lcuda
          rust-lld: error: unable to find library -lnvcuvid
          rust-lld: error: unable to find library -lnvidia-encode
          collect2: error: ld returned 1 exit status
          

error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-m64" "/tmp/rustcii72NM/symbols.o" "<6 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libtest-*,libgetopts-*,librustc_std_workspace_std-*}.rlib" "/home/calvin/src/rave/target/debug/deps/{librave_engine-e2a827911d1bd822,libthiserror-c31e995d4d2456ab,libffmpeg_sys_next-d0d03443a27bde17,libtokio_util-495535f88a74692e,libfutures_util-d653fb3bb168b6e9,libslab-bf54227bd3b10ac1,libfutures_task-9b426e5331f0de27,libbytes-51256f3823e5c3ab,libfutures_core-e7fa090f587e0d16,libfutures_sink-606fc6487acc0e7d,libort-9f40a4dae8e96b14,libort_sys-2c30b9a5507bc559,libndarray-541c3e7fe60a0595,libmatrixmultiply-82f75d05f86948e0,libnum_complex-a48b17186caed1f0,libnum_integer-8e22011dfd2f00f7,libnum_traits-c9bd09bea10dfdf2,librawpointer-632d42a28d05c689,libsmallvec-e1aea5a53633896e,libtracing-27bcbd6d575c4bb3,libtracing_core-1f29259c68130f65,libonce_cell-f3864f3383bfa6d4,libtokio-6a40c18244dcad06,libsignal_hook_registry-c402b8ceae4a9dbb,liberrno-1f0976c6c32f6634,libmio-8a8a169e6a68339d,liblibc-0c2f154968cd63cf,libpin_project_lite-492733ea6c61e226,libcudarc-9f668e51870e9d72,liblibloading-a16efcae9348f444,libcfg_if-59c7dc747326a75e}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lcuda" "-lnvcuvid" "-lnvidia-encode" "-lavutil" "-lavformat" "-lavfilter" "-lavdevice" "-lswscale" "-lswresample" "-lavcodec" "-lstdc++" "-ldl" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustcii72NM/raw-dylibs" "-B<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld" "-fuse-ld=lld" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/usr/local/cuda/lib64" "-L" "/usr/lib/x86_64-linux-gnu" "-L" "/home/calvin/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/d3c01924b801c77ff17d300b24e6dcd46d378348a921a48d96f115f87074fbb1" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/home/calvin/src/rave/target/debug/deps/rave_ffmpeg-95b970996ea7a25a" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: rust-lld: error: unable to find library -lcuda
          rust-lld: error: unable to find library -lnvcuvid
          rust-lld: error: unable to find library -lnvidia-encode
          collect2: error: ld returned 1 exit status
          

error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-m64" "/tmp/rustchu1yrR/symbols.o" "<6 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libtest-*,libgetopts-*,librustc_std_workspace_std-*}.rlib" "/home/calvin/src/rave/target/debug/deps/{librave_engine-e2a827911d1bd822,libthiserror-c31e995d4d2456ab,libffmpeg_sys_next-d0d03443a27bde17,libtokio_util-495535f88a74692e,libfutures_util-d653fb3bb168b6e9,libslab-bf54227bd3b10ac1,libfutures_task-9b426e5331f0de27,libbytes-51256f3823e5c3ab,libfutures_core-e7fa090f587e0d16,libfutures_sink-606fc6487acc0e7d,libort-9f40a4dae8e96b14,libort_sys-2c30b9a5507bc559,libndarray-541c3e7fe60a0595,libmatrixmultiply-82f75d05f86948e0,libnum_complex-a48b17186caed1f0,libnum_integer-8e22011dfd2f00f7,libnum_traits-c9bd09bea10dfdf2,librawpointer-632d42a28d05c689,libsmallvec-e1aea5a53633896e,libtracing-27bcbd6d575c4bb3,libtracing_core-1f29259c68130f65,libonce_cell-f3864f3383bfa6d4,libtokio-6a40c18244dcad06,libsignal_hook_registry-c402b8ceae4a9dbb,liberrno-1f0976c6c32f6634,libmio-8a8a169e6a68339d,liblibc-0c2f154968cd63cf,libpin_project_lite-492733ea6c61e226,libcudarc-9f668e51870e9d72,liblibloading-a16efcae9348f444,libcfg_if-59c7dc747326a75e}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lcuda" "-lnvcuvid" "-lnvidia-encode" "-lavutil" "-lavformat" "-lavfilter" "-lavdevice" "-lswscale" "-lswresample" "-lavcodec" "-lstdc++" "-ldl" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustchu1yrR/raw-dylibs" "-B<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld" "-fuse-ld=lld" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/usr/local/cuda/lib64" "-L" "/usr/lib/x86_64-linux-gnu" "-L" "/home/calvin/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/d3c01924b801c77ff17d300b24e6dcd46d378348a921a48d96f115f87074fbb1" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/home/calvin/src/rave/target/debug/deps/rave_pipeline-953a6549b334fbd1" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: rust-lld: error: unable to find library -lcuda
          rust-lld: error: unable to find library -lnvcuvid
          rust-lld: error: unable to find library -lnvidia-encode
          collect2: error: ld returned 1 exit status
          

error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-m64" "/tmp/rustc5goKSM/symbols.o" "<6 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libtest-*,libgetopts-*,librustc_std_workspace_std-*}.rlib" "/home/calvin/src/rave/target/debug/deps/{librave_engine-e2a827911d1bd822,libthiserror-c31e995d4d2456ab,libffmpeg_sys_next-d0d03443a27bde17,libtokio_util-495535f88a74692e,libfutures_util-d653fb3bb168b6e9,libslab-bf54227bd3b10ac1,libfutures_task-9b426e5331f0de27,libbytes-51256f3823e5c3ab,libfutures_core-e7fa090f587e0d16,libfutures_sink-606fc6487acc0e7d,libort-9f40a4dae8e96b14,libort_sys-2c30b9a5507bc559,libndarray-541c3e7fe60a0595,libmatrixmultiply-82f75d05f86948e0,libnum_complex-a48b17186caed1f0,libnum_integer-8e22011dfd2f00f7,libnum_traits-c9bd09bea10dfdf2,librawpointer-632d42a28d05c689,libsmallvec-e1aea5a53633896e,libtracing-27bcbd6d575c4bb3,libtracing_core-1f29259c68130f65,libonce_cell-f3864f3383bfa6d4,libtokio-6a40c18244dcad06,libsignal_hook_registry-c402b8ceae4a9dbb,liberrno-1f0976c6c32f6634,libmio-8a8a169e6a68339d,liblibc-0c2f154968cd63cf,libpin_project_lite-492733ea6c61e226,libcudarc-9f668e51870e9d72,liblibloading-a16efcae9348f444,libcfg_if-59c7dc747326a75e}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lcuda" "-lnvcuvid" "-lnvidia-encode" "-lavutil" "-lavformat" "-lavfilter" "-lavdevice" "-lswscale" "-lswresample" "-lavcodec" "-lstdc++" "-ldl" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustc5goKSM/raw-dylibs" "-B<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld" "-fuse-ld=lld" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/usr/local/cuda/lib64" "-L" "/usr/lib/x86_64-linux-gnu" "-L" "/home/calvin/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/d3c01924b801c77ff17d300b24e6dcd46d378348a921a48d96f115f87074fbb1" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/home/calvin/src/rave/target/debug/deps/rave_nvcodec-bb22248d357d3628" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: rust-lld: error: unable to find library -lcuda
          rust-lld: error: unable to find library -lnvcuvid
          rust-lld: error: unable to find library -lnvidia-encode
          collect2: error: ld returned 1 exit status
          

error: could not compile `rave-cuda` (lib test) due to 1 previous error
error: could not compile `rave-core` (lib test) due to 1 previous error
error: could not compile `rave-tensorrt` (lib test) due to 1 previous error
error: could not compile `rave-ffmpeg` (lib test) due to 1 previous error
error: could not compile `rave-nvcodec` (lib test) due to 1 previous error
error: could not compile `rave-pipeline` (lib test) due to 1 previous error
error[E0609]: no field `decode_total_us` on type `&PipelineMetrics`
    --> rave-cli/src/main.rs:1069:18
     |
1069 |                 .decode_total_us
     |                  ^^^^^^^^^^^^^^^ unknown field
     |
help: a field with a similar name exists
     |
1069 -                 .decode_total_us
1069 +                 .encode_total_us
     |

error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-m64" "/tmp/rustc54CIXJ/symbols.o" "<6 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "/home/calvin/src/rave/target/debug/deps/{libtracing_subscriber-584477abf7e08aa1,libsharded_slab-318a32a66894a9ef,liblazy_static-fe0ab9ac01c8b763,libmatchers-3e5de94e133fa009,libregex_automata-e60e77ecdee82457,libregex_syntax-4cc0904c69b790cd,libnu_ansi_term-4dc5b16de19bec69,libthread_local-c3e6249ffce06fef,libtracing_log-4d5200fe820a2b94,liblog-1513b431fbdb5117}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libtest-*,libgetopts-*,librustc_std_workspace_std-*}.rlib" "/home/calvin/src/rave/target/debug/deps/{librave_engine-e2a827911d1bd822,libthiserror-c31e995d4d2456ab,libffmpeg_sys_next-d0d03443a27bde17,libtokio_util-495535f88a74692e,libfutures_util-d653fb3bb168b6e9,libslab-bf54227bd3b10ac1,libfutures_task-9b426e5331f0de27,libbytes-51256f3823e5c3ab,libfutures_core-e7fa090f587e0d16,libfutures_sink-606fc6487acc0e7d,libort-9f40a4dae8e96b14,libort_sys-2c30b9a5507bc559,libndarray-541c3e7fe60a0595,libmatrixmultiply-82f75d05f86948e0,libnum_complex-a48b17186caed1f0,libnum_integer-8e22011dfd2f00f7,libnum_traits-c9bd09bea10dfdf2,librawpointer-632d42a28d05c689,libsmallvec-e1aea5a53633896e,libtracing-27bcbd6d575c4bb3,libtracing_core-1f29259c68130f65,libonce_cell-f3864f3383bfa6d4,libtokio-6a40c18244dcad06,libsignal_hook_registry-c402b8ceae4a9dbb,liberrno-1f0976c6c32f6634,libmio-8a8a169e6a68339d,liblibc-0c2f154968cd63cf,libpin_project_lite-492733ea6c61e226,libcudarc-9f668e51870e9d72,liblibloading-a16efcae9348f444,libcfg_if-59c7dc747326a75e,libclap-bd470012ba6b6a33,libclap_builder-cddd0834747adc80,libstrsim-3b2932c6abfac1bf,libanstream-196fa725a9aeaf7c,libanstyle_query-3b9ba84960a1b1db,libis_terminal_polyfill-19c662f0e405e287,libcolorchoice-9b294091a47662bc,libanstyle_parse-fde5d745843450f5,libutf8parse-6c53baf7f51e19b3,libclap_lex-245ef95ba65cbfd1,libanstyle-c8798a1004545484}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lcuda" "-lnvcuvid" "-lnvidia-encode" "-lavutil" "-lavformat" "-lavfilter" "-lavdevice" "-lswscale" "-lswresample" "-lavcodec" "-lstdc++" "-ldl" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustc54CIXJ/raw-dylibs" "-B<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld" "-fuse-ld=lld" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/usr/local/cuda/lib64" "-L" "/usr/lib/x86_64-linux-gnu" "-L" "/home/calvin/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/d3c01924b801c77ff17d300b24e6dcd46d378348a921a48d96f115f87074fbb1" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/home/calvin/src/rave/target/debug/deps/rave-51dc6698f736312b" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: rust-lld: error: unable to find library -lcuda
          rust-lld: error: unable to find library -lnvcuvid
          rust-lld: error: unable to find library -lnvidia-encode
          collect2: error: ld returned 1 exit status
          

warning: rave-engine@2.0.0: Video Codec SDK libs not found in /home/calvin/src/rave/legacy/third_party/nvcodec. Falling back to CUDA lib dir.
error: could not compile `rave-engine` (bin "rave" test) due to 1 previous error
For more information about this error, try `rustc --explain E0609`.
error: could not compile `rave-cli` (bin "rave") due to 1 previous error
error: could not compile `rave-cli` (bin "rave" test) due to 1 previous error
error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-m64" "/tmp/rustcBgWIJL/symbols.o" "<24 object files omitted>" "-Wl,--as-needed" "-Wl,-Bdynamic" "-lcuda" "-lnvcuvid" "-lnvidia-encode" "-Wl,-Bstatic" "/home/calvin/src/rave/target/debug/deps/libthiserror-c31e995d4d2456ab.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libtest-*,libgetopts-*,librustc_std_workspace_std-*}.rlib" "/home/calvin/src/rave/target/debug/deps/{libffmpeg_sys_next-d0d03443a27bde17,libtokio_util-495535f88a74692e,libfutures_util-d653fb3bb168b6e9,libslab-bf54227bd3b10ac1,libfutures_task-9b426e5331f0de27,libbytes-51256f3823e5c3ab,libfutures_core-e7fa090f587e0d16,libfutures_sink-606fc6487acc0e7d,libort-9f40a4dae8e96b14,libort_sys-2c30b9a5507bc559,libndarray-541c3e7fe60a0595,libmatrixmultiply-82f75d05f86948e0,libnum_complex-a48b17186caed1f0,libnum_integer-8e22011dfd2f00f7,libnum_traits-c9bd09bea10dfdf2,librawpointer-632d42a28d05c689,libsmallvec-e1aea5a53633896e,libtracing-27bcbd6d575c4bb3,libtracing_core-1f29259c68130f65,libonce_cell-f3864f3383bfa6d4,libtokio-6a40c18244dcad06,libsignal_hook_registry-c402b8ceae4a9dbb,liberrno-1f0976c6c32f6634,libmio-8a8a169e6a68339d,liblibc-0c2f154968cd63cf,libpin_project_lite-492733ea6c61e226,libcudarc-9f668e51870e9d72,liblibloading-a16efcae9348f444,libcfg_if-59c7dc747326a75e}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lavutil" "-lavformat" "-lavfilter" "-lavdevice" "-lswscale" "-lswresample" "-lavcodec" "-lstdc++" "-ldl" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustcBgWIJL/raw-dylibs" "-B<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld" "-fuse-ld=lld" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/usr/local/cuda/lib64" "-L" "/usr/lib/x86_64-linux-gnu" "-L" "/home/calvin/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/d3c01924b801c77ff17d300b24e6dcd46d378348a921a48d96f115f87074fbb1" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/home/calvin/src/rave/target/debug/deps/rave_engine-cda1e287cd1c48ca" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: rust-lld: error: unable to find library -lcuda
          rust-lld: error: unable to find library -lnvcuvid
          rust-lld: error: unable to find library -lnvidia-encode
          collect2: error: ld returned 1 exit status
          

warning: rave-engine@2.0.0: Video Codec SDK libs not found in /home/calvin/src/rave/legacy/third_party/nvcodec. Falling back to CUDA lib dir.
error: could not compile `rave-engine` (lib test) due to 1 previous error
[exit_code=101]
```

## cargo build --workspace --release
```text
warning: output filename collision at /home/calvin/src/rave/target/release/rave
  |
  = note: the bin target `rave` in package `rave-engine v2.0.0 (/home/calvin/src/rave/legacy/engine-v2)` has the same output filename as the bin target `rave` in package `rave-cli v0.1.0 (/home/calvin/src/rave/rave-cli)`
  = note: this may become a hard error in the future; see <https://github.com/rust-lang/cargo/issues/6313>
  = help: consider changing their names to be unique or compiling them separately
warning: output filename collision at /home/calvin/src/rave/target/release/rave.dwp
  |
  = note: the bin target `rave` in package `rave-engine v2.0.0 (/home/calvin/src/rave/legacy/engine-v2)` has the same output filename as the bin target `rave` in package `rave-cli v0.1.0 (/home/calvin/src/rave/rave-cli)`
  = note: this may become a hard error in the future; see <https://github.com/rust-lang/cargo/issues/6313>
  = help: consider changing their names to be unique or compiling them separately
warning: rave-engine@2.0.0: Video Codec SDK libs not found in /home/calvin/src/rave/legacy/third_party/nvcodec. Falling back to CUDA lib dir.
   Compiling rave-engine v2.0.0 (/home/calvin/src/rave/legacy/engine-v2)
   Compiling rave-cli v0.1.0 (/home/calvin/src/rave/rave-cli)
error[E0609]: no field `decode_total_us` on type `&PipelineMetrics`
    --> rave-cli/src/main.rs:1069:18
     |
1069 |                 .decode_total_us
     |                  ^^^^^^^^^^^^^^^ unknown field
     |
help: a field with a similar name exists
     |
1069 -                 .decode_total_us
1069 +                 .encode_total_us
     |

For more information about this error, try `rustc --explain E0609`.
error: could not compile `rave-cli` (bin "rave") due to 1 previous error
warning: build failed, waiting for other jobs to finish...
error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-m64" "/tmp/rustcKsWCGV/symbols.o" "<17 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "/home/calvin/src/rave/target/release/deps/{libtracing_subscriber-7640bf62ff80c083,libsharded_slab-e42c1efa74f806f5,liblazy_static-0c097b8b36264202,libmatchers-1cb2f99947f703bd,libregex_automata-19085f0b493a9013,libregex_syntax-2e5f141a05055863,libnu_ansi_term-13c2d559ee3bdbd1,libthread_local-5617de847028b676,libtracing_log-c8fdea0eaf27e5ab,liblog-15d2a70ffc062657,librave_engine-6cb808ae1e8a0456,libthiserror-d2a690ac01a9f0c3,libffmpeg_sys_next-472a76cc637c067a,libtokio_util-5e766dc7628579c2,libfutures_util-599fb81138c38a7c,libslab-7c0bef91908aef00,libfutures_task-b0becfdf069a3526,libbytes-28d54ab712b1694a,libfutures_core-3a8937a3af808ea0,libfutures_sink-41efc8f41d436b55,libort-fa9bf77afd9afcf7,libort_sys-28003e652aeb813d,libndarray-92dadf952e5ba733,libmatrixmultiply-8afc2798f7081a69,libnum_complex-ec4c07655071e5d1,libnum_integer-ce656934f9013fc7,libnum_traits-47866403c48e11ae,librawpointer-4e26aba272e348e9,libsmallvec-e332733c476ab62c,libtracing-a94b14a04e4cf123,libtracing_core-671ef22e2aeffbe1,libonce_cell-d025d0ff782d651d,libtokio-4bed8f39cac93493,libsignal_hook_registry-b4fa76fa79ae2c5e,liberrno-570b14efe4de5300,libmio-46f611833af42cc0,liblibc-d9aa4d85030783ab,libpin_project_lite-327c374a3de23524,libcudarc-abd09843090d2b2c,liblibloading-e4c8c252fbe56ada,libcfg_if-ac635eeb7019897d,libclap-1a745ec9a037e9a9,libclap_builder-64298caeb5fdc3aa,libstrsim-714e2b590cb17cd4,libanstream-74b77c23273a9fcd,libanstyle_query-c266a0b01a93860f,libis_terminal_polyfill-e3ea833d4ff0cbb5,libcolorchoice-45fd6469ff0acb77,libanstyle_parse-33a4f1758d5c01c2,libutf8parse-a4288836510ee242,libclap_lex-506fd8acb1ad3b21,libanstyle-8d875f3323e30f52}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lcuda" "-lnvcuvid" "-lnvidia-encode" "-lavutil" "-lavformat" "-lavfilter" "-lavdevice" "-lswscale" "-lswresample" "-lavcodec" "-lstdc++" "-ldl" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustcKsWCGV/raw-dylibs" "-B<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld" "-fuse-ld=lld" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/usr/local/cuda/lib64" "-L" "/usr/lib/x86_64-linux-gnu" "-L" "/home/calvin/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/d3c01924b801c77ff17d300b24e6dcd46d378348a921a48d96f115f87074fbb1" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/home/calvin/src/rave/target/release/deps/rave-7711b1a6fb0ec90b" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-Wl,-O1" "-Wl,--strip-debug" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: rust-lld: error: unable to find library -lcuda
          rust-lld: error: unable to find library -lnvcuvid
          rust-lld: error: unable to find library -lnvidia-encode
          collect2: error: ld returned 1 exit status
          

warning: rave-engine@2.0.0: Video Codec SDK libs not found in /home/calvin/src/rave/legacy/third_party/nvcodec. Falling back to CUDA lib dir.
error: could not compile `rave-engine` (bin "rave") due to 1 previous error
[exit_code=101]
```

## Guardrails
- No existing Rust source files were edited.
- Added missing workspace manifests and compatibility shim crate roots only.
- No pipeline behavior codepaths were intentionally changed.
- No crate refactors beyond manifest/wrapper restoration.
- No CLI flag/UX semantics were intentionally changed.
