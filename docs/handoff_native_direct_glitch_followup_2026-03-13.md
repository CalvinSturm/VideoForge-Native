# Native Direct Glitch Follow-up Handoff

Date: 2026-03-13
Status: active

## Purpose

This handoff captures the current state of the direct native-path corruption investigation after the original visible-bounds fix did not resolve the issue.

## Current Summary

- Native runtime readiness is working again.
- CLI-backed native is clean on the same clip, model, and settings.
- Direct in-process native remains visually glitched.
- The issue is therefore isolated to the direct `engine-v2` path, not the shared native runtime contract, ONNX model, TensorRT runtime, or general native routing.

## Proven Facts

- The frontend/backend readiness contract was repaired by restoring `rave_environment`.
- Native direct runs succeed technically with `engine=native_v2`, `requested=direct`, `executed=direct`, `fallback=false`, and `encoder_mode=nvenc`.
- The same job through native CLI-backed mode works well and produces clean output.
- The original visible-display-vs-coded-dimensions bug was real, but fixing it was not sufficient to solve the remaining direct-path corruption.
- A later experimental NVDEC UV-plane patch was incorrect for this runtime and was reverted.

## Route Comparison Result

Direct native:

- executes successfully
- uses NVENC
- produces glitchy output

CLI-backed native:

- executes successfully
- reports `Engine: Native Via Rave Cli`
- produces clean output

Conclusion:

- the problem is direct-path-specific
- do not treat this issue as resolved
- do not commit the direct-path glitch work as a fix

## Debugging Work Completed

### Readiness and sizing fixes

Files:

- `src-tauri/src/commands/rave.rs`
- `src-tauri/src/lib.rs`
- `src-tauri/src/commands/native_probe.rs`
- `engine-v2/src/codecs/nvdec.rs`

Implemented:

- restored `rave_environment`
- registered the command again in Tauri
- fixed direct-path output sizing to use visible display dimensions instead of coded dimensions
- updated native probe/profile handling to preserve both coded and visible geometry

Result:

- readiness works
- output sizing is more correct
- direct-path corruption still remains

### Stage-by-stage dump instrumentation

Files:

- `engine-v2/src/codecs/nvdec.rs`
- `engine-v2/src/engine/pipeline.rs`
- `engine-v2/src/core/kernels.rs`
- `engine-v2/src/codecs/nvenc.rs`

Debug env flags used:

- `VIDEOFORGE_NVDEC_DEBUG_DUMP=1`
- `VIDEOFORGE_NVENC_DEBUG_DUMP_FRAMES=<N>`
- `VIDEOFORGE_POSTPROCESS_NEUTRAL_CHROMA=1`
- `VIDEOFORGE_NVENC_ALL_INTRA=1`

Implemented dump points:

- first decoded NV12 frame
- first preprocessed RGB frame
- first RGB inference output consumed by postprocess
- first postprocess NV12 frame before encode
- first `N` NVENC handoff NV12 surfaces
- first `N` elementary-stream packets produced by `nvEncLockBitstream`
- per-packet metadata for submitted frame index, locked frame index, packet order, and timestamps

Dump location:

- `artifacts/nvdec_debug/`

Note:

- an earlier version wrote into `src-tauri/artifacts/` and triggered Tauri dev rebuild loops; dumps now target repo-root `artifacts/`.

## What The Dumps Proved

The following stages were inspected and found to be coherent:

- NVDEC decoded NV12
- NV12 -> RGB preprocess output
- RGB inference output consumed by postprocess
- RGB -> NV12 postprocess output
- NVENC submitted NV12 handoff surface
- first encoded elementary-stream packet

This means the issue is not primarily caused by:

- NVDEC surface extraction
- decode/display-area cropping
- preprocess layout assumptions
- inference output interpretation
- RGB -> NV12 postprocess of frame 0
- NVENC direct-resource handoff of frame 0
- muxing of the very first encoded packet

## Latest Multi-Frame Encode Evidence

The encode-side dump window was expanded in `engine-v2/src/codecs/nvenc.rs` so direct-native runs can capture the first `N` NVENC handoff surfaces and bitstream packets with:

- `VIDEOFORGE_NVENC_DEBUG_DUMP_FRAMES=<N>`

### Re-run performed on the actual odd-geometry repro case

Input / model / route:

- `odd_input.mp4` (`406x470`, ~2.0 s, `24000/1001`)
- `weights/2x_SPAN_soft.onnx`
- direct native
- `scale=2`
- `precision=fp16`
- `VIDEOFORGE_NVENC_ALL_INTRA=1`

Artifacts written under:

- `artifacts/nvdec_debug/odd_direct_multiframe_20260313/`
- `artifacts/nvdec_debug/odd_direct_fullencode_20260313/`

### What was validated

1. First 8 odd-input packets:
   - dumped packets `00000` through `00007`
   - dumped handoff surfaces `00000` through `00007`
   - packet metadata showed:
     - `submitted_frame_index == encoded_frame_index` for all dumped packets
     - packet order stayed monotonic
     - no late-frame reorder signal appeared in the locked bitstream metadata

2. Full all-intra odd-input run:
   - dumped all 48 handoff frames and all 48 encoded packets
   - collapsed pitched handoff NV12 surfaces into a dense raw sequence for comparison
   - concatenated dumped HEVC packets into one elementary stream for comparison

3. Frame similarity across the full direct encode segment:
   - `handoff -> dumped bitstream decode` SSIM over all 48 frames stayed high
   - min SSIM: `0.997829`
   - max SSIM: `0.998814`
   - avg SSIM: `0.998628`

4. Mux fidelity spot-checks on later frames:
   - extracted later frames from final `artifacts/odd_direct_fullencode_20260313.mp4`
   - compared against extracted frames from dumped elementary stream
   - later-frame spot checks remained exact for selected late frames (`n=37`, `n=46`, zero-diff SSIM)

### Current conclusion from the expanded dump set

On the current workspace and build, the odd-input direct path does **not** show encode-side corruption in the dumped direct NVENC packets when forced all-intra:

- the pre-encode handoff surfaces remain coherent
- the dumped encoded packets remain coherent through the full 48-frame clip
- the final muxed MP4 matches the dumped encoded stream on spot-checked late frames

That means the earlier hypothesis "later direct NVENC packet generation is where corruption first appears" is **not supported** by the current all-intra odd-input re-run.

## Exact UI Direct Run Comparison

An exact UI repro was then captured with:

- `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1`
- `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`
- `VIDEOFORGE_NVDEC_DEBUG_DUMP=1`
- `VIDEOFORGE_NVENC_DEBUG_DUMP_FRAMES=48`
- no `VIDEOFORGE_NVENC_ALL_INTRA`

Output file:

- `C:\Users\Calvin\Desktop\VideoForge1\odd_input_rave_upscaled.mp4`

Dump directory:

- `artifacts/nvdec_debug/ui_direct_repro_20260313/`

### What this exact run proved

1. The dumped direct elementary stream still tracks the pre-encode handoff closely:
   - `handoff -> dumped bitstream decode` SSIM summary over all 48 frames:
     - min: `0.995698`
     - max: `0.996539`
     - avg: `0.995841`

2. The final UI MP4 from the same run does **not** match the dumped elementary stream:
   - per-frame PNG SSIM against frames extracted from `nvenc_all48.hevc` was extremely poor
   - broad frame SSIM stayed around `~0.27`
   - worst observed frame was frame `06`, `All:0.000785`

3. FFmpeg reported a timestamp-ordering problem while comparing the exact final MP4 against the exact dumped stream:
   - `Application provided invalid, non monotonically increasing dts to muxer in stream 0: 47 >= 47`

4. The dumped packet metadata from the exact UI run shows non-monotonic output timestamps even though frame indices still align:
   - `submitted_frame_index == encoded_frame_index` remained true
   - packet order did not correspond to monotonic `output_pts`
   - examples:
     - packet `00001`: `output_pts=417083`
     - packet `00007`: `output_pts=1`
     - packet `00018`: `output_pts=2`
     - packet `00025`: `output_pts=3`
     - packet `00046`: `output_pts=5`

### Most important narrowing from the exact UI run

For this non-all-intra direct UI repro:

- the direct NVENC handoff surfaces still look coherent
- the dumped direct HEVC elementary stream still looks coherent
- the final UI MP4 diverges sharply from that same dumped HEVC stream

That shifts the most suspicious fault domain from "raw direct NVENC pixels are corrupted" to:

- timestamp / output-order handling across direct NVENC packet delivery
- or the direct streaming mux boundary that turns the dumped Annex B stream into the final MP4

One concrete code clue matches that hypothesis:

- `src-tauri/src/commands/native_streaming_io.rs`
- `StreamingMuxSink::write_packet(...)` currently ignores the encoder-provided `_pts` and `_is_keyframe`
- the sink just pipes raw packets into an FFmpeg copy-mux subprocess

## Mux Log Correlation Result

The direct mux path was then instrumented in:

- `src-tauri/src/commands/native_streaming_io.rs`

The new debug log is written as:

- `artifacts/nvdec_debug/<run>/native_mux_debug.log`

### Exact rerun inspected

Artifacts:

- `artifacts/nvdec_debug/ui_direct_repro_20260313_rerun/native_mux_debug.log`
- matching `artifacts/nvdec_debug/ui_direct_repro_20260313_rerun/nvenc_bitstream_packet_*_hevc.txt`

### What the mux log proved

1. The streaming mux sink is **not** reordering packets on its own:
   - `packet_write_index == packet_index` across the rerun
   - mux packet byte counts matched `bitstream_size`
   - mux `pts` matched dumped `output_pts`

2. The mux sink is faithfully forwarding the same timestamp pattern seen in the dumped NVENC metadata:
   - packet `19`: `pts=7924577`
   - packet `20`: `pts=1`
   - packet `40`: `pts=2`
   - the same wrap / non-monotonic pattern appears in the matching dumped packet metadata

3. FFmpeg warned during mux:
   - `Timestamps are unset in a packet for stream 0. This is deprecated and will stop working in the future. Fix your code to set the timestamps properly`

### Most important narrowing from the mux-log pass

This weakens the theory that the streaming mux sink is inventing bad packet order.

What the rerun now points to instead is:

- direct NVENC packet timing in the normal inter-frame path is already problematic before mux
- the direct mux sink is currently forwarding that timing pattern rather than correcting it
- the remaining likely fault domain is back inside the direct encode output timing path, especially around:
  - `engine-v2/src/codecs/nvenc.rs`
  - `pic_params.inputTimeStamp = frame.pts as u64`
  - `lock_params.outputTimeStamp`

## Updated Best Hypothesis

The remaining issue is now narrower than "direct NVENC is generally corrupting later frames."

Most plausible next targets are:

- timestamp generation / propagation in the direct non-all-intra encode path
- `engine-v2/src/codecs/nvenc.rs`, especially the relationship between submitted frame `pts`, `inputTimeStamp`, and `outputTimeStamp`
- any direct-path encode behavior that causes non-monotonic timestamps while still producing visually coherent elementary-stream packets

The previous hypothesis of "the streaming mux sink is reordering packets incorrectly" is now weakened by the mux-log correlation pass:

- the sink wrote packets in the same order they were dumped
- mux `pts` exactly matched dumped packet `output_pts`
- the non-monotonic timing pattern already existed before the mux boundary

The currently strongest working theory is:

- direct NVENC is producing visually coherent compressed packets
- the catastrophic direct playback bounce was caused by broken direct demux timing before encode and is now fixed by the packet-aware demux port
- the remaining residual issue is smaller: a couple black frames / slight glitches that survive into both the dumped encoded stream and the final MP4
- the current fault domain is now upstream of NVENC packetization and upstream of final mux, likely in the direct path before or at the NVENC handoff surfaces for specific frame indices

## Latest Encode-Side Evidence

The first bitstream packet metadata looked normal:

- `codec_label=hevc`
- `picture_type=3`
- `output_pts=0`
- `bitstream_size=72751`
- `input_width=812`
- `input_height=940`
- `input_pitch=1024`
- `buffer_fmt=1`
- `input_time_stamp=0`

The corresponding dumped elementary stream packet decoded cleanly.

Expanded evidence now adds:

- first 8 dumped odd-input packets stayed aligned with their submitted frame indices
- full 48-frame odd-input all-intra packet dump stayed visually coherent against the pre-encode handoff sequence by SSIM
- later muxed MP4 frames matched late dumped bitstream frames on spot checks
- in the exact non-all-intra rerun, mux-side debug logging showed `packet_write_index`, packet bytes, and `pts` all matched the dumped NVENC packet metadata exactly
- the non-monotonic timestamp pattern therefore originates upstream of the streaming mux sink
- in later direct UI reruns, NVDEC ingest logs already showed wrapped timestamps before encode, e.g. `... 2502498, 1, 417084 ...`
- that same wrapped pattern propagated into NVENC `input_time_stamp` / `output_pts`
- this moved the primary suspect from muxing to the direct demux source in `src-tauri/src/commands/native_streaming_io.rs`
- after the packet-aware demux fix, the catastrophic playback bounce disappeared and direct output became broadly correct
- in the residual direct repro (`artifacts/ui_direct_residual_20260313.mp4`), final MP4 and dumped elementary stream matched on the remaining black-frame intervals
- blackdetect found the same two black spans in both the final MP4 and dumped HEVC stream:
  - `0.000000 -> 0.041708`
  - `0.208542 -> 0.250250`
- handoff-vs-bitstream SSIM for that residual run stayed extremely high over all 48 frames:
  - `count=48 min=0.998746 max=0.999999 avg=0.998909`
- that means the residual black frames are already present in the direct NVENC handoff surfaces / encoded stream path and are not introduced by final muxing

## Validation Status

Commands repeatedly run during this investigation:

- `cd src-tauri && cargo test --workspace`
- `cd src-tauri && cargo test --features native_engine --lib`

Status:

- passing

Known remaining hygiene gap:

- native-feature strict clippy is not yet clean; that is separate from this direct-path corruption bug

## Current Working Tree State

Relevant modified files:

- `engine-v2/src/codecs/nvdec.rs`
- `engine-v2/src/codecs/nvenc.rs`
- `engine-v2/src/core/kernels.rs`
- `engine-v2/src/engine/pipeline.rs`
- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/commands/native_probe.rs`
- `src-tauri/src/commands/rave.rs`
- `src-tauri/src/lib.rs`

Untracked runtime artifacts exist under:

- `artifacts/`
- `src-tauri/artifacts/`

Those are debug outputs, not source changes.

## What Has Been Tried

- restored native readiness command
- fixed visible-vs-coded geometry in native probe and NVDEC
- compared direct native vs CLI-backed native on the same job
- added NVDEC decode dump
- added preprocess RGB dump
- added postprocess-input RGB dump
- added postprocess NV12 dump and plane previews
- added NVENC handoff dump and plane previews
- added first NVENC elementary-stream packet dump
- tried neutral chroma mode
- tried all-intra NVENC debug mode

## What The Next Agent Should Do First

1. Stop spending time on final mux and generic NVENC packet-order theories for this residual issue.
2. Reproduce the residual direct run with targeted upstream dumps:
   - `VIDEOFORGE_NVENC_DEBUG_DUMP_FRAMES=48`
   - `VIDEOFORGE_PIPELINE_DEBUG_DUMP=1`
   - optionally `VIDEOFORGE_POSTPROCESS_KERNEL_DEBUG_DUMP=1` if kernel-input correlation is needed
3. Compare the specific residual black-frame indices against:
   - preprocess output
   - postprocess NV12 output
   - NVENC handoff NV12
   - dumped encoded bitstream
4. Determine whether the black frames first appear:
   - before postprocess completion
   - in postprocess output but not preprocess
   - or only once the frame reaches the NVENC handoff
5. Focus the next code investigation on upstream direct-path continuity for those frame indices:
   - frame metadata / `frame_index` propagation
   - recycled surface reuse
   - postprocess output integrity
   - any direct-path stage where a valid frame could be replaced by a near-empty / black surface

## Next Suggested Direction

Use the residual repro as the new baseline and isolate the first stage that turns black.

Recommended repro env:

- `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1`
- `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`
- `VIDEOFORGE_NVENC_DEBUG_DUMP_FRAMES=48`
- `VIDEOFORGE_PIPELINE_DEBUG_DUMP=1`
- `VIDEOFORGE_NATIVE_MUX_DEBUG=1`
- optionally `VIDEOFORGE_POSTPROCESS_KERNEL_DEBUG_DUMP=1` if postprocess-input correlation is needed

Recommended output/dump targets:

- output MP4: `artifacts/ui_direct_residual_followup.mp4`
- dump dir: `artifacts/nvdec_debug/ui_direct_residual_followup/`

Primary comparison goal:

- identify whether the residual black frames first appear in:
  - preprocess RGB
  - postprocess NV12
  - NVENC handoff NV12
  - encoded HEVC packet decode

Suggested frame targets from the current residual repro:

- frame `0`
- frame `5`

Those correspond to the two black intervals already confirmed in both the dumped HEVC stream and final MP4:

- `0.000000 -> 0.041708`
- `0.208542 -> 0.250250`

If the black frames are already present in preprocess/postprocess outputs:

- investigate upstream frame continuity and surface reuse in `engine-v2/src/engine/pipeline.rs`
- check whether decoded envelopes or recycled output textures are being reused before GPU work completes

If preprocess/postprocess are clean but NVENC handoff is black:

- investigate direct-path output surface ownership / synchronization before encode in `engine-v2/src/codecs/nvenc.rs`

If NVENC handoff is clean and encoded decode is black:

- return to encode-specific investigation, but only for those exact frame indices rather than the whole stream

## Most Important Diagnostic Clues To Preserve

- CLI-backed native is clean; direct native is not.
- Direct frame 0 remains clean through decode, preprocess, inference, postprocess, NVENC handoff, and first encoded packet.
- In the current all-intra odd-input re-run, all 48 dumped direct packets also remained coherent against their pre-encode handoff surfaces.
- In the exact non-all-intra UI repro, the dumped direct elementary stream still remained coherent, but the final UI MP4 diverged sharply from that same stream.
- In the exact rerun with `native_mux_debug.log`, the mux sink forwarded packet order, byte sizes, and `pts` exactly as dumped by NVENC.
- The non-monotonic timing pattern therefore appears to originate upstream of the streaming mux sink, in the direct encode output timing path.
- The direct path has now been ported off the raw/file-backed FFmpeg copy-mux experiment and onto a packet-aware FFmpeg FFI muxer modeled on `third_party/rave/crates/rave-ffmpeg/src/ffmpeg_muxer.rs`.
- `src-tauri` now builds a local FFmpeg accessor shim (`src-tauri/src/ffmpeg_accessors.c`) and uses `ffmpeg-sys-next` directly under the `native_engine` feature for container writes with explicit `pts/dts`.
- The first packet-aware runtime attempt failed early because the muxer was still feeding wrapped NVENC `output_pts` into the container writer even though the direct config uses `b_frames=0`.
- The direct mux path now uses monotonic DTS as the mux PTS when `b_frames=0`, while still logging the raw NVENC `pts` separately as `mux_pts` in `native_mux_debug.log`.
- Engine task-result selection now prefers a real encode/decode failure over a follow-on `ChannelClosed`, so future repros should surface the actual root error if the direct path still fails.
- The direct bitstream source has now also been ported off arbitrary FFmpeg stdout chunking and onto packet-aware FFmpeg demuxing in `src-tauri/src/commands/native_streaming_io.rs`, modeled on `third_party/rave/crates/rave-ffmpeg/src/ffmpeg_demuxer.rs`.
- The direct demux path now opens the input container via FFmpeg APIs, selects the video stream, applies `h264_mp4toannexb` / `hevc_mp4toannexb` when needed, preserves packet boundaries, and rescales real packet timestamps instead of synthesizing one PTS per stdout chunk.
- Validation after the demux port passed:
  - `cargo fmt --manifest-path src-tauri/Cargo.toml`
  - `cargo test --manifest-path src-tauri/Cargo.toml --features native_engine --lib native_streaming_io`
  - `cargo test --manifest-path src-tauri/Cargo.toml --features native_engine --lib`
- After the demux fix, the catastrophic playback bounce is resolved.
- In the residual repro `artifacts/ui_direct_residual_20260313.mp4`, the final MP4 and dumped elementary stream both contain the same two black-frame intervals, so final mux is no longer the fault domain.
- For that same residual repro, handoff-vs-bitstream SSIM remained extremely high (`count=48 min=0.998746 max=0.999999 avg=0.998909`), which moves the remaining fault domain upstream of final mux and away from generic NVENC packet corruption.
- Cleanup after the main fix split the heavy debug switches back into stage-specific env vars:
  - `VIDEOFORGE_NVDEC_DEBUG_DUMP=1` for NVDEC dumps
  - `VIDEOFORGE_NVENC_DEBUG_DUMP_FRAMES=<N>` for NVENC handoff/packet dumps
  - `VIDEOFORGE_PIPELINE_DEBUG_DUMP=1` for preprocess/postprocess surface dumps
  - `VIDEOFORGE_POSTPROCESS_KERNEL_DEBUG_DUMP=1` for postprocess kernel input dumps
  - `VIDEOFORGE_NATIVE_MUX_DEBUG=1` for mux logging

## 2026-03-15 Residual Rerun With Fresh Dump Dir

A fresh direct-native residual rerun was captured with an explicit isolated dump directory and postprocess-kernel dumps enabled.

Run shape:

- direct native
- fresh app process
- `VIDEOFORGE_ENABLE_NATIVE_ENGINE=1`
- `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1`
- `VIDEOFORGE_NVENC_DEBUG_DUMP_FRAMES=48`
- `VIDEOFORGE_PIPELINE_DEBUG_DUMP=1`
- `VIDEOFORGE_NATIVE_MUX_DEBUG=1`
- `VIDEOFORGE_POSTPROCESS_KERNEL_DEBUG_DUMP=1`
- `VIDEOFORGE_NVDEC_DEBUG_DUMP_DIR=.../artifacts/nvdec_debug/ui_direct_residual_20260315_fresh`

Output:

- `artifacts/ui_direct_residual_20260315.mp4`

Dump directory:

- `artifacts/nvdec_debug/ui_direct_residual_20260315_fresh/`

### What this rerun produced

The fresh dump directory contains:

- `nvenc_handoff_00000` through `nvenc_handoff_00047`
- `nvenc_bitstream_packet_00000` through `nvenc_bitstream_packet_00047`
- `preprocess_00000_*`
- `postprocess_00000_*`
- one `postprocess_input_*`
- `native_mux_debug.log`

### What this rerun proves

1. The direct dump path is working again when forced into a fresh dump directory.
2. The pipeline-stage dump path is still effectively single-frame only:
   - only `preprocess_00000_*` was written
   - only `postprocess_00000_*` was written
   - only one `postprocess_input_*` dump was written
3. Full-frame NVENC handoff and bitstream dumps were still captured for all 48 frames.

### Updated stage-of-failure narrowing

Frame `0`:

- this rerun includes `preprocess_00000_*`
- this rerun includes `postprocess_00000_*`
- this rerun includes `nvenc_handoff_00000_*`
- this rerun includes `nvenc_bitstream_packet_00000_*`

Conclusion for frame `0`:

- frame `0` is already black by postprocess output
- frame `0` is also black by NVENC handoff
- therefore the first black stage for frame `0` is **not later than postprocess output**
- this rerun does **not** prove whether frame `0` first becomes black in preprocess output, postprocess kernel input, or during RGB->NV12 postprocess itself

Frame `5`:

- this rerun includes `nvenc_handoff_00005_*`
- this rerun includes `nvenc_bitstream_packet_00005_*`
- this rerun does **not** include `preprocess_00005_*`
- this rerun does **not** include `postprocess_00005_*`
- this rerun does **not** include a frame-indexed `postprocess_input_00005_*`

Conclusion for frame `5`:

- frame `5` is definitely already black by NVENC handoff
- frame `5` also survives into the dumped HEVC packet path
- this rerun still does **not** determine whether frame `5` first becomes black in preprocess output, postprocess input, or postprocess output

### Most important implication

The investigation is now blocked more by instrumentation granularity than by fault-domain uncertainty.

Current direct-path dump behavior is sufficient to prove:

- frame `0` is black by postprocess output
- frame `5` is black by NVENC handoff
- mux is not the residual fault domain
- generic NVENC packet corruption is not the residual fault domain

But current pipeline debug dumping is still not sufficient to answer the key remaining question for frame `5`:

- whether the first black stage is preprocess RGB
- postprocess kernel input RGB
- or postprocess NV12 output

### Next required direction

The next investigation step should stay focused on the direct native path and make the pipeline dump path frame-selective rather than first-frame-only, so the same rerun can capture:

- `preprocess_00005_*`
- `postprocess_input_00005_*`
- `postprocess_00005_*`

Until that is done, the strongest current statement is:

- frame `0` turns black no later than postprocess output
- frame `5` turns black no later than NVENC handoff
