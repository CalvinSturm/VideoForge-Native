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
- the earlier residual issue was smaller: a couple black frames / slight glitches that survived into both the dumped encoded stream and the final MP4
- the latest direct residual recheck did not reproduce black frames, so the residual issue currently appears resolved or at least non-reproducible on the latest rerun
- the remaining work is now confirmation and documentation rather than an active catastrophic corruption hunt

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
- latest direct residual recheck output:
  - `artifacts/ui_direct_residual_202603157.mp4`
- in that latest rerun, user-reported black frames did not reproduce
- that rerun suggests the residual black-frame issue may already be resolved by the later direct-path changes, or was intermittent/run-specific
- later confirmation output:
  - `artifacts/ui_direct_residual_202603158.mp4`
- that later rerun did reproduce a black-frame interval:
  - `0.0834168 -> 0.125125`
- later longer-clip confirmation:
  - user-ran an approximately 8-second clip with no black frames reproduced
- later longer-clip confirmation:
  - user-ran an approximately 17-second clip
  - first couple frames were black
- current best characterization is therefore:
  - the major corruption bug is fixed
  - the remaining residual black-frame issue is intermittent / timing-sensitive rather than a stable always-on failure
  - the residual issue appears biased toward startup / first frames of the run

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
2. First confirm whether the residual issue is still reproducible on the latest code.
3. If it reproduces again, re-run the residual direct run with targeted upstream dumps:
   - `VIDEOFORGE_NVENC_DEBUG_DUMP_FRAMES=48`
   - `VIDEOFORGE_PIPELINE_DEBUG_DUMP=1`
   - optionally `VIDEOFORGE_POSTPROCESS_KERNEL_DEBUG_DUMP=1` if kernel-input correlation is needed
4. Compare the specific residual black-frame indices against:
   - preprocess output
   - postprocess NV12 output
   - NVENC handoff NV12
   - dumped encoded bitstream
5. Determine whether the black frames first appear:
   - before postprocess completion
   - in postprocess output but not preprocess
   - or only once the frame reaches the NVENC handoff
6. Focus the next code investigation on upstream direct-path continuity for those frame indices:
   - frame metadata / `frame_index` propagation
   - recycled surface reuse
   - postprocess output integrity
   - any direct-path stage where a valid frame could be replaced by a near-empty / black surface

If the issue does not reproduce across one or two fresh confirmation runs:

- treat the direct-native corruption investigation as effectively resolved
- keep the stage-specific debug switches available, but avoid more invasive changes unless the issue returns

If the issue reproduces only sporadically across mixed-length clips:

- treat it as an intermittent synchronization / ownership / reuse problem until disproven
- avoid broad architectural changes without a fresh bad repro and stage-correlated dumps
- bias investigation toward startup / first-frame behavior if the bad repro again shows black frames near the beginning of the output

## Next Suggested Direction

Use the latest clean rerun as the new baseline and confirm stability before doing more invasive debugging.

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

Latest known clean recheck:

- output MP4: `artifacts/ui_direct_residual_202603157.mp4`
- user-reported result: no black frames reproduced
- longer-clip follow-up:
  - approximately 8 seconds
  - user-reported result: no black frames reproduced
- longer-clip follow-up:
  - approximately 17 seconds
  - user-reported result: first couple frames were black

Primary comparison goal if the issue returns:

- identify whether the residual black frames first appear in:
  - preprocess RGB
  - postprocess NV12
  - NVENC handoff NV12
  - encoded HEVC packet decode

Suggested frame targets from the last known bad residual repro:

- frame `0`
- frame `5`

Those correspond to the two black intervals already confirmed in both the dumped HEVC stream and final MP4:

- `0.000000 -> 0.041708`
- `0.208542 -> 0.250250`

If the latest clean behavior holds on another confirmation run:

- update this handoff to mark the residual black-frame issue resolved
- keep the packet-aware demux/mux path as the final fix set

If the next bad repro again shows black frames at the beginning only:

- prioritize frame `0`, `1`, and `2`
- treat startup / warmup / first-surface ownership as the leading hypothesis

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

## 2026-03-15 Interpretation Correction

Subsequent manual playback review plus `ffmpeg` `blackdetect` changed the interpretation of the later residual-labelled outputs.

Files checked:

- `artifacts/ui_direct_residual_20260314.mp4`
- `artifacts/ui_direct_residual_20260315.mp4`

What was observed:

- the operator did **not** see the residual black-frame intervals in either of those two later runs
- `blackdetect` did **not** report the previously confirmed black spans on either file
- both files therefore appear clean relative to the earlier confirmed failing residual baseline

Updated interpretation:

- `artifacts/ui_direct_residual_20260313.mp4` remains the last confirmed failing residual baseline
- `artifacts/ui_direct_residual_20260314.mp4` appears clean
- `artifacts/ui_direct_residual_20260315.mp4` appears clean
- the fresh dump set under `artifacts/nvdec_debug/ui_direct_residual_20260315_fresh/` should therefore be treated as an instrumented clean run, **not** as a confirmed failing repro

Implication for the March 15 dump analysis:

- do **not** rely on the March 15 dump set as proof that frame `0` or frame `5` were black in that run
- the March 15 dump set still proves the dump plumbing worked in a fresh process with an isolated dump directory
- the March 15 dump set still shows that pipeline-stage dumping is first-frame-only in practice
- but it does **not** currently answer where a failing frame first turns black, because that run was not a confirmed visual repro

## Updated Primary Goal

The main investigation goal is no longer "isolate a deterministic residual black frame in the latest run."

The primary goal is now:

- reproduce the intermittent direct-native residual condition again under tightly controlled conditions

Recommended reproduction discipline:

- always start from a fully closed app and a fresh terminal
- always use an explicit unique `VIDEOFORGE_NVDEC_DEBUG_DUMP_DIR` per run
- keep input clip, model, route, precision, and output settings fixed
- record the exact env vars used for every run
- immediately classify each run as either:
  - confirmed failing visually / by `blackdetect`
  - or clean

Only after a fresh failing run is captured with matching dumps should the next agent return to frame-stage isolation.

## 2026-03-15 Instrumentation Upgrade

To support the next intermittent startup repro, the direct pipeline dump hooks were widened from first-frame-only to first-`N`-frames-per-run:

- `VIDEOFORGE_PIPELINE_DEBUG_DUMP_FRAMES=<N>`
  - dumps the first `N` preprocess RGB frames
  - dumps the first `N` postprocess NV12 frames
- `VIDEOFORGE_POSTPROCESS_KERNEL_DEBUG_DUMP_FRAMES=<N>`
  - dumps the first `N` postprocess-kernel RGB inputs

Compatibility behavior:

- `VIDEOFORGE_PIPELINE_DEBUG_DUMP=1` still works and now means `N=1`
- `VIDEOFORGE_POSTPROCESS_KERNEL_DEBUG_DUMP=1` still works and now means `N=1`

Important implementation note:

- these dump counters now reset at the start of every `UpscalePipeline::run(...)`
- repeated UI runs in the same app process can now produce fresh startup dumps without restarting only because the old single-frame process-global counters happened to be unused

Immediate practical effect:

- the next bad startup repro can capture frame `0`, `1`, `2`, `3`, etc. across preprocess, postprocess-kernel input, and postprocess output in one run
- this removes the previous instrumentation bottleneck where frame `5` could be seen at NVENC handoff but not at earlier pipeline stages

## 2026-03-15 Mid-End Probe Update

A later direct-native repro used this dump set:

- `artifacts/nvdec_debug/ui_direct_midend_probe_20260315_b`

The corresponding final output path recorded by `native_mux_debug.log` was:

- `C:\Users\Calvin\Music\LeakedBB.com_sheesh_17_processed_v2_rave_upscaled.mp4`

What this run proves:

- `native_mux_debug.log` shows a clean packet-aware mux session
- `mux_mode=packet_aware`
- `pts`, `dts`, and `mux_pts` are monotonic through the full run
- `ffmpeg_exit_status=exit code: 0`
- `blackdetect` reported **no** black intervals in:
  - the final MP4
  - the concatenated dumped HEVC stream

Important comparison result:

- the dump window captured only the first `240` encoded frames, while the final MP4 contains `302` frames
- a bounded frame-by-frame comparison of final MP4 vs dumped HEVC over frames `1..240` matched exactly
- per-frame PNG SSIM over that bounded `1..240` comparison was:
  - `count=240 min=1.000000 max=1.000000 avg=1.000000`

Therefore:

- for this run, the final MP4 does **not** diverge from the dumped encoded stream within the captured `240`-frame window
- any visible glitches seen by the operator in this run must be either:
  - already present in the encoded stream itself
  - or located after frame `240`, beyond the current NVENC dump window

## 2026-03-15 Windowed Upstream Dump Controls

To target intermittent glitches away from startup without dumping an entire clip at preprocess/postprocess stages, the direct pipeline now supports a start-frame window for upstream dumps:

- `VIDEOFORGE_PIPELINE_DEBUG_DUMP_START_FRAME=<F>`
  - applies to both preprocess RGB dumps and postprocess NV12 dumps
- `VIDEOFORGE_POSTPROCESS_KERNEL_DEBUG_DUMP_START_FRAME=<F>`
  - applies to postprocess-kernel RGB input dumps

These env vars work with the existing frame-count controls:

- `VIDEOFORGE_PIPELINE_DEBUG_DUMP_FRAMES=<N>`
- `VIDEOFORGE_POSTPROCESS_KERNEL_DEBUG_DUMP_FRAMES=<N>`

Example meaning:

- `VIDEOFORGE_PIPELINE_DEBUG_DUMP_START_FRAME=150`
- `VIDEOFORGE_PIPELINE_DEBUG_DUMP_FRAMES=24`

will dump frames `150..173` rather than frames `0..23`.

Additional implementation note:

- postprocess-kernel input dumps are now frame-indexed in their filenames, e.g. `postprocess_input_00150_*`

## Current Best Next Step

If the operator sees a middle or middle-end glitch on a roughly 10-second clip, the next rerun should:

- set `VIDEOFORGE_NVENC_DEBUG_DUMP_FRAMES` high enough to cover the whole clip
- set the upstream dump start-frame near the suspected glitch band
- keep the upstream dump frame count modest, e.g. `24`

Suggested starting window for a ~10-second clip:

- start frame around `150`
- window size `24`

That should finally allow same-run comparison across:

- `preprocess_<target>*`
- `postprocess_input_<target>*`
- `postprocess_<target>*`
- `nvenc_handoff_<target>*`
- dumped HEVC decode / final MP4

## 2026-03-15 Windowed Probe Result

A subsequent run used:

- dump dir: `artifacts/nvdec_debug/ui_direct_midend_window_20260315`
- final output: `C:\Users\Calvin\Music\10sectest_rave_upscaled.mp4`

Key observations from that run:

- `native_mux_debug.log` is still clean
- `packet_count=601`
- mux timing is monotonic
- `ffmpeg_exit_status=exit code: 0`

But this run also reproduced the residual startup issue again:

- `blackdetect` on the final MP4 reported:
  - `black_start:0 black_end:0.0166667`
  - `black_start:0.05 black_end:0.0666667`
- `blackdetect` on the concatenated dumped HEVC stream reported the exact same two startup black intervals
- startup PNG comparison between final MP4 and dumped HEVC frames `1..8` matched exactly with SSIM `1.000000`

What this proves:

- the recurring startup black frames are **not** introduced by final mux
- they are already present in the dumped encoded stream
- the direct encode output faithfully carries the bad startup frames forward

What this run does **not** prove:

- where those startup black frames first appear upstream

Reason:

- this run intentionally targeted upstream pipeline/kernel dumps at frame window `150..173`
- the startup-bad frames were therefore outside the captured preprocess / postprocess-input / postprocess dump window

Updated practical implication:

- the residual issue is still biased toward startup in at least some failing runs
- the next targeted upstream capture should move the start-frame window back to `0`
- if the operator later sees a genuinely mid-run glitch without startup blacks, then the window can move forward again

## 2026-03-15 Startup Window Result

A later run used:

- dump dir: `artifacts/nvdec_debug/ui_direct_startup_window_20260315`
- final output: `C:\Users\Calvin\Music\LeakedBB.com_sheesh_17_processed_v2_rave_upscaled.mp4`

This run reproduced:

- a startup black frame
- a separate visual glitch later in the clip

What was confirmed from the startup side:

- `blackdetect` on the final MP4 reported:
  - `black_start:0 black_end:0.0333333 black_duration:0.0333333`
- `blackdetect` on the concatenated dumped HEVC stream reported the exact same startup black interval
- mux remained clean:
  - `packet_count=302`
  - monotonic timing in `native_mux_debug.log`
  - `ffmpeg_exit_status=exit code: 0`

Most important isolation result:

- frame `0` is already black in **preprocess RGB**
- frame `0` is also black in:
  - postprocess-kernel input RGB
  - postprocess NV12 output
  - NVENC handoff NV12

Measured with `ffmpeg` `blackframe` on the dumped images:

- `preprocess_00000_*` -> `pblack:100`
- `postprocess_input_00000_*` -> `pblack:100`
- `postprocess_00000_*` -> `pblack:100`
- `nvenc_handoff_00000_*` -> `pblack:100`

Frames `1` and `2` in those same startup dumps were **not** flagged as black by the same check.

Current strongest statement:

- the first confirmed bad stage for the startup black frame is now **no later than preprocess output**
- the remaining unresolved question for startup is whether frame `0` was already black at decode ingress

Implication:

- the next startup-focused repro should enable `VIDEOFORGE_NVDEC_DEBUG_DUMP=1`
- that should finally determine whether frame `0` is already black in the decoded NV12 surface, or whether the first corruption occurs between decode output and preprocess output

Important limitation of this run:

- the separate middle glitch was not isolated upstream, because this run only captured frames `0..11` at preprocess / postprocess stages

## 2026-03-15 NVDEC-Enabled Clean Counterexample

A later rerun used:

- dump dir: `artifacts/nvdec_debug/ui_direct_startup_nvdec_20260315`
- final output: `artifacts/ui_direct_startup_nvdec_20260315.mp4`

This run did **not** reproduce the startup black frame.

What was checked:

- `blackdetect` on the final MP4 returned no black intervals
- `native_mux_debug.log` remained clean:
  - `packet_count=302`
  - monotonic timing
  - `ffmpeg_exit_status=exit code: 0`
- NVDEC frame `0` was dumped and converted from pitched NV12 to a dense NV12 image for inspection
- `blackframe` did **not** flag:
  - decoded `frame_00000`
  - `preprocess_00000`
  - `postprocess_00000`

Implication:

- this run is a clean counterexample with `VIDEOFORGE_NVDEC_DEBUG_DUMP=1` enabled
- it does **not** answer whether the startup-black failure begins in NVDEC or after NVDEC, because the failure did not reproduce
- the remaining required capture is now very specific:
- rerun with the same NVDEC-enabled startup command until the startup black frame reproduces
- then compare decoded `frame_00000` against `preprocess_00000`

## 2026-03-15 NVDEC Windowed Dump Upgrade

The NVDEC dump path now matches the newer pipeline/kernel dump controls.

New env vars:

- `VIDEOFORGE_NVDEC_DEBUG_DUMP_FRAMES=<N>`
- `VIDEOFORGE_NVDEC_DEBUG_DUMP_START_FRAME=<F>`

Behavior:

- `VIDEOFORGE_NVDEC_DEBUG_DUMP=1` still works and remains backward-compatible as a single-frame dump (`N=1`, `F=0`)
- when `VIDEOFORGE_NVDEC_DEBUG_DUMP_FRAMES` is set, NVDEC will dump a frame window starting at `VIDEOFORGE_NVDEC_DEBUG_DUMP_START_FRAME`

Example:

- `VIDEOFORGE_NVDEC_DEBUG_DUMP_FRAMES=4`
- `VIDEOFORGE_NVDEC_DEBUG_DUMP_START_FRAME=5`

will dump decoded frames `5..8`.

Why this matters:

- the latest failing NVDEC-enabled retry did **not** black out frame `0`
- `blackdetect` on `artifacts/ui_direct_startup_nvdec_retry_20260315.mp4` reported:
  - `black_start:0.166667 black_end:0.2 black_duration:0.0333333`
- that maps to frame `5`
- frame `5` was already black in:
  - `preprocess_00005_*`
  - `postprocess_input_00005_*`
  - `postprocess_00005_*`
  - `nvenc_handoff_00005_*`
- frame `6` was clean across those same stages

Updated immediate next step:

- rerun with:
  - `VIDEOFORGE_NVDEC_DEBUG_DUMP_FRAMES=8`
  - `VIDEOFORGE_NVDEC_DEBUG_DUMP_START_FRAME=4`
- keep the startup pipeline/kernel window enabled
- if the same frame-`5` failure reproduces, compare:
  - decoded `frame_00005_*`
  - `preprocess_00005_*`

That next run should finally answer whether frame `5` is already black at NVDEC output or first turns black between NVDEC and preprocess.

## 2026-03-15 `third_party/rave` Reference Comparison

The clean reference path in `third_party/rave` was inspected specifically at the decode → preprocess boundary:

- `third_party/rave/crates/rave-nvcodec/src/nvdec.rs`
- `third_party/rave/crates/rave-pipeline/src/pipeline.rs`

Most important comparison result:

- the strongest corruption-capable mismatch was **not** in demux or mux
- the strongest mismatch was in **decoded NV12 buffer lifetime during preprocess**

Reference `rave` behavior:

- preprocess waits on the decode event
- launches `nv12_to_rgb`
- explicitly calls `GpuContext::sync_stream(&ctx.preprocess_stream)` before recycling the decoded NV12 input buffer

Earlier `engine-v2` behavior:

- preprocess waited on `decode_ready`
- launched `prepare()`
- recorded `preprocess_ready`
- immediately recycled the decoded NV12 input buffer back to the pool

Why that looked suspicious:

- recording `preprocess_ready` is enough to protect downstream inference
- it is **not** enough to make the decoded NV12 input safe for immediate reuse
- if the buffer pool reissued that NV12 allocation while `preprocess_stream` was still reading it, that would create exactly the kind of intermittent early-frame corruption seen in this investigation

Patch applied:

- `engine-v2/src/engine/pipeline.rs`
- preprocess now explicitly calls `GpuContext::sync_stream(&ctx.preprocess_stream)` before `frame.texture.try_recycle(ctx)`

Validation:

- `cargo fmt --manifest-path engine-v2/Cargo.toml`
- `cargo test --manifest-path engine-v2/Cargo.toml --lib`

Current hypothesis after the `rave` comparison:

- the decode→preprocess input-buffer reuse race was the most defensible remaining root-cause candidate in `engine-v2`
- the next required step is a direct-native repro rerun after this patch to see whether the intermittent early black/glitch issue disappears or becomes materially rarer
