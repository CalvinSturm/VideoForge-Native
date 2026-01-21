docs/PLAYBACK_SYNC_CONTRACT.md
# Playback Synchronization Contract

## Overview
This document defines the strict synchronization requirements for VideoForge's split-view and comparison playback modes. Violating this contract destroys user trust in the visual enhancement results.

## 1. Frame Parity
**Requirement:** The Source view and Result view must display exactly the same temporal frame index at all times.
- **Acceptable Variance:** 0ms.
- **Failure Mode:** If the Result frame for time `T` is not ready, the Source frame must pause at `T` until the Result is available. Use a loading spinner on the Result side; never let Source run ahead.

## 2. Seek Determinism
**Requirement:** Clicking the timeline at timestamp `X` must update both views to timestamp `X` in the same render cycle.
- **Implementation:** React state `currentTime` drives both video elements.
- **Event Loop:**
  1. User seeks to `T`.
  2. `videoState.currentTime` updates to `T`.
  3. Source `<video>` seeks to `T`.
  4. Result `<video>` seeks to `T`.
  5. Playback resumes only when both emit `onSeeked`.

## 3. Trim Bounds
**Requirement:** Playback must never occur outside the `[TrimStart, TrimEnd]` interval.
- **Looping:** When `currentTime >= TrimEnd`, the engine must immediately reset `currentTime` to `TrimStart`.
- **Constraint:** It is illegal for `currentTime` to be `< TrimStart` or `> TrimEnd` during active playback.

## 4. Playback Rate
**Requirement:** Variable playback rates (0.5x, 2x) must apply atomically to both players.
- **Synchronization:** Setting `playbackRate` on the source must trigger the exact same setting on the result target immediately.

## 5. Drift Correction
**Requirement:** A background interval (every 250ms) must check the delta between `Source.currentTime` and `Result.currentTime`.
- **Threshold:** If `delta > 0.04s` (approx 1 frame @ 25fps), pause both players, sync to the Source timestamp, and resume.

## 6. Audio Routing
**Requirement:** Only the **Source** audio track is audible by default.
- **Result Audio:** Muted by default to prevent phasing/echo.
- **Toggle:** User may toggle which track is active, but never both simultaneously.