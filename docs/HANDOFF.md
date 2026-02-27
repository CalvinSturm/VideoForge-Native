# VideoForge — Project Handoff

> Generated on: 2026-02-26
> Current Phase: Codebase Optimization (P0 tasks)

---

## 📌 Context

Following a structural audit of the VideoForge codebase, we identified a prioritised list of
P0/P1/P2 optimizations spanning performance, architecture, reliability, and build/DX.
We began executing the **P0 (Highest Priority)** tasks focused on reliability and dead-code removal.

**Reference documentation produced during the audit:**

| Document | Purpose |
|----------|---------|
| [CODEBASE.md](file:///C:/Users/Calvin/Desktop/VideoForge1/CODEBASE.md) | Comprehensive architectural map of the full stack (Rust, Python, React, Native GPU, IPC) |
| [ARCHITECTURE_GUIDE.md](file:///C:/Users/Calvin/Desktop/VideoForge1/ARCHITECTURE_GUIDE.md) | Contributor-facing runtime model, data-flow diagrams, and operational guide |

---

## ✅ Completed

### 1. TypeScript Error Resolution (P0)

Fixed all 30 component-level TypeScript errors that were causing a broken `npx tsc --noEmit` build:

| File | Errors | Fix |
|------|--------|-----|
| `Timeline.tsx` | 7 | Undefined-safety in `parseTimeInput` logic and `getRulerTicks` `.find()` methods. Added `?? 0` fallbacks and non-null assertions. |
| `PreviewPanel.tsx` | 2 | Implicit `any` parameters on `onSeek` and `onTrimChange` callbacks. Added explicit types. |
| `ToggleGroup.tsx` | 20+ | Removed ~320 lines of corrupted file concatenation — a duplicate `Timeline` component and inline `CropTool` were accidentally pasted below the main export, causing duplicate identifier errors. |
| `VideoEngine.tsx` | 1 | Removed an old 22-line prototype definition that collided with the actual typed export. |
| `useTauriEvents.ts` | — | Fixed `exactOptionalPropertyTypes` collisions by wrapping `\|\|` checks with explicit `?? 0` fallbacks. |

**Status:** `npx tsc --noEmit` now passes for all UI components. Only environment config type-stubs remain (CSS modules, Node types).

---

### 2. Rewire `model_manager.py` to Extracted Subpackages (P0) ✅

Replaced all inline definitions in `model_manager.py` with imports from the already-extracted `architectures/` and `loaders/` subpackages:

* **1687 → 782 lines** (−905 lines of dead inline code removed)
* 9 call-sites updated: `_build_rcan` → `build_rcan`, `_load_via_spandrel` → `load_via_spandrel`, etc.
* `RCAN`, `EDSR`, `remap_edsr_keys` re-exported for `shm_worker.py` backward compatibility
* `python -m py_compile model_manager.py` passes

### 3. Extract `App.tsx` Hooks (P1 #7) ✅

Extracted the 945-line monolithic `App.tsx` into three focused hooks:

* **`useVideoState`** (82 lines) — video/edit state, input handling, `VideoState` memoization
* **`useRaveIntegration`** (155 lines) — RAVE environment, error parsing, CLI arg builders
* **`useUpscaleJob`** (262 lines) — upscale/export/validate/preview job orchestration
* Moved `ModelInfo`, `RaveCommandJson`, `RaveEnvironmentJson`, `RaveErrorPayload` to `types.ts`
* **App.tsx: 945 → 330 lines** (−615 lines, 65% reduction)
* `npx tsc --noEmit` passes (zero new errors)

### 4. Harden Batch Inference Pipeline (P1 #3) ✅

The batch plumbing (`_frame_loop` → `_collect_ready_slots` → `_process_batch` → `inference_batch`) already existed with `MAX_BATCH_SIZE=3`. This task hardened it for production:

| Change | Detail |
|--------|--------|
| `--batch-size` CLI flag | Users can now control batch size at startup (1=disable, max=ring_size) |
| Batch telemetry | Frame loop logs avg batch size and frame count every 100 frames |
| Dynamic OOM recovery | `inference_batch` halves `MAX_BATCH_SIZE` on CUDA OOM instead of just logging |
| Redundant SHM re-reads | `_process_batch` reuses cached `inputs_rgb[]` for research layer + spatial map instead of re-reading from mmap (saves 2 copies per batch frame) |
| Unit tests | 15 tests in `test_batch_inference.py` covering `inference_batch`, `_collect_ready_slots`, OOM fallback, and CLI parsing |

**Verification:** `python -m py_compile python/shm_worker.py` passes. All 15 tests pass.

### 5. Enable Strict TypeScript (P1 #15) ✅

`strict: true` was already enabled in `tsconfig.json`. Fixed the remaining 7 environment type-stub errors that prevented `npx tsc --noEmit` from passing:

| Change | Detail |
|--------|--------|
| `@types/node` installed | Provides types for `process.env`, `path`, `__dirname` used by `modelClassification.ts` and `vite.config.ts` |
| `tsconfig.json` `types: ["node"]` | Includes Node type definitions in compilation |
| `src/env.d.ts` created | Ambient CSS module declarations (`*.css`) + Vite client reference |

**Status:** `npx tsc --noEmit` now passes with **zero errors** (down from 7).

---

### Recent Completions

* **Task #8: Split `AIUpscaleNode.tsx`** (Architecture P2) ✅
  * Extracted 5 sub-components into `upscale/` directory: `CollapsibleSection`, `ArchitectureCard`, `ScaleToggle`, `PipelineToggle`, `ResolutionPresetButton`.
  * Added `IconWarning` to shared `panel/Icons.tsx`.
  * **AIUpscaleNode.tsx: 974 → 619 lines** (−355 lines, 36% reduction).
  * `npx tsc --noEmit` passes with zero errors.
* **Task #9: Extract Inference Stages from `shm_worker.py`** (Architecture P2) ✅
  * Created `inference_engine.py` with `inference()`, `inference_batch()`, `PreallocBuffers`, `PinnedStagingBuffers`, `configure_precision()`, and `enforce_deterministic_mode()`.
  * Removed 231 lines of redundant shadowed code from `shm_worker.py` (**2057 → 1826 lines**).
  * `shm_worker.py` re-exports all extracted symbols for backward compatibility — existing tests and `cuda_streams.py` continue to work unchanged.
  * `python -m py_compile` passes for both `shm_worker.py` and `inference_engine.py`.
* **Task #4: CUDA Stream Pipelining** (Optimization P2)
  * Created `cuda_streams.py` with `CudaStreamPipeline` class — double-buffered transfer/compute overlap using 2 CUDA streams + event synchronization.
  * Integrated stream-pipelined path into `_process_batch()` in `shm_worker.py`, gated by `--cuda-streams` CLI flag.
  * Auto-enables `--pinned-memory` when `--cuda-streams` is used (required for async DMA).
  * Pipeline automatically refreshes on model reload via `update_model()`.
  * 10 unit tests in `test_cuda_streams.py` covering CPU rejection, GPU integration, pipeline reuse, telemetry, and CLI parsing.
* **Task #6: FFmpeg Hardware Pipeline** (Optimization P2)
  * Enhanced `NvencCapabilities` probe detecting H.264, HEVC, and AV1.
  * Added `-hwaccel_output_format cuda` to `VideoDecoder` for GPU-resident frames.
  * Implemented 5-second first-frame timeout validation for NVDEC with automatic software fallback to catch unsupported codecs.
  * Added automatic internal codec selection between `av1_nvenc`, `hevc_nvenc`, and `h264_nvenc` based on resolution, format, and capabilities.
* **Task #5: Pinned Memory for SHM** (Optimization P2)
  * Created `PinnedStagingBuffers` class in `shm_worker.py`.
  * Added optional `--pinned-memory` flag to `AIWorker`.
  * Bypassed host-pageable bottlenecks in `inference()` and `inference_batch()`.

---

## ⏸️ Next Immediate Step — P2: Pick from Split AIUpscaleNode, Type-safe IPC, or Extract inference stages

## 🚀 Optimization Roadmap

### 🔥 Performance (Highest Impact)

| # | Task | Effort | Impact | Details |
|---|------|--------|--------|---------|
| 3 | ~~**Batch inference in SHM frame loop**~~ | ~~4h~~ | ~~20–40% throughput~~ | ✅ Done — `_frame_loop` → `_collect_ready_slots` → `_process_batch` → `inference_batch` with `MAX_BATCH_SIZE=3`, `--batch-size` CLI flag, OOM recovery, and 15 unit tests. |
| 4 | ~~**CUDA stream pipelining**~~ | ~~8h~~ | ~~30% latency~~ | ✅ Done — `CudaStreamPipeline` class in `cuda_streams.py`, double-buffered transfer/compute overlap, `--cuda-streams` CLI flag. |
| 5 | ~~**Pinned memory for SHM**~~ | ~~4h~~ | ~~2× CPU↔GPU copy~~ | ✅ Done — `PinnedStagingBuffers` class + `--pinned-memory` CLI flag |
| 6 | ~~**FFmpeg hardware pipeline**~~ | ~~6h~~ | ~~Avoids CPU round-trips~~ | ✅ Done — Enhanced `NvencCapabilities` probe, `-hwaccel_output_format cuda`, automatic codec selection, and robust fallback. |

### 🏗️ Architecture (Medium Effort, High Payoff)

| # | Task | Effort | Impact | Details |
|---|------|--------|--------|---------|
| 7 | ~~**Extract `App.tsx` state into hooks**~~ | ~~3h~~ | ~~Maintainability + perf~~ | ✅ Done — Extracted `useVideoState`, `useRaveIntegration`, `useUpscaleJob`, `useTauriEvents` hooks (945 → 388 lines). |
| 8 | ~~**Split `AIUpscaleNode.tsx`**~~ | ~~2h~~ | ~~Consistency~~ | ✅ Done — 5 sub-components extracted to `upscale/` directory (974 → 619 lines, −36%). |
| 9 | ~~**Extract inference stages from `shm_worker.py`**~~ | ~~4h~~ | ~~Modularity~~ | ✅ Done — Created `inference_engine.py`, removed 231 lines of redundant code from `shm_worker.py` (2057 → 1826 lines). |
| 10 | ~~**Type-safe IPC**~~ | ~~4h~~ | ~~Eliminates schema drift~~ | ✅ Done — Added `CommandKind` constants + typed payload structs on both Rust & Python. Replaced all `RequestEnvelope::new` with `RequestEnvelope::typed(kinds::*, ...)`. |
| 11 | ~~**Lazy research layer**~~ ✅ | 2h | Faster startup | Done — deferred `research_layer` import to first use via `_load_research_layer()` accessor. UI hidden by default (`showResearchParams=false`). |

### 🛡️ Reliability

| # | Task | Effort | Impact | Details |
|---|------|--------|--------|---------|
| 12 | ~~**Error boundaries around panels**~~ | ~~1h~~ | ~~Fault isolation~~ | ✅ Done — Created `PanelErrorBoundary.tsx` wrapping all 4 mosaic panels. Shows error + retry button instead of crashing the app. |
| 13 | ~~**SHM corruption detection**~~ | ~~2h~~ | ~~Data integrity~~ | ✅ Done — Added `transition_slot_state` CAS helper. All 6 slot transitions in decoder/poll/encoder now CAS-guarded with tracing warnings on violations. |
| 14 | ~~**Graceful `event_sync.py` degradation**~~ | ~~1h~~ | ~~Observability~~ | ✅ Done — Added `EventSyncMetrics` dataclass with degradation tracking (timestamps, reasons, counters). Enhanced `EventSync` and `AIWorker` inline methods with wait/signal/polling counters and periodic diagnostic logging in `_frame_loop`. 19 unit tests in `test_event_sync.py`. |

### 📦 Build & DX

| # | Task | Effort | Impact | Details |
|---|------|--------|--------|---------|
| 15 | ~~**Enable `strict: true` in `tsconfig.json`**~~ ✅ | 2h | Catches bugs at compile time | Done — already enabled (see completed item #5). `npx tsc --noEmit` passes with zero errors. |
| 16 | ~~**Add Python `pyproject.toml`**~~ ✅ | 30m | Reproducible envs | Done — `python/pyproject.toml` created with PEP 621 metadata and pinned deps. |
| 17 | ~~**Unused import sweep**~~ ✅ | 30m | Clean build | Done — removed 4 unused imports (`random`, `tempfile`, `traceback` in `shm_worker.py`; `traceback` in `research_layer.py`). TSX components were clean. |
| 18 | ~~**Consolidate CI test coverage**~~ ✅ | 1h | SHM regression gate | Done — added `--shm-roundtrip` smoke test step to `gpu-rave` CI lane. |

---

## 📊 Priority Summary

| Priority | # | Task | Est. Effort | Impact |
|----------|---|------|-------------|--------|
| 🔴 P0 | 2 | Rewire `model_manager.py` to subpackages | 1h | Clean build, −500 lines, faster module load |
| 🟡 P1 | 3 | Batch inference | 4h | 20–40% throughput gain |
| 🟡 P1 | 7 | Extract `App.tsx` hooks | 3h | Maintainability + fewer re-renders |
| 🟡 P1 | 15 | Enable strict TypeScript | 2h | Catches bugs at compile time |
| 🟢 P2 | 4 | CUDA stream pipelining | 8h | 30% latency reduction |
| 🟢 P2 | 8 | Split `AIUpscaleNode.tsx` | 2h | Consistent panel pattern |
| 🟢 P2 | 10 | Type-safe IPC | 4h | Eliminates schema drift |

> **Recommended next action:** Complete P0 #2 (rewire `model_manager.py`), then move to P1 #3 (batch inference) for the biggest performance win.
