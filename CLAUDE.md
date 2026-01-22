# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VideoForge is a local-first, deterministic visual enhancement engine for professional image and video upscaling. It combines:
- **Rust** (Tauri 2.0) for orchestration, process lifecycle, and the video pipeline
- **Python** (PyTorch/RealESRGAN) for AI inference in a separate sidecar process
- **React/TypeScript** (Vite) for the UI with BlueprintJS and react-mosaic tiled layout
- **FFmpeg** for video decoding/encoding (H.264 NVENC)

Core philosophy: ML models are implementation tools, not decision-makers. Determinism, user control, and preview-before-commit are primary constraints.

## Build Commands

```bash
# Install UI dependencies (required first time)
npm run ui-install

# Development (launches Tauri with Vite hot-reload)
npm run dev

# Production build
npm run build

# Run Rust tests
cd src-tauri && cargo test

# Type-check the UI
cd ui && npx tsc --noEmit
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ React/TypeScript UI (ui/src/)                       │
│ Tauri invoke() calls, Zustand state                 │
└──────────────────┬──────────────────────────────────┘
                   │ Tauri IPC
                   ▼
┌─────────────────────────────────────────────────────┐
│ Rust Backend (src-tauri/src/)                       │
│ Pipeline coordination, SHM ring buffer, FFmpeg/     │
│ Python process spawn/kill, Zenoh pub/sub            │
└──────┬──────────────────────────┬───────────────────┘
       │                          │
       ▼                          ▼
   FFmpeg (decode/encode)    Python sidecar (python/)
                             RealESRGAN inference
```

**IPC Pattern**: Zero-copy frame passing via shared memory (memmap2). 3-slot ring buffer allows parallel decode/infer/encode stages. Zenoh MessagePack for Rust↔Python signaling.

## Key Files

**Rust Backend (src-tauri/src/)**
- `lib.rs` - Main Tauri setup, all exposed commands (`upscale_request`, `export_request`, `check_engine_status`, `install_engine`, `get_models`)
- `video_pipeline.rs` - FFmpeg decoder/encoder abstractions, video probing
- `shm.rs` - VideoShm struct for ring buffer SHM management
- `edit_config.rs` - EditConfig (trim, crop, color, rotation)

**React Frontend (ui/src/)**
- `App.tsx` - Main component with Mosaic layout
- `components/InputOutputPanel.tsx` - File picker, model selection
- `components/PreviewPanel.tsx` - Video/image preview with crop overlay
- `Store/useJobStore.tsx` - Zustand job queue state

**Python Worker (python/)**
- `shm_worker.py` - Main inference loop, Zenoh subscriber
- `upscale_engine.py` - Core upscaling logic with tiling

## Data Flow (Video Upscale)

1. FFmpeg decodes input → raw RGBA frames
2. Rust writes frames to SHM input slot
3. Rust signals Python via Zenoh
4. Python processes frame via RealESRGAN GPU
5. Python writes result to SHM output slot
6. Rust reads SHM output → FFmpeg stdin
7. FFmpeg encodes → final MP4 (NVENC)

## Platform Notes

- Windows-primary: Python runtime resolves to `AppData/Local/VideoForge/python/`
- Assumes NVIDIA GPU with NVENC support
- FFmpeg and FFprobe must be in PATH
- Model weights scanned from `weights/` directories relative to installation
