# Deterministic AI Upscaling – VideoForge v1.0

**Editor-grade image and video enhancement with user-controlled model options.**

This project provides a high-performance, local AI upscaling worker for professional pipelines, emphasizing reproducibility, efficiency, and flexibility.

---

## Core Philosophy

- **Privacy First:** All processing happens locally on the user’s GPU.
- **Zero-Copy Pipeline:** Video frames move between decoder, AI engine, and encoder via **Shared Memory (SHM)** to avoid unnecessary serialization.
- **User Authority:** Users can control all pipeline steps, including trimming, cropping, resizing, pausing, resuming, and stopping processes.
- **Flexible Model Selection:** Users choose which AI model to run based on workflow needs.

---

## Available Models

VideoForge supports multiple upscaling models. Each model runs locally and can be selected per job:

- **RCAN / EDSR** – Deterministic models focused on structural fidelity and reproducibility.
- **RealESRGAN** – GAN-based model designed for high-quality perceptual upscaling. Slight visual variation may occur.

Users can switch models at any point in the pipeline.

---

## High-Level Overview

**VideoForge** is a local-first, high-performance visual enhancement engine for images and videos.

### Functional Architecture

**1. Orchestration (Rust / Tauri)**
- Manages Python worker lifecycle
- Allocates the SHM ring buffer
- Controls FFmpeg decode/encode
- Computes ETA and emits progress events

**2. AI Engine (Python / PyTorch)**
- Runs as a standalone process
- Loads selected models
- Reads frames from SHM, performs upscaling, writes results back
- Communicates via **Zenoh** IPC for low latency

**3. UI Layer (React / TypeScript)**
- Mosaic-style professional layout
- Job queue for batch processing
- Interactive preview with cropping and trimming controls

### Video Data Flow

1. Rust spawns Python worker and performs a handshake via Zenoh
2. Rust requests SHM setup from Python worker
3. Processing loop:
   - **Decode:** FFmpeg → Rust → SHM input slot
   - **Inference:** Python reads SHM → selected model → writes SHM output slot
   - **Encode:** Rust reads SHM → FFmpeg encoder
4. Cleanup: SHM files removed, worker terminated

---

## UI / UX Design

- **Left Panel:** Inputs, outputs, model selection, resolution/FPS toggle
- **Center Panel:** Video viewport with crop overlay and timeline trimming
- **Right Panel:** Job queue with ETA and progress bars
- **Footer:** GPU status, technical details, panic kill switch

**Notable Features**
- Real-time ETA estimation
- Crash recovery via watchdog and kill switch
- Visual progress feedback and error notifications

---

## Technical Details

**Python Runtime Resolution**
- Bundled runtime for distribution builds
- Local virtual environment for development
- User-installable runtime under `%APPDATA%/VideoForge/python`

**Safe Upscaler Wrapper**
- Explicit model weight loading, handles nested dictionaries
- Enforces correct color space (BGR) for OpenCV compatibility
- Supports tile-based processing with mirror padding for consistency

**Shared Memory (SHM)**
- Ring buffer with multiple slots
- Rust `mpsc` channels enforce producer/consumer safety
- Efficient zero-copy data transfer to minimize overhead

---

## Determinism and Model Notes

- RCAN and EDSR are deterministic; same input produces same output
- RealESRGAN may exhibit slight perceptual variation due to GAN-based inference
- Users can select the model that best fits the required workflow

---

## License

[Specify license here]

---

## Contributing

Contributions are welcome. When contributing models or pipeline changes, ensure that determinism and performance characteristics are preserved where applicable.

