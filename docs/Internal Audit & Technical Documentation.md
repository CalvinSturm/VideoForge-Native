# Internal Audit & Technical Documentation VideoForge

## 1. High-Level Overview

VideoForge is a local-first, high-performance visual enhancement engine designed for upscaling images and videos using AI models (specifically RealESRGAN variants).

Purpose
To provide a professional, deterministic, and privacy-focused alternative to cloud-based upscalers. It allows users to fix blur, increase resolution, and manipulate video properties (trim, crop, rotate) without uploading assets to a third-party server.

Design Philosophy
- Local Authority The user owns the compute (GPU) and the data.
- Pipeline-Centric Operations are explicit, ordered steps (Decode → Enhance → Encode).
- Performance Utilizes Shared Memory (SHM) and Zero-Copy protocols to bypass the typical overhead of IPC data serialization.
- Modular UI A Mosaic tiling interface allows users to customize their workspace.

---

## 2. Functional Breakdown

### Core Systems
1.  Orchestration Layer (RustTauri)
    -   Acts as the application controller.
    -   Manages the lifecycle of child processes (FFmpeg, Python).
    -   Handles window management and native OS dialogs.
    -   Coordinates the Shared Memory (SHM) ring buffer.

2.  AI Inference Engine (Python)
    -   Runs as a sidecar process (`shm_worker.py`).
    -   Loads PyTorch models (RealESRGAN).
    -   Reads raw pixel data directly from SHM, processes it, and writes back to SHM.
    -   Communicates statuscommands via `Zenoh` (IPC).

3.  Video Pipeline (FFmpeg)
    -   Decoding Extracts raw RGBA frames from video files.
    -   Encoding Compresses processed frames back into high-quality video (H.264HEVC) via NVENC (NVIDIA Encoder).

4.  Frontend (ReactTypeScript)
    -   Provides the user interface for configuration, preview, and job management.
    -   Communicates with Rust via Tauri's invoke system and event listeners.

### Data Flow (Video Upscale Job)
1.  User Input File selected, config set (Trim, Crop, Model), Start clicked.
2.  Initialization Rust calculates output dimensions and requests Python to allocate an SHM file.
3.  Pipeline Start
    -   FFmpeg Decoder starts streaming raw frames to Rust.
    -   Rust writes frames into the SHM Ring Buffer (Input Slot).
4.  Inference
    -   Rust signals Python via Zenoh.
    -   Python reads SHM, runs AI model, writes to SHM (Output Slot).
5.  Encoding
    -   Rust reads the Output Slot.
    -   Rust pipes data to FFmpeg Encoder stdin.
6.  Completion Final video file is finalized; UI is updated.

---

## 3. UI  UX Analysis

### Layout Structure
The UI uses `react-mosaic-component` to create a tiled, professional dashboard look.
-   Left Panel (Configuration) Controls for InputOutput paths, Model selection, Scaling factors (1x, 2x, 4x), and Editing tools (Trim, Crop, Transform).
-   Center Panel (Viewport) Interactive preview area. Supports drag-to-crop overlays and media playback.
-   Right Panel (Process Queue) List of active, pending, and completed jobs with progress bars.
-   Bottom Panel (System Telemetry) Scrolling log window showing system events, GPU status, and errors.

### User Flow
1.  Load Drag & Drop or Browse file.
2.  Configure Select model (e.g., `RealESRGAN_x4plus`) and editing constraints.
3.  Preview (Optional) Users can render a 3-second sample or view a static frame to verify settings.
4.  Execute Click Start Processing.
5.  Monitor Watch the Process Queue and Telemetry for real-time feedback.

### Feedback Systems
-   Visual Scanning animation on progress bars (cyberpunk aesthetic).
-   Toast Notifications Non-blocking popups for errorssuccess (top-right).
-   Logs Detailed text logs for power users (e.g., `[GPU] Processing Frame 100500`).
-   Panic Button A specific UI control in the footer to force-kill backend processes if the pipeline hangs.

---

## 4. Visual Design & Color Scheme

Aesthetic Cyberpunk Industrial  High-Performance Tool.

Color Palette
-   Backgrounds Deep industrial graysblacks (`#09090b`, `#111113`).
-   Primary Accent Neon Green (`#00ff88` - often associated with NVIDIAPerformance).
-   Text High-contrast whitegray (`#ededed`) for readability; `JetBrains Mono` for data.
-   DangerError Muted Red (`#3f1818` bg, `#ff6b6b` text).

Design Language
-   Typography `Inter` for UI elements; `JetBrains Mono` for technical data and logs.
-   Shapes Sharp corners (radius 4px), thin borders (`1px`), dense information density.
-   Theme Supports LightDark modes, though Dark mode is the primary intent.

---

## 5. Interaction & Feel

-   Responsiveness The UI logic is decoupled from heavy processing. React state remains snappy even when the backend is maxing out the GPU.
-   Tactility
    -   The Mosaic windows allow resizingrearranging, giving a custom workstation feel.
    -   Interactive cropping overlay feels precise (handles for cornersedges).
-   Tone The application feels serious and engineering-focused. It avoids gamification in favor of raw telemetry and control.

---

## 6. Technical Architecture

### Tech Stack
-   Frontend React 19, TypeScript, Vite, BlueprintJS (UI components), Zustand (State).
-   Backend Rust (Tauri 2.0), Tokio (Async runtime), Zenoh (IPC), Memmap2 (SHM).
-   AI Runtime Python 3.10+, PyTorch, BasicsrRealESRGAN.
-   Video Engine FFmpeg (invoked as child process).

### Key Patterns
-   Zero-Copy Architecture
    -   Video frames are not serialized to JSONBase64 for IPC (except for single images).
    -   Frames move via `System RAM - SHM - GPU VRAM` to minimize copying.
-   Ring Buffer Uses a 3-slot ring buffer in SHM to allow parallel Decoding, Inference, and Encoding (Pipelining).
-   Sidecar Pattern Python is treated as an unstable worker process; Rust manages its lifecycle and restarts it if necessary.
-   Direct FFmpeg Piping Rust manages `stdin``stdout` pipes to FFmpeg, giving granular control over flow.

---

## 7. Strengths

1.  Performance The architecture avoids the Electron Bloat of serializing large binary data. Passing pointers to Shared Memory is significantly faster for 4K video.
2.  User Control The Panic Button and explicit process management (TaskkillPkill) show a commitment to user authority over runaway processes.
3.  UX Layout The Mosaic tile system is excellent for complex workflows, allowing users to hide logs or expand the preview as needed.
4.  Visual Polish The styling is consistent, modern, and distinct (VideoForge branding).