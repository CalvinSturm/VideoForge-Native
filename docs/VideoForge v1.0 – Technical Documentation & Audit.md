This is a wise engineering decision. **RealESRGAN_x4plus** and **RealESRGAN_x4plus_anime_6B** are the industry standards for a reason: they are robust, predictable, and their input/output behavior is consistent.

By focusing on these for **v1.0**, you guarantee a reliable user experience without the "wild west" compatibility issues of experimental community models.

Here is the **Final Technical Documentation** for your v1.0 release. This document reflects the current state of the codebase, the architecture improvements we implemented (SHM, ETA, Dynamic Paths), and the decision to standardize on RealESRGAN.

---

# VideoForge v1.0 – Technical Documentation & Audit

**Date:** January 15, 2026
**Version:** 1.0 (Release Candidate)
**Architecture:** Rust (Tauri) + Python (PyTorch) + React

---

## 1. High-Level Overview

**VideoForge** is a local-first, high-performance visual enhancement engine. It allows users to upscale images and videos using AI (RealESRGAN) without uploading data to the cloud.

### Core Philosophy
*   **Privacy First:** All processing happens locally on the user's GPU.
*   **Zero-Copy Pipeline:** Video frames are passed between the decoder, AI engine, and encoder via **Shared Memory (SHM)** to avoid serialization overhead.
*   **User Authority:** The user controls the pipeline (Trim, Crop, Resize) and processes (Pause, Resume, Kill).

---

## 2. Functional Architecture

### System Modules
1.  **Orchestration (Rust):**
    *   Manages the lifecycle of the Python sidecar.
    *   Allocates the Shared Memory Ring Buffer.
    *   Controls FFmpeg for decoding and encoding.
    *   Calculates ETA and broadcasts progress events.
2.  **AI Engine (Python):**
    *   Runs as a standalone process (`shm_worker.py`).
    *   Loads PyTorch models.
    *   Reads raw RGB/BGR data from SHM, upscales it, and writes back.
    *   Communication via **Zenoh** (low-latency IPC).
3.  **UI Layer (React/TypeScript):**
    *   **Mosaic Layout:** Tiled, professional interface.
    *   **Job Queue:** Manages state for batch processing.
    *   **Interactive Preview:** Real-time cropping and trimming controls.

### Data Flow (Video Pipeline)
1.  **Init:** Rust spawns Python -> Handshake via Zenoh -> Model Load.
2.  **SHM Setup:** Rust requests Python to map a memory file (`tempfile`).
3.  **Loop:**
    *   **Decode:** FFmpeg streams raw frames to Rust -> SHM Input Slot.
    *   **Inference:** Rust signals Python -> Python reads SHM -> Upscales -> Writes SHM Output Slot.
    *   **Encode:** Rust reads SHM Output -> Pipes to FFmpeg Encoder (NVENC/H.264).
4.  **Completion:** Cleanup of SHM files and process termination.

---

## 3. UI / UX Analysis

### Layout
*   **Left Panel (Config):** Inputs, Outputs, Model Selection, Resolution/FPS toggle.
*   **Center Panel (Viewport):** Video player with drag-to-crop overlay and timeline trimming.
*   **Right Panel (Queue):** Active jobs list with ETA and progress bars.
*   **Footer (Status):** Global GPU status, Tech Specs toggle, Panic Button (Kill Switch).

### Key Features
*   **ETA Estimation:** Real-time calculation of "Time Remaining" based on processing speed.
*   **Crash Recovery:** If the Python worker hangs, the "Panic Button" or app restart cleans up zombie processes via `taskkill`.
*   **Visual Feedback:** "Scanning" animations on progress bars; Toast notifications for errors.

---

## 4. Technical Implementation Details

### Path Resolution (Hybrid System)
The system uses a robust resolution strategy to find the Python runtime:
1.  **Distribution Mode:** Checks for a bundled `python/` folder next to the executable (for the `.exe` installer).
2.  **Dev Mode:** Checks for a local `venv310` or `venv` folder in the project root.
3.  **Installer Logic:** If the engine is missing, the UI prompts the user to download and unzip the runtime to `%APPDATA%\VideoForge\python`.

### The "Safe" Upscaler Wrapper
To support standard RealESRGAN models properly:
*   **Weight Loading:** The Python worker recursively searches `.pth` files for weights, handling nested dictionaries (`params`, `params_ema`, `net_g`).
*   **Color Space:** The worker enforces **BGR** input (for OpenCV compatibility) and handles the standard RealESRGAN output format reliably.

### Shared Memory (SHM)
*   **Structure:** A Ring Buffer with 3 slots.
*   **Sync:** `mpsc` channels in Rust coordinate the producer (Decoder), worker (AI), and consumer (Encoder) to prevent race conditions.

---

## 5. Strengths & Weaknesses

### Strengths
*   **Performance:** SHM pipeline eliminates the bottleneck of serializing 4K frames to JSON/Base64.
*   **Portability:** The installer logic allows for a small initial download (~20MB) with an optional engine download (~1.5GB).
*   **Modularity:** The UI is decoupled from the backend; the backend is decoupled from the specific AI implementation via Zenoh.

### Weaknesses / Constraints
*   **Hardware Requirement:** Requires an NVIDIA GPU (CUDA) for reasonable performance. CPU fallback exists but is extremely slow.
*   **Model Compatibility:** Highly specialized for `RRDBNet` (RealESRGAN) architectures. SwinIR/Transformer models require specific tweaking (disabled for v1.0).

---

## 6. Preparation for Release (Action Items)

To finalize **v1.0** with RealESRGAN only:

1.  **Clean Weights Folder:**
    *   Navigate to your `weights` folder.
    *   **Keep:** `RealESRGAN_x4plus.pth`, `RealESRGAN_x4plus_anime_6B.pth`.
    *   **Delete:** `SwinIR...`, `VimeoScale...`, `TSSM...`.
    *   *Result:* The UI dropdown will automatically show only the working models.

2.  **Verify `App.tsx` Defaults:**
    *   Ensure the fallback list in `ui/src/App.tsx` only lists the reliable models:
    ```typescript
    setAvailableModels([
        "RealESRGAN_x4plus.pth",
        "RealESRGAN_x2plus.pth",
        "RealESRGAN_x4plus_anime_6B.pth",
        "RealESRGAN_x2plus_anime_6B.pth"
        
    ]);
    ```

3.  **Build:**
    *   Run `npm run tauri build` to generate the final `.exe`.

---

## Model Readiness Summary

**Status:** **READY FOR RELEASE (v1.0)**

I fully understand the system. We have successfully:
1.  **Stabilized the Pipeline:** Reverted to the reliable RealESRGAN architecture.
2.  **Fixed Architecture:** Solved the "hardcoded path" issue for distribution.
3.  **Enhanced UX:** Added ETA, Toast notifications, and a professional Mosaic layout.

The codebase is now in a clean, maintainable state for a public release.