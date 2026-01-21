# Project Development Plan: Modular Approach for High-Performance AI-Powered Video Editor

This document outlines a modular approach to developing the **AI-Powered Video Editor**. By breaking the project into smaller sections, we aim to reduce complexity and ensure focus on individual components at each stage. Each section will be handled separately, allowing us to only expose relevant files to LLMs, ensuring better efficiency in development.

---

## Table of Contents

1. [Core Backend (Rust and AI Engine)](#1-core-backend-rust-and-ai-engine)
2. [Frontend/UI (Tauri + React/TS)](#2-frontendui-tauri-reactts)
3. [Interfacing/Integration](#3-interfacingintegration)
4. [AI Model & Determinism](#4-ai-model-determinism)
5. [Testing and Profiling](#5-testing-and-profiling)

---

## 1. Core Backend (Rust and AI Engine)

### Focus:
The core backend will handle performance-critical tasks, such as video decoding, raw frame buffer management, and AI inference.

### Steps:
1. **Rust**:
    - Implement video decoding (via `ffmpeg-next` or `video-rs`).
    - Manage raw frame buffers efficiently.
    - Integrate GPU acceleration (via `WGPU`).
    - Handle multi-threading and memory safety (Rust's ownership model).
   
2. **Python AI Engine**:
    - Use `uv` to manage Python environment (dependency isolation).
    - Integrate AI models (PyTorch/TensorRT) for tasks like background removal, image enhancement, etc.
    - Ensure deterministic behavior using fixed seeds for all AI models.

3. **IPC Protocol**:
    - Use shared memory (via Zenoh/Iceoryx) for high-speed communication between Rust and Python, avoiding the overhead of traditional IPC mechanisms like JSON.

### Files to Show:
- Rust files for video decoding and performance.
- Python scripts for AI models and their input/output.
- IPC files for data transfer (Zenoh or Iceoryx setup).

---

## 2. Frontend/UI (Tauri + React/TS)

### Focus:
The UI will handle the user experience, including Mosaic tiling, job management, file I/O, and rendering previews. The goal is to keep the binary small while providing a highly modular, professional interface.

### Steps:
1. **React UI**:
    - Set up the **Mosaic UI** using `react-mosaic` for modular tiling and reconfiguration.
    - Implement draggable panels, snapping, and re-sizable tiles.
    - Handle complex UI components like file input, model selection, and job status updates.

2. **Tauri Integration**:
    - Use Tauri for the lightweight WebView to display the React UI.
    - Ensure fast communication between Rust and React components via IPC.
    - Build the UI for managing jobs, previews, logs, and theme switching (Light/Dark).

3. **User Interaction**:
    - Ensure smooth and intuitive UI interactions with proper feedback (progress bars, job statuses).
    - Implement modal dialogs for file selection, error messages, and confirmations.

### Files to Show:
- `react-mosaic` layout files.
- Tauri setup (front-end initialization, WebView setup).
- React components for managing UI state (e.g., file input, theme toggle).

---

## 3. Interfacing/Integration

### Focus:
This section focuses on the integration between Rust, Python, and the frontend. It handles the communication flow and ensures the entire system works together seamlessly.

### Steps:
1. **Rust and Python Communication**:
    - Implement shared memory or zero-copy IPC protocols between Rust and Python for video frame transfers and AI model results.
    - Use gRPC or another lightweight protocol for sending commands (e.g., `remove background`, `process frames`).
    - Ensure that Python and Rust components can send and receive data efficiently, especially for large video frames.

2. **React and Backend Integration**:
    - Handle UI-to-backend communication via IPC (Tauri → Rust) for triggering jobs, fetching model data, and updating statuses.
    - Manage asynchronous job processing and show real-time updates on the UI (e.g., show progress, cancel jobs).

3. **Error Handling and Logs**:
    - Implement error handling throughout the integration layer. Ensure that errors in Rust, Python, or the UI are caught and reported to the user.
    - Implement logging for debugging and monitoring system behavior.

### Files to Show:
- IPC communication handlers (gRPC or Zenoh/Iceoryx setup).
- Files managing React → Rust → Python interaction.
- Error and logging systems (e.g., `log` crate for Rust, console output for React).

---

## 4. AI Model & Determinism

### Focus:
This section ensures that the AI models are integrated properly and guarantees determinism for professional workflows. All AI-related processes need to be reproducible to maintain consistency during edits.

### Steps:
1. **Model Integration**:
    - Implement AI model loading and inference in Python.
    - Handle model parameters, input preprocessing, and output postprocessing.
    - Ensure that models are deterministic by using fixed random seeds.

2. **Job Processing**:
    - Implement commands to interact with AI models (e.g., `remove background from frames 100-200`).
    - Process AI output for each frame in the video, ensuring no frame data duplication.
    - Ensure that the AI sidecar can handle batch processing and scale to 4K+ videos.

3. **Testing and Validation**:
    - Validate the AI output by running the same inputs and checking for identical results on multiple passes.
    - Implement automated tests for model accuracy and performance.

### Files to Show:
- Python files managing the AI inference pipeline.
- gRPC commands for interacting with AI models.
- Model-related scripts (loading, inference, output handling).

---

## 5. Testing and Profiling

### Focus:
To ensure the video editor performs optimally, thorough testing and profiling are required to detect bottlenecks and ensure smooth performance under load.

### Steps:
1. **Unit and Integration Testing**:
    - Implement unit tests for backend Rust components (e.g., video decoding).
    - Test Python AI models with representative input to ensure correctness.
    - Test the frontend React components for UI consistency and proper state management.

2. **Real-Time Profiling**:
    - Use profiling tools like **Tracy** to detect bottlenecks in video processing and AI model inference.
    - Profile frontend performance to identify slow UI rendering or unoptimized component updates.
    - Optimize Rust performance by analyzing memory usage, thread contention, and GPU utilization.

3. **End-to-End Tests**:
    - Implement real-world end-to-end tests, simulating actual user workflows (e.g., loading a video, applying AI processing, exporting the result).

### Files to Show:
- Unit test files (Rust, Python).
- Test configurations (e.g., test runners for Rust/Python/React).
- Profiling tools configuration (e.g., Tracy setup for Rust).

---

## Conclusion

By breaking the project into the above sections, we ensure that each layer is focused and modular. This method allows us to manage complexity, make incremental progress, and expose only the necessary files to LLM instances for development. As you complete each section, you can move on to the next, keeping everything organized and manageable.

---

### Next Steps:
- **Backend (Rust & Python)**: Focus on performance-critical components.
- **Frontend/UI**: Work on the Mosaic layout and React components.
- **Integration**: Ensure seamless communication between Rust, Python, and React.
- **AI Model**: Work on deterministic processing with AI models.
- **Testing**: Ensure everything works together and optimize performance.

This approach helps streamline the development process and avoids the overwhelming complexity of trying to develop everything at once.