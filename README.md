# VideoForge-Native

> Local-first AI video and image enhancement with a native GPU engine path.

VideoForge-Native is a desktop app for enhancing video and images on your own machine.

It is built for people who want faster local workflows, direct control over models and settings, and a clearer path from experiment to repeatable result without sending media to the cloud.

## Why it matters

Most AI media tools make you give something up:

- privacy
- control
- performance
- repeatability

VideoForge-Native is built to narrow those tradeoffs.

It gives you:

- local-first processing
- GPU-backed enhancement workflows
- direct control over models, precision, and output settings
- inspectable runtime behavior
- an optional native `engine-v2` path for higher-performance video execution

## What it does

- runs local video and image enhancement jobs
- supports GPU-backed upscaling and enhancement workflows
- lets you compare model and runtime paths for quality and speed
- keeps sensitive media workflows off the cloud
- supports practical desktop workflows with trim, crop, timeline, and output controls

## Architecture at a glance

VideoForge-Native is built around three layers:

- **Desktop app**: Tauri + React UI for setup, preview, controls, and jobs
- **Inference paths**: Python worker path for broad model support and optional native `engine-v2` path for faster video execution
- **Local processing stack**: FFmpeg, GPU acceleration, on-disk artifacts, and local model management

## Status

Active local-first project focused on:

- image and video enhancement workflows
- native-engine execution paths
- inspectable runtime behavior
- model and runtime flexibility without cloud dependency
- practical desktop workflows for repeatable local processing

## Documentation

Start here:

- `docs/README.md`
- `docs/architecture_status_truth.md`
- `docs/capability_matrix.md`
- `docs/runtime_path_contracts.md`
