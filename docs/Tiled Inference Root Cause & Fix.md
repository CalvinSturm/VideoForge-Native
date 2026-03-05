# Tiled Inference — Root Cause & Fix

## Problem

Tiled inference (`--tile-size 256 --tile-pad 32`) produced black frames, black tiles, and green frames with transformer models.

## Root Cause: `cuMemAllocAsync` Cross-Stream Race

cudarc's `CudaDevice::alloc_zeros()` internally uses **`cuMemAllocAsync`** + **`cuMemsetD8Async`** on the device's **default stream** (stream 0). But NVRTC kernels (`crop_tile`, `place_tile`) launch on **`inference_stream`** — a different CUDA stream with no ordering guarantees relative to stream 0.

Without an inter-stream sync, the kernel could execute **before the allocation completed**, reading/writing non-existent memory. This caused:

- **Intermittent zero output** — kernel writes went to unallocated memory (silent no-op)
- **Black tiles at specific positions** — only tiles whose allocation timing lost the race were affected
- **Random black frames** — entire frames where the input buffer allocation hadn't completed

## Fix

### 1. Sync after fresh allocations in the pool allocator

```diff
 // context.rs — GpuContext::alloc() pool miss path
 let buf = self.device.alloc_zeros::<u8>(bucket_size)?;
+self.device.synchronize()?;  // ensure async alloc completes
 self.vram.on_alloc(bucket_size);
```

This single fix covers **every caller** across the pipeline.

### 2. Sync after fresh allocation in `crop_tile`

```diff
 // kernels.rs — crop_tile (bypassed pool for debugging)
 let output_buf = ctx.device().alloc_zeros::<u8>(out_bytes)?;
+ctx.device().synchronize()?;  // ensure async alloc completes
```

### 3. Serialized tile processing

Rewrote `tiled_inference` to process one tile fully (crop→infer→place→drop) before starting the next. Fixes OutputRing contention when total tiles exceeds `ring_size=8`.

## Files Changed

| File | Change |
|------|--------|
| [context.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-core/src/context.rs) | `synchronize()` after `alloc_zeros` in pool miss |
| [kernels.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-cuda/src/kernels.rs) | `synchronize()` after `alloc_zeros` in `crop_tile` |
| [pipeline.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-pipeline/src/pipeline.rs) | Serialized tile processing, input keepalive |

## Debugging Journey (15 attempts)

| # | Attempt | Insight |
|---|---------|---------|
| 1–2 | Stream sync / device sync | Not a stream ordering issue |
| 3–4 | GPU readback diagnostics | Crop kernel intermittently produces zeros |
| 5–6 | memcpy bypass | Raw FFI also affected (same alloc race) |
| 7 | **Full-frame bypass** | ✅ Works — confirmed bug is in tiling |
| 8–9 | bind_to_thread / block_in_place | Not thread migration |
| 10–11 | Keepalive / pool bypass | Not buffer reuse |
| 12 | **Serialized tiles** | ⚠️ Fixed line artifacts (ring contention) |
| 13 | Per-tile diagnostics | tx=0 crop zeros 80% of frames |
| 14 | Deep-copy input | Still zeros — deep copy also used async alloc |
| 15 | **`synchronize()` after `alloc_zeros`** | ✅ **ROOT CAUSE FIXED** |
