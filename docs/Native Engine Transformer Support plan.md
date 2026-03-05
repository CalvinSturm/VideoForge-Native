# Native Engine Transformer Support

Port Python engine's transformer optimizations to the native Rust engine (rave), bringing DAT2 performance from ~35s/frame to ~1-3s/frame.

## Design Decisions (Approved)

- **Tile crop/place: Custom CUDA kernels** (not cudarc sub-slicing) — planar RGB needs 2D strided access across 3 planes; cudarc slices are 1D contiguous. NVRTC infrastructure already exists. ~40 lines of kernel code.
- **Auto-suggest tile_size=256 for transformers** — backend detects dynamic spatial axes and suggests 256px tiles (matching Python's `preferred_tile_size`). CLI `--tile-size` overrides for power users.
- **`run_simple()` is a safety net only** — 2-5ms host roundtrip per call, ~10-20ms total with tiling (~4 tiles at 406×470). Acceptable overhead vs 35s/frame. Tiling is the performance path.

## Proposed Changes

### Phase 1 — Tiled Inference (highest impact)

The Python engine splits transformer inputs into 256px overlapping tiles, runs inference per-tile, and merges results. The native engine processes the full image in one pass, causing O(n²) attention cost on the entire spatial dimension.

---

#### [MODIFY] [pipeline.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-pipeline/src/pipeline.rs)

**Add `TileConfig` to `PipelineConfig`:**

```rust
pub struct TileConfig {
    /// Tile size in pixels (0 = disabled / full-frame).
    pub tile_size: u32,
    /// Overlap padding in pixels for seamless tile boundaries.
    pub tile_pad: u32,
}
```

Add `pub tile_config: TileConfig` to `PipelineConfig` with default `{ tile_size: 0, tile_pad: 32 }`.

**Add tiling logic to `inference_stage`:**

Before calling `backend.process()`, check if `tile_config.tile_size > 0`. If so:

1. Split the preprocessed RGB planar texture into overlapping tiles (CUDA kernel or CPU crop)
2. Run `backend.process()` on each tile independently
3. Merge tiles back into a single output texture, blending overlap regions
4. Continue with existing postprocess (RGB→NV12)

The tiling loop replaces the single `backend.process_batch(textures)` call with:

```
for each tile:
    tile_input = crop_texture(input, tile_x, tile_y, tile_size + 2*tile_pad)
    tile_output = backend.process(tile_input)
    place_tile(output, tile_output, tile_x*scale, tile_y*scale, tile_pad*scale)
```

> [!NOTE]
> **Locked in: Custom CUDA kernels** for crop/place. Planar RGB (`[R][G][B]` planes) requires strided 2D access that cudarc 1D slices can't express. The kernels handle pitch/stride naturally and can strip overlap padding in a single pass.

#### [MODIFY] [kernels.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-cuda/src/kernels.rs)

Add two new CUDA kernels:

- `crop_tile_planar_f32(src, dst, src_w, src_h, tile_x, tile_y, tile_w, tile_h)` — extract a tile from RGB planar F32 texture
- `place_tile_planar_f32(src, dst, dst_w, dst_h, tile_x, tile_y, tile_w, tile_h, pad)` — place tile output into destination, cropping overlap padding

#### [MODIFY] [main.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/rave-cli/src/main.rs)

Add CLI arguments to `UpscaleArgs`:

```rust
#[arg(long, default_value_t = 0)]
tile_size: u32,

#[arg(long, default_value_t = 32)]
tile_pad: u32,
```

Wire into `PipelineConfig::tile_config`.

#### [MODIFY] [tensorrt.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-tensorrt/src/tensorrt.rs)

After `probe_has_dynamic_spatial()` detects a transformer model, set the default tile size:

```rust
if ep_mode == OrtEpMode::CudaOnly && self.probe_has_dynamic_spatial() {
    // Suggest 256px tiles for transformer models
    info!("Suggesting tile_size=256 for transformer model");
}
```

Expose a method `recommended_tile_size() -> u32` that returns 256 for dynamic-axis models, 0 (disabled) otherwise.

#### [MODIFY] [rave.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/src-tauri/src/commands/rave.rs)

Pass `--tile-size 256 --tile-pad 32` when the model is detected as a transformer (can use filename heuristics matching the Python `_TRANSFORMER_KEYS`).

---

### Phase 2 — CUDA EP FP16

#### [MODIFY] [tensorrt.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-tensorrt/src/tensorrt.rs)

In `build_cuda_session()`, apply FP16 graph optimization when precision policy is `Fp16`:

```rust
fn build_cuda_session(&self) -> Result<Session> {
    let cuda_ep = CUDAExecutionProvider::default()
        .with_device_id(self.device_id);
    
    let mut builder = Session::builder()?
        .with_execution_providers([cuda_ep.build().error_on_failure()])?
        .with_intra_threads(1)?;
    
    // Enable FP16 graph optimization for CUDA EP
    if matches!(self.precision_policy, PrecisionPolicy::Fp16) {
        builder = builder.with_optimization_level(ort::session::GraphOptimizationLevel::All)?;
    }
    
    builder.commit_from_file(&self.model_path).map_err(Into::into)
}
```

> [!NOTE]  
> ORT CUDA EP FP16 works differently from TRT EP — it uses graph-level cast insertion. The model weights stay FP32 but intermediate activations are computed in FP16 via ORT's `GraphOptimizationLevel::All` + transformer-specific optimizer passes. This gives ~1.5-2× speedup without model conversion.

---

### Phase 3 — OOM-Safe `session.run()` Fallback

#### [MODIFY] [tensorrt.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-tensorrt/src/tensorrt.rs)

Add a `run_simple()` method alongside `run_io_bound()`:

```rust
fn run_simple(
    session: &Session,
    meta: &ModelMetadata,
    input: &GpuTexture,
    ctx: &GpuContext,
) -> Result<GpuTexture> {
    // 1. DtoH copy input to host
    // 2. session.run() with CPU tensors (ORT handles GPU internally)
    // 3. HtoD copy output back to device
    // Returns a freshly allocated GpuTexture
}
```

In `process()` / `process_batch()`, catch ORT errors from `run_io_bound()` and fall back to `run_simple()`:

```rust
match Self::run_io_bound(...) {
    Ok(()) => { /* use ring buffer output */ }
    Err(e) if e.is_oom_like() => {
        warn!("IO binding failed: {e}, falling back to session.run()");
        return Self::run_simple(...);
    }
    Err(e) => return Err(e),
}
```

> [!WARNING]
> `run_simple()` has a host roundtrip penalty (~2-5ms per frame). It's a safety net, not a performance path. The primary fix for OOM is tiling (Phase 1).

---

### Phase 4 — Deadlock Probe

#### [MODIFY] [tensorrt.rs](file:///c:/Users/Calvin/Desktop/VideoForge1/third_party/rave/crates/rave-tensorrt/src/tensorrt.rs)

Add a `probe_inference_deadlock()` method in `initialize()`, called after session build for CUDA EP models:

```rust
fn probe_inference_deadlock(&self, session: &Session, meta: &ModelMetadata) -> bool {
    // 1. Allocate a tiny 64×64 input
    // 2. Spawn a thread that runs session.run()
    // 3. Wait with 20-second timeout
    // 4. If thread is still alive → deadlock detected → return false
    // Return true if probe completed successfully
}
```

If the probe fails, `initialize()` returns an error with a clear message suggesting the Python engine for this model.

---

## Verification Plan

### Manual Testing (all phases)

Each phase should be tested with the DAT2 model (`4xRealWebPhoto_v4_dat2_fp32_opset17.onnx`) using a 2-second test video.

**Phase 1 verification:**

1. Build rave: `cd third_party\rave && cargo build --release`
2. Delete stale cache: `Remove-Item -Recurse weights\trt_cache\` (if exists)
3. Run with tiles: `third_party\rave\target\release\rave.exe upscale --model weights\4xRealWebPhoto_v4_dat2_fp32_opset17.onnx --tile-size 256 --tile-pad 32 --input <test_video> --output <output_path> --json`
4. **Expected**: completes in ~30-90s (vs 10min+ without tiling), no green output, no crash
5. Compare output visually — should have no tile seam artifacts

**Phase 2 verification:**

1. Same as Phase 1 but add `--precision fp16`
2. **Expected**: ~1.5-2× faster than FP32, minor quality difference (acceptable for SR)

**Phase 3 verification:**

1. Test with an artificially large tile size (e.g., `--tile-size 2048`) to trigger OOM
2. **Expected**: graceful fallback to `run_simple()` with warning in logs, no process crash

**Phase 4 verification:**

1. If a model is known to deadlock CUDA EP (some DAT2 variants), the probe should detect it within 20s and return an error
2. **Expected**: clear error message within 20s, no hang

### Regression Testing

Run with the Span CNN model to verify no regression:

```
third_party\rave\target\release\rave.exe upscale --model weights\2x_SPAN_soft.onnx --input <test_video> --output <output_path> --json
```

**Expected**: TRT EP used (not CUDA EP), fast inference (~0.1s/frame), no tiling applied
