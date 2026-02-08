/**
 * SpatialMapOverlay — Client-side spatial routing map visualization.
 *
 * Computes Sobel gradient magnitude + local variance on the preview image,
 * then classifies each pixel as Edge / Texture / Flat using the current
 * research parameters (edge_threshold, texture_threshold).
 *
 * Recomputes whenever:
 *   - The "research-params-changed" Tauri event fires (user moves a slider)
 *   - The source image changes
 *   - The component becomes visible
 *
 * Algorithm mirrors Python SpatialRouter.compute_routing_masks():
 *   1. Grayscale conversion
 *   2. 3×3 Sobel → gradient magnitude → normalize to [0,1]
 *   3. 5×5 box filter → local variance → normalize to [0,1]
 *   4. Soft sigmoid classification with thresholds
 *   5. LUT-indexed RGBA write to canvas
 */

import { useRef, useEffect, useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// ---------------------------------------------------------------------------
// CLASSIFICATION COLORS — RGBA bytes
// ---------------------------------------------------------------------------
const FLAT_R = 40, FLAT_G = 120, FLAT_B = 255, FLAT_A = 178;       // Blue
const TEX_R = 60, TEX_G = 220, TEX_B = 80, TEX_A = 178;           // Green
const EDGE_R = 255, EDGE_G = 60, EDGE_B = 60, EDGE_A = 178;       // Red
const CLEAR_A = 0;

// ---------------------------------------------------------------------------
// SPATIAL ROUTING (mirrors Python SpatialRouter)
// ---------------------------------------------------------------------------

/** Sigmoid approximation: 1 / (1 + exp(-x*10)) */
function sigmoid10(x: number): number {
  const ex = Math.exp(-x * 10);
  return 1 / (1 + ex);
}

function computeSpatialMap(
  gray: Float32Array,
  w: number,
  h: number,
  edgeThreshold: number,
  textureThreshold: number,
  out: Uint8ClampedArray,
): void {
  // --- Sobel gradient magnitude ---
  const gradMag = new Float32Array(w * h);
  let gradMax = 1e-8;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const tl = gray[(y - 1) * w + (x - 1)]!;
      const tc = gray[(y - 1) * w + x]!;
      const tr = gray[(y - 1) * w + (x + 1)]!;
      const ml = gray[y * w + (x - 1)]!;
      const mr = gray[y * w + (x + 1)]!;
      const bl = gray[(y + 1) * w + (x - 1)]!;
      const bc = gray[(y + 1) * w + x]!;
      const br = gray[(y + 1) * w + (x + 1)]!;
      const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      const mag = Math.sqrt(gx * gx + gy * gy);
      gradMag[y * w + x] = mag;
      if (mag > gradMax) gradMax = mag;
    }
  }

  // --- Local variance (5×5 box) ---
  const localVar = new Float32Array(w * h);
  let varMax = 1e-8;
  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      let sum = 0, sumSq = 0;
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const v = gray[(y + dy) * w + (x + dx)]!;
          sum += v;
          sumSq += v * v;
        }
      }
      const mean = sum / 25;
      const variance = Math.max(0, sumSq / 25 - mean * mean);
      localVar[y * w + x] = variance;
      if (variance > varMax) varMax = variance;
    }
  }

  // --- Classify + write RGBA ---
  for (let i = 0; i < w * h; i++) {
    const gradNorm = gradMag[i]! / gradMax;
    const varNorm = localVar[i]! / varMax;

    const edgeMask = sigmoid10(gradNorm - edgeThreshold);
    const textureIndicator = sigmoid10(varNorm - textureThreshold);
    const textureMask = textureIndicator * (1 - edgeMask);
    // flat = 1 - edge - texture (clamped)

    const idx = i * 4;
    if (edgeMask > 0.5) {
      out[idx] = EDGE_R; out[idx + 1] = EDGE_G; out[idx + 2] = EDGE_B; out[idx + 3] = EDGE_A;
    } else if (textureMask > 0.3) {
      out[idx] = TEX_R; out[idx + 1] = TEX_G; out[idx + 2] = TEX_B; out[idx + 3] = TEX_A;
    } else {
      out[idx] = FLAT_R; out[idx + 1] = FLAT_G; out[idx + 2] = FLAT_B; out[idx + 3] = FLAT_A;
    }
  }
}

// ---------------------------------------------------------------------------
// COMPONENT
// ---------------------------------------------------------------------------

interface SpatialMapOverlayProps {
  visible?: boolean;
  /** Tauri-converted file:// URL for the preview image. */
  imageSrc?: string;
}

export const SpatialMapOverlay: React.FC<SpatialMapOverlayProps> = ({
  visible = true,
  imageSrc,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [params, setParams] = useState<{ edge_threshold: number; texture_threshold: number }>({
    edge_threshold: 0.5,
    texture_threshold: 0.2,
  });

  // Cache the loaded image to avoid reloading on every param change
  const imgCacheRef = useRef<{ src: string; img: HTMLImageElement } | null>(null);

  // ------------------------------------------------------------------
  // LISTEN FOR PARAM CHANGES
  // ------------------------------------------------------------------

  useEffect(() => {
    if (!visible) return;
    let unlisten: (() => void) | undefined;
    let cleaned = false;

    // Fetch initial params from backend
    invoke<Record<string, unknown>>("get_research_config").then((cfg) => {
      if (cleaned || !cfg) return;
      const et = typeof cfg.edge_threshold === "number" ? cfg.edge_threshold : 0.5;
      const tt = typeof cfg.texture_threshold === "number" ? cfg.texture_threshold : 0.2;
      setParams({ edge_threshold: et, texture_threshold: tt });
    }).catch(() => {});

    // Listen for live updates
    const setup = async () => {
      unlisten = await listen<Record<string, unknown>>("research-params-changed", (event) => {
        if (cleaned) return;
        const p = event.payload;
        if (p && typeof p === "object") {
          const et = typeof p.edge_threshold === "number" ? p.edge_threshold : 0.5;
          const tt = typeof p.texture_threshold === "number" ? p.texture_threshold : 0.2;
          setParams({ edge_threshold: et, texture_threshold: tt });
        }
      });
    };
    setup();
    return () => { cleaned = true; unlisten?.(); };
  }, [visible]);

  // ------------------------------------------------------------------
  // COMPUTE + RENDER
  // ------------------------------------------------------------------

  const render = useCallback(() => {
    if (!visible || !imageSrc) return;

    const doRender = (img: HTMLImageElement) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      // Down-sample for performance (max 512px on longest side)
      const maxDim = 512;
      let sw = img.naturalWidth;
      let sh = img.naturalHeight;
      if (sw === 0 || sh === 0) return;
      const ratio = Math.min(1, maxDim / Math.max(sw, sh));
      const w = Math.round(sw * ratio);
      const h = Math.round(sh * ratio);

      // Draw to offscreen canvas to get pixel data
      const offscreen = document.createElement("canvas");
      offscreen.width = w;
      offscreen.height = h;
      const offCtx = offscreen.getContext("2d")!;
      offCtx.drawImage(img, 0, 0, w, h);
      const srcData = offCtx.getImageData(0, 0, w, h);

      // Convert to grayscale float [0,1]
      const gray = new Float32Array(w * h);
      for (let i = 0; i < w * h; i++) {
        const r = srcData.data[i * 4]!;
        const g = srcData.data[i * 4 + 1]!;
        const b = srcData.data[i * 4 + 2]!;
        gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      }

      // Set canvas size
      canvas.width = w;
      canvas.height = h;

      // Compute spatial map
      const imgData = new ImageData(w, h);
      computeSpatialMap(gray, w, h, params.edge_threshold, params.texture_threshold, imgData.data);

      const ctx = canvas.getContext("2d");
      if (ctx) ctx.putImageData(imgData, 0, 0);
    };

    // Use cached image if same src
    if (imgCacheRef.current && imgCacheRef.current.src === imageSrc) {
      doRender(imgCacheRef.current.img);
      return;
    }

    // Load image
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgCacheRef.current = { src: imageSrc, img };
      doRender(img);
    };
    img.src = imageSrc;
  }, [visible, imageSrc, params]);

  useEffect(() => { render(); }, [render]);

  if (!visible || !imageSrc) return null;

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        pointerEvents: "none",
        zIndex: 10,
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "contain",
          imageRendering: "auto",
        }}
      />
    </div>
  );
};
