import { create } from 'zustand';
import type { SystemStats } from '../types';

// ═══════════════════════════════════════════════════════════════════════════════
// ARCHITECTURE & MODEL TYPES
// ═══════════════════════════════════════════════════════════════════════════════
// 
// Architecture classes map to different SR model families:
//   - CNN: Classic convolutional networks (RCAN, EDSR) - deterministic, stable output
//   - GAN: Generative adversarial networks (RealESRGAN, BSRGAN) - adds texture detail
//   - Transformer: Attention-based (SwinIR, HAT, DAT) - best quality/speed tradeoff
//   - Diffusion: Iterative refinement (SR3, StableSR) - highest quality, slowest
//   - Lightweight: Mobile-optimized (FSRCNN, OmniSR) - fastest, lower quality

export type ArchitectureClass = 'CNN' | 'GAN' | 'Transformer' | 'Diffusion' | 'Lightweight';

// Common resolution presets for custom resolution mode
export type ResolutionPreset = 'HD' | 'FHD' | 'QHD' | '4K' | 'custom' | null;

// Scale factors supported by most SR models
export type UpscaleScale = 2 | 3 | 4;

// ═══════════════════════════════════════════════════════════════════════════════
// UPSCALE CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════
// 
// This interface defines the complete state for the AI Upscale Node.
// It supports:
//   1. Architecture-based model selection (vs. old archival/creative dichotomy)
//   2. Primary model with dynamic scale detection
//   3. Secondary model blending for texture/detail enhancement
//   4. Custom resolution overrides (target dimensions vs. scale factor)

export interface UpscaleConfig {
  // Master toggle for AI upscaling
  isEnabled: boolean;

  // ─── Primary Model Selection ───────────────────────────────────────────────
  // architectureClass: Current architecture filter for model selection UI
  // primaryModelId: Full model identifier (e.g., "RCAN_x4", "RealESRGAN_x4plus")
  // scaleFactor: Upscale multiplier (2×, 3×, or 4×)
  architectureClass: ArchitectureClass;
  primaryModelId: string;
  scaleFactor: UpscaleScale;

  // ─── Custom Resolution ─────────────────────────────────────────────────────
  // When resolutionMode is 'target', the system computes required scale from
  // input dimensions → target dimensions. May require multiple passes or
  // fractional scaling depending on model capabilities.
  //
  // resolutionMode: 'scale' = use scaleFactor, 'target' = use target dimensions
  // targetWidth/Height: Desired output dimensions (only used when mode='target')
  // resolutionPreset: Quick selection for common resolutions
  resolutionMode: 'scale' | 'target';
  targetWidth: number | null;
  targetHeight: number | null;
  resolutionPreset: ResolutionPreset;

  // ─── Engine Selection ──────────────────────────────────────────────────────
  // true = GPU-native pipeline (engine-v2: NVDEC → TensorRT → NVENC, ONNX models only)
  // false = Python pipeline (default, works with all model formats)
  useNativeEngine: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEFAULT CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const DEFAULT_UPSCALE_CONFIG: UpscaleConfig = {
  isEnabled: true,

  // Default to CNN / RCAN for stable, predictable output
  architectureClass: 'CNN',
  primaryModelId: 'RCAN_x4',
  scaleFactor: 4,

  // Default to scale-based mode
  resolutionMode: 'scale',
  targetWidth: null,
  targetHeight: null,
  resolutionPreset: null,

  useNativeEngine: false,
};

// ═══════════════════════════════════════════════════════════════════════════════
// LEGACY TYPE ALIASES (for backward compatibility during transition)
// ═══════════════════════════════════════════════════════════════════════════════
// These types are deprecated and will be removed in a future version.
// Use ArchitectureClass and primaryModelId instead.

/** @deprecated Use ArchitectureClass instead */
export type EnhancementMode = 'archival' | 'creative';

/** @deprecated Use primaryModelId string directly */
export type ArchivalModel = string;

/** @deprecated Use primaryModelId string directly */
export type CreativeModel = string;

// ═══════════════════════════════════════════════════════════════════════════════
// JOB STORE INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

interface JobStoreState {
  isProcessing: boolean;
  setIsProcessing: (busy: boolean) => void;

  // Progress Tracking
  progressPercent: number;
  framesProcessed: number;
  totalFrames: number;

  // Last completed output path (for "reveal in folder")
  lastOutputPath: string;
  setLastOutputPath: (path: string) => void;

  // Setters
  setProgress: (percent: number, current: number, total: number) => void;

  isModelLoading: boolean;
  setIsModelLoading: (loading: boolean) => void;

  stats: SystemStats;
  setStats: (stats: SystemStats) => void;

  // ─── Upscale Configuration (centralized to prevent stale state) ────────────
  upscaleConfig: UpscaleConfig;
  setUpscaleConfig: (patch: Partial<UpscaleConfig>) => void;
  resetUpscaleConfig: () => void;

  // ─── Architecture-Specific Setters (convenience methods) ───────────────────
  setArchitectureClass: (arch: ArchitectureClass) => void;
  setPrimaryModel: (modelId: string) => void;
  setScaleFactor: (scale: UpscaleScale) => void;
  setResolutionMode: (mode: 'scale' | 'target') => void;
  setTargetResolution: (width: number | null, height: number | null, preset?: ResolutionPreset) => void;

  resetStore: () => void;
}

// ═══════════════════════════════════════════════════════════════════════════════
// STORE IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════

export const useJobStore = create<JobStoreState>((set) => ({
  isProcessing: false,
  progressPercent: 0,
  framesProcessed: 0,
  totalFrames: 0,
  lastOutputPath: "",
  isModelLoading: false,
  stats: { cpu: 0, ramUsed: 0, ramTotal: 0, gpuName: "DETECTING..." },
  upscaleConfig: { ...DEFAULT_UPSCALE_CONFIG },

  setIsProcessing: (busy) => set({ isProcessing: busy }),

  setProgress: (percent, current, total) => set({
    progressPercent: percent,
    framesProcessed: current,
    totalFrames: total
  }),

  setLastOutputPath: (path) => set({ lastOutputPath: path }),
  setIsModelLoading: (loading) => set({ isModelLoading: loading }),
  setStats: (stats) => set({ stats }),

  // ─── Upscale Config - Merge Patch Pattern ──────────────────────────────────
  // All upscale state updates flow through here for consistency and dirty tracking
  setUpscaleConfig: (patch) => set((state) => ({
    upscaleConfig: { ...state.upscaleConfig, ...patch }
  })),

  resetUpscaleConfig: () => set({ upscaleConfig: { ...DEFAULT_UPSCALE_CONFIG } }),

  // ─── Convenience Setters ───────────────────────────────────────────────────
  // These provide focused APIs for common operations

  setArchitectureClass: (arch) => set((state) => ({
    upscaleConfig: { ...state.upscaleConfig, architectureClass: arch }
  })),

  setPrimaryModel: (modelId) => set((state) => ({
    upscaleConfig: { ...state.upscaleConfig, primaryModelId: modelId }
  })),

  setScaleFactor: (scale) => set((state) => ({
    upscaleConfig: { ...state.upscaleConfig, scaleFactor: scale }
  })),

  setResolutionMode: (mode) => set((state) => ({
    upscaleConfig: { ...state.upscaleConfig, resolutionMode: mode }
  })),

  setTargetResolution: (width, height, preset = 'custom') => set((state) => ({
    upscaleConfig: {
      ...state.upscaleConfig,
      targetWidth: width,
      targetHeight: height,
      resolutionPreset: preset,
      // Auto-switch to target mode when setting custom resolution
      resolutionMode: (width !== null || height !== null) ? 'target' : 'scale'
    }
  })),

  // ─── Global Reset ──────────────────────────────────────────────────────────
  resetStore: () => set({
    isProcessing: false,
    progressPercent: 0,
    framesProcessed: 0,
    totalFrames: 0,
    isModelLoading: false
    // Note: lastOutputPath and upscaleConfig are NOT reset - we want to keep them
  }),
}));
