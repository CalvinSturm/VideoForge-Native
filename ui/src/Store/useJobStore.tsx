import { create } from 'zustand';
import type { SystemStats } from '../types';

// Upscale configuration types
export type EnhancementMode = 'archival' | 'creative';
export type ArchivalModel = 'RCAN' | 'EDSR';
export type CreativeModel = 'REALISTIC' | 'ANIME';
export type UpscaleScale = 2 | 3 | 4;

export interface UpscaleConfig {
  isEnabled: boolean;
  mode: EnhancementMode;
  archivalModel: ArchivalModel;
  creativeModel: CreativeModel;
  scaleFactor: UpscaleScale;
}

const DEFAULT_UPSCALE_CONFIG: UpscaleConfig = {
  isEnabled: true,
  mode: 'archival',
  archivalModel: 'RCAN',
  creativeModel: 'REALISTIC',
  scaleFactor: 4
};

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

  // Upscale Configuration (centralized to prevent stale state)
  upscaleConfig: UpscaleConfig;
  setUpscaleConfig: (patch: Partial<UpscaleConfig>) => void;
  resetUpscaleConfig: () => void;

  resetStore: () => void;
}

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

  // Upscale config - merge patch pattern for clean updates
  setUpscaleConfig: (patch) => set((state) => ({
    upscaleConfig: { ...state.upscaleConfig, ...patch }
  })),
  resetUpscaleConfig: () => set({ upscaleConfig: { ...DEFAULT_UPSCALE_CONFIG } }),

  resetStore: () => set({
    isProcessing: false,
    progressPercent: 0,
    framesProcessed: 0,
    totalFrames: 0,
    isModelLoading: false
    // Note: lastOutputPath and upscaleConfig are NOT reset - we want to keep them
  }),
}));
