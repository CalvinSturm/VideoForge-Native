import { create } from 'zustand';
import type { SystemStats } from '../types';

interface JobStoreState {
  isProcessing: boolean;
  setIsProcessing: (busy: boolean) => void;

  // Progress Tracking
  progressPercent: number;
  framesProcessed: number;
  totalFrames: number;

  // Setters
  setProgress: (percent: number, current: number, total: number) => void;

  isModelLoading: boolean;
  setIsModelLoading: (loading: boolean) => void;

  stats: SystemStats;
  setStats: (stats: SystemStats) => void;

  resetStore: () => void;
}

export const useJobStore = create<JobStoreState>((set) => ({
  isProcessing: false,
  progressPercent: 0,
  framesProcessed: 0,
  totalFrames: 0,
  isModelLoading: false,
  stats: { cpu: 0, ramUsed: 0, ramTotal: 0, gpuName: "DETECTING..." },

  setIsProcessing: (busy) => set({ isProcessing: busy }),

  setProgress: (percent, current, total) => set({
    progressPercent: percent,
    framesProcessed: current,
    totalFrames: total
  }),

  setIsModelLoading: (loading) => set({ isModelLoading: loading }),
  setStats: (stats) => set({ stats }),

  resetStore: () => set({
    isProcessing: false,
    progressPercent: 0,
    framesProcessed: 0,
    totalFrames: 0,
    isModelLoading: false
  }),
}));
