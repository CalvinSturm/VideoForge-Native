export type UpscaleMode = "image" | "batch" | "video";
export type JobStatus = "queued" | "running" | "paused" | "done" | "cancelled" | "error";

export interface ColorSettings {
  brightness: number;  // -1.0 to 1.0 (0 = no change)
  contrast: number;    // -1.0 to 1.0 (0 = no change)
  saturation: number;  // -1.0 to 1.0 (0 = no change)
  gamma: number;       // 0.1 to 10.0 (1.0 = no change)
}

export interface EditState {
  trimStart: number;
  trimEnd: number;
  crop: { x: number; y: number; width: number; height: number; applied: boolean } | null;
  aspectRatio?: number | null;
  rotation: 0 | 90 | 180 | 270;
  flipH: boolean;
  flipV: boolean;
  fps: number; // 0 = Source
  color: ColorSettings;
}

export interface Job {
  id: string;
  command: string;
  status: JobStatus;
  progress: number; // Percentage 0-100
  framesProcessed?: number; // NEW
  totalFrames?: number;     // NEW
  statusMessage: string;
  paused: boolean;
  errorMessage?: string;
  outputPath?: string;
  eta?: number;
}

export interface Toast {
  id: string;
  message: string;
  type: "success" | "error" | "info" | "warning";
}

export interface VideoState {
  src: string;
  currentTime: number;
  setCurrentTime: (t: number) => void;
  duration: number;
  setDuration: (d: number) => void;
  inputWidth: number;
  inputHeight: number;
  setInputDimensions: (w: number, h: number) => void;
  sourceFps?: number; // NEW: Track Source FPS
  trimStart: number;
  setTrimStart: (t: number) => void;
  trimEnd: number;
  setTrimEnd: (t: number) => void;
  crop: { x: number; y: number; width: number; height: number };
  setCrop: (c: { x: number; y: number; width: number; height: number }) => void;
  samplePreview: string | null;
  renderSample: () => void;
  clearPreview: () => void;
  renderedRange?: { start: number; end: number } | null; // NEW: Track separately from trim
}

export interface SystemStats {
  cpu: number;
  ramUsed: number;
  ramTotal: number;
  gpuName: string;
}
