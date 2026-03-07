export type UpscaleMode = "image" | "batch" | "video";
export type JobStatus = "queued" | "running" | "paused" | "done" | "cancelled" | "error";

export interface RavePolicy {
  profile?: string;
  strict_invariants?: boolean;
  strict_vram_limit?: boolean;
  strict_no_host_copies?: boolean;
  determinism_policy?: string;
  ort_reexec_gate?: boolean;
}

export interface RaveCommandJson {
  output?: string;
  policy?: RavePolicy;
  host_copy_audit_enabled?: boolean;
  host_copy_audit_disable_reason?: string | null;
  [key: string]: unknown;
}

export interface NativeUpscaleResultJson {
  output_path: string;
  engine: string;
  encoder_mode: string;
  encoder_detail?: string | null;
  frames_processed: number;
  audio_preserved: boolean;
  trt_cache_enabled: boolean;
  trt_cache_dir?: string | null;
  requested_executor?: string | null;
  executed_executor?: string | null;
  direct_attempted?: boolean;
  fallback_used?: boolean;
  fallback_reason_code?: string | null;
  fallback_reason_message?: string | null;
}

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
  totalFrames?: number;
  statusMessage: string;
  paused: boolean;
  errorMessage?: string;
  errorCategory?: string;
  errorHint?: string;
  outputPath?: string;
  eta?: number;
  startedAt?: number;   // ms timestamp when job entered running state
  completedAt?: number; // ms timestamp when job reached done/error/cancelled
  policy?: RavePolicy;
  hostCopyAuditEnabled?: boolean;
  hostCopyAuditDisableReason?: string | null;
  nativeEngine?: string;
  encoderMode?: string;
  encoderDetail?: string | null;
  framesProcessed?: number;
  audioPreserved?: boolean;
  trtCacheEnabled?: boolean;
  trtCacheDir?: string | null;
  requestedExecutor?: string | null;
  executedExecutor?: string | null;
  directAttempted?: boolean;
  fallbackUsed?: boolean;
  fallbackReasonCode?: string | null;
  fallbackReasonMessage?: string | null;
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
