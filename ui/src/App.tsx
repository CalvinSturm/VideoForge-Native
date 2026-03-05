/* ui/src/App.tsx */
import React, { useState, useMemo, useEffect, useRef, useCallback } from "react";
import {
  Mosaic,
  MosaicWindow,
  getLeaves,
  createBalancedTreeFromLeaves
} from "react-mosaic-component";
import type { MosaicNode } from "react-mosaic-component";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

import { useJobStore } from './Store/useJobStore';
import { useViewLayoutStore } from './Store/viewLayoutStore';
import type { PanelId } from './Store/viewLayoutStore';
import { useTauriEvents } from './hooks/useTauriEvents';

import { InputOutputPanel } from "./components/InputOutputPanel";
import { JobsPanel } from "./components/JobsPanel";
import { LogsPanel } from "./components/LogsPanel";
import { PreviewPanel } from "./components/PreviewPanel";
import { StatusFooter } from "./components/StatusFooter";
import { TitleBar } from "./components/TitleBar";
import { PanelHeader } from "./components/PanelHeader";
import { EmptyState } from "./components/EmptyState";

import type { Toast, UpscaleMode, Job, VideoState, EditState, RavePolicy } from './types';

interface ModelInfo {
  id: string;
  scale: number;
  filename: string;
  format: string;  // "onnx" | "pytorch"
  path: string;
}

interface RaveCommandJson {
  output?: string;
  policy?: RavePolicy;
  host_copy_audit_enabled?: boolean;
  host_copy_audit_disable_reason?: string | null;
  [key: string]: unknown;
}

interface RaveErrorPayload {
  category?: string;
  message?: string;
  detail?: string;
  next_action?: string;
}

// Fallback model list when engine discovery fails
const DEFAULT_MODELS = ["RCAN_x4", "EDSR_x4", "RealESRGAN_x4plus"];

const DEFAULT_LAYOUT: MosaicNode<PanelId> = {
  direction: "row",
  first: "SETTINGS",
  second: {
    direction: "row",
    first: {
      direction: "column",
      first: "PREVIEW",
      second: "ACTIVITY",
      splitPercentage: 75,
    },
    second: "QUEUE",
    splitPercentage: 80,
  },
  splitPercentage: 24
};

// Panel accent colors for dock strips
const DOCK_COLORS: Record<string, string> = {
  SETTINGS: '#3b82f6',
  PREVIEW: '#00ff88',
  QUEUE: '#f59e0b',
  ACTIVITY: '#a855f7'
};

const DockStrip = ({ position, onClick, label, icon, panelId }: {
  position: 'right' | 'bottom' | 'left',
  onClick: () => void,
  label: string,
  icon: React.ReactNode,
  panelId?: string
}) => {
  const [isHovered, setIsHovered] = React.useState(false);
  const isVertical = position === 'right' || position === 'left';
  const accentColor = panelId ? DOCK_COLORS[panelId] || '#00ff88' : '#00ff88';

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      title={`Open ${label}`}
      style={{
        position: 'absolute',
        [position]: 0,
        [isVertical ? 'top' : 'left']: '50%',
        transform: `${isVertical ? 'translateY(-50%)' : 'translateX(-50%)'} ${isHovered ? 'scale(1.02)' : 'scale(1)'}`,
        [isVertical ? 'width' : 'height']: '28px',
        borderRadius: position === 'right' ? '6px 0 0 6px' : position === 'left' ? '0 6px 6px 0' : '6px 6px 0 0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        zIndex: 50,
        padding: isVertical ? '14px 0' : '0 14px',
        background: isHovered
          ? `linear-gradient(135deg, ${accentColor}20, var(--dock-bg))`
          : 'var(--dock-bg)',
        backdropFilter: 'blur(8px)',
        border: isHovered
          ? `1px solid ${accentColor}60`
          : '1px solid var(--dock-border)',
        boxShadow: isHovered
          ? `0 4px 20px ${accentColor}30, 0 2px 8px rgba(0,0,0,0.4)`
          : '0 4px 12px rgba(0,0,0,0.4)',
        transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)'
      }}
    >
      {/* Accent indicator line */}
      <div style={{
        position: 'absolute',
        [position === 'right' ? 'right' : position === 'left' ? 'left' : 'bottom']: 0,
        [isVertical ? 'top' : 'left']: '20%',
        [isVertical ? 'bottom' : 'right']: '20%',
        [isVertical ? 'width' : 'height']: '2px',
        background: `linear-gradient(${isVertical ? '180deg' : '90deg'}, transparent, ${accentColor}, transparent)`,
        opacity: isHovered ? 1 : 0.5,
        transition: 'opacity 0.2s ease'
      }} />

      <div style={{
        writingMode: isVertical ? 'vertical-rl' : 'horizontal-tb',
        transform: position === 'right' ? 'rotate(180deg)' : 'none',
        fontSize: '10px',
        fontWeight: 600,
        letterSpacing: '0.08em',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        color: isHovered ? '#fff' : 'var(--text-secondary)',
        transition: 'color 0.15s ease'
      }}>
        <span style={{
          color: isHovered ? accentColor : 'inherit',
          transition: 'color 0.15s ease'
        }}>
          {icon}
        </span>
        {label}
      </div>
    </div>
  );
};


const App: React.FC = () => {
  const [mode, setMode] = useState<UpscaleMode>("image");
  const [inputPath, setInputPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [model, setModel] = useState("RealESRGAN_x4plus.pth");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [modelInfoMap, setModelInfoMap] = useState<Map<string, ModelInfo>>(new Map());
  const [loadingModel, setLoadingModel] = useState(false);
  const [showTechSpecs, setShowTechSpecs] = useState(false);
  const [showResearchParams, setShowResearchParams] = useState(true);
  const [darkMode, setDarkMode] = useState(true);
  const [viewMode, setViewMode] = useState<'edit' | 'preview'>('edit');
  const [jobs, setJobs] = useState<Job[]>([]);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const [previewFile, setPreviewFile] = useState<string | null>(null);
  const [isEngineReady, setIsEngineReady] = useState(false);
  const [installProgress, setInstallProgress] = useState(0);
  const [isInstalling, setIsInstalling] = useState(false);
  const [checkingEngine, setCheckingEngine] = useState(true);

  const [mosaicValue, setMosaicValue] = useState<MosaicNode<PanelId> | null>(DEFAULT_LAYOUT);
  const { panels, setAllPanels, togglePanel, openPanel } = useViewLayoutStore();

  const [editState, setEditState] = useState<EditState>({
    trimStart: 0, trimEnd: 0, crop: null, rotation: 0, flipH: false, flipV: false, fps: 0,
    color: { brightness: 0, contrast: 0, saturation: 0, gamma: 1.0 }
  });
  const [inputDims, setInputDims] = useState({ w: 0, h: 0 });
  const { setIsProcessing, setLastOutputPath, upscaleConfig } = useJobStore();

  // Helper: Extract scale factor from model string (robust fallback)
  const getScaleFromModel = (modelId?: string): number => {
    if (!modelId) return 4;
    const match = modelId.match(/x(\d)/);
    return match?.[1] ? parseInt(match[1], 10) : 4;
  };

  // --- Keybinds ---
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check for Ctrl (or Command on Mac)
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case '1':
            e.preventDefault();
            togglePanel('SETTINGS');
            break;
          case '2':
            e.preventDefault();
            togglePanel('PREVIEW');
            break;
          case '3':
            e.preventDefault();
            togglePanel('QUEUE');
            break;
          case '4':
            e.preventDefault();
            togglePanel('ACTIVITY');
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePanel]);

  useEffect(() => { checkEngine(); }, []);
  const checkEngine = async () => {
    try {
      const ready = await invoke<boolean>("check_engine_status");
      setIsEngineReady(ready);
    } catch (e) {
      console.error("Engine check failed:", e);
      setIsEngineReady(false);
    } finally {
      setCheckingEngine(false);
    }
  };

  useTauriEvents({ setJobs, setLogs, setActiveJob, setLoadingModel, addToast: (msg: string, type: string) => addToast(msg, type as any) });

  const handleMosaicChange = (newNode: MosaicNode<PanelId> | null) => {
    setMosaicValue(newNode);
    if (!newNode) return;

    const visibleIds = getLeaves(newNode);
    const newPanelState = { ...panels };
    (Object.keys(newPanelState) as PanelId[]).forEach(k => newPanelState[k] = false);
    visibleIds.forEach(id => newPanelState[id] = true);
    setAllPanels(newPanelState);
  };

  useEffect(() => {
    let currentTree = mosaicValue;
    const visibleInTree = currentTree ? getLeaves(currentTree) : [];
    const visibleIdsInStore = (Object.keys(panels) as PanelId[]).filter(id => panels[id]);

    const setA = new Set(visibleIdsInStore);
    const setB = new Set(visibleInTree);
    const areEqual = setA.size === setB.size && [...setA].every(val => setB.has(val));

    if (!areEqual) {
      if (visibleIdsInStore.length === 0) {
        setMosaicValue(null);
      } else {
        let centerNode: MosaicNode<PanelId> | null = null;
        if (panels.PREVIEW && panels.ACTIVITY) {
          centerNode = { direction: 'column', first: 'PREVIEW', second: 'ACTIVITY', splitPercentage: 75 };
        } else if (panels.PREVIEW) {
          centerNode = 'PREVIEW';
        } else if (panels.ACTIVITY) {
          centerNode = 'ACTIVITY';
        }

        let middleNode: MosaicNode<PanelId> | null = null;
        if (centerNode && panels.QUEUE) {
          middleNode = { direction: 'row', first: centerNode, second: 'QUEUE', splitPercentage: 80 };
        } else if (centerNode) {
          middleNode = centerNode;
        } else if (panels.QUEUE) {
          middleNode = 'QUEUE';
        }

        let rootNode: MosaicNode<PanelId> | null = null;
        if (panels.SETTINGS && middleNode) {
          rootNode = { direction: 'row', first: 'SETTINGS', second: middleNode, splitPercentage: 20 };
        } else if (panels.SETTINGS) {
          rootNode = 'SETTINGS';
        } else {
          rootNode = middleNode;
        }

        setMosaicValue(rootNode);
      }
    }
  }, [panels]);

  useEffect(() => {
    if (isEngineReady) {
      invoke<ModelInfo[]>("get_models").then(models => {
        setAvailableModels(models.length > 0 ? models.map(m => m.id) : DEFAULT_MODELS);
        setModelInfoMap(new Map(models.map(m => [m.id, m])));
      }).catch(() => setAvailableModels(DEFAULT_MODELS));
    } else {
      setAvailableModels(DEFAULT_MODELS);
    }
  }, [isEngineReady]);

  const handleNewInput = useCallback((path: string) => {
    setInputPath(path);
    if (/\.(mp4|mkv|mov|avi|webm)$/i.test(path)) {
      setMode('video');
    } else {
      setMode('image');
    }
    setEditState({
      trimStart: 0, trimEnd: 0, crop: null, rotation: 0, flipH: false, flipV: false, fps: 0,
      color: { brightness: 0, contrast: 0, saturation: 0, gamma: 1.0 }
    });
    setVideoTime(0); setVideoDuration(0); setInputDims({ w: 0, h: 0 });
    setViewMode('edit'); setOutputPath(""); setPreviewFile(null); setActiveJob(null);
    setRenderedRange(null); // Reset rendered range
  }, []);

  // --- Global File Drop Listener (Tauri v2) ---
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    const setupListener = async () => {
      // In Tauri v2, the event is 'tauri://drag-drop' and the payload is { paths: string[], position: { x: number, y: number } }
      unlisten = await listen<{ paths: string[] }>('tauri://drag-drop', (event) => {
        const { paths } = event.payload;
        if (paths && paths.length > 0 && paths[0]) {
          handleNewInput(paths[0]);
        }
      });
    };
    setupListener();
    return () => { if (unlisten) unlisten(); };
  }, [handleNewInput]);

  const [videoTime, setVideoTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [renderedRange, setRenderedRange] = useState<{ start: number; end: number } | null>(null);

  const videoState: VideoState = useMemo(() => ({
    src: inputPath, currentTime: videoTime, setCurrentTime: setVideoTime, duration: videoDuration, setDuration: setVideoDuration,
    inputWidth: inputDims.w, inputHeight: inputDims.h, setInputDimensions: (w, h) => setInputDims({ w, h }),
    trimStart: editState.trimStart, trimEnd: editState.trimEnd, setTrimStart: (t) => setEditState(p => ({ ...p, trimStart: t })), setTrimEnd: (t) => setEditState(p => ({ ...p, trimEnd: t })),
    crop: { x: 0, y: 0, width: 0, height: 0 }, setCrop: () => { }, samplePreview: previewFile,
    renderSample: () => { }, clearPreview: () => setPreviewFile(null),
    renderedRange // Pass to downstream components
  }), [inputPath, videoTime, videoDuration, editState, inputDims, model, previewFile, renderedRange]);

  const addToast = (m: string, t: string) => setToasts(prev => [...prev, { id: Math.random().toString(), message: m, type: t as any }]);
  const pickInput = async () => { const s = await open({ multiple: false }); if (s && typeof s === 'string') handleNewInput(s); };
  const pickOutput = async () => {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const selected = await save({
      title: "Select Output Location",
      filters: [{ name: 'Media', extensions: ['mp4', 'png', 'jpg'] }]
    });
    if (selected) setOutputPath(selected);
  };

  const getRustEditConfig = () => ({
    trim_start: editState.trimStart, trim_end: editState.trimEnd, crop: editState.crop,
    rotation: editState.rotation, flip_h: editState.flipH, flip_v: editState.flipV, fps: editState.fps,
    color: editState.color
  });

  const hintForCategory = (category: string): string | undefined => {
    switch (category) {
      case "policy_violation":
        return "Enable strict-profile requirements (audit/no-host-copies) or switch profile.";
      case "provider_loader_error":
        return "Check CUDA/driver/ORT/TensorRT provider installation and loader paths.";
      case "runtime_dependency_missing":
        return "Install missing runtime dependency and rerun.";
      case "input_contract_error":
        return "Fix input/CLI contract values (for example keep max_batch at 1).";
      case "inference_error":
        return "Model failed during inference. May use unsupported ops or exceed VRAM. Try a smaller model or fp16 precision.";
      case "codec_error":
        return "Codec error during decode/encode. Verify input format is supported and FFmpeg/NVENC/NVDEC are available.";
      case "pipeline_error":
        return "Processing pipeline failed. Check activity log for details.";
      default:
        return undefined;
    }
  };

  const buildCategorizedError = (category: string, message: string): { category: string; message: string; detail?: string; nextAction?: string } => {
    const hint = hintForCategory(category);
    return hint ? { category, message, nextAction: hint } : { category, message };
  };

  const parseRaveError = (err: unknown): { category: string; message: string; detail?: string; nextAction?: string } => {
    const raw = String(err);
    try {
      const payload = JSON.parse(raw) as RaveErrorPayload;
      if (payload && typeof payload === "object" && payload.category && payload.message) {
        return {
          category: payload.category,
          message: payload.message,
          ...(payload.detail !== undefined ? { detail: payload.detail } : {})
          , ...(payload.next_action !== undefined ? { nextAction: payload.next_action } : {})
        };
      }
    } catch {
      // fall back to heuristic classification below
    }

    const lower = raw.toLowerCase();
    if (lower.includes("strict no-host-copies") || lower.includes("host copy audit")) {
      return buildCategorizedError("policy_violation", raw);
    }
    if (lower.includes("provider") || lower.includes("onnxruntime") || lower.includes("tensorrt")) {
      return buildCategorizedError("provider_loader_error", raw);
    }
    if (lower.includes("missing") || lower.includes("not found")) {
      return buildCategorizedError("runtime_dependency_missing", raw);
    }
    if (lower.includes("max_batch")) {
      return buildCategorizedError("input_contract_error", raw);
    }
    return buildCategorizedError("runtime_error", raw);
  };

  const defaultRaveOutputPath = (input: string) => {
    const ext = input.split('.').pop()?.toLowerCase();
    if (ext) return input.replace(new RegExp(`\\.${ext}$`), `_rave_upscaled.mp4`);
    return `${input}_rave_upscaled.mp4`;
  };

  const buildRaveUpscaleArgs = (params: {
    input: string;
    output: string;
    modelPath: string;
    architectureClass?: string;
  }): string[] => {
    const args = [
      "-i", params.input,
      "-m", params.modelPath,
      "-o", params.output,
      "--precision", "fp16",
      "--progress", "jsonl"
    ];
    // Auto-inject tiling for transformer models (DAT2, SwinIR, HAT)
    if (params.architectureClass === "Transformer") {
      args.push("--tile-size", "256");
    }
    return args;
  };

  const buildRaveBenchmarkArgs = (params: {
    input: string;
    modelPath: string;
    architectureClass?: string;
  }): string[] => {
    const args = [
      "-i", params.input,
      "-m", params.modelPath,
      "--skip-encode",
      "--dry-run",
      "--progress", "jsonl"
    ];
    if (params.architectureClass === "Transformer") {
      args.push("--tile-size", "256");
    }
    return args;
  };

  const startUpscale = async () => {
    if (!inputPath) return addToast("Select an input file first!", "error");

    // Guard: If AI upscale is disabled, this shouldn't be called
    if (!upscaleConfig.isEnabled) {
      return addToast("AI Upscale is bypassed. Enable it to upscale.", "warning");
    }

    const jobId = Date.now().toString();
    const newJob: Job = { id: jobId, command: `Upscale: ${inputPath.split(/[\\/]/).pop()}`, status: "running", progress: 0, statusMessage: "Initializing...", paused: false, eta: 0, startedAt: Date.now() };
    setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);
    if (!panels.QUEUE) openPanel('QUEUE');

    // Use store's scaleFactor as source of truth, with model string as fallback
    const activeScale = upscaleConfig.scaleFactor || getScaleFromModel(model);

    // Build comprehensive upscale payload with all new fields
    const upscalePayload = {
      inputPath,
      outputPath,
      model: upscaleConfig.primaryModelId || model, // Use store's model as source of truth
      editConfig: getRustEditConfig(),
      scale: activeScale,
      // New architecture-aware fields
      architectureClass: upscaleConfig.architectureClass,
      // Secondary model blending
      secondaryModel: null, // Managed via ResearchConfig
      blendAlpha: 0, // Managed via ResearchConfig
      // Custom resolution overrides
      resolutionMode: upscaleConfig.resolutionMode,
      targetWidth: upscaleConfig.resolutionMode === 'target' ? upscaleConfig.targetWidth : null,
      targetHeight: upscaleConfig.resolutionMode === 'target' ? upscaleConfig.targetHeight : null,
    };

    console.debug('[App] Upscale payload:', upscalePayload);

    try {
      const selectedModel = upscaleConfig.primaryModelId || model;
      const info = modelInfoMap.get(selectedModel);

      let resultPath: string;
      let policy: RavePolicy | undefined;
      let hostCopyAuditEnabled: boolean | undefined;
      let hostCopyAuditDisableReason: string | null | undefined;
      const canUseNative = upscaleConfig.useNativeEngine && mode === 'video';
      if (canUseNative) {
        if (info?.format === "onnx") {
          // Route native-video path through rave-cli JSON contract.
          const resolvedOutputPath = outputPath?.trim() ? outputPath : defaultRaveOutputPath(inputPath);
          if (showTechSpecs) {
            try {
              const benchmark = await invoke<RaveCommandJson>("rave_benchmark", {
                args: buildRaveBenchmarkArgs({
                  input: inputPath,
                  modelPath: info.path,
                  architectureClass: upscaleConfig.architectureClass
                }),
                strictAudit: true,
                mockRun: false,
                uiOptIn: true
              });
              setLogs(prev => [
                ...prev,
                `[RAVE] benchmark dry-run fps=${String(benchmark.fps ?? "n/a")} policy=${JSON.stringify(benchmark.policy ?? {})}`
              ]);
            } catch (benchErr) {
              setLogs(prev => [...prev, `[RAVE] benchmark dry-run failed: ${benchErr}`]);
            }
          }

          const raveResult = await invoke<RaveCommandJson>("rave_upscale", {
            args: buildRaveUpscaleArgs({
              input: inputPath,
              output: resolvedOutputPath,
              modelPath: info.path,
              architectureClass: upscaleConfig.architectureClass
            }),
            strictAudit: true,
            mockRun: false,
            uiOptIn: true
          });
          if (!raveResult.output || typeof raveResult.output !== "string") {
            throw new Error("rave_upscale did not return a valid output path");
          }
          resultPath = raveResult.output;
          policy = raveResult.policy;
          hostCopyAuditEnabled = raveResult.host_copy_audit_enabled;
          hostCopyAuditDisableReason = raveResult.host_copy_audit_disable_reason;
        } else {
          // Native engine selected but current model is not ONNX — fall back.
          addToast(
            `Native engine requires an ONNX model. "${selectedModel}" is PyTorch-only — using Python pipeline.`,
            "warning"
          );
          resultPath = await invoke<string>("upscale_request", upscalePayload);
        }
      } else {
        resultPath = await invoke<string>("upscale_request", upscalePayload);
      }
      setLogs(prev => [...prev, `[SYSTEM] Job ${jobId} finished.`]);
      const finishedJob: Job = {
        ...newJob,
        status: 'done',
        progress: 100,
        outputPath: resultPath,
        eta: 0,
        completedAt: Date.now(),
        ...(policy ? { policy } : {}),
        ...(typeof hostCopyAuditEnabled === "boolean" ? { hostCopyAuditEnabled } : {}),
        ...(hostCopyAuditDisableReason !== undefined ? { hostCopyAuditDisableReason } : {})
      };
      setJobs(prev => prev.map(j => j.id === jobId ? finishedJob : j));
      setActiveJob(finishedJob);
      setLastOutputPath(resultPath);
    } catch (err) {
      const normalized = parseRaveError(err);
      const msg = `[${normalized.category}] ${normalized.message}`;
      addToast(`Error: ${msg}`, "error");
      setLogs(prev => [...prev, `[ERROR][RAVE][${normalized.category}] Job ${jobId} failed: ${normalized.message}`]);
      setJobs(prev => prev.map(j => j.id === jobId ? {
        ...j,
        status: 'error',
        statusMessage: msg,
        errorCategory: normalized.category,
        ...(normalized.nextAction ? { errorHint: normalized.nextAction } : {}),
        errorMessage: normalized.detail ? `${normalized.message} :: ${normalized.detail}` : normalized.message,
        completedAt: Date.now()
      } : j));
    } finally { setIsProcessing(false); }
  };

  const onExportEdited = async () => {
    if (!inputPath) return addToast("Select input first!", "error");
    const jobId = Date.now().toString();
    const newJob: Job = { id: jobId, command: `Transcode: ${inputPath.split(/[\\/]/).pop()}`, status: "running", progress: 0, statusMessage: "Encoding...", paused: false, eta: 0, startedAt: Date.now() };
    setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);
    if (!panels.QUEUE) openPanel('QUEUE');

    try {
      const resultPath = await invoke<string>("export_request", { inputPath, outputPath, editConfig: getRustEditConfig(), scale: 1 });
      setLogs(prev => [...prev, `[SYSTEM] Export ${jobId} complete.`]);
      addToast("Export Completed", "success");
      const finishedJob: Job = { ...newJob, status: 'done', progress: 100, outputPath: resultPath, eta: 0, completedAt: Date.now() };
      setJobs(prev => prev.map(j => j.id === jobId ? finishedJob : j));
      setActiveJob(finishedJob);
      setLastOutputPath(resultPath);
    } catch (err) {
      const normalized = parseRaveError(err);
      const msg = `[${normalized.category}] ${normalized.message}`;
      addToast(`Error: ${msg}`, "error");
      setLogs(prev => [...prev, `[ERROR][${normalized.category}] Export ${jobId} failed: ${normalized.message}`]);
      setJobs(prev => prev.map(j => j.id === jobId ? {
        ...j,
        status: 'error',
        statusMessage: msg,
        errorCategory: normalized.category,
        ...(normalized.nextAction ? { errorHint: normalized.nextAction } : {}),
        errorMessage: normalized.detail ? `${normalized.message} :: ${normalized.detail}` : normalized.message,
        completedAt: Date.now()
      } : j));
    } finally { setIsProcessing(false); }
  };

  const startRaveValidate = async () => {
    if (!inputPath) return addToast("Select an input file first!", "error");

    const jobId = `validate_${Date.now().toString()}`;
    const newJob: Job = {
      id: jobId,
      command: "Validate: production_strict (mock)",
      status: "running",
      progress: 0,
      statusMessage: "Validating strict policy...",
      paused: false,
      eta: 0,
      startedAt: Date.now()
    };
    setJobs(prev => [...prev, newJob]);
    setActiveJob(newJob);
    setIsProcessing(true);
    if (!panels.QUEUE) openPanel('QUEUE');

    try {
      const result = await invoke<RaveCommandJson>("rave_validate", {
        fixture: null,
        profile: "production_strict",
        bestEffort: true,
        strictAudit: true,
        mockRun: true
      });

      const policy = result.policy;
      const hostCopyAuditEnabled = result.host_copy_audit_enabled;
      const hostCopyAuditDisableReason = result.host_copy_audit_disable_reason;
      const skipped = result.skipped === true;

      const finishedJob: Job = {
        ...newJob,
        status: "done",
        progress: 100,
        statusMessage: skipped ? "Validation skipped" : "Validation passed",
        completedAt: Date.now(),
        ...(policy ? { policy } : {}),
        ...(typeof hostCopyAuditEnabled === "boolean" ? { hostCopyAuditEnabled } : {}),
        ...(hostCopyAuditDisableReason !== undefined ? { hostCopyAuditDisableReason } : {})
      };
      setJobs(prev => prev.map(j => j.id === jobId ? finishedJob : j));
      setActiveJob(finishedJob);
      setLogs(prev => [
        ...prev,
        `[RAVE] validate ok=${String(result.ok ?? "unknown")} skipped=${String(result.skipped ?? false)} policy=${JSON.stringify(policy ?? {})}`
      ]);
      addToast(skipped ? "Validate completed (skipped)" : "Validate passed", skipped ? "warning" : "success");
    } catch (err) {
      const normalized = parseRaveError(err);
      setJobs(prev => prev.map(j => j.id === jobId ? {
        ...j,
        status: "error",
        statusMessage: `[${normalized.category}] ${normalized.message}`,
        errorCategory: normalized.category,
        ...(normalized.nextAction ? { errorHint: normalized.nextAction } : {}),
        errorMessage: normalized.detail ? `${normalized.message} :: ${normalized.detail}` : normalized.message,
        completedAt: Date.now()
      } : j));
      setLogs(prev => [...prev, `[RAVE][${normalized.category}] validate failed: ${normalized.message}`]);
      addToast(`Validate failed: [${normalized.category}] ${normalized.message}`, "error");
    } finally {
      setIsProcessing(false);
    }
  };

  const renderPreviewSample = async () => {
    if (!inputPath || mode !== 'video') return;
    addToast("Rendering 2s Sample...", "info"); setPreviewFile(null);
    const start = Math.max(0, videoTime); const safeDuration = videoDuration > 0 ? videoDuration : 1000; const end = Math.min(safeDuration, start + 2.0);
    const previewConfig = { ...getRustEditConfig(), trim_start: start, trim_end: end };
    let activeScale = upscaleConfig.scaleFactor || getScaleFromModel(model);
    const jobId = "preview_" + Date.now().toString().slice(-6);
    const newJob: Job = { id: jobId, command: `PREVIEW SAMPLE`, status: "running", progress: 0, statusMessage: "Rendering...", paused: false, eta: 0, startedAt: Date.now() };
    setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);

    // Build comprehensive preview payload with all new fields
    const previewPayload = {
      inputPath,
      outputPath: "",
      model: upscaleConfig.primaryModelId || model,
      editConfig: previewConfig,
      scale: activeScale,
      // New architecture-aware fields
      architectureClass: upscaleConfig.architectureClass,
      // Secondary model blending
      secondaryModel: null, // Managed via ResearchConfig
      blendAlpha: 0, // Managed via ResearchConfig
      // Custom resolution overrides (use scale mode for previews typically)
      resolutionMode: upscaleConfig.resolutionMode,
      targetWidth: upscaleConfig.resolutionMode === 'target' ? upscaleConfig.targetWidth : null,
      targetHeight: upscaleConfig.resolutionMode === 'target' ? upscaleConfig.targetHeight : null,
    };

    try {
      const resultPath = await invoke<string>("upscale_request", previewPayload);
      setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'done', progress: 100, outputPath: resultPath, completedAt: Date.now() } : j));
      setActiveJob(prev => prev?.id === jobId ? { ...prev, status: 'done', outputPath: resultPath, progress: 100 } : prev);
      setPreviewFile(resultPath); addToast("Preview Ready", "success");
      setRenderedRange({ start, end }); // Set specific sample range
    } catch (e) {
      const normalized = parseRaveError(e);
      addToast(`Preview Failed: [${normalized.category}] ${normalized.message}`, "error");
      setLogs(prev => [...prev, `[ERROR][${normalized.category}] Preview ${jobId} failed: ${normalized.message}`]);
      setJobs(prev => prev.map(j => j.id === jobId ? {
        ...j,
        status: 'error',
        statusMessage: `[${normalized.category}] ${normalized.message}`,
        errorCategory: normalized.category,
        ...(normalized.nextAction ? { errorHint: normalized.nextAction } : {}),
        errorMessage: normalized.detail ? `${normalized.message} :: ${normalized.detail}` : normalized.message,
        completedAt: Date.now()
      } : j));
    } finally { setIsProcessing(false); }
  };

  // --- POLISH: Clear Completed Logic ---
  const clearCompletedJobs = () => {
    setJobs(prev => prev.filter(j => j.status === 'running' || j.status === 'queued' || j.status === 'paused'));
  };

  // --- Cancel / Dismiss Job Logic ---
  const handleCancelJob = async (id: string) => {
    const job = jobs.find(j => j.id === id);
    if (!job) return;

    if (job.status === 'running' || job.status === 'paused') {
      try {
        // Assume backend has a cancellation command, or just mark as cancelled in UI if backend is fire-and-forget
        // For now, we'll mark as cancelled. If backend support exists, invoke it here.
        // await invoke('cancel_job', { jobId: id }); 
        setJobs(prev => prev.map(j => j.id === id ? { ...j, status: 'cancelled', progress: 0, eta: 0, completedAt: Date.now() } : j));
        setLogs(prev => [...prev, `[SYSTEM] Job ${id} cancelled by user.`]);
        if (activeJob?.id === id) setActiveJob(null);
        setIsProcessing(false);
      } catch (err) {
        addToast("Failed to cancel job", "error");
      }
    } else {
      // Dismiss (Delete from list)
      setJobs(prev => prev.filter(j => j.id !== id));
      if (activeJob?.id === id) setActiveJob(null);
    }
  };

  const isValidPaths = !!inputPath;

  const tileComponents = useMemo(() => {
    const completeVideoState = { ...videoState, renderSample: renderPreviewSample };
    return {
      SETTINGS: <InputOutputPanel mode={mode} setMode={setMode} pickInput={pickInput} inputPath={inputPath} pickOutput={pickOutput} outputPath={outputPath} model={model} setModel={setModel} availableModels={availableModels} loadingModel={loadingModel} loadModel={() => { }} startUpscale={startUpscale} onRunValidate={startRaveValidate} isValidPaths={isValidPaths} showTech={showTechSpecs} showResearchParams={showResearchParams} videoState={completeVideoState} editState={editState} setEditState={setEditState} onExportEdited={onExportEdited} viewMode={viewMode} setViewMode={setViewMode} />,
      PREVIEW: <PreviewPanel inputPreview={inputPath} activeJob={activeJob} videoState={completeVideoState} onFileDrop={handleNewInput} mode={mode} editState={editState} setEditState={setEditState} viewMode={viewMode} setViewMode={setViewMode} showTech={showTechSpecs} />,
      // Updated to pass clearCompleted
      QUEUE: <JobsPanel jobs={jobs} pauseJob={() => { }} cancelJob={handleCancelJob} resumeJob={() => { }} clearCompleted={clearCompletedJobs} showTech={showTechSpecs} />,
      ACTIVITY: <LogsPanel logs={logs} setLogs={setLogs} darkMode={darkMode} logsEndRef={logsEndRef} />
    };
  }, [mode, inputPath, outputPath, model, availableModels, loadingModel, isValidPaths, showTechSpecs, showResearchParams, videoState, editState, viewMode, jobs, activeJob, logs, upscaleConfig, modelInfoMap]);

  const renderTile = useCallback((id: PanelId, path: any[]) => {
    return (
      <MosaicWindow<PanelId>
        title={id} path={path}
        renderToolbar={(props) => (
          <div style={{ width: '100%', height: '100%' }}>
            <PanelHeader title={props.title} onClose={() => togglePanel(id)} />
          </div>
        )}
      >
        {tileComponents[id]}
      </MosaicWindow>
    );
  }, [tileComponents, togglePanel]);

  if (!isEngineReady && !checkingEngine) return <div>Install Engine Screen...</div>;

  return (
    <div className={`app-container ${darkMode ? "bp5-dark" : "light-mode-container"}`} style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <TitleBar />

      {/* Reduced outer padding to 4px to match inner gaps */}
      <div style={{ flex: 1, overflow: 'hidden', padding: '4px', position: 'relative' }}>
        <Mosaic<PanelId>
          renderTile={renderTile}
          value={mosaicValue}
          onChange={handleMosaicChange}
          className={darkMode ? "mosaic-blueprint-theme" : "mosaic-light-theme"}
          zeroStateView={<EmptyState />}
        />

        {/* DOCK STRIPS */}
        {!panels.SETTINGS && (
          <DockStrip
            position="left"
            label="SETTINGS"
            panelId="SETTINGS"
            icon={<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>}
            onClick={() => openPanel('SETTINGS')}
          />
        )}
        {!panels.QUEUE && (
          <DockStrip
            position="right"
            label="QUEUE"
            panelId="QUEUE"
            icon={<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 6h16M4 12h16M4 18h16" /></svg>}
            onClick={() => openPanel('QUEUE')}
          />
        )}
        {!panels.ACTIVITY && (
          <DockStrip
            position="bottom"
            label="ACTIVITY"
            panelId="ACTIVITY"
            icon={<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>}
            onClick={() => openPanel('ACTIVITY')}
          />
        )}
      </div>
      <StatusFooter
        toggleTheme={() => setDarkMode(!darkMode)}
        darkMode={darkMode}
        showTechSpecs={showTechSpecs}
        setShowTechSpecs={setShowTechSpecs}
        showResearchParams={showResearchParams}
        setShowResearchParams={setShowResearchParams}
      />
    </div>
  );
};

export default App;
