/* ui/src/App.tsx */
import React, { useState, useMemo, useEffect, useRef, useCallback } from "react";
import {
  Mosaic,
  MosaicWindow,
  MosaicNode,
  getLeaves,
  createBalancedTreeFromLeaves
} from "react-mosaic-component";
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

import type { Toast, UpscaleMode, Job, VideoState, EditState } from './types';

const SUPPORTED_MODELS = [
  "RealESRGAN_x4plus.pth",
  "RealESRGAN_x4plus_anime_6B.pth",
  "RealESRGAN_x2plus.pth",
  "RealESRGAN_x2plus_anime_6B.pth"
];

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
  splitPercentage: 20
};

const DockStrip = ({ position, onClick, label, icon }: { position: 'right' | 'bottom' | 'left', onClick: () => void, label: string, icon: React.ReactNode }) => {
  const isVertical = position === 'right' || position === 'left';
  return (
    <div
      onClick={onClick}
      title={`Open ${label}`}
      className="dock-strip"
      style={{
        position: 'absolute',
        [position]: 0,
        [isVertical ? 'top' : 'left']: '50%',
        transform: isVertical ? 'translateY(-50%)' : 'translateX(-50%)',
        [isVertical ? 'width' : 'height']: '24px',
        borderRadius: position === 'right' ? '4px 0 0 4px' : position === 'left' ? '0 4px 4px 0' : '4px 4px 0 0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        zIndex: 50,
        padding: isVertical ? '12px 0' : '0 12px',
        boxShadow: 'var(--shadow-md)'
      }}
    >
      <div style={{
        writingMode: isVertical ? 'vertical-rl' : 'horizontal-tb',
        transform: position === 'right' ? 'rotate(180deg)' : 'none',
        fontSize: '10px', fontWeight: 600, letterSpacing: '0.1em',
        display: 'flex', alignItems: 'center', gap: '8px'
      }}>
        {icon}
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
  const [loadingModel, setLoadingModel] = useState(false);
  const [showTechSpecs, setShowTechSpecs] = useState(false);
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
  const { setIsProcessing, setCurrentFrame } = useJobStore();

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

  useTauriEvents({ setJobs, setLogs, setActiveJob, setLoadingModel, setCurrentFrame, addToast: (msg: string, type: string) => addToast(msg, type as any) });

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
      invoke<any[]>("get_models").then(models => {
        const filtered = models.map(m => m.id).filter(id => SUPPORTED_MODELS.includes(id));
        setAvailableModels(filtered.length === 0 ? SUPPORTED_MODELS : filtered);
      }).catch(e => setAvailableModels(SUPPORTED_MODELS));
    } else {
      setAvailableModels(SUPPORTED_MODELS);
    }
  }, [isEngineReady]);

  const handleNewInput = (path: string) => {
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
  };

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

  const startUpscale = async () => {
    if (!inputPath) return addToast("Select an input file first!", "error");
    const jobId = Date.now().toString();
    const newJob: Job = { id: jobId, command: `Upscale: ${inputPath.split(/[\\/]/).pop()}`, status: "running", progress: 0, statusMessage: "Initializing...", paused: false, eta: 0 };
    setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);
    if (!panels.QUEUE) openPanel('QUEUE');

    let activeScale = 4;
    if (model.includes("x2")) activeScale = 2;

    try {
      const resultPath = await invoke<string>("upscale_request", { inputPath, outputPath, model, editConfig: getRustEditConfig(), scale: activeScale });
      setLogs(prev => [...prev, `[SYSTEM] Job ${jobId} finished.`]);
      const finishedJob: Job = { ...newJob, status: 'done', progress: 100, outputPath: resultPath, eta: 0 };
      setJobs(prev => prev.map(j => j.id === jobId ? finishedJob : j));
      setActiveJob(finishedJob);
    } catch (err) {
      addToast(`Error: ${err}`, "error");
      setLogs(prev => [...prev, `[ERROR] Job ${jobId} failed: ${err}`]);
      setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'error', statusMessage: String(err) } : j));
    } finally { setIsProcessing(false); }
  };

  const onExportEdited = async () => {
    if (!inputPath) return addToast("Select input first!", "error");
    const jobId = Date.now().toString();
    const newJob: Job = { id: jobId, command: `Transcode: ${inputPath.split(/[\\/]/).pop()}`, status: "running", progress: 0, statusMessage: "Encoding...", paused: false, eta: 0 };
    setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);
    if (!panels.QUEUE) openPanel('QUEUE');

    try {
      const resultPath = await invoke<string>("export_request", { inputPath, outputPath, editConfig: getRustEditConfig(), scale: 1 });
      setLogs(prev => [...prev, `[SYSTEM] Export ${jobId} complete.`]);
      addToast("Export Completed", "success");
      const finishedJob: Job = { ...newJob, status: 'done', progress: 100, outputPath: resultPath, eta: 0 };
      setJobs(prev => prev.map(j => j.id === jobId ? finishedJob : j));
      setActiveJob(finishedJob);
    } catch (err) {
      addToast(`Error: ${err}`, "error");
      setLogs(prev => [...prev, `[ERROR] Export ${jobId} failed: ${err}`]);
      setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'error', statusMessage: String(err) } : j));
    } finally { setIsProcessing(false); }
  };

  const renderPreviewSample = async () => {
    if (!inputPath || mode !== 'video') return;
    addToast("Rendering 2s Sample...", "info"); setPreviewFile(null);
    const start = Math.max(0, videoTime); const safeDuration = videoDuration > 0 ? videoDuration : 1000; const end = Math.min(safeDuration, start + 2.0);
    const previewConfig = { ...getRustEditConfig(), trim_start: start, trim_end: end };
    let activeScale = 4; if (model.includes("x2")) activeScale = 2;
    const jobId = "preview_" + Date.now().toString().slice(-6);
    const newJob: Job = { id: jobId, command: `PREVIEW SAMPLE`, status: "running", progress: 0, statusMessage: "Rendering...", paused: false, eta: 0 };
    setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);
    try {
      const resultPath = await invoke<string>("upscale_request", { inputPath, outputPath: "", model, editConfig: previewConfig, scale: activeScale });
      setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'done', progress: 100, outputPath: resultPath } : j));
      setActiveJob(prev => prev?.id === jobId ? { ...prev, status: 'done', outputPath: resultPath, progress: 100 } : prev);
      setPreviewFile(resultPath); addToast("Preview Ready", "success");
      setRenderedRange({ start, end }); // Set specific sample range
    } catch (e) {
      addToast("Preview Failed: " + e, "error");
      setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'error', errorMessage: String(e) } : j));
    } finally { setIsProcessing(false); }
  };

  // --- POLISH: Clear Completed Logic ---
  const clearCompletedJobs = () => {
    setJobs(prev => prev.filter(j => j.status === 'running' || j.status === 'queued' || j.status === 'paused'));
  };

  const isValidPaths = !!inputPath;

  const tileComponents = useMemo(() => {
    const completeVideoState = { ...videoState, renderSample: renderPreviewSample };
    return {
      SETTINGS: <InputOutputPanel mode={mode} setMode={setMode} pickInput={pickInput} inputPath={inputPath} pickOutput={pickOutput} outputPath={outputPath} model={model} setModel={setModel} availableModels={availableModels} loadingModel={loadingModel} loadModel={() => { }} startUpscale={startUpscale} isValidPaths={isValidPaths} showTech={showTechSpecs} videoState={completeVideoState} editState={editState} setEditState={setEditState} onExportEdited={onExportEdited} viewMode={viewMode} setViewMode={setViewMode} />,
      PREVIEW: <PreviewPanel inputPreview={inputPath} activeJob={activeJob} videoState={completeVideoState} onFileDrop={handleNewInput} mode={mode} editState={editState} setEditState={setEditState} viewMode={viewMode} setViewMode={setViewMode} showTech={showTechSpecs} />,
      // Updated to pass clearCompleted
      QUEUE: <JobsPanel jobs={jobs} pauseJob={() => { }} cancelJob={() => { }} resumeJob={() => { }} clearCompleted={clearCompletedJobs} showTech={showTechSpecs} />,
      ACTIVITY: <LogsPanel logs={logs} setLogs={setLogs} darkMode={darkMode} logsEndRef={logsEndRef} />
    };
  }, [mode, inputPath, outputPath, model, availableModels, loadingModel, isValidPaths, showTechSpecs, videoState, editState, viewMode, jobs, activeJob, logs]);

  const renderTile = useCallback((id: PanelId, path: any[]) => {
    return (
      <MosaicWindow<PanelId>
        title={id} path={path}
        renderToolbar={(props) => (
          <div style={{ width: '100%', height: '100%' }}>
            <PanelHeader title={props.title} onSplit={props.onSplit} onClose={() => togglePanel(id)} />
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
          zeroStateView={<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#555', fontSize: '12px' }}>All panels closed. Use View menu to restore.</div>}
        />

        {/* DOCK STRIPS */}
        {!panels.SETTINGS && (
          <DockStrip
            position="left"
            label="SETTINGS"
            // Use a settings/sliders icon
            icon={<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>}
            onClick={() => openPanel('SETTINGS')}
          />
        )}
        {!panels.QUEUE && (
          <DockStrip
            position="right"
            label="QUEUE"
            icon={<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 6h16M4 12h16M4 18h16" /></svg>}
            onClick={() => openPanel('QUEUE')}
          />
        )}
        {!panels.ACTIVITY && (
          <DockStrip
            position="bottom"
            label="ACTIVITY"
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
        outputPath={outputPath}
      />
    </div>
  );
};

export default App;
