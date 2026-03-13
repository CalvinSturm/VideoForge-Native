import React, { useMemo, useState, useRef, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { EditState, VideoState, UpscaleMode } from "../types";
import { SignalSummary } from "./SignalSummary";
import { AIUpscaleNode } from "./AIUpscaleNode";
import {
  HF_METHODS,
  RESEARCH_DEFAULTS,
  RESEARCH_PRESETS,
  type ResearchConfig,
} from "./inputOutputPanel/researchConfig";
import {
  useJobStore,
  type UpscaleScale,
} from "../Store/useJobStore";
import { extractFamily, extractScale } from "../utils/modelClassification";
import {
  ASPECT_RATIOS,
  FPS_OPTIONS,
  getSmartResInfo,
  truncateModelName,
} from "./inputOutputPanel/panelHelpers";
import {
  IconCamera,
  IconCheck,
  IconChevronDown,
  IconClock,
  IconCpu,
  IconCrop,
  IconExport,
  IconFile,
  IconFilm,
  IconFlash,
  IconFlipH,
  IconFlipV,
  IconImport,
  IconInfo,
  IconLock,
  IconMove,
  IconPalette,
  IconPlay,
  IconPlus,
  IconRotateCCW,
  IconRotateCW,
  IconSave,
  IconShield,
  IconSparkles,
  IconX,
} from "./inputOutputPanel/panelIcons";
import {
  ColorSlider,
  PipelineConnector,
  PipelineNode,
  SmartPath,
  Tooltip,
} from "./inputOutputPanel/panelPrimitives";
import { ResearchControlsSection } from "./inputOutputPanel/ResearchControlsSection";
import { InputSourceSection } from "./inputOutputPanel/InputSourceSection";
import { GeometryControlsSection } from "./inputOutputPanel/GeometryControlsSection";
import { PostProcessingSection } from "./inputOutputPanel/PostProcessingSection";
import { PanelActionFooter } from "./inputOutputPanel/PanelActionFooter";

// --- TOAST COMPONENT ---
const ToastNotification: React.FC<{ message: string; visible: boolean; onDismiss: () => void; duration?: number }> = ({
  message, visible, onDismiss, duration = 3000
}) => {
  const [render, setRender] = useState(visible);

  useEffect(() => {
    if (visible) {
      setRender(true);
      // Auto-dismiss after duration
      const timer = setTimeout(() => onDismiss(), duration);
      return () => clearTimeout(timer);
    } else {
      const hideTimer = setTimeout(() => setRender(false), 300);
      return () => clearTimeout(hideTimer);
    }
  }, [visible, duration, onDismiss]);

  if (!render) return null;

  return (
    <div style={{
      position: 'absolute', top: '16px', right: '16px', zIndex: 2000,
      background: 'var(--toast-bg)', border: '1px solid var(--toast-border)',
      borderLeft: '3px solid var(--brand-primary)',
      borderRadius: '4px', padding: '10px 12px',
      boxShadow: 'var(--shadow-md)',
      display: 'flex', alignItems: 'center', gap: '10px',
      transform: visible ? 'translateX(0)' : 'translateX(100%)',
      opacity: visible ? 1 : 0,
      transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
      maxWidth: '260px'
    }}>
      <span style={{ fontSize: '11px', color: 'var(--toast-text)', fontFamily: 'var(--font-sans)', fontWeight: 500 }}>{message}</span>
      <button onClick={onDismiss} style={{
        background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '0 4px', fontSize: '14px'
      }}>×</button>
    </div>
  );
};

// --- SUB-COMPONENTS ---

const SelectionCard = ({ selected, onClick, title, subtitle, icon, disabled, badge }: {
  selected: boolean;
  onClick: () => void;
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  disabled?: boolean;
  badge?: React.ReactNode;
}) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        justifyContent: "center",
        height: "60px",
        padding: "10px 12px",
        minWidth: "90px",
        background: selected && !disabled
          ? "linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,255,136,0.05))"
          : isHovered && !disabled
            ? "rgba(255,255,255,0.04)"
            : "rgba(255,255,255,0.02)",
        border: selected && !disabled
          ? "1px solid rgba(0,255,136,0.5)"
          : "1px solid rgba(255,255,255,0.08)",
        borderRadius: "8px",
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "all 0.2s ease",
        position: "relative",
        overflow: "hidden",
        opacity: disabled ? 0.4 : 1,
        boxShadow: selected && !disabled
          ? "0 4px 16px rgba(0,255,136,0.15), inset 0 1px 0 rgba(255,255,255,0.05)"
          : "inset 0 1px 0 rgba(255,255,255,0.03)"
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px", width: '100%' }}>
        <div style={{
          color: selected && !disabled ? "var(--brand-primary)" : "var(--text-muted)",
          transition: "color 0.15s"
        }}>
          {icon}
        </div>
        <span style={{
          fontWeight: 700,
          fontSize: "11px",
          color: selected && !disabled ? "var(--text-primary)" : "var(--text-secondary)",
          fontFamily: 'var(--font-sans)',
          letterSpacing: '0.02em',
          flex: 1
        }}>
          {title}
        </span>
        {badge}
      </div>
      <span style={{
        fontSize: "9px",
        color: selected && !disabled ? "var(--brand-primary)" : "var(--text-muted)",
        marginLeft: "24px",
        fontFamily: 'var(--font-mono)',
        letterSpacing: "0.05em",
        opacity: 0.8
      }}>
        {subtitle}
      </span>

      {/* Selection indicator */}
      {selected && !disabled && (
        <>
          <div style={{
            position: "absolute",
            top: "6px",
            right: "6px",
            width: "16px",
            height: "16px",
            borderRadius: "50%",
            background: "var(--brand-primary)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#000",
            boxShadow: "0 2px 6px rgba(0,255,136,0.3)"
          }}>
            <IconCheck />
          </div>
          {/* Glow effect */}
          <div style={{
            position: "absolute",
            inset: 0,
            background: "radial-gradient(circle at 50% 100%, rgba(0,255,136,0.1), transparent 60%)",
            pointerEvents: "none"
          }} />
        </>
      )}
    </button>
  );
};

// Model families and IDs are now derived dynamically from availableModels.

interface InputOutputPanelProps {
  mode: UpscaleMode; setMode: (mode: UpscaleMode) => void;
  pickInput: () => void; inputPath: string;
  pickOutput: () => void; outputPath: string;
  model: string; setModel: (model: string) => void;
  availableModels: string[]; loadingModel: boolean; loadModel: (model: string) => void;
  startUpscale: () => void; isValidPaths: boolean;
  onRunValidate: () => void;
  videoState: VideoState; editState: EditState; setEditState: (state: EditState) => void;
  onExportEdited: () => void; showTech: boolean;
  showResearchParams?: boolean;
  viewMode: 'edit' | 'preview'; setViewMode: (mode: 'edit' | 'preview') => void;
}

export const InputOutputPanel: React.FC<InputOutputPanelProps> = ({
  mode, setMode, pickInput, inputPath, pickOutput, outputPath,
  model, setModel, loadModel, availableModels, loadingModel,
  startUpscale, isValidPaths, onRunValidate, videoState, editState, setEditState, onExportEdited,
  showTech, showResearchParams = true,
}) => {

  const panelRef = useRef<HTMLDivElement>(null);

  // ==========================================
  // UPSCALE CONFIG FROM STORE (Centralized State)
  // ==========================================
  const { upscaleConfig, setUpscaleConfig } = useJobStore();

  const isAIActive = upscaleConfig.isEnabled;
  const modelFamily = extractFamily(upscaleConfig.primaryModelId);
  const upscaleFactor = upscaleConfig.scaleFactor;
  const useNativeEngine = upscaleConfig.useNativeEngine;

  const setIsAIActive = (enabled: boolean) => setUpscaleConfig({ isEnabled: enabled });
  const setUseNativeEngine = (enabled: boolean) => setUpscaleConfig({ useNativeEngine: enabled });

  // Local UI state (not persisted)
  const [toastState, setToastState] = useState<{ msg: string; visible: boolean }>({ msg: '', visible: false });
  const [researchConfig, setResearchConfigLocal] = useState<ResearchConfig>(RESEARCH_DEFAULTS);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const showToast = (msg: string) => {
    setToastState({ msg, visible: true });
    setTimeout(() => setToastState(s => ({ ...s, visible: false })), 4000);
  };

  // Fetch research config from backend on mount
  useEffect(() => {
    invoke<ResearchConfig>("get_research_config")
      .then(setResearchConfigLocal)
      .catch(() => { });
  }, []);

  const updateResearchParam = useCallback((key: keyof ResearchConfig, value: number | string | boolean) => {
    setResearchConfigLocal(prev => ({ ...prev, [key]: value }));
    invoke("update_research_param", { key, value }).catch(() => { });
  }, []);

  const applyResearchPreset = useCallback((presetName: string) => {
    const preset = RESEARCH_PRESETS[presetName];
    if (!preset) return;
    const newConfig = { ...researchConfig, ...preset };
    setResearchConfigLocal(newConfig);
    invoke("set_research_config", { config: newConfig }).catch(() => { });
  }, [researchConfig]);

  // --- DERIVED STATE ---
  const activeScale = upscaleFactor;
  const sourceW = videoState.inputWidth || 0;
  const sourceH = videoState.inputHeight || 0;
  const sourceFps = videoState.sourceFps || 30;

  const isCropActive = !!editState.crop;
  const isCropApplied = (editState.crop as any)?.applied === true;

  const cropW = editState.crop ? Math.round(sourceW * editState.crop.width) : sourceW;
  const cropH = editState.crop ? Math.round(sourceH * editState.crop.height) : sourceH;

  const isRotated = editState.rotation === 90 || editState.rotation === 270;
  const finalInW = isRotated ? cropH : cropW;
  const finalInH = isRotated ? cropW : cropH;

  const targetW = isAIActive ? finalInW * activeScale : finalInW;
  const targetH = isAIActive ? finalInH * activeScale : finalInH;
  const targetFps = editState.fps === 0 ? sourceFps : editState.fps;

  const toggleCrop = () => {
    if (isCropActive) {
      setEditState({ ...editState, crop: null, aspectRatio: null });
    } else {
      setEditState({ ...editState, crop: { x: 0.1, y: 0.1, width: 0.8, height: 0.8, applied: false } as any });
    }
  };

  const applyCrop = () => {
    if (!editState.crop) return;
    const newApplied = !editState.crop.applied;
    setEditState({ ...editState, crop: { ...editState.crop, applied: newApplied } as any });
  };

  const applyAspectRatio = (val: number | null) => {
    let newState = { ...editState, aspectRatio: val };
    if (!val || !sourceW) { setEditState(newState); return; }
    const ratio = val * (sourceH / sourceW);
    let w = 0.8; let h = w / ratio;
    if (h > 0.8) { h = 0.8; w = h * ratio; }
    setEditState({ ...newState, crop: { x: (1 - w) / 2, y: (1 - h) / 2, width: w, height: h, applied: false } as any });
  };

  const sourceInfo = getSmartResInfo(sourceW, sourceH);
  const targetInfo = getSmartResInfo(targetW, targetH);
  const strSourceFps = mode === 'video' ? `${sourceFps} FPS` : '';
  const strTargetFps = mode === 'video' ? `${targetFps} FPS` : '';

  const hasGeometryEdits = isCropActive || editState.rotation !== 0 || editState.flipH || editState.flipV;
  const hasMotionEdits = editState.fps !== 0;
  const hasColorEdits = Math.abs(editState.color.brightness) > 0.001 ||
    Math.abs(editState.color.contrast) > 0.001 ||
    Math.abs(editState.color.saturation) > 0.001 ||
    Math.abs(editState.color.gamma - 1.0) > 0.001;
  const hasEdits = hasGeometryEdits || hasMotionEdits || hasColorEdits;

  let mainActionLabel = "NO ACTION";
  let mainActionHandler = () => { };
  let isMainActionDisabled = !isValidPaths;
  let isHighIntensity = false;

  // Build a descriptive label for what edits are active
  const activeEditNames: string[] = [];
  if (isCropActive) activeEditNames.push("Crop");
  if (hasColorEdits) activeEditNames.push("Color");
  if (editState.rotation !== 0 || editState.flipH || editState.flipV) activeEditNames.push("Transform");
  if (hasMotionEdits) activeEditNames.push("FPS");

  if (isAIActive) {
    if (hasEdits) {
      mainActionLabel = `RENDER: ${activeEditNames.join(" + ")} + ${modelFamily} ${activeScale}×`;
    } else {
      mainActionLabel = `RENDER: ${modelFamily} ${activeScale}× UPSCALE`;
    }
    mainActionHandler = startUpscale;
    isHighIntensity = true;
  } else if (hasEdits) {
    mainActionLabel = `EXPORT: ${activeEditNames.join(" + ")}`;
    mainActionHandler = onExportEdited;
    isHighIntensity = false;
  } else {
    mainActionLabel = "SELECT TOOL OR AI";
    isMainActionDisabled = true;
  }

  const buttonStyle = isHighIntensity
    ? { height: '44px', fontSize: '12px' }
    : { height: '44px', fontSize: '12px', background: '#ededed', color: '#000', border: 'none', boxShadow: '0 0 10px rgba(255,255,255,0.2)' };
  const canRunValidate = mode === "video" && isAIActive;

  // Model display name for SignalSummary
  const modelDisplayLabel = isAIActive ? `${modelFamily} ${activeScale}×` : undefined;

  return (
    <div ref={panelRef} style={{
      display: "flex",
      flexDirection: "column",
      background: "var(--panel-bg)",
      height: "100%",
      overflow: "hidden",
      position: 'relative'
    }}>
      {/* Global Styles */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
          50% { opacity: 0.6; transform: translate(-50%, -50%) scale(1.2); }
        }
        .pipeline-scroll::-webkit-scrollbar { width: 6px; }
        .pipeline-scroll::-webkit-scrollbar-track { background: transparent; }
        .pipeline-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
        .pipeline-scroll::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
      `}</style>

      <ToastNotification message={toastState.msg} visible={toastState.visible} onDismiss={() => setToastState(s => ({ ...s, visible: false }))} />

      {/* Pipeline Header */}
      <div style={{
        padding: "12px 16px",
        borderBottom: "1px solid var(--panel-border)",
        background: "var(--section-bg)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexShrink: 0
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{
            width: "28px",
            height: "28px",
            borderRadius: "8px",
            background: "linear-gradient(135deg, var(--brand-primary)30, var(--brand-primary)10)",
            border: "1px solid var(--brand-primary)40",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--brand-primary)"
          }}>
            <IconCpu />
          </div>
          <div>
            <h2 style={{
              margin: 0,
              fontSize: "12px",
              fontWeight: 700,
              color: "var(--text-primary)",
              letterSpacing: "0.02em"
            }}>
              Processing Pipeline
            </h2>
            <p style={{
              margin: 0,
              fontSize: "9px",
              color: "var(--text-muted)",
              marginTop: "2px"
            }}>
              {hasEdits || isAIActive
                ? `${activeEditNames.length + (isAIActive ? 1 : 0)} stage${(activeEditNames.length + (isAIActive ? 1 : 0)) > 1 ? 's' : ''} active`
                : 'No processing stages active'}
            </p>
          </div>
        </div>

        {/* Asset type badge */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 10px',
          borderRadius: '6px',
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.08)'
        }}>
          <div style={{ color: 'var(--brand-primary)' }}>
            {mode === 'video' ? <IconFilm /> : <IconFile />}
          </div>
          <span style={{
            fontSize: '10px',
            fontWeight: 600,
            color: 'var(--text-secondary)',
            letterSpacing: '0.05em'
          }}>
            {mode.toUpperCase()}
          </span>
        </div>
      </div>

      <div className="pipeline-scroll" style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", padding: "12px" }}>

        <InputSourceSection
          inputPath={inputPath}
          mode={mode}
          pickInput={pickInput}
          sourceFps={sourceFps}
          sourceInfo={sourceInfo}
          sourceW={sourceW}
        />

        {/* AI UPSCALE NODE */}
        <PipelineNode
          title="AI Upscale"
          icon={<IconSparkles />}
          nodeNumber={2}
          isActive={isAIActive}
          accentColor="var(--brand-primary)"
          badge={
            <div style={{
              fontSize: '9px',
              padding: '2px 6px',
              borderRadius: '4px',
              background: 'rgba(255,255,255,0.1)',
              fontWeight: 600,
              letterSpacing: '0.05em'
            }}>
              {upscaleConfig.architectureClass}
            </div>
          }
          extra={
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{
                fontSize: '9px',
                color: isAIActive ? 'var(--brand-primary)' : 'var(--text-muted)',
                fontWeight: 700,
                letterSpacing: '0.05em'
              }}>
                {isAIActive ? 'ENABLED' : 'BYPASS'}
              </span>
              <div
                role="switch"
                aria-checked={isAIActive}
                aria-label="Toggle AI Pipeline"
                tabIndex={0}
                onClick={(e) => { e.stopPropagation(); setIsAIActive(!isAIActive); }}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); setIsAIActive(!isAIActive); } }}
                style={{
                  width: '36px',
                  height: '20px',
                  borderRadius: '10px',
                  background: isAIActive
                    ? 'linear-gradient(135deg, var(--brand-primary), rgba(0,255,136,0.7))'
                    : 'rgba(255,255,255,0.08)',
                  border: isAIActive
                    ? '1px solid var(--brand-primary)'
                    : '1px solid rgba(255,255,255,0.1)',
                  position: 'relative',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: isAIActive
                    ? '0 2px 8px rgba(0,255,136,0.3)'
                    : 'inset 0 1px 3px rgba(0,0,0,0.3)',
                  outline: 'none'
                }}
              >
                <div style={{
                  width: '16px',
                  height: '16px',
                  borderRadius: '50%',
                  background: '#fff',
                  position: 'absolute',
                  top: '1px',
                  left: isAIActive ? '17px' : '1px',
                  transition: 'left 0.2s cubic-bezier(0.4, 0.0, 0.2, 1)',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.3)'
                }} />
              </div>

              {/* Engine selector — only visible when AI is active */}
              {isAIActive && (
                <>
                  <div style={{ width: '1px', height: '14px', background: 'rgba(255,255,255,0.12)', margin: '0 2px' }} />
                  <span style={{
                    fontSize: '9px',
                    color: useNativeEngine ? '#fbbf24' : 'var(--text-muted)',
                    fontWeight: 700,
                    letterSpacing: '0.05em'
                  }}>
                    {useNativeEngine ? 'NATIVE' : 'PYTHON'}
                  </span>
                  <div
                    role="switch"
                    aria-checked={useNativeEngine}
                    aria-label="Toggle Native Engine"
                    tabIndex={0}
                    onClick={(e) => { e.stopPropagation(); setUseNativeEngine(!useNativeEngine); }}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); setUseNativeEngine(!useNativeEngine); } }}
                    style={{
                      width: '36px',
                      height: '20px',
                      borderRadius: '10px',
                      background: useNativeEngine
                        ? 'linear-gradient(135deg, #fbbf24, rgba(251,191,36,0.7))'
                        : 'rgba(255,255,255,0.08)',
                      border: useNativeEngine
                        ? '1px solid #fbbf24'
                        : '1px solid rgba(255,255,255,0.1)',
                      position: 'relative',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      boxShadow: useNativeEngine
                        ? '0 2px 8px rgba(251,191,36,0.3)'
                        : 'inset 0 1px 3px rgba(0,0,0,0.3)',
                      outline: 'none'
                    }}
                  >
                    <div style={{
                      width: '16px',
                      height: '16px',
                      borderRadius: '50%',
                      background: '#fff',
                      position: 'absolute',
                      top: '1px',
                      left: useNativeEngine ? '17px' : '1px',
                      transition: 'left 0.2s cubic-bezier(0.4, 0.0, 0.2, 1)',
                      boxShadow: '0 1px 2px rgba(0,0,0,0.3)'
                    }} />
                  </div>
                </>
              )}
            </div>
          }
        >
          <AIUpscaleNode
            videoState={videoState}
            availableModels={availableModels}
            onModelChange={setModel}
            loadModel={loadModel}
            isLoading={loadingModel}
            showTech={showTech}
            showResearchParams={showResearchParams}
            pipelineFeatures={{
              adr_enabled: researchConfig.adr_enabled,
              temporal_enabled: researchConfig.temporal_enabled,
              luma_only: researchConfig.luma_only,
              sharpen_strength: researchConfig.sharpen_strength,
            }}
            onPipelineToggle={(key, value) => updateResearchParam(key as keyof ResearchConfig, value)}
          />
        </PipelineNode>

        <ResearchControlsSection
          advancedOpen={advancedOpen}
          applyResearchPreset={applyResearchPreset}
          availableModels={availableModels}
          isAIActive={isAIActive}
          mode={mode}
          researchConfig={researchConfig}
          setAdvancedOpen={setAdvancedOpen}
          showResearchParams={showResearchParams}
          updateResearchParam={updateResearchParam}
        />

        <PipelineConnector isActive={isAIActive} />

        <GeometryControlsSection
          applyAspectRatio={applyAspectRatio}
          applyCrop={applyCrop}
          editState={editState}
          isAIActive={isAIActive}
          isCropActive={isCropActive}
          isCropApplied={isCropApplied}
          setEditState={setEditState}
          toggleCrop={toggleCrop}
        />

        <PostProcessingSection
          activeScale={activeScale}
          editState={editState}
          hasColorEdits={hasColorEdits}
          hasEdits={hasEdits}
          hasMotionEdits={hasMotionEdits}
          isAIActive={isAIActive}
          isCropActive={isCropActive}
          mode={mode}
          modelDisplayLabel={modelDisplayLabel || ""}
          modelFamily={modelFamily}
          outputPath={outputPath}
          pickOutput={pickOutput}
          setEditState={setEditState}
          showToast={showToast}
          sourceInfo={sourceInfo}
          strSourceFps={strSourceFps}
          strTargetFps={strTargetFps}
          targetFps={targetFps}
          targetInfo={targetInfo}
        />

      </div >

      <PanelActionFooter
        buttonStyle={buttonStyle}
        canRunValidate={canRunValidate}
        isHighIntensity={isHighIntensity}
        isMainActionDisabled={isMainActionDisabled}
        isValidPaths={isValidPaths}
        mainActionHandler={mainActionHandler}
        mainActionLabel={mainActionLabel}
        mode={mode}
        onRunValidate={onRunValidate}
        videoState={videoState}
      />
    </div >
  );
};
