import React, { useMemo, useState, useRef, useEffect } from "react";
import type { EditState, VideoState, UpscaleMode } from "../types";
import { SignalSummary } from "./SignalSummary";

// --- ICONS ---
// (Icons are unchanged, keeping them brief for this output)
const IconCamera = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" /><circle cx="12" cy="13" r="4" /></svg>;
const IconAnimation = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" /></svg>;
const IconImport = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>;
const IconSave = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" /><polyline points="17 21 17 13 7 13 7 21" /><polyline points="7 3 7 8 15 8" /></svg>;
const IconPlay = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg>;
const IconFlash = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>;
const IconFile = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><polyline points="13 2 13 9 20 9" /></svg>;
const IconFilm = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" /><line x1="7" y1="2" x2="7" y2="22" /><line x1="17" y1="2" x2="17" y2="22" /><line x1="2" y1="12" x2="22" y2="12" /><line x1="2" y1="7" x2="7" y2="7" /><line x1="2" y1="17" x2="7" y2="17" /><line x1="17" y1="17" x2="22" y2="17" /><line x1="17" y1="7" x2="22" y2="7" /></svg>;

// --- CONFIGURATION ---
const MODEL_MAP = {
  real: { quality: "RealESRGAN_x4plus.pth", speed: "RealESRGAN_x2plus.pth" },
  anime: { quality: "RealESRGAN_x4plus_anime_6B.pth", speed: "RealESRGAN_x2plus_anime_6B.pth" }
};

const ASPECT_RATIOS = [
  { label: "FREE", value: null },
  { label: "16:9", value: 16 / 9 },
  { label: "9:16", value: 9 / 16 },
  { label: "4:5", value: 0.8 },
  { label: "1:1", value: 1 },
  { label: "2.35:1", value: 2.35 },
];

const FPS_OPTIONS = [
  { value: 0, label: "NATIVE", sub: "SOURCE" },
  { value: 30, label: "30 FPS", sub: "STD" },
  { value: 60, label: "60 FPS", sub: "SMOOTH" },
  { value: 120, label: "120 FPS", sub: "SLOW-MO" },
];

const getModelScale = (name: string): number => name.includes("x2") ? 2 : 4;

const getSmartResInfo = (w: number, h: number) => {
  if (w === 0 || h === 0) return { label: "---", detail: "" };
  const min = Math.min(w, h);
  const dims = `${w} × ${h}`;
  if (min >= 4320) return { label: "8K UHD", detail: dims };
  if (min >= 2160) return { label: "4K UHD", detail: dims };
  if (min >= 1440) return { label: "1440p QHD", detail: dims };
  if (min >= 1080) return { label: "1080p FHD", detail: dims };
  if (min >= 720) return { label: "720p HD", detail: dims };
  if (min >= 480) return { label: "480p SD", detail: dims };
  return { label: dims, detail: "" };
};

// --- NEW COMPONENT: Smart Path Truncation ---
const SmartPath: React.FC<{ path: string; placeholder?: string }> = ({ path, placeholder }) => {
  if (!path) return <bdo dir="ltr">{placeholder || ""}</bdo>;

  const formatPath = (p: string) => {
    // If it's reasonably short, just show it
    if (p.length < 45) return p;

    // Split logic
    const parts = p.split(/[/\\]/);
    if (parts.length < 3) return p; // Can't compact effectively

    const filename = parts.pop();
    const drive = parts.shift();
    // Reconstruct with middle truncation
    return `${drive}\\...\\${filename}`;
  };

  return (
    <span style={{ fontFamily: 'var(--font-mono)', direction: 'ltr', whiteSpace: 'nowrap' }}>
      {formatPath(path)}
    </span>
  );
};

// --- SUB-COMPONENTS ---
const SelectionCard = ({ selected, onClick, title, subtitle, icon, disabled }: any) => (
  <button onClick={onClick} disabled={disabled} style={{
    flex: 1, display: "flex", flexDirection: "column", alignItems: "flex-start", justifyContent: "center",
    height: "64px", padding: "8px 12px",
    background: selected ? "rgba(0, 255, 136, 0.05)" : "var(--panel-bg)",
    border: selected ? "1px solid var(--brand-primary)" : "1px solid rgba(255,255,255,0.08)",
    borderTop: selected ? "1px solid var(--brand-primary)" : "1px solid rgba(255,255,255,0.12)",
    borderRadius: "6px", cursor: disabled ? "not-allowed" : "pointer",
    transition: "all 0.15s ease", position: "relative", overflow: "hidden",
    opacity: disabled ? 0.3 : 1,
    boxShadow: selected ? "0 0 10px rgba(0,255,136,0.1)" : "0 2px 4px rgba(0,0,0,0.2)"
  }}>
    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px" }}>
      <div style={{ color: selected && !disabled ? "var(--brand-primary)" : "var(--text-muted)" }}>{icon}</div>
      <span style={{ fontWeight: 700, fontSize: "11px", color: selected && !disabled ? "var(--text-primary)" : "var(--text-secondary)", fontFamily: 'var(--font-sans)', letterSpacing: '0.02em' }}>{title}</span>
    </div>
    <span style={{ fontSize: "9px", color: "var(--text-muted)", marginLeft: "24px", fontFamily: 'var(--font-sans)' }}>{subtitle}</span>
    {selected && !disabled && <div style={{ position: "absolute", top: 0, right: 0, width: 0, height: 0, borderTop: "8px solid var(--brand-primary)", borderLeft: "8px solid transparent" }} />}
  </button>
);

const ToggleGroup = ({ options, value, onChange, disabled }: any) => (
  <div style={{
    display: "flex", background: "#050505", padding: "3px", borderRadius: "6px",
    border: "1px solid rgba(255,255,255,0.08)", boxShadow: "inset 0 2px 4px rgba(0,0,0,0.5)",
    opacity: disabled ? 0.5 : 1, pointerEvents: disabled ? 'none' : 'auto'
  }}>
    {options.map((opt: any) => {
      const isActive = value === opt.value;
      return (
        <button key={opt.label} onClick={() => onChange(opt.value)}
          className={isActive ? "toggle-active" : ""}
          style={{
            flex: 1, height: "30px", border: "none", borderRadius: "4px",
            background: "transparent", color: "var(--text-muted)",
            fontSize: "10px", fontWeight: 500, fontFamily: 'var(--font-sans)',
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', lineHeight: 1.1,
            boxShadow: isActive ? "0 2px 4px rgba(0,0,0,0.4)" : "none"
          }}
        >
          <span>{opt.label}</span>
          {opt.sub && <span style={{ fontSize: '8px', opacity: 0.6, fontFamily: 'var(--font-mono)' }}>{opt.sub}</span>}
        </button>
      );
    })}
  </div>
);

const Section = ({ title, children, defaultOpen = true, extra }: any) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  return (
    <div style={{
      marginBottom: "16px", background: "var(--panel-bg)", border: "1px solid var(--panel-border)",
      borderTop: "1px solid rgba(255,255,255,0.1)", borderRadius: "6px", overflow: "hidden", flexShrink: 0,
      boxShadow: "0 4px 6px -1px rgba(0,0,0,0.3)"
    }}>
      <div onClick={() => setIsOpen(!isOpen)} style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        cursor: "pointer", padding: "10px 12px", background: "rgba(255, 255, 255, 0.02)",
        userSelect: "none", transition: "background 0.2s",
        borderBottom: isOpen ? "1px solid var(--panel-border)" : "none", height: "36px"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flex: 1 }}>
          <h3 style={{ margin: 0, color: "var(--text-secondary)", fontSize: "10px" }}>{title}</h3>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {extra}
          <span style={{ fontSize: "8px", color: "var(--text-muted)", transform: isOpen ? "rotate(0deg)" : "rotate(-90deg)" }}>▼</span>
        </div>
      </div>
      {isOpen && <div style={{ padding: "14px", display: "flex", flexDirection: "column", gap: "14px" }}>{children}</div>}
    </div>
  );
};

const ColorSlider = ({ label, value, onChange, min = -1, max = 1, step = 0.01, formatValue }: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  formatValue?: (v: number) => string;
}) => {
  const defaultFormat = (v: number) => {
    if (min === -1 && max === 1) return `${v >= 0 ? '+' : ''}${Math.round(v * 100)}%`;
    return v.toFixed(2);
  };
  const displayValue = formatValue ? formatValue(value) : defaultFormat(value);
  const isDefault = min === -1 && max === 1 ? Math.abs(value) < 0.01 : Math.abs(value - 1) < 0.01;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <label style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.05em' }}>{label}</label>
        <span style={{
          fontSize: '10px', fontFamily: 'var(--font-mono)',
          color: isDefault ? 'var(--text-muted)' : 'var(--brand-primary)', fontWeight: 500
        }}>{displayValue}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{
          width: '100%', height: '4px', WebkitAppearance: 'none', appearance: 'none',
          background: `linear-gradient(to right, var(--brand-primary) 0%, var(--brand-primary) ${((value - min) / (max - min)) * 100}%, #333 ${((value - min) / (max - min)) * 100}%, #333 100%)`,
          borderRadius: '2px', cursor: 'pointer', outline: 'none'
        }}
      />
    </div>
  );
};

interface InputOutputPanelProps {
  mode: UpscaleMode; setMode: (mode: UpscaleMode) => void;
  pickInput: () => void; inputPath: string;
  pickOutput: () => void; outputPath: string;
  model: string; setModel: (model: string) => void;
  availableModels: string[]; loadingModel: boolean; loadModel: (model: string) => void;
  startUpscale: () => void; isValidPaths: boolean;
  videoState: VideoState; editState: EditState; setEditState: (state: EditState) => void;
  onExportEdited: () => void; showTech: boolean;
  viewMode: 'edit' | 'preview'; setViewMode: (mode: 'edit' | 'preview') => void;
}

export const InputOutputPanel: React.FC<InputOutputPanelProps> = ({
  mode, setMode, pickInput, inputPath, pickOutput, outputPath,
  model, setModel, loadModel, availableModels,
  startUpscale, isValidPaths, videoState, editState, setEditState, onExportEdited,
}) => {

  const panelRef = useRef<HTMLDivElement>(null);
  const [isAIActive, setIsAIActive] = useState(true);

  // Intent State: We track the user's intended scale specifically
  // This allows us to map "Anime Speed (2x)" to an x4 model but request 2x output
  const [scalingMode, setScalingMode] = useState<'auto' | 2 | 4>('auto');

  // Auto-correct invalid model selection (e.g. if file was removed)
  useEffect(() => {
    if (availableModels.length > 0 && !availableModels.includes(model)) {
      console.log(`[UI] Model ${model} not available. Attempting fallback...`);

      const isAnime = model.toLowerCase().includes("anime");
      const isX2 = model.toLowerCase().includes("x2");

      // Determine fallback target
      let fallback = isAnime ? MODEL_MAP.anime.quality : MODEL_MAP.real.quality;

      // If we are specifically fixing the "Animation 2x" case:
      if (isAnime && isX2) {
        fallback = MODEL_MAP.anime.speed; // This now points to x4 file
        setScalingMode(2); // Preserve 2x intent
      }

      if (availableModels.includes(fallback)) {
        console.log(`[UI] Switching to fallback: ${fallback}`);
        setModel(fallback);
        loadModel(fallback);
      } else if (availableModels.includes(MODEL_MAP.real.quality)) {
        // Ultimate fallback
        setModel(MODEL_MAP.real.quality);
        loadModel(MODEL_MAP.real.quality);
      }
    }
  }, [availableModels, model, setModel, loadModel]);

  // --- DERIVED STATE ---
  const activeScale = useMemo(() => {
    if (scalingMode !== 'auto') return scalingMode;
    return getModelScale(model);
  }, [model, scalingMode]);
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

  const isAnime = model.toLowerCase().includes("anime");
  // isSpeed is now determined by the active scale, not just the filename
  const isSpeed = activeScale === 2;
  const currentPerf = isSpeed ? 'speed' : 'quality';

  const setIntent = (type: 'real' | 'anime', perf: 'quality' | 'speed') => {
    const targetFile = MODEL_MAP[type][perf];
    // Explicitly set scaling mode based on perf button
    setScalingMode(perf === 'speed' ? 2 : 4);

    setModel(targetFile);
    loadModel(targetFile);
  };

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
      mainActionLabel = `RENDER: ${activeEditNames.join(" + ")} + AI ${activeScale}×`;
    } else {
      mainActionLabel = `RENDER: AI ${activeScale}× UPSCALE`;
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

  return (
    <div ref={panelRef} style={{ display: "flex", flexDirection: "column", background: "transparent", height: "100%", overflow: "hidden" }}>
      <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", padding: "16px" }}>

        <Section title="Input Assets">
          <div style={{ marginBottom: "8px", display: 'flex', alignItems: 'center' }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: '8px',
              padding: '4px 8px', borderRadius: '4px',
              backgroundColor: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              fontSize: '10px', fontWeight: 600, color: 'var(--text-secondary)',
              fontFamily: 'var(--font-sans)', letterSpacing: '0.05em'
            }}>
              <span style={{ opacity: 0.5, fontSize: '9px' }}>ASSET TYPE:</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--brand-primary)' }}>
                {mode === 'video' ? <IconFilm /> : <IconFile />}
                <span>{mode.toUpperCase()}</span>
              </div>
            </div>
          </div>

          <div onClick={pickInput} title={inputPath} style={{
            background: "var(--input-bg)", border: "1px solid var(--input-border)", borderRadius: "4px",
            height: "40px", display: "flex", alignItems: "center", cursor: "pointer", padding: "0 12px", gap: "10px",
            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.5)"
          }}>
            <div style={{ color: "var(--text-muted)" }}><IconImport /></div>
            <div style={{
              flex: 1, fontSize: "11px", color: inputPath ? "var(--text-primary)" : "var(--text-muted)",
              overflow: "hidden", textAlign: "left",
            }}>
              {/* FIX 3: Smart Path Truncation */}
              <SmartPath path={inputPath} placeholder="Select Source File..." />
            </div>
          </div>
        </Section>

        <Section title="Upscaling & Enhancement"
          extra={
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '9px', color: isAIActive ? 'var(--brand-primary)' : 'var(--text-muted)', fontWeight: 700 }}>{isAIActive ? 'ON' : 'OFF'}</span>
              <div onClick={(e) => { e.stopPropagation(); setIsAIActive(!isAIActive); }}
                style={{
                  width: '28px', height: '16px', borderRadius: '10px',
                  background: isAIActive ? 'var(--brand-primary)' : '#1a1a1c',
                  border: '1px solid rgba(255,255,255,0.1)',
                  position: 'relative', cursor: 'pointer', transition: 'all 0.2s',
                  boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.5)'
                }}>
                <div style={{
                  width: '12px', height: '12px', borderRadius: '50%', background: isAIActive ? 'black' : '#555',
                  position: 'absolute', top: '1px', left: isAIActive ? '13px' : '1px', transition: 'left 0.2s',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.5)'
                }} />
              </div>
            </div>
          }
        >
          <div style={{ display: "flex", gap: "8px", opacity: isAIActive ? 1 : 0.4, pointerEvents: isAIActive ? 'auto' : 'none', transition: 'opacity 0.2s' }}>
            <SelectionCard title="REALISTIC" subtitle="PHOTO / FILM" icon={<IconCamera />} selected={!isAnime} disabled={!isAIActive} onClick={() => setIntent('real', currentPerf)} />
            <SelectionCard title="ANIMATION" subtitle="2D / LINE ART" icon={<IconAnimation />} selected={isAnime} disabled={!isAIActive} onClick={() => setIntent('anime', currentPerf)} />
          </div>
          <div style={{ marginTop: '12px', opacity: isAIActive ? 1 : 0.4, pointerEvents: isAIActive ? 'auto' : 'none', transition: 'opacity 0.2s' }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
              <label className="label-text" style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, fontFamily: 'var(--font-sans)' }}>UPSCALE FACTOR</label>
            </div>
            <ToggleGroup
              value={currentPerf}
              onChange={(v: any) => setIntent(isAnime ? 'anime' : 'real', v)}
              options={[{ label: "QUALITY", sub: "4x", value: "quality" }, { label: "SPEED", sub: "2x", value: "speed" }]}
            />
          </div>
        </Section>

        <Section title="Crop Tool"
          extra={
            isCropActive && (
              <button onClick={(e) => { e.stopPropagation(); applyCrop(); }}
                className={!isCropApplied ? "toggle-active" : ""}
                style={{
                  height: '22px', fontSize: '9px', padding: '0 10px', borderRadius: '4px',
                  border: '1px solid var(--panel-border)', background: isCropApplied ? 'transparent' : 'var(--brand-primary)',
                  color: isCropApplied ? 'var(--brand-primary)' : 'black'
                }}
              >
                {isCropApplied ? "EDIT" : "DONE"}
              </button>
            )
          }
        >
          {!isCropActive ? (
            <button onClick={toggleCrop} style={{ width: '100%', height: '32px', border: '1px dashed var(--panel-border)', color: 'var(--text-secondary)', fontSize: '10px', background: 'rgba(255,255,255,0.02)' }}>
              + ENABLE CROP
            </button>
          ) : (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px' }}>
                {ASPECT_RATIOS.map(ar => (
                  <button key={ar.label} onClick={() => applyAspectRatio(ar.value)}
                    className={editState.aspectRatio === ar.value ? "toggle-active" : ""}
                    style={{
                      fontSize: '9px', height: '26px', borderRadius: '4px',
                      background: "var(--input-bg)", border: "1px solid var(--input-border)", color: "var(--text-muted)",
                      boxShadow: "0 1px 2px rgba(0,0,0,0.2)"
                    }}
                  >{ar.label}</button>
                ))}
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '4px' }}>
                <button onClick={toggleCrop} style={{ color: '#ef4444', background: 'transparent', fontSize: '9px', border: 'none', cursor: 'pointer' }}>REMOVE</button>
              </div>
            </>
          )}
        </Section>

        <Section title="Color Grading"
          extra={
            (editState.color.brightness !== 0 || editState.color.contrast !== 0 ||
              editState.color.saturation !== 0 || editState.color.gamma !== 1.0) && (
              <button onClick={(e) => {
                e.stopPropagation();
                setEditState({
                  ...editState,
                  color: { brightness: 0, contrast: 0, saturation: 0, gamma: 1.0 }
                });
              }}
                style={{
                  height: '22px', fontSize: '9px', padding: '0 10px', borderRadius: '4px',
                  border: '1px solid var(--panel-border)', background: 'transparent',
                  color: 'var(--text-muted)', cursor: 'pointer'
                }}>RESET</button>
            )
          }
        >
          <ColorSlider
            label="BRIGHTNESS"
            value={editState.color.brightness}
            onChange={(v) => setEditState({ ...editState, color: { ...editState.color, brightness: v } })}
          />
          <ColorSlider
            label="CONTRAST"
            value={editState.color.contrast}
            onChange={(v) => setEditState({ ...editState, color: { ...editState.color, contrast: v } })}
          />
          <ColorSlider
            label="SATURATION"
            value={editState.color.saturation}
            onChange={(v) => setEditState({ ...editState, color: { ...editState.color, saturation: v } })}
          />
          <ColorSlider
            label="GAMMA"
            value={editState.color.gamma}
            min={0.1}
            max={3.0}
            step={0.01}
            onChange={(v) => setEditState({ ...editState, color: { ...editState.color, gamma: v } })}
            formatValue={(v) => v.toFixed(2)}
          />
        </Section>

        {mode === 'video' && (
          <Section title="Temporal Processing">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px' }}>
              {FPS_OPTIONS.map(opt => (
                <button key={opt.value} onClick={() => setEditState({ ...editState, fps: opt.value })}
                  className={editState.fps === opt.value ? "toggle-active" : ""}
                  style={{
                    height: '48px', borderRadius: '6px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1px',
                    background: "var(--input-bg)", color: "var(--text-secondary)", border: "1px solid var(--input-border)",
                    boxShadow: "inset 0 1px 3px rgba(0,0,0,0.3)"
                  }}
                >
                  <span style={{ fontWeight: 700, fontSize: '10px' }}>{opt.label}</span>
                  <span style={{ fontSize: '7px', opacity: 0.6, letterSpacing: '0.05em', fontFamily: 'var(--font-mono)' }}>{opt.sub}</span>
                </button>
              ))}
            </div>
          </Section>
        )}

        <Section title="Render Targets">
          <SignalSummary
            sourceResolution={sourceInfo.label} sourceDetail={sourceInfo.detail} sourceFps={strSourceFps}
            targetResolution={targetInfo.label} targetDetail={targetInfo.detail} targetFps={strTargetFps}
          />

          {/* Active Edits Display */}
          {(hasEdits || isAIActive) && (
            <div style={{
              display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '12px',
              padding: '8px 10px', background: 'rgba(0,0,0,0.3)', borderRadius: '4px',
              border: '1px solid rgba(255,255,255,0.05)'
            }}>
              <span style={{ fontSize: '8px', color: 'var(--text-muted)', fontWeight: 600, width: '100%', marginBottom: '4px', letterSpacing: '0.05em' }}>
                ACTIVE EDITS:
              </span>
              {isCropActive && (
                <span style={{
                  fontSize: '9px', padding: '3px 8px', borderRadius: '3px',
                  background: 'rgba(59, 130, 246, 0.2)', color: '#60a5fa', fontWeight: 600,
                  border: '1px solid rgba(59, 130, 246, 0.3)'
                }}>CROP</span>
              )}
              {hasColorEdits && (
                <span style={{
                  fontSize: '9px', padding: '3px 8px', borderRadius: '3px',
                  background: 'rgba(168, 85, 247, 0.2)', color: '#c084fc', fontWeight: 600,
                  border: '1px solid rgba(168, 85, 247, 0.3)'
                }}>COLOR</span>
              )}
              {(editState.rotation !== 0 || editState.flipH || editState.flipV) && (
                <span style={{
                  fontSize: '9px', padding: '3px 8px', borderRadius: '3px',
                  background: 'rgba(236, 72, 153, 0.2)', color: '#f472b6', fontWeight: 600,
                  border: '1px solid rgba(236, 72, 153, 0.3)'
                }}>TRANSFORM</span>
              )}
              {hasMotionEdits && (
                <span style={{
                  fontSize: '9px', padding: '3px 8px', borderRadius: '3px',
                  background: 'rgba(234, 179, 8, 0.2)', color: '#fbbf24', fontWeight: 600,
                  border: '1px solid rgba(234, 179, 8, 0.3)'
                }}>FPS: {targetFps}</span>
              )}
              {isAIActive && (
                <span style={{
                  fontSize: '9px', padding: '3px 8px', borderRadius: '3px',
                  background: 'rgba(0, 255, 136, 0.15)', color: 'var(--brand-primary)', fontWeight: 600,
                  border: '1px solid rgba(0, 255, 136, 0.3)'
                }}>AI {activeScale}×</span>
              )}
            </div>
          )}

          {/* No edits message */}
          {!hasEdits && !isAIActive && (
            <div style={{
              fontSize: '10px', color: 'var(--text-muted)', textAlign: 'center',
              padding: '8px', background: 'rgba(0,0,0,0.2)', borderRadius: '4px',
              marginBottom: '12px', fontStyle: 'italic'
            }}>
              No edits active — enable a tool above
            </div>
          )}

          <div>
            <label className="label-text" style={{ marginBottom: '6px', display: 'block', fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, fontFamily: 'var(--font-sans)' }}>EXPORT PATH</label>
            <div onClick={pickOutput} title={outputPath} style={{
              background: "var(--input-bg)", border: "1px solid var(--input-border)", borderRadius: "4px",
              height: "36px", display: "flex", alignItems: "center", cursor: "pointer", padding: "0 10px", gap: "10px",
              boxShadow: "inset 0 2px 4px rgba(0,0,0,0.5)"
            }}>
              <div style={{ color: "var(--text-muted)" }}><IconSave /></div>
              <div style={{
                flex: 1, fontSize: "10px", color: "var(--text-secondary)",
                overflow: "hidden", textAlign: "left",
              }}>
                {/* FIX 3: Smart Path Truncation */}
                <SmartPath path={outputPath} placeholder="Auto-Generated" />
              </div>
            </div>
          </div>
        </Section>

      </div>

      <div style={{ padding: "16px", borderTop: "1px solid var(--panel-border)", background: "rgba(0,0,0,0.2)", display: "flex", flexDirection: "column", gap: "12px", flexShrink: 0 }}>
        {mode === 'video' && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "8px", alignItems: 'center' }}>
            <button
              className="action-secondary"
              onClick={videoState.renderSample}
              disabled={!isValidPaths}
              style={{
                borderColor: "rgba(255,255,255,0.15)", color: "white",
                background: "rgba(255,255,255,0.02)", fontWeight: 600,
                display: 'flex', gap: '8px', alignItems: 'center', justifyContent: 'center',
                boxShadow: "0 2px 4px rgba(0,0,0,0.2)"
              }}
            >
              <IconPlay />
              PREVIEW 2s
            </button>
          </div>
        )}
        <button
          className={isHighIntensity ? "action-primary" : ""}
          onClick={mainActionHandler}
          disabled={isMainActionDisabled}
          style={buttonStyle}
        >
          {isHighIntensity ? mainActionLabel : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 800 }}>
              <IconFlash /> {mainActionLabel}
            </div>
          )}
        </button>
      </div>
    </div>
  );
};
