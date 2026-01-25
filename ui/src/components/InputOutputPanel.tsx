import React, { useMemo, useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import type { EditState, VideoState, UpscaleMode } from "../types";
import { SignalSummary } from "./SignalSummary";

// --- ICONS ---
const IconCamera = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" /><circle cx="12" cy="13" r="4" /></svg>;
const IconRotateCW = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 4 23 10 17 10" /><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" /></svg>;
const IconRotateCCW = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 2.12-9.36L1 10" /></svg>;
const IconFlipH = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3v18" /><path d="M16 7l4 5-4 5" /><path d="M8 7l-4 5 4 5" /></svg>;
const IconFlipV = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12h18" /><path d="M7 8L12 4l5 4" /><path d="M7 16l5 4 5-4" /></svg>;
const IconImport = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>;
const IconSave = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" /><polyline points="17 21 17 13 7 13 7 21" /><polyline points="7 3 7 8 15 8" /></svg>;
const IconPlay = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg>;
const IconFlash = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>;
const IconFile = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><polyline points="13 2 13 9 20 9" /></svg>;
const IconFilm = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" /><line x1="7" y1="2" x2="7" y2="22" /><line x1="17" y1="2" x2="17" y2="22" /><line x1="2" y1="12" x2="22" y2="12" /><line x1="2" y1="7" x2="7" y2="7" /><line x1="2" y1="17" x2="7" y2="17" /><line x1="17" y1="17" x2="22" y2="17" /><line x1="17" y1="7" x2="22" y2="7" /></svg>;
const IconShield = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>;
const IconSparkles = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z" /><path d="M5 19l.5 1.5L7 21l-1.5.5L5 23l-.5-1.5L3 21l1.5-.5L5 19z" /><path d="M19 12l.5 1.5L21 14l-1.5.5L19 16l-.5-1.5L17 14l1.5-.5L19 12z" /></svg>;
const IconLock = () => <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" /></svg>;
const IconInfo = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" /></svg>;
const IconCheck = () => <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>;
const IconCrop = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 2v14a2 2 0 0 0 2 2h14" /><path d="M18 22V8a2 2 0 0 0-2-2H2" /></svg>;
const IconPalette = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="13.5" cy="6.5" r=".5" fill="currentColor" /><circle cx="17.5" cy="10.5" r=".5" fill="currentColor" /><circle cx="8.5" cy="7.5" r=".5" fill="currentColor" /><circle cx="6.5" cy="12.5" r=".5" fill="currentColor" /><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.555C21.965 6.012 17.461 2 12 2z" /></svg>;
const IconMove = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="5 9 2 12 5 15" /><polyline points="9 5 12 2 15 5" /><polyline points="15 19 12 22 9 19" /><polyline points="19 9 22 12 19 15" /><line x1="2" y1="12" x2="22" y2="12" /><line x1="12" y1="2" x2="12" y2="22" /></svg>;
const IconClock = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>;
const IconCpu = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2" /><rect x="9" y="9" width="6" height="6" /><line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" /><line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" /><line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="14" x2="23" y2="14" /><line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" /></svg>;
const IconExport = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>;
const IconChevronDown = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9" /></svg>;
const IconPlus = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>;
const IconX = () => <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>;

// --- CONFIGURATION ---
type AIModel = 'RCAN' | 'EDSR';
type CreativeModel = 'REALISTIC' | 'ANIME';
type EnhancementMode = 'archival' | 'creative';
type UpscaleScale = 2 | 3 | 4;

// Canonical model identifiers sent to backend
// Format: {MODEL}_x{SCALE} - backend resolves to actual weight files
const ARCHIVAL_MAP: Record<AIModel, Record<UpscaleScale, string>> = {
  RCAN: { 2: "RCAN_x2", 3: "RCAN_x3", 4: "RCAN_x4" },
  EDSR: { 2: "EDSR_x2", 3: "EDSR_x3", 4: "EDSR_x4" }
};

// Creative mode uses RealESRGAN (only 2x and 4x for realistic, 4x only for anime)
const CREATIVE_MAP: Record<CreativeModel, Partial<Record<UpscaleScale, string>>> = {
  REALISTIC: { 2: "RealESRGAN_x2plus", 4: "RealESRGAN_x4plus" },
  ANIME: { 4: "RealESRGAN_x4plus_anime_6B" }  // Note: No 2x anime model exists in official RealESRGAN
};

// Scale options with labels (ascending order for intuitive progression)
const SCALE_OPTIONS: { value: UpscaleScale; label: string; sub: string }[] = [
  { value: 2, label: "2×", sub: "FAST" },
  { value: 3, label: "3×", sub: "BALANCED" },
  { value: 4, label: "4×", sub: "QUALITY" }
];

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

// --- TOOLTIP COMPONENT (Portal-based with viewport boundary detection) ---
const TOOLTIP_WIDTH = 200;
const TOOLTIP_PADDING = 8;

const Tooltip: React.FC<{ text: string; children: React.ReactNode; position?: 'top' | 'bottom' }> = ({
  text,
  children,
  position = 'top'
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const [style, setStyle] = useState<React.CSSProperties>({});
  const [arrowStyle, setArrowStyle] = useState<React.CSSProperties>({});
  const triggerRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const updatePosition = useCallback(() => {
    if (!triggerRef.current) return;

    const rect = triggerRef.current.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    // Calculate ideal center position
    let x = rect.left + rect.width / 2;
    let y = position === 'top' ? rect.top - 8 : rect.bottom + 8;

    // Calculate tooltip bounds
    const tooltipLeft = x - TOOLTIP_WIDTH / 2;
    const tooltipRight = x + TOOLTIP_WIDTH / 2;

    // Adjust horizontal position to stay in viewport
    let offsetX = 0;
    if (tooltipLeft < TOOLTIP_PADDING) {
      offsetX = TOOLTIP_PADDING - tooltipLeft;
    } else if (tooltipRight > viewportWidth - TOOLTIP_PADDING) {
      offsetX = (viewportWidth - TOOLTIP_PADDING) - tooltipRight;
    }

    // Flip to bottom if not enough space on top
    let actualPosition = position;
    if (position === 'top' && y < 60) {
      actualPosition = 'bottom';
      y = rect.bottom + 8;
    } else if (position === 'bottom' && y + 60 > viewportHeight) {
      actualPosition = 'top';
      y = rect.top - 8;
    }

    setStyle({
      position: 'fixed',
      left: x + offsetX,
      top: y,
      transform: `translateX(-50%) translateY(${actualPosition === 'top' ? '-100%' : '0'})`,
      width: TOOLTIP_WIDTH,
      padding: '8px 12px',
      background: '#1a1a1c',
      border: '1px solid rgba(255,255,255,0.15)',
      borderRadius: '6px',
      fontSize: '10px',
      color: '#e0e0e0',
      whiteSpace: 'normal' as const,
      zIndex: 99999,
      boxShadow: '0 4px 12px rgba(0,0,0,0.6)',
      pointerEvents: 'none' as const,
      lineHeight: 1.4
    });

    // Arrow points to trigger, offset if tooltip was shifted
    setArrowStyle({
      position: 'absolute',
      [actualPosition === 'top' ? 'bottom' : 'top']: '-5px',
      left: `calc(50% - ${offsetX}px)`,
      transform: 'translateX(-50%) rotate(45deg)',
      width: '8px',
      height: '8px',
      background: '#1a1a1c',
      border: '1px solid rgba(255,255,255,0.15)',
      borderTop: actualPosition === 'top' ? 'none' : '1px solid rgba(255,255,255,0.15)',
      borderLeft: actualPosition === 'top' ? 'none' : '1px solid rgba(255,255,255,0.15)',
      borderBottom: actualPosition === 'top' ? '1px solid rgba(255,255,255,0.15)' : 'none',
      borderRight: actualPosition === 'top' ? '1px solid rgba(255,255,255,0.15)' : 'none',
    });
  }, [position]);

  const showTooltip = () => {
    timeoutRef.current = setTimeout(() => {
      updatePosition();
      setIsMounted(true);
      requestAnimationFrame(() => setIsVisible(true));
    }, 400);
  };

  const hideTooltip = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsVisible(false);
    setTimeout(() => setIsMounted(false), 100);
  };

  const tooltipContent = isMounted && createPortal(
    <div style={{ ...style, opacity: isVisible ? 1 : 0, transition: 'opacity 100ms ease' }}>
      {text}
      <div style={arrowStyle} />
    </div>,
    document.body
  );

  return (
    <div
      ref={triggerRef}
      style={{ display: 'inline-flex' }}
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
      onFocus={showTooltip}
      onBlur={hideTooltip}
    >
      {children}
      {tooltipContent}
    </div>
  );
};

// --- TOAST COMPONENT ---
const ToastNotification: React.FC<{ message: string; visible: boolean; onDismiss: () => void }> = ({ message, visible, onDismiss }) => {
  const [render, setRender] = useState(visible);

  useEffect(() => {
    if (visible) setRender(true);
    else setTimeout(() => setRender(false), 300); // Wait for exit animation
  }, [visible]);

  if (!render) return null;

  return (
    <div style={{
      position: 'absolute', top: '16px', right: '16px', zIndex: 2000,
      background: '#1a1a1c', border: '1px solid rgba(255,255,255,0.1)',
      borderLeft: '3px solid #fbbf24', // Amber warning color
      borderRadius: '4px', padding: '10px 12px',
      boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
      display: 'flex', alignItems: 'center', gap: '10px',
      transform: visible ? 'translateX(0)' : 'translateX(100%)',
      opacity: visible ? 1 : 0,
      transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
      maxWidth: '260px'
    }}>
      <span style={{ fontSize: '11px', color: '#ededed', fontFamily: 'var(--font-sans)', fontWeight: 500 }}>{message}</span>
      <button onClick={onDismiss} style={{
        background: 'transparent', border: 'none', color: '#aaa', cursor: 'pointer', padding: '0 4px', fontSize: '14px'
      }}>×</button>
    </div>
  );
};

// --- SMART PATH TRUNCATION ---
const SmartPath: React.FC<{ path: string; placeholder?: string }> = ({ path, placeholder }) => {
  if (!path) return <bdo dir="ltr">{placeholder || ""}</bdo>;

  const formatPath = (p: string) => {
    if (p.length < 45) return p;
    const parts = p.split(/[/\\]/);
    if (parts.length < 3) return p;
    const filename = parts.pop();
    const drive = parts.shift();
    return `${drive}\\...\\${filename}`;
  };

  return (
    <span style={{ fontFamily: 'var(--font-mono)', direction: 'ltr', whiteSpace: 'nowrap' }}>
      {formatPath(path)}
    </span>
  );
};

// --- SUB-COMPONENTS ---

// Connection line between pipeline nodes
const PipelineConnector = ({ isActive = false }: { isActive?: boolean }) => (
  <div style={{
    display: "flex",
    justifyContent: "center",
    padding: "2px 0",
    position: "relative"
  }}>
    <div style={{
      width: "2px",
      height: "12px",
      background: isActive
        ? "linear-gradient(180deg, var(--brand-primary)60, var(--brand-primary)30)"
        : "linear-gradient(180deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05))",
      borderRadius: "1px"
    }} />
    {isActive && (
      <div style={{
        position: "absolute",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        width: "6px",
        height: "6px",
        borderRadius: "50%",
        background: "var(--brand-primary)",
        boxShadow: "0 0 8px var(--brand-primary)",
        animation: "pulse 2s infinite"
      }} />
    )}
  </div>
);

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

const ToggleGroup = ({ options, value, onChange, disabled }: {
  options: { label: string; sub?: string; value: any; disabled?: boolean }[];
  value: any;
  onChange: (v: any) => void;
  disabled?: boolean;
}) => (
  <div style={{
    display: "flex",
    gap: "4px",
    background: "rgba(0,0,0,0.3)",
    padding: "4px",
    borderRadius: "8px",
    border: "1px solid rgba(255,255,255,0.06)",
    boxShadow: "inset 0 2px 4px rgba(0,0,0,0.3)",
    opacity: disabled ? 0.5 : 1,
    pointerEvents: disabled ? 'none' : 'auto'
  }}>
    {options.map((opt) => {
      const isActive = value === opt.value;
      const isOptDisabled = opt.disabled;
      return (
        <button
          key={opt.label}
          onClick={() => !isOptDisabled && onChange(opt.value)}
          disabled={isOptDisabled}
          style={{
            flex: 1,
            height: "36px",
            border: "none",
            borderRadius: "6px",
            minWidth: 0,
            background: isActive && !isOptDisabled
              ? "linear-gradient(135deg, var(--brand-primary), rgba(0,255,136,0.8))"
              : "transparent",
            color: isActive && !isOptDisabled
              ? "#000"
              : isOptDisabled
                ? "var(--text-muted)"
                : "var(--text-secondary)",
            fontSize: "10px",
            fontWeight: isActive ? 800 : 600,
            fontFamily: 'var(--font-sans)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            lineHeight: 1.2,
            boxShadow: isActive && !isOptDisabled
              ? "0 2px 8px rgba(0,255,136,0.4)"
              : "none",
            opacity: isOptDisabled ? 0.3 : 1,
            cursor: isOptDisabled ? 'not-allowed' : 'pointer',
            padding: '0 8px',
            transition: "all 0.15s ease"
          }}
        >
          <span>{opt.label}</span>
          {opt.sub && (
            <span style={{
              fontSize: '8px',
              opacity: isActive ? 0.7 : 0.5,
              fontFamily: 'var(--font-mono)',
              marginTop: '1px'
            }}>
              {opt.sub}
            </span>
          )}
        </button>
      );
    })}
  </div>
);

// --- PIPELINE NODE COMPONENT ---
// A visual node card for the processing pipeline with status indicator
const PipelineNode = ({
  title,
  icon,
  children,
  defaultOpen = true,
  extra,
  badge,
  isActive = false,
  accentColor = 'var(--brand-primary)',
  nodeNumber
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
  extra?: React.ReactNode;
  badge?: React.ReactNode;
  isActive?: boolean;
  accentColor?: string;
  nodeNumber?: number;
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      style={{
        marginBottom: "8px",
        background: isActive
          ? `linear-gradient(135deg, ${accentColor}08, transparent 60%)`
          : "linear-gradient(180deg, #141416, #111113)",
        border: isActive
          ? `1px solid ${accentColor}40`
          : "1px solid rgba(255,255,255,0.06)",
        borderRadius: "10px",
        overflow: "hidden",
        flexShrink: 0,
        boxShadow: isActive
          ? `0 4px 20px ${accentColor}15, inset 0 1px 0 rgba(255,255,255,0.04)`
          : "0 2px 8px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04)",
        transition: "all 0.2s ease",
        position: "relative"
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Active indicator line */}
      {isActive && (
        <div style={{
          position: "absolute",
          left: 0,
          top: 0,
          bottom: 0,
          width: "3px",
          background: `linear-gradient(180deg, ${accentColor}, ${accentColor}60)`,
          borderRadius: "3px 0 0 3px"
        }} />
      )}

      {/* Header */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          cursor: "pointer",
          padding: "10px 12px",
          paddingLeft: isActive ? "15px" : "12px",
          background: isHovered ? "rgba(255,255,255,0.02)" : "transparent",
          userSelect: "none",
          transition: "background 0.15s",
          borderBottom: isOpen ? "1px solid rgba(255,255,255,0.04)" : "none",
          minHeight: "40px",
          gap: "8px"
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px", flex: 1, minWidth: 0 }}>
          {/* Node number indicator */}
          {nodeNumber !== undefined && (
            <div style={{
              width: "20px",
              height: "20px",
              borderRadius: "6px",
              background: isActive ? accentColor : "rgba(255,255,255,0.08)",
              color: isActive ? "#000" : "var(--text-muted)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "10px",
              fontWeight: 700,
              fontFamily: "var(--font-mono)",
              flexShrink: 0
            }}>
              {nodeNumber}
            </div>
          )}

          {/* Icon */}
          <div style={{
            color: isActive ? accentColor : "var(--text-muted)",
            opacity: isActive ? 1 : 0.7,
            transition: "all 0.15s"
          }}>
            {icon}
          </div>

          {/* Title */}
          <h3 style={{
            margin: 0,
            color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
            fontSize: "11px",
            fontWeight: 600,
            letterSpacing: "0.02em",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis"
          }}>
            {title}
          </h3>

          {badge}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
          {extra}
          <div style={{
            color: "var(--text-muted)",
            transform: isOpen ? "rotate(0deg)" : "rotate(-90deg)",
            transition: 'transform 0.2s ease',
            opacity: 0.5
          }}>
            <IconChevronDown />
          </div>
        </div>
      </div>

      {/* Content */}
      <div style={{
        maxHeight: isOpen ? '2000px' : '0',
        overflow: 'hidden',
        transition: 'max-height 0.3s ease-out'
      }}>
        <div style={{ padding: "14px", display: "flex", flexDirection: "column", gap: "14px" }}>
          {children}
        </div>
      </div>
    </div>
  );
};

// Legacy Section wrapper for backwards compatibility
const Section = ({ title, children, defaultOpen = true, extra, badge }: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  extra?: React.ReactNode;
  badge?: React.ReactNode;
}) => (
  <PipelineNode
    title={title}
    icon={null}
    defaultOpen={defaultOpen}
    extra={extra}
    badge={badge}
  >
    {children}
  </PipelineNode>
);

const ColorSlider = ({ label, value, onChange, min = -1, max = 1, step = 0.01, formatValue, icon }: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  formatValue?: (v: number) => string;
  icon?: React.ReactNode;
}) => {
  const [isDragging, setIsDragging] = useState(false);

  const defaultFormat = (v: number) => {
    if (min === -1 && max === 1) return `${v >= 0 ? '+' : ''}${Math.round(v * 100)}%`;
    return v.toFixed(2);
  };
  const displayValue = formatValue ? formatValue(value) : defaultFormat(value);
  const isDefault = min === -1 && max === 1 ? Math.abs(value) < 0.01 : Math.abs(value - 1) < 0.01;
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '8px',
      padding: '10px 12px',
      background: isDragging ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.2)',
      borderRadius: '8px',
      border: isDragging ? '1px solid rgba(0,255,136,0.2)' : '1px solid rgba(255,255,255,0.04)',
      transition: 'all 0.15s ease'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {icon && <span style={{ color: 'var(--text-muted)', opacity: 0.6 }}>{icon}</span>}
          <label style={{
            fontSize: '10px',
            color: 'var(--text-secondary)',
            fontWeight: 600,
            letterSpacing: '0.03em'
          }}>
            {label}
          </label>
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}>
          <span style={{
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            color: isDefault ? 'var(--text-muted)' : 'var(--brand-primary)',
            fontWeight: 600,
            minWidth: '48px',
            textAlign: 'right'
          }}>
            {displayValue}
          </span>
          {!isDefault && (
            <button
              onClick={() => onChange(min === -1 ? 0 : 1)}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                padding: '2px',
                display: 'flex',
                alignItems: 'center',
                opacity: 0.6,
                transition: 'opacity 0.15s'
              }}
              title="Reset to default"
            >
              <IconX />
            </button>
          )}
        </div>
      </div>

      {/* Custom slider track */}
      <div style={{ position: 'relative', height: '6px' }}>
        {/* Track background */}
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'rgba(255,255,255,0.08)',
          borderRadius: '3px',
          overflow: 'hidden'
        }}>
          {/* Fill */}
          <div style={{
            position: 'absolute',
            left: 0,
            top: 0,
            bottom: 0,
            width: `${percentage}%`,
            background: isDefault
              ? 'rgba(255,255,255,0.15)'
              : 'linear-gradient(90deg, var(--brand-primary)80, var(--brand-primary))',
            borderRadius: '3px',
            transition: isDragging ? 'none' : 'width 0.1s ease'
          }} />
        </div>

        {/* Native input (invisible but functional) */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          onMouseDown={() => setIsDragging(true)}
          onMouseUp={() => setIsDragging(false)}
          onMouseLeave={() => setIsDragging(false)}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            opacity: 0,
            cursor: 'pointer',
            margin: 0
          }}
        />

        {/* Custom thumb */}
        <div style={{
          position: 'absolute',
          left: `${percentage}%`,
          top: '50%',
          transform: 'translate(-50%, -50%)',
          width: isDragging ? '14px' : '12px',
          height: isDragging ? '14px' : '12px',
          borderRadius: '50%',
          background: isDefault ? '#666' : 'var(--brand-primary)',
          border: '2px solid #fff',
          boxShadow: isDragging
            ? '0 0 12px rgba(0,255,136,0.5), 0 2px 6px rgba(0,0,0,0.4)'
            : '0 2px 4px rgba(0,0,0,0.3)',
          pointerEvents: 'none',
          transition: isDragging ? 'none' : 'all 0.1s ease'
        }} />
      </div>
    </div>
  );
};

// --- DETERMINISTIC BADGE ---
const DeterministicBadge: React.FC<{ mode: EnhancementMode }> = ({ mode }) => {
  if (mode === 'archival') {
    return (
      <Tooltip text="Frame-stable output. Identical results on every run. Recommended for professional workflows.">
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '3px',
          padding: '2px 5px', borderRadius: '3px',
          background: 'rgba(0, 255, 136, 0.1)',
          border: '1px solid rgba(0, 255, 136, 0.2)',
          fontSize: '7px', fontWeight: 600, color: 'var(--brand-primary)',
          letterSpacing: '0.03em', cursor: 'help', whiteSpace: 'nowrap', flexShrink: 0
        }}>
          <IconCheck />
          STABLE
        </div>
      </Tooltip>
    );
  }
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: '3px',
      padding: '2px 5px', borderRadius: '3px',
      background: 'rgba(251, 191, 36, 0.1)',
      border: '1px solid rgba(251, 191, 36, 0.2)',
      fontSize: '7px', fontWeight: 600, color: '#fbbf24',
      letterSpacing: '0.03em', whiteSpace: 'nowrap', flexShrink: 0
    }}>
      GAN
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

  // New state for deterministic upscaler UI
  const [enhancementMode, setEnhancementMode] = useState<EnhancementMode>('archival');
  const [aiModel, setAiModel] = useState<AIModel>('RCAN');
  const [creativeModel, setCreativeModel] = useState<CreativeModel>('REALISTIC');
  const [upscaleFactor, setUpscaleFactor] = useState<UpscaleScale>(4);
  const [toastState, setToastState] = useState<{ msg: string; visible: boolean }>({ msg: '', visible: false });

  const showToast = (msg: string) => {
    setToastState({ msg, visible: true });
    setTimeout(() => setToastState(s => ({ ...s, visible: false })), 4000);
  };

  // Compute the full model identifier for the backend
  const computedModelId = useMemo((): string => {
    if (enhancementMode === 'archival') {
      return ARCHIVAL_MAP[aiModel][upscaleFactor];
    }
    // Creative mode - fallback to 4x if 3x not available
    const creativeId = CREATIVE_MAP[creativeModel][upscaleFactor];
    if (!creativeId && upscaleFactor === 3) {
      return CREATIVE_MAP[creativeModel][4] ?? CREATIVE_MAP[creativeModel][2] ?? "RealESRGAN_x4plus";
    }
    return creativeId ?? CREATIVE_MAP[creativeModel][4] ?? "RealESRGAN_x4plus";
  }, [enhancementMode, aiModel, creativeModel, upscaleFactor]);

  // Sync with parent when our computed model changes
  useEffect(() => {
    if (computedModelId !== model) {
      setModel(computedModelId);
      loadModel(computedModelId);
    }
  }, [computedModelId, model, setModel, loadModel]);

  // Check model availability
  const isRCANAvailable = useMemo(() => {
    return availableModels.some(m => m.toUpperCase().includes('RCAN'));
  }, [availableModels]);

  const isEDSRAvailable = useMemo(() => {
    return availableModels.some(m => m.toUpperCase().includes('EDSR'));
  }, [availableModels]);

  const isScale4Available = useMemo(() => {
    if (enhancementMode === 'creative') return true; // Creative always has 4x
    const currentModelBase = aiModel;
    return availableModels.some(m =>
      m.toUpperCase().includes(currentModelBase) && m.includes('4')
    );
  }, [availableModels, aiModel, enhancementMode]);

  const isScale3Available = useMemo(() => {
    // Creative mode doesn't have 3x
    if (enhancementMode === 'creative') return false;
    const currentModelBase = aiModel;
    return availableModels.some(m =>
      m.toUpperCase().includes(currentModelBase) && m.includes('3')
    );
  }, [availableModels, aiModel, enhancementMode]);

  const isScale2Available = useMemo(() => {
    if (enhancementMode === 'creative') {
      // Anime only has 4x, Realistic has 2x and 4x
      return creativeModel === 'REALISTIC';
    }
    const currentModelBase = aiModel;
    return availableModels.some(m =>
      m.toUpperCase().includes(currentModelBase) && m.includes('2')
    );
  }, [availableModels, aiModel, enhancementMode, creativeModel]);

  const isRealisticAvailable = useMemo(() => {
    return availableModels.some(m => m.includes('RealESRGAN') && !m.includes('anime'));
  }, [availableModels]);

  const isAnimeAvailable = useMemo(() => {
    return availableModels.some(m => m.includes('anime'));
  }, [availableModels]);

  // Auto-fallback if selected model isn't available
  useEffect(() => {
    if (enhancementMode === 'archival') {
      if (aiModel === 'RCAN' && !isRCANAvailable && isEDSRAvailable) {
        setAiModel('EDSR');
        showToast("RCAN unavailable, using EDSR");
      }
    } else {
      // Creative mode fallback
      if (creativeModel === 'REALISTIC' && !isRealisticAvailable && isAnimeAvailable) {
        setCreativeModel('ANIME');
        showToast("RealESRGAN unavailable, using Anime model");
      }
      // Creative mode doesn't support 3x - fall back to 4x
      if (upscaleFactor === 3) {
        setUpscaleFactor(4);
        showToast("3x not available in Creative mode, using 4x");
      }
      // Anime mode only has 4x - fall back from 2x
      if (creativeModel === 'ANIME' && upscaleFactor === 2) {
        setUpscaleFactor(4);
        showToast("Anime 2x not available, using 4x");
      }
    }
  }, [enhancementMode, aiModel, creativeModel, upscaleFactor, isRCANAvailable, isEDSRAvailable, isRealisticAvailable, isAnimeAvailable]);

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
      mainActionLabel = `RENDER: ${activeEditNames.join(" + ")} + ${enhancementMode === 'archival' ? aiModel : creativeModel} ${activeScale}×`;
    } else {
      mainActionLabel = `RENDER: ${enhancementMode === 'archival' ? aiModel : creativeModel} ${activeScale}× UPSCALE`;
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

  // Model display name for SignalSummary
  // Model display name for SignalSummary
  const modelDisplayLabel = isAIActive ? `${enhancementMode === 'archival' ? aiModel : creativeModel} ${activeScale}×` : undefined;

  return (
    <div ref={panelRef} style={{
      display: "flex",
      flexDirection: "column",
      background: "linear-gradient(180deg, #0c0c0e 0%, #0a0a0c 100%)",
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
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        background: "linear-gradient(180deg, rgba(255,255,255,0.02), transparent)",
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

        {/* INPUT NODE */}
        <PipelineNode
          title="Source Input"
          icon={<IconImport />}
          nodeNumber={1}
          isActive={!!inputPath}
          accentColor="#3b82f6"
        >
          <div
            onClick={pickInput}
            title={inputPath}
            style={{
              background: "linear-gradient(135deg, rgba(59,130,246,0.1), transparent)",
              border: inputPath ? "1px solid rgba(59,130,246,0.3)" : "1px dashed rgba(255,255,255,0.15)",
              borderRadius: "8px",
              height: "48px",
              display: "flex",
              alignItems: "center",
              cursor: "pointer",
              padding: "0 14px",
              gap: "12px",
              transition: "all 0.2s ease"
            }}
          >
            <div style={{
              width: "32px",
              height: "32px",
              borderRadius: "6px",
              background: inputPath ? "rgba(59,130,246,0.2)" : "rgba(255,255,255,0.05)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: inputPath ? "#3b82f6" : "var(--text-muted)"
            }}>
              {inputPath ? (mode === 'video' ? <IconFilm /> : <IconFile />) : <IconPlus />}
            </div>
            <div style={{
              flex: 1,
              fontSize: "11px",
              color: inputPath ? "var(--text-primary)" : "var(--text-muted)",
              overflow: "hidden",
              textAlign: "left",
            }}>
              <SmartPath path={inputPath} placeholder="Click to select source file..." />
            </div>
            {inputPath && (
              <div style={{
                fontSize: "9px",
                color: "#3b82f6",
                fontWeight: 600,
                padding: "3px 8px",
                background: "rgba(59,130,246,0.15)",
                borderRadius: "4px"
              }}>
                LOADED
              </div>
            )}
          </div>

          {/* Source info */}
          {sourceW > 0 && (
            <div style={{
              display: "flex",
              gap: "8px",
              marginTop: "4px"
            }}>
              <div style={{
                flex: 1,
                padding: "8px 10px",
                background: "rgba(0,0,0,0.2)",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.04)"
              }}>
                <div style={{ fontSize: "8px", color: "var(--text-muted)", marginBottom: "2px" }}>RESOLUTION</div>
                <div style={{ fontSize: "11px", color: "var(--text-primary)", fontWeight: 600 }}>{sourceInfo.label}</div>
                <div style={{ fontSize: "9px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{sourceInfo.detail}</div>
              </div>
              {mode === 'video' && (
                <div style={{
                  flex: 1,
                  padding: "8px 10px",
                  background: "rgba(0,0,0,0.2)",
                  borderRadius: "6px",
                  border: "1px solid rgba(255,255,255,0.04)"
                }}>
                  <div style={{ fontSize: "8px", color: "var(--text-muted)", marginBottom: "2px" }}>FRAME RATE</div>
                  <div style={{ fontSize: "11px", color: "var(--text-primary)", fontWeight: 600 }}>{sourceFps} FPS</div>
                </div>
              )}
            </div>
          )}
        </PipelineNode>

        <PipelineConnector isActive={!!inputPath} />

        {/* AI UPSCALE NODE */}
        <PipelineNode
          title="AI Upscale"
          icon={<IconSparkles />}
          nodeNumber={2}
          isActive={isAIActive}
          accentColor="var(--brand-primary)"
          badge={<DeterministicBadge mode={enhancementMode} />}
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
                  background: isAIActive ? '#000' : '#555',
                  position: 'absolute',
                  top: '1px',
                  left: isAIActive ? '17px' : '1px',
                  transition: 'left 0.2s ease',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.3)'
                }} />
              </div>
            </div>
          }
        >
          {/* Enhancement Mode */}
          <div style={{ opacity: isAIActive ? 1 : 0.4, pointerEvents: isAIActive ? 'auto' : 'none', transition: 'opacity 0.2s' }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
              <label style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.05em' }}>ENHANCEMENT MODE</label>
            </div>
            <div style={{ display: "flex", gap: "6px" }}>
              <Tooltip text="Frame-stable output. Identical results on every run. Recommended for professional workflows.">
                <SelectionCard
                  title="ARCHIVAL"
                  subtitle="STABLE"
                  icon={<IconShield />}
                  selected={enhancementMode === 'archival'}
                  disabled={!isAIActive}
                  onClick={() => setEnhancementMode('archival')}
                />
              </Tooltip>
              <Tooltip text="AI-enhanced output with more detail. Best for creative upscaling.">
                <SelectionCard
                  title="CREATIVE"
                  subtitle="GAN"
                  icon={<IconSparkles />}
                  selected={enhancementMode === 'creative'}
                  disabled={!isAIActive}
                  onClick={() => setEnhancementMode('creative')}
                />
              </Tooltip>
            </div>
          </div>

          {/* AI Model / Style Selector */}
          <div style={{ opacity: isAIActive ? 1 : 0.4, pointerEvents: isAIActive ? 'auto' : 'none', transition: 'opacity 0.2s' }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: 'center', marginBottom: "6px" }}>
              <label style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.05em' }}>
                {enhancementMode === 'archival' ? 'AI MODEL' : 'STYLE'}
              </label>
              <Tooltip text={enhancementMode === 'archival'
                ? "RCAN: Best quality, slower. EDSR: Faster, lighter. Both produce identical results across runs."
                : "REALISTIC: General photos & video (2× or 4×). ANIME: Illustrations & cel animation (4× only)."}>
                <div style={{ cursor: 'help', color: 'var(--text-muted)', opacity: 0.6 }}>
                  <IconInfo />
                </div>
              </Tooltip>
            </div>
            {enhancementMode === 'archival' ? (
              <div style={{ display: "flex", gap: "6px" }}>
                <Tooltip text="Residual Channel Attention Network. Best balance of speed and quality.">
                  <button
                    onClick={() => setAiModel('RCAN')}
                    className={aiModel === 'RCAN' ? "toggle-active" : ""}
                    disabled={!isRCANAvailable}
                    style={{
                      flex: 1, height: '44px', borderRadius: '5px', minWidth: 0,
                      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1px',
                      background: aiModel === 'RCAN' ? "var(--brand-dim)" : "var(--input-bg)",
                      border: aiModel === 'RCAN' ? "1px solid var(--brand-primary)" : "1px solid var(--input-border)",
                      color: aiModel === 'RCAN' ? "var(--text-primary)" : "var(--text-secondary)",
                      opacity: !isRCANAvailable ? 0.4 : 1,
                      cursor: !isRCANAvailable ? 'not-allowed' : 'pointer',
                      transition: 'all 0.1s ease'
                    }}
                  >
                    <span style={{ fontWeight: 700, fontSize: '10px' }}>RCAN</span>
                    <span style={{ fontSize: '7px', opacity: 0.6, fontFamily: 'var(--font-mono)' }}>BALANCED</span>
                  </button>
                </Tooltip>
                <Tooltip text="Enhanced Deep Residual Network. Faster processing, slightly less detail recovery.">
                  <button
                    onClick={() => setAiModel('EDSR')}
                    className={aiModel === 'EDSR' ? "toggle-active" : ""}
                    disabled={!isEDSRAvailable}
                    style={{
                      flex: 1, height: '44px', borderRadius: '5px', minWidth: 0,
                      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1px',
                      background: aiModel === 'EDSR' ? "var(--brand-dim)" : "var(--input-bg)",
                      border: aiModel === 'EDSR' ? "1px solid var(--brand-primary)" : "1px solid var(--input-border)",
                      color: aiModel === 'EDSR' ? "var(--text-primary)" : "var(--text-secondary)",
                      opacity: !isEDSRAvailable ? 0.4 : 1,
                      cursor: !isEDSRAvailable ? 'not-allowed' : 'pointer',
                      transition: 'all 0.1s ease'
                    }}
                  >
                    <span style={{ fontWeight: 700, fontSize: '10px' }}>EDSR</span>
                    <span style={{ fontSize: '7px', opacity: 0.6, fontFamily: 'var(--font-mono)' }}>FAST</span>
                  </button>
                </Tooltip>
              </div>
            ) : (
              <div style={{ display: "flex", gap: "6px" }}>
                <Tooltip text="RealESRGAN General. Generates texture details for a sharper, more detailed look.">
                  <button
                    onClick={() => setCreativeModel('REALISTIC')}
                    className={creativeModel === 'REALISTIC' ? "toggle-active" : ""}
                    disabled={!isRealisticAvailable}
                    style={{
                      flex: 1, height: '44px', borderRadius: '5px', minWidth: 0,
                      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1px',
                      background: creativeModel === 'REALISTIC' ? "var(--brand-dim)" : "var(--input-bg)",
                      border: creativeModel === 'REALISTIC' ? "1px solid var(--brand-primary)" : "1px solid var(--input-border)",
                      color: creativeModel === 'REALISTIC' ? "var(--text-primary)" : "var(--text-secondary)",
                      opacity: !isRealisticAvailable ? 0.4 : 1,
                      cursor: !isRealisticAvailable ? 'not-allowed' : 'pointer',
                      transition: 'all 0.1s ease'
                    }}
                  >
                    <span style={{ fontWeight: 700, fontSize: '10px' }}>PHOTO</span>
                    <span style={{ fontSize: '7px', opacity: 0.6, fontFamily: 'var(--font-mono)' }}>DETAIL</span>
                  </button>
                </Tooltip>
                <Tooltip text="RealESRGAN Anime. Optimized for illustrations and flat colors.">
                  <button
                    onClick={() => setCreativeModel('ANIME')}
                    className={creativeModel === 'ANIME' ? "toggle-active" : ""}
                    disabled={!isAnimeAvailable}
                    style={{
                      flex: 1, height: '44px', borderRadius: '5px', minWidth: 0,
                      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1px',
                      background: creativeModel === 'ANIME' ? "var(--brand-dim)" : "var(--input-bg)",
                      border: creativeModel === 'ANIME' ? "1px solid var(--brand-primary)" : "1px solid var(--input-border)",
                      color: creativeModel === 'ANIME' ? "var(--text-primary)" : "var(--text-secondary)",
                      opacity: !isAnimeAvailable ? 0.4 : 1,
                      cursor: !isAnimeAvailable ? 'not-allowed' : 'pointer',
                      transition: 'all 0.1s ease'
                    }}
                  >
                    <span style={{ fontWeight: 700, fontSize: '10px' }}>ANIME</span>
                    <span style={{ fontSize: '7px', opacity: 0.6, fontFamily: 'var(--font-mono)' }}>2D ART</span>
                  </button>
                </Tooltip>
              </div>
            )}
          </div>

          {/* Upscale Factor */}
          <div style={{ opacity: isAIActive ? 1 : 0.4, pointerEvents: isAIActive ? 'auto' : 'none', transition: 'opacity 0.2s' }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
              <label style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.05em' }}>UPSCALE FACTOR</label>
              <span style={{ fontSize: '9px', color: 'var(--brand-primary)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                {activeScale}× ENLARGEMENT
              </span>
            </div>
            <ToggleGroup
              value={upscaleFactor}
              onChange={(v: UpscaleScale) => setUpscaleFactor(v)}
              options={[
                { label: "2×", sub: "FAST", value: 2, disabled: !isScale2Available },
                { label: "3×", sub: "BALANCED", value: 3, disabled: !isScale3Available },
                { label: "4×", sub: "QUALITY", value: 4, disabled: !isScale4Available }
              ]}
              disabled={!isAIActive}
            />
          </div>
        </PipelineNode>

        <PipelineConnector isActive={isAIActive} />

        {/* CROP NODE */}
        <PipelineNode
          title="Crop & Frame"
          icon={<IconCrop />}
          nodeNumber={3}
          isActive={isCropActive}
          accentColor="#3b82f6"
          extra={
            isCropActive && (
              <button
                onClick={(e) => { e.stopPropagation(); applyCrop(); }}
                style={{
                  height: '24px',
                  fontSize: '9px',
                  padding: '0 12px',
                  borderRadius: '6px',
                  border: isCropApplied ? '1px solid rgba(59,130,246,0.3)' : 'none',
                  background: isCropApplied ? 'transparent' : 'linear-gradient(135deg, #3b82f6, #2563eb)',
                  color: isCropApplied ? '#3b82f6' : 'white',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  boxShadow: !isCropApplied ? '0 2px 8px rgba(59,130,246,0.3)' : 'none'
                }}
              >
                {isCropApplied ? "EDIT" : "APPLY"}
              </button>
            )
          }
        >
          {!isCropActive ? (
            <button
              onClick={toggleCrop}
              style={{
                width: '100%',
                height: '44px',
                border: '1px dashed rgba(59,130,246,0.3)',
                color: '#3b82f6',
                fontSize: '10px',
                fontWeight: 600,
                background: 'rgba(59,130,246,0.05)',
                borderRadius: '8px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                transition: 'all 0.15s ease'
              }}
            >
              <IconPlus /> ENABLE CROP TOOL
            </button>
          ) : (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px' }}>
                {ASPECT_RATIOS.map(ar => (
                  <button
                    key={ar.label}
                    onClick={() => applyAspectRatio(ar.value)}
                    style={{
                      fontSize: '10px',
                      height: '34px',
                      borderRadius: '6px',
                      background: editState.aspectRatio === ar.value
                        ? 'linear-gradient(135deg, rgba(59,130,246,0.3), rgba(59,130,246,0.1))'
                        : 'rgba(255,255,255,0.03)',
                      border: editState.aspectRatio === ar.value
                        ? '1px solid rgba(59,130,246,0.5)'
                        : '1px solid rgba(255,255,255,0.08)',
                      color: editState.aspectRatio === ar.value ? '#60a5fa' : 'var(--text-muted)',
                      fontWeight: editState.aspectRatio === ar.value ? 700 : 500,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    {ar.label}
                  </button>
                ))}
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '4px' }}>
                <button
                  onClick={toggleCrop}
                  style={{
                    color: '#ef4444',
                    background: 'rgba(239,68,68,0.1)',
                    fontSize: '9px',
                    border: '1px solid rgba(239,68,68,0.2)',
                    borderRadius: '4px',
                    padding: '4px 10px',
                    cursor: 'pointer',
                    fontWeight: 600,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}
                >
                  <IconX /> REMOVE
                </button>
              </div>
            </>
          )}
        </PipelineNode>

        <PipelineConnector isActive={isCropActive} />

        {/* TRANSFORM NODE */}
        <PipelineNode
          title="Transform"
          icon={<IconMove />}
          nodeNumber={4}
          isActive={editState.rotation !== 0 || editState.flipH || editState.flipV}
          accentColor="#ec4899"
          extra={
            (editState.rotation !== 0 || editState.flipH || editState.flipV) && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setEditState({ ...editState, rotation: 0, flipH: false, flipV: false });
                }}
                style={{
                  height: '24px',
                  fontSize: '9px',
                  padding: '0 10px',
                  borderRadius: '6px',
                  border: '1px solid rgba(236,72,153,0.3)',
                  background: 'transparent',
                  color: '#ec4899',
                  cursor: 'pointer',
                  fontWeight: 600
                }}
              >
                RESET
              </button>
            )
          }
        >
          <div>
            <label style={{ fontSize: '10px', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.03em', marginBottom: '8px', display: 'block' }}>ROTATION</label>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <button onClick={() => {
                const rotations: (0 | 90 | 180 | 270)[] = [0, 90, 180, 270];
                const idx = rotations.indexOf(editState.rotation);
                const newRotation = rotations[(idx + 3) % 4] as 0 | 90 | 180 | 270;
                setEditState({ ...editState, rotation: newRotation });
              }}
                style={{
                  height: '38px',
                  width: '44px',
                  borderRadius: '6px',
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px solid rgba(255,255,255,0.08)',
                  color: 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
                title="Rotate Counter-Clockwise"
              >
                <IconRotateCCW />
              </button>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '4px', flex: 1 }}>
                {([0, 90, 180, 270] as const).map(deg => (
                  <button
                    key={deg}
                    onClick={() => setEditState({ ...editState, rotation: deg })}
                    style={{
                      fontSize: '10px',
                      height: '38px',
                      borderRadius: '6px',
                      background: editState.rotation === deg
                        ? 'linear-gradient(135deg, rgba(236,72,153,0.3), rgba(236,72,153,0.1))'
                        : 'rgba(255,255,255,0.03)',
                      border: editState.rotation === deg
                        ? '1px solid rgba(236,72,153,0.5)'
                        : '1px solid rgba(255,255,255,0.08)',
                      color: editState.rotation === deg ? '#f472b6' : 'var(--text-muted)',
                      fontWeight: editState.rotation === deg ? 700 : 500,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    {deg}°
                  </button>
                ))}
              </div>
              <button onClick={() => {
                const rotations: (0 | 90 | 180 | 270)[] = [0, 90, 180, 270];
                const idx = rotations.indexOf(editState.rotation);
                const newRotation = rotations[(idx + 1) % 4] as 0 | 90 | 180 | 270;
                setEditState({ ...editState, rotation: newRotation });
              }}
                style={{
                  height: '38px',
                  width: '44px',
                  borderRadius: '6px',
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px solid rgba(255,255,255,0.08)',
                  color: 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
                title="Rotate Clockwise"
              >
                <IconRotateCW />
              </button>
            </div>
          </div>
          <div>
            <label style={{ fontSize: '10px', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.03em', marginBottom: '8px', display: 'block' }}>FLIP / MIRROR</label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px' }}>
              <button
                onClick={() => setEditState({ ...editState, flipH: !editState.flipH })}
                style={{
                  fontSize: '10px',
                  height: '40px',
                  borderRadius: '6px',
                  background: editState.flipH
                    ? 'linear-gradient(135deg, rgba(236,72,153,0.3), rgba(236,72,153,0.1))'
                    : 'rgba(255,255,255,0.03)',
                  border: editState.flipH
                    ? '1px solid rgba(236,72,153,0.5)'
                    : '1px solid rgba(255,255,255,0.08)',
                  color: editState.flipH ? '#f472b6' : 'var(--text-muted)',
                  fontWeight: editState.flipH ? 700 : 500,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <IconFlipH /> HORIZONTAL
              </button>
              <button
                onClick={() => setEditState({ ...editState, flipV: !editState.flipV })}
                style={{
                  fontSize: '10px',
                  height: '40px',
                  borderRadius: '6px',
                  background: editState.flipV
                    ? 'linear-gradient(135deg, rgba(236,72,153,0.3), rgba(236,72,153,0.1))'
                    : 'rgba(255,255,255,0.03)',
                  border: editState.flipV
                    ? '1px solid rgba(236,72,153,0.5)'
                    : '1px solid rgba(255,255,255,0.08)',
                  color: editState.flipV ? '#f472b6' : 'var(--text-muted)',
                  fontWeight: editState.flipV ? 700 : 500,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <IconFlipV /> VERTICAL
              </button>
            </div>
          </div>
        </PipelineNode>

        <PipelineConnector isActive={editState.rotation !== 0 || editState.flipH || editState.flipV} />

        {/* COLOR GRADING NODE */}
        <PipelineNode
          title="Color Grading"
          icon={<IconPalette />}
          nodeNumber={5}
          isActive={hasColorEdits}
          accentColor="#a855f7"
          extra={
            hasColorEdits && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setEditState({
                    ...editState,
                    color: { brightness: 0, contrast: 0, saturation: 0, gamma: 1.0 }
                  });
                }}
                style={{
                  height: '24px',
                  fontSize: '9px',
                  padding: '0 10px',
                  borderRadius: '6px',
                  border: '1px solid rgba(168,85,247,0.3)',
                  background: 'transparent',
                  color: '#a855f7',
                  cursor: 'pointer',
                  fontWeight: 600
                }}
              >
                RESET ALL
              </button>
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
        </PipelineNode>

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
            modelLabel={modelDisplayLabel || ""}
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
                }}>{enhancementMode === 'archival' ? aiModel : creativeModel} {activeScale}×</span>
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
