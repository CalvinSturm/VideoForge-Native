import React, { useMemo, useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
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

// --- CONFIGURATION ---
// Model families and IDs are now derived dynamically from availableModels.
// NOTE: extractFamily and extractScale are now imported from ../utils/modelClassification


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
      background: 'var(--panel-bg)',
      border: '1px solid var(--panel-border)',
      borderRadius: '6px',
      fontSize: '10px',
      color: 'var(--text-primary)',
      whiteSpace: 'normal' as const,
      zIndex: 99999,
      boxShadow: 'var(--shadow-md)',
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
      background: 'var(--panel-bg)',
      border: '1px solid var(--panel-border)',
      borderTop: actualPosition === 'top' ? 'none' : '1px solid var(--panel-border)',
      borderLeft: actualPosition === 'top' ? 'none' : '1px solid var(--panel-border)',
      borderBottom: actualPosition === 'top' ? '1px solid var(--panel-border)' : 'none',
      borderRight: actualPosition === 'top' ? '1px solid var(--panel-border)' : 'none',
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
          ? `linear-gradient(135deg, rgba(0,255,136,0.03), transparent 60%)`
          : "var(--node-bg)",
        border: isActive
          ? `1px solid rgba(0,255,136,0.35)`
          : "1px solid var(--node-border)",
        borderRadius: "10px",
        overflow: "hidden",
        flexShrink: 0,
        boxShadow: isActive
          ? `0 4px 20px rgba(0,255,136,0.12), inset 0 1px 0 rgba(255,255,255,0.04)`
          : "var(--shadow-md)",
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
          background: isHovered ? "var(--node-bg-hover)" : "transparent",
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

const ColorSlider = ({ label, value, onChange, min = -1, max = 1, step = 0.01, formatValue, icon, accentColor = 'var(--brand-primary)' }: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  formatValue?: (v: number) => string;
  icon?: React.ReactNode;
  accentColor?: string;
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
      border: isDragging ? `1px solid ${accentColor}40` : '1px solid rgba(255,255,255,0.04)',
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
            color: isDefault ? 'var(--text-muted)' : accentColor,
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
              : `linear-gradient(90deg, ${accentColor}80, ${accentColor})`,
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
          background: isDefault ? '#666' : accentColor,
          border: '2px solid #fff',
          boxShadow: isDragging
            ? `0 0 12px ${accentColor}80, 0 2px 6px rgba(0,0,0,0.4)`
            : '0 2px 4px rgba(0,0,0,0.3)',
          pointerEvents: 'none',
          transition: isDragging ? 'none' : 'all 0.1s ease'
        }} />
      </div>
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

        {/* RESEARCH PARAMETERS NODE - only when AI Upscale is active and research params are enabled */}
        {
          isAIActive && showResearchParams && (() => {
            // Dependency booleans
            const hasSecondary = researchConfig.secondary_model !== 'None';
            const isAdrAvailable = hasSecondary;
            const isAdrActive = hasSecondary && researchConfig.adr_enabled;
            const isTemporalActive = researchConfig.temporal_enabled;
            const isVideoMode = mode === 'video';

            // Shared disabled wrapper style helper
            const disabledWrap = (disabled: boolean): React.CSSProperties => ({
              opacity: disabled ? 0.35 : 1,
              pointerEvents: disabled ? 'none' : 'auto',
              transition: 'opacity 0.2s ease',
            });

            // Reusable switch component
            const ResearchSwitch = ({ checked, onChange: onSwitchChange, disabled }: { checked: boolean; onChange: () => void; disabled?: boolean }) => (
              <div
                role="switch"
                aria-checked={checked}
                tabIndex={disabled ? -1 : 0}
                onClick={disabled ? undefined : onSwitchChange}
                onKeyDown={disabled ? undefined : (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSwitchChange(); } }}
                style={{
                  width: '32px', height: '18px', borderRadius: '9px', position: 'relative',
                  cursor: disabled ? 'not-allowed' : 'pointer',
                  background: checked && !disabled ? 'rgba(245,158,11,0.4)' : 'rgba(255,255,255,0.08)',
                  border: checked && !disabled ? '1px solid rgba(245,158,11,0.5)' : '1px solid rgba(255,255,255,0.1)',
                  transition: 'all 0.2s ease', outline: 'none',
                  opacity: disabled ? 0.5 : 1,
                }}
              >
                <div style={{
                  width: '14px', height: '14px', borderRadius: '50%',
                  background: checked && !disabled ? '#f59e0b' : 'rgba(255,255,255,0.3)',
                  position: 'absolute', top: '1px',
                  left: checked && !disabled ? '15px' : '1px',
                  transition: 'all 0.2s ease',
                }} />
              </div>
            );

            return (
              <>
                <PipelineConnector isActive={true} />
                <PipelineNode
                  title="Research Parameters"
                  icon={<IconCpu />}
                  nodeNumber={3}
                  isActive={true}
                  accentColor="#f59e0b"
                  defaultOpen={false}
                  extra={
                    <div style={{ display: 'flex', gap: '4px' }}>
                      {(['performance', 'balanced', 'quality'] as const).map(p => (
                        <button
                          key={p}
                          onClick={(e) => { e.stopPropagation(); applyResearchPreset(p); }}
                          style={{
                            height: '22px',
                            fontSize: '8px',
                            padding: '0 8px',
                            borderRadius: '4px',
                            border: researchConfig.preset === p
                              ? '1px solid rgba(245,158,11,0.5)'
                              : '1px solid rgba(255,255,255,0.1)',
                            background: researchConfig.preset === p
                              ? 'rgba(245,158,11,0.15)'
                              : 'transparent',
                            color: researchConfig.preset === p ? '#f59e0b' : 'var(--text-muted)',
                            fontWeight: 700,
                            cursor: 'pointer',
                            letterSpacing: '0.05em',
                            transition: 'all 0.15s ease',
                            textTransform: 'uppercase',
                          }}
                        >
                          {p === 'performance' ? 'PERF' : p === 'balanced' ? 'BAL' : 'QUAL'}
                        </button>
                      ))}
                    </div>
                  }
                >
                  {/* 1. Summary Bar */}
                  {(() => {
                    const activeFeatures: string[] = [];
                    if (hasSecondary) activeFeatures.push('2ND MODEL');
                    if (researchConfig.adr_enabled && hasSecondary) activeFeatures.push('ADR');
                    if (researchConfig.temporal_enabled && isVideoMode) activeFeatures.push('TEMPORAL');
                    if (researchConfig.luma_only && hasSecondary) activeFeatures.push('LUMA');
                    if (researchConfig.sharpen_strength > 0) activeFeatures.push('SHARP');

                    const modifiedCount = (Object.keys(RESEARCH_DEFAULTS) as (keyof ResearchConfig)[]).filter(k => {
                      const cur = researchConfig[k];
                      const def = RESEARCH_DEFAULTS[k];
                      if (typeof cur === 'number' && typeof def === 'number') return Math.abs(cur - def) > 0.001;
                      return cur !== def;
                    }).length;

                    return (
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        padding: '8px 10px',
                        background: 'rgba(245,158,11,0.05)',
                        borderRadius: '6px',
                        border: '1px solid rgba(245,158,11,0.15)',
                        marginBottom: '8px',
                        flexWrap: 'wrap',
                      }}>
                        {activeFeatures.length > 0 ? activeFeatures.map(f => (
                          <span key={f} style={{
                            fontSize: '7px', fontWeight: 700, padding: '2px 5px', borderRadius: '3px',
                            background: 'rgba(245,158,11,0.15)', border: '1px solid rgba(245,158,11,0.3)',
                            color: '#f59e0b', letterSpacing: '0.04em',
                          }}>{f}</span>
                        )) : (
                          <span style={{ fontSize: '8px', color: 'var(--text-muted)', fontWeight: 600 }}>
                            DEFAULT CONFIG
                          </span>
                        )}
                        {modifiedCount > 0 && (
                          <span style={{
                            fontSize: '8px', color: 'var(--text-muted)', marginLeft: 'auto',
                            fontFamily: 'var(--font-mono)',
                          }}>
                            {modifiedCount} modified
                          </span>
                        )}
                      </div>
                    );
                  })()}

                  {/* 2. SECONDARY MODEL — gates downstream features */}
                  <div style={{ marginBottom: '4px' }}>
                    <div style={{ fontSize: '8px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.08em', padding: '4px 0 6px', borderBottom: '1px solid rgba(255,255,255,0.04)', marginBottom: '4px' }}>
                      SECONDARY MODEL
                    </div>
                    <Tooltip text="Select a secondary (GAN/texture) model for dual-model blending. 'None' uses only the primary model. Enables ADR detail injection and luma blending." position="bottom">
                      <div style={{
                        padding: '8px 12px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px',
                        border: '1px solid rgba(255,255,255,0.04)',
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                          <label style={{ fontSize: '10px', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.03em' }}>
                            MODEL
                          </label>
                          <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: '#f59e0b', fontWeight: 600 }}>
                            {researchConfig.secondary_model}
                          </span>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '4px', maxHeight: '120px', overflowY: 'auto' }}>
                          {["None", ...availableModels].map(m => (
                            <button
                              key={m}
                              onClick={() => updateResearchParam('secondary_model', m)}
                              title={m}
                              style={{
                                fontSize: '8px', height: '28px', borderRadius: '5px',
                                background: researchConfig.secondary_model === m
                                  ? 'rgba(245,158,11,0.15)' : 'rgba(255,255,255,0.03)',
                                border: researchConfig.secondary_model === m
                                  ? '1px solid rgba(245,158,11,0.4)' : '1px solid rgba(255,255,255,0.08)',
                                color: researchConfig.secondary_model === m ? '#f59e0b' : 'var(--text-muted)',
                                fontWeight: researchConfig.secondary_model === m ? 700 : 500,
                                cursor: 'pointer', transition: 'all 0.15s ease',
                                letterSpacing: '0.03em',
                                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                              }}
                            >
                              {m === 'None' ? 'NONE' : truncateModelName(m)}
                            </button>
                          ))}
                        </div>
                      </div>
                    </Tooltip>
                  </div>

                  {/* 3. DETAIL ENHANCEMENT — gated by secondary model */}
                  <div style={{ marginBottom: '4px' }}>
                    <div style={{ fontSize: '8px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.08em', padding: '4px 0 6px', borderBottom: '1px solid rgba(255,255,255,0.04)', marginBottom: '4px' }}>
                      DETAIL ENHANCEMENT
                    </div>
                    <div style={disabledWrap(!isAdrAvailable)}>
                      <Tooltip text="Enable Adaptive Detail Residual. Extracts high-frequency texture from the secondary (GAN) model and injects it into the primary (structure) output for richer surface detail." position="bottom">
                        <div style={{
                          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                          padding: '8px 12px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px',
                          border: '1px solid rgba(255,255,255,0.04)', marginBottom: '4px',
                        }}>
                          <div>
                            <label style={{ fontSize: '10px', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.03em' }}>
                              ADR ENABLED
                            </label>
                            {!isAdrAvailable && (
                              <div style={{ fontSize: '8px', color: 'rgba(245,158,11,0.6)', marginTop: '2px' }}>
                                Requires secondary model
                              </div>
                            )}
                          </div>
                          <ResearchSwitch
                            checked={researchConfig.adr_enabled}
                            onChange={() => updateResearchParam('adr_enabled', !researchConfig.adr_enabled)}
                            disabled={!isAdrAvailable}
                          />
                        </div>
                      </Tooltip>
                    </div>
                    <div style={disabledWrap(!isAdrActive)}>
                      <Tooltip text="How much GAN high-frequency texture to inject into the structure output. 0 = no detail injection, 1 = full GAN residual. Requires ADR enabled with a secondary model. Default 0.50." position="bottom">
                        <ColorSlider label="DETAIL STRENGTH" value={researchConfig.detail_strength} onChange={(v) => updateResearchParam('detail_strength', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                      </Tooltip>
                      <Tooltip text="Blend only the luminance (Y) channel in YCbCr space. Preserves the structure model's colour accuracy while injecting GAN brightness detail. Prevents colour shifts." position="bottom">
                        <div style={{
                          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                          padding: '8px 12px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px',
                          border: '1px solid rgba(255,255,255,0.04)', marginBottom: '4px',
                        }}>
                          <label style={{ fontSize: '10px', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.03em' }}>
                            LUMA ONLY
                          </label>
                          <ResearchSwitch
                            checked={researchConfig.luma_only}
                            onChange={() => updateResearchParam('luma_only', !researchConfig.luma_only)}
                            disabled={!isAdrActive}
                          />
                        </div>
                      </Tooltip>
                    </div>
                  </div>

                  {/* 4. POST-PROCESSING — always available */}
                  <div style={{ marginBottom: '4px' }}>
                    <div style={{ fontSize: '8px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.08em', padding: '4px 0 6px', borderBottom: '1px solid rgba(255,255,255,0.04)', marginBottom: '4px' }}>
                      POST-PROCESSING
                    </div>
                    <Tooltip text="Sobel edge mask strength for spatially-varying blend. Higher values apply stronger blending on edges, weaker on flat regions. 0 = uniform blend. Default 0.30." position="bottom">
                      <ColorSlider label="EDGE STRENGTH" value={researchConfig.edge_strength} onChange={(v) => updateResearchParam('edge_strength', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                    <Tooltip text="GPU unsharp mask intensity applied after blending. Adds crispness to the final output. 0 = disabled, higher = sharper. Can amplify noise at high values. Default 0.00." position="bottom">
                      <ColorSlider label="SHARPEN" value={researchConfig.sharpen_strength} onChange={(v) => updateResearchParam('sharpen_strength', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                  </div>

                  {/* 5. TEMPORAL — video-mode aware */}
                  <div style={{ marginBottom: '4px' }}>
                    <div style={{
                      fontSize: '8px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.08em',
                      padding: '4px 0 6px', borderBottom: '1px solid rgba(255,255,255,0.04)', marginBottom: '4px',
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    }}>
                      <span>TEMPORAL</span>
                      {!isVideoMode && (
                        <span style={{ fontSize: '7px', color: 'rgba(245,158,11,0.5)', fontWeight: 600, letterSpacing: '0.03em' }}>
                          VIDEO MODE ONLY
                        </span>
                      )}
                    </div>
                    <div style={disabledWrap(!isVideoMode)}>
                      <Tooltip text="Enable exponential moving average (EMA) temporal stabilization across frames. Reduces inter-frame flicker in video upscaling." position="bottom">
                        <div style={{
                          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                          padding: '8px 12px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px',
                          border: '1px solid rgba(255,255,255,0.04)', marginBottom: '4px',
                        }}>
                          <label style={{ fontSize: '10px', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.03em' }}>
                            TEMPORAL EMA
                          </label>
                          <ResearchSwitch
                            checked={researchConfig.temporal_enabled}
                            onChange={() => updateResearchParam('temporal_enabled', !researchConfig.temporal_enabled)}
                            disabled={!isVideoMode}
                          />
                        </div>
                      </Tooltip>
                      <div style={disabledWrap(!isTemporalActive)}>
                        <Tooltip text="EMA smoothing factor. Lower = more smoothing (more temporal averaging). Higher = faster response to new frames. Default 0.90." position="bottom">
                          <ColorSlider label="TEMPORAL ALPHA" value={researchConfig.temporal_alpha} onChange={(v) => updateResearchParam('temporal_alpha', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                        </Tooltip>
                        <Tooltip text="Flush all temporal EMA buffers. Use after seeking, changing clips, or when ghosting artifacts appear." position="bottom">
                          <button
                            onClick={() => { invoke("reset_temporal_buffer").catch(() => { }); }}
                            disabled={!isTemporalActive}
                            style={{
                              width: '100%', height: '30px', fontSize: '9px', fontWeight: 700,
                              borderRadius: '6px', border: '1px solid rgba(245,158,11,0.3)',
                              background: 'rgba(245,158,11,0.08)', color: '#f59e0b',
                              cursor: isTemporalActive ? 'pointer' : 'not-allowed',
                              letterSpacing: '0.05em',
                              transition: 'all 0.15s ease', marginTop: '4px',
                            }}
                          >
                            RESET TEMPORAL BUFFER
                          </button>
                        </Tooltip>
                      </div>
                    </div>
                  </div>

                  {/* 6. MODEL WEIGHTS — always active */}
                  <div style={{ marginBottom: '4px' }}>
                    <div style={{ fontSize: '8px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.08em', padding: '4px 0 6px', borderBottom: '1px solid rgba(255,255,255,0.04)', marginBottom: '4px' }}>
                      MODEL WEIGHTS
                    </div>
                    <Tooltip text="Weight for structural fidelity (edges, geometry). Higher values preserve hard lines and shapes at the cost of softer textures. Default 0.50." position="bottom">
                      <ColorSlider label="STRUCTURE" value={researchConfig.alpha_structure} onChange={(v) => updateResearchParam('alpha_structure', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                    <Tooltip text="Weight for texture detail recovery. Controls how aggressively fine surface detail (fabric, skin pores, grain) is reconstructed. Default 0.30." position="bottom">
                      <ColorSlider label="TEXTURE" value={researchConfig.alpha_texture} onChange={(v) => updateResearchParam('alpha_texture', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                    <Tooltip text="Weight for perceptual similarity. Optimizes output to look natural to the human eye rather than pixel-exact. Higher values may smooth fine detail. Default 0.15." position="bottom">
                      <ColorSlider label="PERCEPTUAL" value={researchConfig.alpha_perceptual} onChange={(v) => updateResearchParam('alpha_perceptual', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                    <Tooltip text="Weight for diffusion-based refinement pass. Adds subtle generative detail but can introduce hallucinated content at high values. Keep low for archival work. Default 0.05." position="bottom">
                      <ColorSlider label="DIFFUSION" value={researchConfig.alpha_diffusion} onChange={(v) => updateResearchParam('alpha_diffusion', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                  </div>

                  {/* 7. FREQUENCY BAND — always active */}
                  <div style={{ marginBottom: '4px' }}>
                    <div style={{ fontSize: '8px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.08em', padding: '4px 0 6px', borderBottom: '1px solid rgba(255,255,255,0.04)', marginBottom: '4px' }}>
                      FREQUENCY BAND
                    </div>
                    <Tooltip text="Amplification of low-frequency content (smooth gradients, large shapes). Values above 1.0 boost, below 1.0 attenuate. Increase to strengthen broad tonal structure. Default 1.00." position="bottom">
                      <ColorSlider label="LOW FREQ" value={researchConfig.low_freq_strength} onChange={(v) => updateResearchParam('low_freq_strength', v)} min={0} max={2} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                    <Tooltip text="Amplification of mid-frequency content (medium detail, object contours). Controls the body of visible sharpness. Boost for crisper mid-detail, reduce to soften. Default 1.00." position="bottom">
                      <ColorSlider label="MID FREQ" value={researchConfig.mid_freq_strength} onChange={(v) => updateResearchParam('mid_freq_strength', v)} min={0} max={2} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                    <Tooltip text="Amplification of high-frequency content (fine edges, noise, micro-texture). Higher values sharpen fine detail but may amplify noise or ringing artifacts. Default 1.00." position="bottom">
                      <ColorSlider label="HIGH FREQ" value={researchConfig.high_freq_strength} onChange={(v) => updateResearchParam('high_freq_strength', v)} min={0} max={2} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                  </div>

                  {/* 8. HALLUCINATION — always active */}
                  <div style={{ marginBottom: '4px' }}>
                    <div style={{ fontSize: '8px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.08em', padding: '4px 0 6px', borderBottom: '1px solid rgba(255,255,255,0.04)', marginBottom: '4px' }}>
                      HALLUCINATION
                    </div>
                    <Tooltip text="How aggressively the detector flags AI-generated detail as hallucinated. Higher values catch more false detail but may suppress legitimate reconstruction. Default 1.00." position="bottom">
                      <ColorSlider label="SENSITIVITY" value={researchConfig.h_sensitivity} onChange={(v) => updateResearchParam('h_sensitivity', v)} min={0} max={3} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                    <Tooltip text="Strength of blending applied to regions flagged as hallucinated. At 1.0, flagged regions are fully replaced with the source. Lower values allow partial AI detail to remain. Default 0.50." position="bottom">
                      <ColorSlider label="BLEND REDUCTION" value={researchConfig.h_blend_reduction} onChange={(v) => updateResearchParam('h_blend_reduction', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                  </div>

                  {/* 9. SPATIAL ROUTING — always active */}
                  <div style={{ marginBottom: '4px' }}>
                    <div style={{ fontSize: '8px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.08em', padding: '4px 0 6px', borderBottom: '1px solid rgba(255,255,255,0.04)', marginBottom: '4px' }}>
                      SPATIAL ROUTING
                    </div>
                    <Tooltip text="How strongly edge-detected regions prefer the structure-preserving model branch. Higher values keep hard edges sharper but may introduce stairstepping on diagonal lines. Default 0.70." position="bottom">
                      <ColorSlider label="EDGE BIAS" value={researchConfig.edge_model_bias} onChange={(v) => updateResearchParam('edge_model_bias', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                    <Tooltip text="How strongly textured regions prefer the texture-recovery model branch. Increase for richer surface detail in complex areas (foliage, fabric). Default 0.70." position="bottom">
                      <ColorSlider label="TEXTURE BIAS" value={researchConfig.texture_model_bias} onChange={(v) => updateResearchParam('texture_model_bias', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                    <Tooltip text="Suppresses AI enhancement in flat, low-detail regions (sky, walls) to prevent noise amplification and false texture. Higher values apply more suppression. Default 0.30." position="bottom">
                      <ColorSlider label="FLAT SUPPRESSION" value={researchConfig.flat_region_suppression} onChange={(v) => updateResearchParam('flat_region_suppression', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                    </Tooltip>
                  </div>

                  {/* 10. ADVANCED — collapsed by default */}
                  <div>
                    <button
                      onClick={() => setAdvancedOpen(!advancedOpen)}
                      style={{
                        width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        padding: '6px 0', border: 'none', background: 'none', cursor: 'pointer',
                        borderBottom: '1px solid rgba(255,255,255,0.04)', marginBottom: advancedOpen ? '4px' : '0',
                      }}
                    >
                      <span style={{ fontSize: '8px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.08em' }}>
                        ADVANCED
                      </span>
                      <span style={{
                        color: 'var(--text-muted)', transform: advancedOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                        transition: 'transform 0.15s ease',
                      }}>
                        <IconChevronDown />
                      </span>
                    </button>
                    {advancedOpen && (
                      <div>
                        <Tooltip text="Gaussian blur sigma for the low-frequency band separation. Larger values capture broader structures in the low band. Increase for smoother tonal rolloff. Default 4.0." position="bottom">
                          <ColorSlider label="LOW SIGMA" value={researchConfig.freq_low_sigma} onChange={(v) => updateResearchParam('freq_low_sigma', v)} min={0.5} max={10} step={0.1} accentColor="#f59e0b" formatValue={(v) => v.toFixed(1)} />
                        </Tooltip>
                        <Tooltip text="Gaussian blur sigma for the mid-frequency band separation. Controls the cutoff between mid and high detail. Lower values shift more content into the high band. Default 1.5." position="bottom">
                          <ColorSlider label="MID SIGMA" value={researchConfig.freq_mid_sigma} onChange={(v) => updateResearchParam('freq_mid_sigma', v)} min={0.5} max={5} step={0.1} accentColor="#f59e0b" formatValue={(v) => v.toFixed(1)} />
                        </Tooltip>
                        <Tooltip text="Gradient magnitude threshold for classifying a pixel as an edge. Pixels above this threshold are routed to the edge model branch. Lower values detect more edges. Default 0.50." position="bottom">
                          <ColorSlider label="EDGE THRESHOLD" value={researchConfig.edge_threshold} onChange={(v) => updateResearchParam('edge_threshold', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                        </Tooltip>
                        <Tooltip text="Local variance threshold for classifying a region as textured. Pixels above this threshold are routed to the texture model branch. Lower values classify more area as textured. Default 0.20." position="bottom">
                          <ColorSlider label="TEXTURE THRESHOLD" value={researchConfig.texture_threshold} onChange={(v) => updateResearchParam('texture_threshold', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                        </Tooltip>
                        <Tooltip text="Blend ratio between spatial routing and frequency-band routing. At 0.0 only spatial routing is used; at 1.0 only frequency bands drive the blend. Default 0.50." position="bottom">
                          <ColorSlider label="SPATIAL-FREQ MIX" value={researchConfig.spatial_freq_mix} onChange={(v) => updateResearchParam('spatial_freq_mix', v)} min={0} max={1} step={0.01} accentColor="#f59e0b" formatValue={(v) => v.toFixed(2)} />
                        </Tooltip>

                        {/* HF Method dropdown */}
                        <Tooltip text="Algorithm used to extract high-frequency detail. Laplacian: second-order edges, general purpose. Sobel: first-order gradient, sharper edges. Highpass: simple subtraction, fast. FFT: spectral domain, most precise but slowest." position="bottom">
                          <div style={{
                            display: 'flex', flexDirection: 'column', gap: '6px',
                            padding: '10px 12px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px',
                            border: '1px solid rgba(255,255,255,0.04)',
                          }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <label style={{ fontSize: '10px', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.03em' }}>
                                HF METHOD
                              </label>
                              <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: '#f59e0b', fontWeight: 600 }}>
                                {researchConfig.hf_method.toUpperCase()}
                              </span>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '4px' }}>
                              {HF_METHODS.map(method => (
                                <button
                                  key={method}
                                  onClick={() => updateResearchParam('hf_method', method)}
                                  style={{
                                    fontSize: '9px', height: '28px', borderRadius: '5px',
                                    background: researchConfig.hf_method === method
                                      ? 'rgba(245,158,11,0.15)' : 'rgba(255,255,255,0.03)',
                                    border: researchConfig.hf_method === method
                                      ? '1px solid rgba(245,158,11,0.4)' : '1px solid rgba(255,255,255,0.08)',
                                    color: researchConfig.hf_method === method ? '#f59e0b' : 'var(--text-muted)',
                                    fontWeight: researchConfig.hf_method === method ? 700 : 500,
                                    cursor: 'pointer', transition: 'all 0.15s ease',
                                    textTransform: 'uppercase', letterSpacing: '0.03em',
                                  }}
                                >
                                  {method === 'highpass' ? 'HP' : method === 'laplacian' ? 'LAP' : method.toUpperCase()}
                                </button>
                              ))}
                            </div>
                          </div>
                        </Tooltip>
                      </div>
                    )}
                  </div>
                </PipelineNode>
              </>
            );
          })()
        }

        <PipelineConnector isActive={isAIActive} />

        {/* CROP NODE */}
        <PipelineNode
          title="Crop & Frame"
          icon={<IconCrop />}
          nodeNumber={isAIActive ? 4 : 3}
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
          nodeNumber={isAIActive ? 5 : 4}
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
          nodeNumber={isAIActive ? 6 : 5}
          isActive={hasColorEdits}
          accentColor="#a855f7"
          extra={
            <div style={{ display: 'flex', gap: '6px' }}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  // Auto Grade: Apply conservative color corrections
                  // These are sensible defaults that work well for most footage
                  setEditState({
                    ...editState,
                    color: {
                      brightness: 0.05,    // Slight lift
                      contrast: 0.08,      // Gentle contrast boost
                      saturation: 0.05,    // Subtle saturation
                      gamma: 1.0           // Keep gamma neutral
                    }
                  });
                  showToast('Auto Grade applied (conservative)');
                }}
                style={{
                  height: '24px',
                  fontSize: '9px',
                  padding: '0 10px',
                  borderRadius: '6px',
                  border: '1px solid rgba(0,255,136,0.3)',
                  background: 'linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,255,136,0.05))',
                  color: 'var(--brand-primary)',
                  cursor: 'pointer',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}
                title="Apply automatic color grading"
              >
                <IconSparkles /> AUTO
              </button>
              {hasColorEdits && (
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
                  RESET
                </button>
              )}
            </div>
          }
        >
          <ColorSlider
            label="BRIGHTNESS"
            value={editState.color.brightness}
            onChange={(v) => setEditState({ ...editState, color: { ...editState.color, brightness: v } })}
            accentColor="#fbbf24"
          />
          <ColorSlider
            label="CONTRAST"
            value={editState.color.contrast}
            onChange={(v) => setEditState({ ...editState, color: { ...editState.color, contrast: v } })}
            accentColor="#22d3ee"
          />
          <ColorSlider
            label="SATURATION"
            value={editState.color.saturation}
            onChange={(v) => setEditState({ ...editState, color: { ...editState.color, saturation: v } })}
            accentColor="#f472b6"
          />
          <ColorSlider
            label="GAMMA"
            value={editState.color.gamma}
            min={0.1}
            max={3.0}
            step={0.01}
            onChange={(v) => setEditState({ ...editState, color: { ...editState.color, gamma: v } })}
            formatValue={(v) => v.toFixed(2)}
            accentColor="#a78bfa"
          />
        </PipelineNode>

        <PipelineConnector isActive={hasColorEdits} />

        {/* TEMPORAL NODE - Video only */}
        {
          mode === 'video' && (
            <PipelineNode
              title="Frame Rate"
              icon={<IconClock />}
              nodeNumber={isAIActive ? 7 : 6}
              isActive={hasMotionEdits}
              accentColor="#eab308"
              extra={
                hasMotionEdits && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setEditState({ ...editState, fps: 0 });
                    }}
                    style={{
                      height: '24px',
                      fontSize: '9px',
                      padding: '0 10px',
                      borderRadius: '6px',
                      border: '1px solid rgba(234,179,8,0.3)',
                      background: 'transparent',
                      color: '#eab308',
                      cursor: 'pointer',
                      fontWeight: 600
                    }}
                  >
                    RESET
                  </button>
                )
              }
            >
              <ToggleGroup
                value={editState.fps}
                onChange={(v: number) => setEditState({ ...editState, fps: v })}
                options={[
                  { label: "NATIVE", sub: "SOURCE", value: 0 },
                  { label: "30", sub: "FPS", value: 30 },
                  { label: "60", sub: "FPS", value: 60 },
                  { label: "120", sub: "FPS", value: 120 }
                ]}
              />
            </PipelineNode>
          )
        }

        <PipelineConnector isActive={mode === 'video' ? hasMotionEdits : hasColorEdits} />

        {/* OUTPUT NODE */}
        <PipelineNode
          title="Export Output"
          icon={<IconExport />}
          nodeNumber={mode === 'video' ? (isAIActive ? 8 : 7) : (isAIActive ? 7 : 6)}
          isActive={!!outputPath}
          accentColor="#10b981"
        >
          <SignalSummary
            sourceResolution={sourceInfo.label} sourceDetail={sourceInfo.detail} sourceFps={strSourceFps}
            targetResolution={targetInfo.label} targetDetail={targetInfo.detail} targetFps={strTargetFps}
            modelLabel={modelDisplayLabel || ""}
          />

          {/* Active Edits Display */}
          {(hasEdits || isAIActive) && (
            <div style={{
              display: 'flex', flexWrap: 'wrap', gap: '6px',
              padding: '10px 12px', background: 'rgba(0,0,0,0.25)', borderRadius: '8px',
              border: '1px solid rgba(255,255,255,0.06)'
            }}>
              <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600, width: '100%', marginBottom: '4px', letterSpacing: '0.05em' }}>
                PIPELINE STAGES:
              </span>
              {isAIActive && (
                <span style={{
                  fontSize: '9px', padding: '4px 10px', borderRadius: '4px',
                  background: 'rgba(0, 255, 136, 0.12)', color: 'var(--brand-primary)', fontWeight: 600,
                  border: '1px solid rgba(0, 255, 136, 0.25)',
                  display: 'flex', alignItems: 'center', gap: '4px'
                }}>
                  <IconSparkles /> {modelFamily} {activeScale}×
                </span>
              )}
              {isCropActive && (
                <span style={{
                  fontSize: '9px', padding: '4px 10px', borderRadius: '4px',
                  background: 'rgba(59, 130, 246, 0.12)', color: '#60a5fa', fontWeight: 600,
                  border: '1px solid rgba(59, 130, 246, 0.25)'
                }}>CROP</span>
              )}
              {(editState.rotation !== 0 || editState.flipH || editState.flipV) && (
                <span style={{
                  fontSize: '9px', padding: '4px 10px', borderRadius: '4px',
                  background: 'rgba(236, 72, 153, 0.12)', color: '#f472b6', fontWeight: 600,
                  border: '1px solid rgba(236, 72, 153, 0.25)'
                }}>TRANSFORM</span>
              )}
              {hasColorEdits && (
                <span style={{
                  fontSize: '9px', padding: '4px 10px', borderRadius: '4px',
                  background: 'rgba(168, 85, 247, 0.12)', color: '#c084fc', fontWeight: 600,
                  border: '1px solid rgba(168, 85, 247, 0.25)'
                }}>COLOR</span>
              )}
              {hasMotionEdits && (
                <span style={{
                  fontSize: '9px', padding: '4px 10px', borderRadius: '4px',
                  background: 'rgba(234, 179, 8, 0.12)', color: '#fbbf24', fontWeight: 600,
                  border: '1px solid rgba(234, 179, 8, 0.25)'
                }}>{targetFps} FPS</span>
              )}
            </div>
          )}

          {/* No edits message */}
          {!hasEdits && !isAIActive && (
            <div style={{
              fontSize: '10px', color: 'var(--text-muted)', textAlign: 'center',
              padding: '12px', background: 'rgba(0,0,0,0.2)', borderRadius: '6px',
              border: '1px dashed rgba(255,255,255,0.08)',
              fontStyle: 'italic'
            }}>
              No pipeline stages active — enable a tool above
            </div>
          )}

          {/* Export Path */}
          <div
            onClick={pickOutput}
            title={outputPath}
            style={{
              background: "linear-gradient(135deg, rgba(16,185,129,0.1), transparent)",
              border: outputPath ? "1px solid rgba(16,185,129,0.3)" : "1px dashed rgba(255,255,255,0.15)",
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
              background: outputPath ? "rgba(16,185,129,0.2)" : "rgba(255,255,255,0.05)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: outputPath ? "#10b981" : "var(--text-muted)"
            }}>
              <IconSave />
            </div>
            <div style={{
              flex: 1,
              fontSize: "11px",
              color: outputPath ? "var(--text-primary)" : "var(--text-muted)",
              overflow: "hidden",
              textAlign: "left",
            }}>
              <SmartPath path={outputPath} placeholder="Click to set export location..." />
            </div>
            {outputPath && (
              <div style={{
                fontSize: "9px",
                color: "#10b981",
                fontWeight: 600,
                padding: "3px 8px",
                background: "rgba(16,185,129,0.15)",
                borderRadius: "4px"
              }}>
                SET
              </div>
            )}
          </div>
        </PipelineNode>

      </div >

      <div style={{ padding: "16px", borderTop: "1px solid var(--panel-border)", background: "var(--section-bg)", display: "flex", flexDirection: "column", gap: "12px", flexShrink: 0 }}>
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
            <button
              className="action-secondary"
              onClick={onRunValidate}
              disabled={!isValidPaths || !canRunValidate}
              title={canRunValidate ? "Run strict policy validation in mock mode" : "Enable AI Upscale on video input to validate"}
              style={{
                borderColor: "rgba(59,130,246,0.4)", color: "#bfdbfe",
                background: "rgba(59,130,246,0.08)", fontWeight: 700,
                display: 'flex', gap: '8px', alignItems: 'center', justifyContent: 'center',
                boxShadow: "0 2px 4px rgba(0,0,0,0.2)"
              }}
            >
              <IconShield />
              VALIDATE STRICT (MOCK)
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
    </div >
  );
};
