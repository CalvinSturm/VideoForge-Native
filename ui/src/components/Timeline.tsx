import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';

interface TimelineProps {
  duration: number;
  currentTime: number;
  trimStart: number;
  trimEnd: number;
  renderedRange?: { start: number; end: number } | null | undefined;
  onSeek: (time: number) => void;
  onTrimChange: (start: number, end: number) => void;
  hasAudio?: boolean;
  onInteractionStart?: () => void;
  onInteractionEnd?: () => void;
}

// --- CONSTANTS ---
const FRAME_RATE = 30; // Assumed frame rate for frame-based navigation
const FRAME_DURATION = 1 / FRAME_RATE;

// --- UTILITIES ---

const formatTimeFull = (seconds: number) => {
  if (!Number.isFinite(seconds)) return "00:00.00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
};

const formatRulerTime = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
};

const parseTimeInput = (str: string, maxDuration: number) => {
  const parts = str.split(':');
  let seconds = 0;
  if (parts.length === 2) {
    seconds = parseInt(parts[0] ?? "0", 10) * 60 + parseFloat(parts[1] ?? "0");
  } else {
    seconds = parseFloat(str);
  }
  return Math.min(Math.max(0, seconds), maxDuration);
};

// --- SUB-COMPONENTS ---

const TimeInput = ({ time, max, onChange, label, highlight = false }: any) => {
  const [val, setVal] = useState(formatTimeFull(time));
  const [isEditing, setIsEditing] = useState(false);
  const [isFocused, setIsFocused] = useState(false);

  useEffect(() => { if (!isEditing) setVal(formatTimeFull(time)); }, [time, isEditing]);

  const commit = () => {
    const s = parseTimeInput(val, max);
    onChange(s);
    setVal(formatTimeFull(s));
    setIsEditing(false);
  };

  return (
    <div style={{
      display: 'flex', alignItems: 'center',
      background: isFocused ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.03)',
      borderRadius: '6px', overflow: 'hidden',
      border: isFocused ? '1px solid rgba(0, 255, 136, 0.3)' : '1px solid rgba(255,255,255,0.08)',
      transition: 'all 0.15s ease',
      boxShadow: isFocused ? '0 0 0 2px rgba(0, 255, 136, 0.1)' : 'none'
    }}>
      <div style={{
        padding: '0 8px', fontSize: '9px', fontWeight: 700, letterSpacing: '0.5px',
        color: highlight ? '#000' : 'var(--text-muted)',
        background: highlight ? 'var(--brand-primary)' : 'rgba(255,255,255,0.05)',
        height: '26px', display: 'flex', alignItems: 'center', cursor: 'default',
        borderRight: '1px solid rgba(255,255,255,0.05)'
      }}>
        {label}
      </div>
      <input
        type="text" value={val}
        onChange={(e) => setVal(e.target.value)}
        onFocus={() => { setIsEditing(true); setIsFocused(true); }}
        onBlur={() => { commit(); setIsFocused(false); }}
        onKeyDown={(e) => { if (e.key === 'Enter') { e.currentTarget.blur(); } }}
        style={{
          background: 'transparent', border: 'none', outline: 'none',
          color: highlight ? 'var(--brand-primary)' : '#ededed',
          fontFamily: '"JetBrains Mono", monospace', fontSize: '11px',
          width: '76px', textAlign: 'center', height: '26px',
          padding: '0 4px'
        }}
      />
    </div>
  );
};

// --- ICONS ---

const TrimStartIcon = ({ isHovered }: { isHovered?: boolean }) => (
  <svg width="12" height="32" viewBox="0 0 12 32" fill="none" style={{
    filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.4))',
    transition: 'transform 0.1s ease',
    transform: isHovered ? 'scale(1.1)' : 'scale(1)'
  }}>
    <rect x="0" y="0" width="12" height="32" rx="2" fill="var(--brand-primary)" />
    <rect x="0" y="0" width="4" height="32" fill="var(--brand-primary)" />
    <path d="M3 11 L3 21" stroke="rgba(0,0,0,0.4)" strokeWidth="2" strokeLinecap="round" />
    <path d="M6 11 L6 21" stroke="rgba(0,0,0,0.25)" strokeWidth="1" strokeLinecap="round" />
  </svg>
);

const TrimEndIcon = ({ isHovered }: { isHovered?: boolean }) => (
  <svg width="12" height="32" viewBox="0 0 12 32" fill="none" style={{
    filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.4))',
    transition: 'transform 0.1s ease',
    transform: isHovered ? 'scale(1.1)' : 'scale(1)'
  }}>
    <rect x="0" y="0" width="12" height="32" rx="2" fill="var(--brand-primary)" />
    <rect x="8" y="0" width="4" height="32" fill="var(--brand-primary)" />
    <path d="M9 11 L9 21" stroke="rgba(0,0,0,0.4)" strokeWidth="2" strokeLinecap="round" />
    <path d="M6 11 L6 21" stroke="rgba(0,0,0,0.25)" strokeWidth="1" strokeLinecap="round" />
  </svg>
);

const PlayheadIcon = ({ isDragging }: { isDragging?: boolean }) => (
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{
    filter: isDragging ? 'drop-shadow(0 0 6px rgba(255,255,255,0.8))' : 'drop-shadow(0 2px 3px rgba(0,0,0,0.5))',
    transition: 'filter 0.15s ease'
  }}>
    <path d="M7 13L1 7V1H13V7L7 13Z" fill={isDragging ? '#fff' : '#f0f0f0'} stroke="rgba(0,0,0,0.3)" strokeWidth="0.5" />
  </svg>
);

// Frame navigation buttons
const FrameButton = ({ direction, onClick, disabled }: { direction: 'prev' | 'next', onClick: () => void, disabled?: boolean }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    title={direction === 'prev' ? 'Previous Frame (←)' : 'Next Frame (→)'}
    style={{
      background: 'rgba(255,255,255,0.05)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '4px',
      color: disabled ? 'var(--text-muted)' : 'var(--text-secondary)',
      cursor: disabled ? 'not-allowed' : 'pointer',
      padding: '4px 8px',
      fontSize: '10px',
      fontFamily: 'var(--font-mono)',
      display: 'flex',
      alignItems: 'center',
      gap: '2px',
      transition: 'all 0.15s ease',
      opacity: disabled ? 0.5 : 1
    }}
    onMouseEnter={(e) => { if (!disabled) e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; }}
    onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; }}
  >
    {direction === 'prev' ? '◀' : '▶'}
  </button>
);

// Zoom control button
const ZoomButton = ({ type, onClick, disabled }: { type: 'in' | 'out' | 'fit', onClick: () => void, disabled?: boolean }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    title={type === 'in' ? 'Zoom In (Ctrl+Scroll)' : type === 'out' ? 'Zoom Out' : 'Fit to View'}
    style={{
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: '3px',
      color: disabled ? 'var(--text-muted)' : 'var(--text-secondary)',
      cursor: disabled ? 'not-allowed' : 'pointer',
      padding: '2px 6px',
      fontSize: '11px',
      fontWeight: 600,
      transition: 'all 0.15s ease',
      opacity: disabled ? 0.5 : 1
    }}
    onMouseEnter={(e) => { if (!disabled) e.currentTarget.style.background = 'rgba(255,255,255,0.08)'; }}
    onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; }}
  >
    {type === 'in' ? '+' : type === 'out' ? '−' : '⊡'}
  </button>
);

// --- MAIN COMPONENT ---

export const Timeline: React.FC<TimelineProps> = ({ duration, currentTime, trimStart, trimEnd, onSeek, onTrimChange, hasAudio = true, onInteractionStart, onInteractionEnd }) => {
  const trackRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [hoveredHandle, setHoveredHandle] = useState<'start' | 'end' | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragTarget, setDragTarget] = useState<string | null>(null);

  const propsRef = useRef({ duration, currentTime, trimStart, trimEnd, onSeek, onTrimChange, onInteractionStart, onInteractionEnd });
  useEffect(() => { propsRef.current = { duration, currentTime, trimStart, trimEnd, onSeek, onTrimChange, onInteractionStart, onInteractionEnd }; });

  const dragState = useRef<{
    active: boolean;
    target: 'playhead' | 'start' | 'end' | 'range' | null;
    startX: number;
    startVal: number;
    endVal: number;
  }>({ active: false, target: null, startX: 0, startVal: 0, endVal: 0 });

  // --- KEYBOARD NAVIGATION ---
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if timeline container is focused or no input is focused
      const activeElement = document.activeElement;
      if (activeElement?.tagName === 'INPUT' || activeElement?.tagName === 'TEXTAREA') return;

      const { trimStart: ts, trimEnd: te, currentTime: ct, onSeek: seek } = propsRef.current;

      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          const prevTime = e.shiftKey ? ct - 1 : ct - FRAME_DURATION;
          seek(Math.max(ts, prevTime));
          break;
        case 'ArrowRight':
          e.preventDefault();
          const nextTime = e.shiftKey ? ct + 1 : ct + FRAME_DURATION;
          seek(Math.min(te, nextTime));
          break;
        case 'Home':
          e.preventDefault();
          seek(ts);
          break;
        case 'End':
          e.preventDefault();
          seek(te);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Frame navigation handlers
  const stepFrame = useCallback((direction: 'prev' | 'next') => {
    const { trimStart: ts, trimEnd: te, currentTime: ct, onSeek: seek } = propsRef.current;
    if (direction === 'prev') {
      seek(Math.max(ts, ct - FRAME_DURATION));
    } else {
      seek(Math.min(te, ct + FRAME_DURATION));
    }
  }, []);

  // --- ADAPTIVE TICKS ---
  const getRulerTicks = useMemo(() => {
    const safeDuration = Math.max(duration, 0.1);
    const visibleSeconds = safeDuration / zoom;
    const targetInterval = visibleSeconds / 10;

    const intervals = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
    const majorStep = intervals.find(i => i >= targetInterval) ?? intervals[intervals.length - 1] ?? 300;

    let minorStep = majorStep / 5;
    if (majorStep <= 0.1) minorStep = majorStep / 2;
    if (safeDuration / minorStep > 1000) minorStep = majorStep; // Optimize

    const ticks = [];
    const epsilon = minorStep / 10;

    for (let t = 0; t <= safeDuration + epsilon; t += minorStep) {
      const time = Math.abs(Math.round(t * 1000) / 1000);
      const nearestMajor = Math.round(time / majorStep) * majorStep;
      const isMajor = Math.abs(time - nearestMajor) < epsilon;
      ticks.push({ time, pct: (time / safeDuration) * 100, isMajor });
    }
    return ticks;
  }, [duration, zoom]);

  const getTimeFromEvent = (clientX: number, currentDuration: number) => {
    if (!trackRef.current) return 0;
    const rect = trackRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const safeDur = Math.max(currentDuration, 0.1);
    return (x / rect.width) * safeDur;
  };

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      e.stopPropagation();
      setZoom(z => Math.max(1, Math.min(20, z - e.deltaY * 0.005)));
    }
  }, []);

  const handleMouseDown = (e: React.MouseEvent, target: 'playhead' | 'start' | 'end' | 'range') => {
    e.stopPropagation();
    e.preventDefault();

    // Notify Interaction Start
    if (propsRef.current.onInteractionStart) propsRef.current.onInteractionStart();

    // Track drag state for visual feedback
    setIsDragging(true);
    setDragTarget(target);

    const currentDuration = propsRef.current.duration;
    dragState.current = {
      active: true, target,
      startX: getTimeFromEvent(e.clientX, currentDuration),
      startVal: propsRef.current.trimStart,
      endVal: propsRef.current.trimEnd
    };
    window.addEventListener('mousemove', onWindowMouseMove, { capture: true });
    window.addEventListener('mouseup', onWindowMouseUp, { capture: true });
  };

  const onWindowMouseMove = useCallback((e: MouseEvent) => {
    if (!dragState.current.active) return;
    e.preventDefault(); e.stopPropagation();
    const { duration, onSeek, onTrimChange } = propsRef.current;
    const safeDur = Math.max(duration, 0.1);
    const t = getTimeFromEvent(e.clientX, duration);
    const { target, startX, startVal, endVal } = dragState.current;
    const delta = t - startX;

    if (target === 'playhead') {
      let val = Math.max(propsRef.current.trimStart, Math.min(propsRef.current.trimEnd, t));
      onSeek(val);
    } else if (target === 'start') {
      onTrimChange(Math.min(Math.max(0, t), endVal - 0.1), endVal);
    } else if (target === 'end') {
      onTrimChange(startVal, Math.max(Math.min(safeDur, t), startVal + 0.1));
    } else if (target === 'range') {
      let ns = startVal + delta, ne = endVal + delta;
      if (ns < 0) { ns = 0; ne = endVal - startVal; }
      if (ne > safeDur) { ne = safeDur; ns = safeDur - (endVal - startVal); }
      onTrimChange(ns, ne);
    }
  }, []);

  const onWindowMouseUp = useCallback((e: MouseEvent) => {
    if (dragState.current.active) {
      e.preventDefault(); e.stopPropagation();
      dragState.current.active = false;

      // Reset drag visual state
      setIsDragging(false);
      setDragTarget(null);

      // Notify Interaction End
      if (propsRef.current.onInteractionEnd) propsRef.current.onInteractionEnd();

      window.removeEventListener('mousemove', onWindowMouseMove, { capture: true });
      window.removeEventListener('mouseup', onWindowMouseUp, { capture: true });
    }
  }, [onWindowMouseMove]);

  useEffect(() => {
    return () => {
      window.removeEventListener('mousemove', onWindowMouseMove, { capture: true });
      window.removeEventListener('mouseup', onWindowMouseUp, { capture: true });
    };
  }, [onWindowMouseMove, onWindowMouseUp]);

  const safeDur = Math.max(duration, 0.1);
  const startPct = (trimStart / safeDur) * 100;
  const endPct = (trimEnd / safeDur) * 100;
  const currPct = (currentTime / safeDur) * 100;
  const widthPct = Math.max(endPct - startPct, 0);
  const isTrimmed = trimStart > 0 || (trimEnd > 0 && trimEnd < duration);

  // Calculate frame number for display
  const currentFrame = Math.floor(currentTime * FRAME_RATE);
  const totalFrames = Math.floor(duration * FRAME_RATE);

  return (
    <div ref={containerRef} className="timeline-wrapper" style={{ display: 'flex', flexDirection: 'column', width: '100%', userSelect: 'none', touchAction: 'none' }} tabIndex={0}>
      <style>{`
        .timeline-scroll-area::-webkit-scrollbar { height: 8px; background: #0a0a0c; }
        .timeline-scroll-area::-webkit-scrollbar-track { background: linear-gradient(to bottom, #121214, #18181b); border-radius: 4px; margin: 0 16px; }
        .timeline-scroll-area::-webkit-scrollbar-thumb { background: linear-gradient(to bottom, #3f3f46, #2a2a30); border-radius: 4px; border: 1px solid #18181b; }
        .timeline-scroll-area::-webkit-scrollbar-thumb:hover { background: linear-gradient(to bottom, #52525b, #3f3f46); }
        .timeline-wrapper:focus { outline: none; }
        .timeline-wrapper:focus-visible { box-shadow: inset 0 0 0 1px rgba(0, 255, 136, 0.3); }
        @keyframes playhead-pulse { 0%, 100% { box-shadow: 0 0 4px rgba(255,255,255,0.3); } 50% { box-shadow: 0 0 8px rgba(255,255,255,0.6); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateX(-50%) translateY(4px); } to { opacity: 1; transform: translateX(-50%) translateY(0); } }
      `}</style>

      {/* DASHBOARD TOOLBAR */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: '6px', padding: '6px 8px', minHeight: '36px',
        borderBottom: '1px solid var(--panel-border)',
        background: 'var(--timeline-toolbar-bg)',
        borderRadius: '6px 6px 0 0'
      }}>
        {/* LEFT: TRIM RANGE + FRAME NAV */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <FrameButton direction="prev" onClick={() => stepFrame('prev')} disabled={currentTime <= trimStart} />
          <TimeInput label="IN" time={trimStart} max={trimEnd} onChange={(v: any) => { onTrimChange(v, trimEnd); if (currentTime < v) onSeek(v); }} />
          <div style={{ width: '12px', height: '2px', background: 'linear-gradient(to right, var(--brand-primary), transparent)', borderRadius: '1px' }}></div>
          <TimeInput label="OUT" time={trimEnd} max={duration} onChange={(v: any) => { onTrimChange(trimStart, v); if (currentTime > v) onSeek(v); }} />
          <FrameButton direction="next" onClick={() => stepFrame('next')} disabled={currentTime >= trimEnd} />
        </div>

        {/* CENTER: HEAD + FRAME COUNT */}
        <div style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <TimeInput label="HEAD" time={currentTime} max={trimEnd} highlight onChange={(v: any) => onSeek(Math.max(trimStart, Math.min(trimEnd, v)))} />
          <div style={{
            fontSize: '9px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
            background: 'rgba(255,255,255,0.03)', padding: '4px 8px', borderRadius: '4px',
            border: '1px solid rgba(255,255,255,0.05)'
          }}>
            <span style={{ color: 'var(--text-secondary)' }}>F</span> {currentFrame} <span style={{ color: 'var(--text-muted)' }}>/ {totalFrames}</span>
          </div>
        </div>

        {/* RIGHT: TOOLS */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {isTrimmed && (
            <button onClick={() => onTrimChange(0, duration)}
              title="Reset Trim Range"
              style={{
                background: 'rgba(255, 100, 100, 0.1)', border: '1px solid rgba(255, 100, 100, 0.2)',
                color: '#ff9999', borderRadius: '4px', padding: '0 10px',
                fontSize: '9px', cursor: 'pointer', height: '26px', fontWeight: 600,
                display: 'flex', alignItems: 'center', gap: '4px',
                transition: 'all 0.15s ease'
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255, 100, 100, 0.2)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255, 100, 100, 0.1)'; }}
            >
              <span style={{ fontSize: '10px' }}>✕</span> RESET
            </button>
          )}
          <div style={{ display: 'flex', gap: '2px', alignItems: 'center' }}>
            <ZoomButton type="out" onClick={() => setZoom(z => Math.max(1, z - 0.5))} disabled={zoom <= 1} />
            <div style={{
              fontSize: '9px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
              minWidth: '44px', textAlign: 'center', padding: '0 4px'
            }}>
              {Math.round(zoom * 100)}%
            </div>
            <ZoomButton type="in" onClick={() => setZoom(z => Math.min(20, z + 0.5))} disabled={zoom >= 20} />
            {zoom > 1 && <ZoomButton type="fit" onClick={() => setZoom(1)} />}
          </div>
        </div>
      </div>

      {/* TRACK CONTAINER */}
      <div ref={scrollContainerRef} onWheel={handleWheel} className="timeline-scroll-area"
        style={{
          width: '100%', overflowX: 'auto', overflowY: 'hidden', position: 'relative',
          paddingTop: '32px', paddingBottom: '14px', paddingLeft: '16px', paddingRight: '16px',
          boxSizing: 'border-box',
          background: 'var(--timeline-bg)',
          borderRadius: '0 0 6px 6px'
        }}>

        <div ref={trackRef} style={{
          width: `${100 * zoom}%`, height: '28px', position: 'relative',
          transition: isDragging ? 'none' : 'width 0.2s ease'
        }}
          onMouseMove={(e) => { if (trackRef.current) setHoverTime(getTimeFromEvent(e.clientX, duration)); }}
          onMouseLeave={() => { setHoverTime(null); setHoveredHandle(null); }}
          onMouseDown={(e) => {
            if (e.target === trackRef.current || (e.target as HTMLElement).className.includes('bg-layer')) {
              handleMouseDown(e, 'playhead');
              const t = getTimeFromEvent(e.clientX, duration);
              onSeek(Math.max(trimStart, Math.min(trimEnd, t)));
            }
          }}>

          {/* RULER */}
          <div style={{ position: 'absolute', top: -18, left: 0, right: 0, height: '18px', pointerEvents: 'none', zIndex: 1 }}>
            {getRulerTicks.map((tick, i) => (
              <div key={i} style={{
                position: 'absolute', left: `${tick.pct}%`, bottom: 0,
                transform: 'translateX(-50%)', display: 'flex', flexDirection: 'column', alignItems: 'center'
              }}>
                {tick.isMajor && (
                  <span style={{
                    fontSize: '9px', color: '#71717a', marginBottom: '3px',
                    fontFamily: 'var(--font-mono)', fontWeight: 500,
                    textShadow: '0 1px 2px rgba(0,0,0,0.5)'
                  }}>{formatRulerTime(tick.time)}</span>
                )}
                <div style={{
                  width: tick.isMajor ? 1 : 1,
                  height: tick.isMajor ? 8 : 4,
                  background: tick.isMajor
                    ? 'linear-gradient(to bottom, #71717a, #52525b)'
                    : '#3f3f46',
                  borderRadius: '0 0 1px 1px'
                }} />
              </div>
            ))}
          </div>

          {/* GROOVE / TRACK BED */}
          <div className="bg-layer" style={{
            position: 'absolute', top: 6, bottom: 6, left: 0, right: 0,
            background: 'var(--timeline-track-bg)',
            borderRadius: '6px', overflow: 'hidden',
            border: '1px solid var(--panel-border)',
            zIndex: 2,
            boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.6), inset 0 -1px 2px rgba(255,255,255,0.02)'
          }}>
            {/* Waveform placeholder with better styling */}
            {hasAudio && (
              <div style={{
                width: '100%', height: '100%', opacity: 0.12,
                backgroundImage: `
                  repeating-linear-gradient(90deg,
                    transparent 0px, transparent 3px,
                    rgba(255,255,255,0.3) 3px, rgba(255,255,255,0.3) 4px,
                    transparent 4px, transparent 6px,
                    rgba(255,255,255,0.5) 6px, rgba(255,255,255,0.5) 7px,
                    transparent 7px, transparent 12px
                  )
                `,
                maskImage: 'linear-gradient(to bottom, transparent 15%, black 40%, black 60%, transparent 85%)'
              }} />
            )}
          </div>

          {/* TRIMMED REGIONS (Disabled Zones) */}
          <div style={{
            position: 'absolute', top: 6, bottom: 6, left: 0, width: `${startPct}%`,
            background: 'linear-gradient(to right, rgba(0,0,0,0.85), rgba(0,0,0,0.75))',
            backdropFilter: 'grayscale(1) brightness(0.5)',
            pointerEvents: 'none', zIndex: 5, borderRadius: '6px 0 0 6px',
            borderRight: startPct > 0 ? '1px solid rgba(0, 255, 136, 0.3)' : 'none'
          }} />
          <div style={{
            position: 'absolute', top: 6, bottom: 6, left: `${endPct}%`, right: 0,
            background: 'linear-gradient(to left, rgba(0,0,0,0.85), rgba(0,0,0,0.75))',
            backdropFilter: 'grayscale(1) brightness(0.5)',
            pointerEvents: 'none', zIndex: 5, borderRadius: '0 6px 6px 0',
            borderLeft: endPct < 100 ? '1px solid rgba(0, 255, 136, 0.3)' : 'none'
          }} />

          {/* ACTIVE TRIM RANGE */}
          <div onMouseDown={(e) => handleMouseDown(e, 'range')} style={{
            position: 'absolute', top: 7, bottom: 7, left: `${startPct}%`, width: `${widthPct}%`,
            background: dragTarget === 'range'
              ? 'rgba(0, 255, 136, 0.35)'
              : 'rgba(0, 255, 136, 0.18)',
            cursor: dragTarget === 'range' ? 'grabbing' : 'grab', zIndex: 6,
            borderRadius: '4px',
            boxShadow: dragTarget === 'range'
              ? '0 0 20px rgba(0, 255, 136, 0.5), inset 0 0 10px rgba(0, 255, 136, 0.2)'
              : '0 0 12px rgba(0, 255, 136, 0.3)',
            transition: isDragging ? 'none' : 'background 0.15s ease, box-shadow 0.15s ease'
          }} />

          {/* TRIM HANDLES */}
          <div
            onMouseDown={(e) => handleMouseDown(e, 'start')}
            onMouseEnter={() => setHoveredHandle('start')}
            onMouseLeave={() => setHoveredHandle(null)}
            style={{
              position: 'absolute', left: `${startPct}%`, top: -2, height: '36px', width: '48px', marginLeft: '-24px',
              cursor: 'ew-resize', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: isDragging ? 'none' : 'left 0.05s ease'
            }}
            title="Drag to set in point (I)"
          >
            <TrimStartIcon isHovered={hoveredHandle === 'start' || dragTarget === 'start'} />
          </div>
          <div
            onMouseDown={(e) => handleMouseDown(e, 'end')}
            onMouseEnter={() => setHoveredHandle('end')}
            onMouseLeave={() => setHoveredHandle(null)}
            style={{
              position: 'absolute', left: `${endPct}%`, top: -2, height: '36px', width: '48px', marginLeft: '-24px',
              cursor: 'ew-resize', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: isDragging ? 'none' : 'left 0.05s ease'
            }}
            title="Drag to set out point (O)"
          >
            <TrimEndIcon isHovered={hoveredHandle === 'end' || dragTarget === 'end'} />
          </div>

          {/* PLAYHEAD */}
          <div
            onMouseDown={(e) => handleMouseDown(e, 'playhead')}
            style={{
              position: 'absolute', left: `${currPct}%`, top: -6, bottom: -4,
              width: '48px', marginLeft: '-24px',
              cursor: dragTarget === 'playhead' ? 'grabbing' : 'grab', zIndex: 110,
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              transition: isDragging ? 'none' : 'left 0.05s ease'
            }}
            title="Drag to scrub (or click timeline)"
          >
            <PlayheadIcon isDragging={dragTarget === 'playhead'} />
            <div style={{
              width: dragTarget === 'playhead' ? 2 : 1.5,
              height: '100%',
              background: dragTarget === 'playhead'
                ? '#fff'
                : 'linear-gradient(to bottom, #fff, rgba(255,255,255,0.7))',
              boxShadow: dragTarget === 'playhead'
                ? '0 0 8px rgba(255,255,255,0.8), 0 0 16px rgba(255,255,255,0.4)'
                : '0 0 6px rgba(0,0,0,0.8)',
              marginTop: '-3px',
              borderRadius: '1px',
              transition: 'width 0.1s ease, box-shadow 0.1s ease'
            }} />
          </div>

          {/* HOVER TIME TOOLTIP */}
          {hoverTime !== null && !dragState.current.active && (
            <div style={{
              position: 'absolute', left: `${(hoverTime / safeDur) * 100}%`, top: -38,
              transform: 'translateX(-50%)',
              background: 'linear-gradient(to bottom, #2a2a2e, #222226)',
              padding: '4px 10px', borderRadius: '6px',
              fontSize: '10px', border: '1px solid rgba(255,255,255,0.1)',
              pointerEvents: 'none', zIndex: 120,
              color: '#fff', fontFamily: 'var(--font-mono)',
              boxShadow: '0 4px 12px rgba(0,0,0,0.5), 0 2px 4px rgba(0,0,0,0.3)',
              whiteSpace: 'nowrap',
              animation: 'fadeIn 0.1s ease'
            }}>
              <span style={{ color: 'var(--brand-primary)', fontWeight: 600 }}>{formatTimeFull(hoverTime)}</span>
              <span style={{ color: 'var(--text-muted)', marginLeft: '6px', fontSize: '9px' }}>
                F{Math.floor(hoverTime * FRAME_RATE)}
              </span>
            </div>
          )}

          {/* HOVER LINE */}
          {hoverTime !== null && !dragState.current.active && (
            <div style={{
              position: 'absolute', left: `${(hoverTime / safeDur) * 100}%`, top: 6, bottom: 6,
              width: 1, background: 'rgba(255,255,255,0.3)',
              pointerEvents: 'none', zIndex: 4,
              transform: 'translateX(-50%)'
            }} />
          )}
        </div>
      </div>

      {/* KEYBOARD HINTS (shown on focus) */}
      <div style={{
        display: 'flex', justifyContent: 'center', gap: '16px',
        padding: '4px 0', fontSize: '9px', color: 'var(--text-muted)',
        opacity: 0.5
      }}>
        <span>← → Frame step</span>
        <span>Shift+← → 1s step</span>
        <span>Home/End Jump to trim</span>
        <span>Ctrl+Scroll Zoom</span>
      </div>
    </div>
  );
};
