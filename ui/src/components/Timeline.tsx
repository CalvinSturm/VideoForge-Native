import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';

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
    seconds = parseInt(parts[0]) * 60 + parseFloat(parts[1]);
  } else {
    seconds = parseFloat(str);
  }
  return Math.min(Math.max(0, seconds), maxDuration);
};

// --- SUB-COMPONENTS ---

const TimeInput = ({ time, max, onChange, label, highlight = false }: any) => {
  const [val, setVal] = useState(formatTimeFull(time));
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => { if (!isEditing) setVal(formatTimeFull(time)); }, [time, isEditing]);

  const commit = () => {
    const s = parseTimeInput(val, max);
    onChange(s);
    setVal(formatTimeFull(s));
    setIsEditing(false);
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', background: 'rgba(255,255,255,0.03)', borderRadius: '4px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{
        padding: '0 6px', fontSize: '9px', fontWeight: 700,
        color: highlight ? '#000' : 'var(--text-muted)',
        background: highlight ? 'var(--brand-primary)' : 'rgba(255,255,255,0.05)',
        height: '24px', display: 'flex', alignItems: 'center', cursor: 'default'
      }}>
        {label}
      </div>
      <input
        type="text" value={val}
        onChange={(e) => setVal(e.target.value)}
        onFocus={() => setIsEditing(true)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') { e.currentTarget.blur(); } }}
        style={{
          background: 'transparent', border: 'none', outline: 'none',
          color: highlight ? 'var(--brand-primary)' : '#ededed',
          fontFamily: '"JetBrains Mono", monospace', fontSize: '11px',
          width: '72px', textAlign: 'center', height: '24px',
          padding: '0'
        }}
      />
    </div>
  );
};

// --- ICONS ---

const TrimStartIcon = () => (
  <svg width="14" height="28" viewBox="0 0 14 28" fill="none" style={{ filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.5))' }}>
    <path d="M0 0 H14 V5 H6 V23 H14 V28 H0 Z" fill="var(--brand-primary)" />
    <path d="M2 10 H4 M2 14 H4 M2 18 H4" stroke="rgba(0,0,0,0.5)" strokeWidth="2" strokeLinecap="round" />
  </svg>
);

const TrimEndIcon = () => (
  <svg width="14" height="28" viewBox="0 0 14 28" fill="none" style={{ filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.5))' }}>
    <path d="M14 0 H0 V5 H8 V23 H0 V28 H14 Z" fill="var(--brand-primary)" />
    <path d="M10 10 H12 M10 14 H12 M10 18 H12" stroke="rgba(0,0,0,0.5)" strokeWidth="2" strokeLinecap="round" />
  </svg>
);

const PlayheadIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.5))' }}>
    <path d="M8 15L2 9V1H14V9L8 15Z" fill="#ffffff" />
  </svg>
);

// --- MAIN COMPONENT ---

export const Timeline: React.FC<any> = ({ duration, currentTime, trimStart, trimEnd, onSeek, onTrimChange, hasAudio = true, onInteractionStart, onInteractionEnd }) => {
  const trackRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [hoverTime, setHoverTime] = useState<number | null>(null);

  const propsRef = useRef({ duration, currentTime, trimStart, trimEnd, onSeek, onTrimChange, onInteractionStart, onInteractionEnd });
  useEffect(() => { propsRef.current = { duration, currentTime, trimStart, trimEnd, onSeek, onTrimChange, onInteractionStart, onInteractionEnd }; });

  const dragState = useRef<{
    active: boolean;
    target: 'playhead' | 'start' | 'end' | 'range' | null;
    startX: number;
    startVal: number;
    endVal: number;
  }>({ active: false, target: null, startX: 0, startVal: 0, endVal: 0 });

  // --- ADAPTIVE TICKS ---
  const getRulerTicks = useMemo(() => {
    const safeDuration = Math.max(duration, 0.1);
    const visibleSeconds = safeDuration / zoom;
    const targetInterval = visibleSeconds / 10;

    const intervals = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
    let majorStep = intervals.find(i => i >= targetInterval) || intervals[intervals.length - 1];

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

  return (
    <div className="timeline-wrapper" style={{ display: 'flex', flexDirection: 'column', width: '100%', userSelect: 'none', touchAction: 'none' }}>
      <style>{`
        .timeline-scroll-area::-webkit-scrollbar { height: 6px; background: #09090b; }
        .timeline-scroll-area::-webkit-scrollbar-track { background: #18181b; border-radius: 3px; margin: 0 16px; }
        .timeline-scroll-area::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 3px; border: 1px solid #18181b; }
        .timeline-scroll-area::-webkit-scrollbar-thumb:hover { background: #52525b; }
      `}</style>

      {/* DASHBOARD TOOLBAR */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: '4px', padding: '4px', height: '32px',
        borderBottom: '1px solid rgba(255,255,255,0.06)', background: '#09090b'
      }}>
        {/* LEFT: RANGE */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <TimeInput label="IN" time={trimStart} max={trimEnd} onChange={(v: any) => { onTrimChange(v, trimEnd); if (currentTime < v) onSeek(v); }} />
          <div style={{ width: '8px', height: '1px', background: '#3f3f46' }}></div>
          <TimeInput label="OUT" time={trimEnd} max={duration} onChange={(v: any) => { onTrimChange(trimStart, v); if (currentTime > v) onSeek(v); }} />
        </div>

        {/* CENTER: HEAD */}
        <div style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)' }}>
          <TimeInput label="HEAD" time={currentTime} max={trimEnd} highlight onChange={(v: any) => onSeek(Math.max(trimStart, Math.min(trimEnd, v)))} />
        </div>

        {/* RIGHT: TOOLS */}
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          {isTrimmed && (
            <button onClick={() => onTrimChange(0, duration)}
              title="Reset Trim Range"
              style={{
                background: 'transparent', border: '1px solid #3f3f46',
                color: '#a1a1aa', borderRadius: '3px', padding: '0 8px',
                fontSize: '9px', cursor: 'pointer', height: '24px', fontWeight: 600,
                display: 'flex', alignItems: 'center'
              }}>
              RESET TRIM
            </button>
          )}
          <div style={{ fontSize: '9px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span>ZOOM: {Math.round(zoom * 100)}%</span>
            {zoom > 1 && <button onClick={() => setZoom(1)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--brand-primary)', fontWeight: 700, padding: 0 }}>[RESET]</button>}
          </div>
        </div>
      </div>

      {/* TRACK CONTAINER */}
      <div ref={scrollContainerRef} onWheel={handleWheel} className="timeline-scroll-area"
        style={{ width: '100%', overflowX: 'auto', overflowY: 'hidden', position: 'relative', paddingTop: '28px', paddingBottom: '12px', paddingLeft: '16px', paddingRight: '16px', boxSizing: 'border-box' }}>

        <div ref={trackRef} style={{ width: `${100 * zoom}%`, height: '24px', position: 'relative' }}
          onMouseMove={(e) => { if (trackRef.current) setHoverTime(getTimeFromEvent(e.clientX, duration)); }}
          onMouseLeave={() => setHoverTime(null)}
          onMouseDown={(e) => {
            if (e.target === trackRef.current || (e.target as HTMLElement).className.includes('bg-layer')) {
              handleMouseDown(e, 'playhead');
              const t = getTimeFromEvent(e.clientX, duration);
              onSeek(Math.max(trimStart, Math.min(trimEnd, t)));
            }
          }}>

          {/* RULER */}
          <div style={{ position: 'absolute', top: -14, left: 0, right: 0, height: '14px', pointerEvents: 'none', zIndex: 1 }}>
            {getRulerTicks.map((tick, i) => (
              <div key={i} style={{ position: 'absolute', left: `${tick.pct}%`, bottom: 0, transform: 'translateX(-50%)', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                {tick.isMajor && (
                  <span style={{ fontSize: '9px', color: '#a1a1aa', marginBottom: '2px', fontFamily: 'var(--font-mono)', fontWeight: 500 }}>{formatRulerTime(tick.time)}</span>
                )}
                <div style={{ width: 1, height: tick.isMajor ? 6 : 3, background: tick.isMajor ? '#a1a1aa' : '#52525b', opacity: tick.isMajor ? 0.8 : 0.5 }} />
              </div>
            ))}
          </div>

          {/* GROOVE */}
          <div className="bg-layer" style={{ position: 'absolute', top: 8, bottom: 8, left: 0, right: 0, background: '#09090b', borderRadius: '4px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.08)', zIndex: 2, boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.8)' }}>
            {hasAudio && <div style={{ width: '100%', height: '100%', opacity: 0.15, backgroundImage: 'repeating-linear-gradient(90deg, transparent 0, transparent 2px, #fff 2px, #fff 3px)' }} />}
          </div>

          {/* HANDLES & ZONES */}
          {/* Enhanced Glow Style here */}
          <div onMouseDown={(e) => handleMouseDown(e, 'range')} style={{
            position: 'absolute', top: 9, bottom: 9, left: `${startPct}%`, width: `${widthPct}%`,
            background: 'rgba(0, 255, 136, 0.25)', // Brighter green
            cursor: 'grab', zIndex: 6,
            borderLeft: '1px solid var(--brand-primary)', borderRight: '1px solid var(--brand-primary)',
            boxShadow: '0 0 10px rgba(0, 255, 136, 0.4)' // Green glow
          }} />

          <div style={{ position: 'absolute', top: 8, bottom: 8, left: 0, width: `${startPct}%`, background: 'rgba(0,0,0,0.7)', backdropFilter: 'grayscale(1)', pointerEvents: 'none', zIndex: 5, borderRadius: '4px 0 0 4px' }} />
          <div style={{ position: 'absolute', top: 8, bottom: 8, left: `${endPct}%`, right: 0, background: 'rgba(0,0,0,0.7)', backdropFilter: 'grayscale(1)', pointerEvents: 'none', zIndex: 5, borderRadius: '0 4px 4px 0' }} />

          <div onMouseDown={(e) => handleMouseDown(e, 'start')} style={{ position: 'absolute', left: `${startPct}%`, top: -4, height: '32px', width: '60px', marginLeft: '-30px', cursor: 'ew-resize', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }} title="Drag Start"><TrimStartIcon /></div>
          <div onMouseDown={(e) => handleMouseDown(e, 'end')} style={{ position: 'absolute', left: `${endPct}%`, top: -4, height: '32px', width: '60px', marginLeft: '-30px', cursor: 'ew-resize', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }} title="Drag End"><TrimEndIcon /></div>

          <div onMouseDown={(e) => handleMouseDown(e, 'playhead')} style={{ position: 'absolute', left: `${currPct}%`, top: -6, bottom: -2, width: '60px', marginLeft: '-30px', cursor: 'grab', zIndex: 110, display: 'flex', flexDirection: 'column', alignItems: 'center' }} title="Scrub">
            <PlayheadIcon />
            <div style={{ width: 1, height: '100%', background: '#fff', boxShadow: '0 0 4px rgba(0,0,0,0.8)', marginTop: '-4px' }} />
          </div>

          {hoverTime !== null && !dragState.current.active && (
            <div style={{ position: 'absolute', left: `${(hoverTime / safeDur) * 100}%`, top: -30, transform: 'translateX(-50%)', background: '#222', padding: '2px 6px', borderRadius: '4px', fontSize: '10px', border: '1px solid #444', pointerEvents: 'none', zIndex: 120, color: '#fff', fontFamily: 'monospace', boxShadow: '0 2px 8px rgba(0,0,0,0.5)', whiteSpace: 'nowrap' }}>
              {formatTimeFull(hoverTime)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
