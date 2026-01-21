import React from 'react';

interface SignalSummaryProps {
  sourceResolution: string;
  sourceDetail?: string; // e.g. "1920x1080"
  sourceFps: string;
  targetResolution: string;
  targetDetail?: string; // e.g. "3840x2160"
  targetFps: string;
  className?: string;
}

export const SignalSummary: React.FC<SignalSummaryProps> = ({
  sourceResolution,
  sourceDetail,
  sourceFps,
  targetResolution,
  targetDetail,
  targetFps,
  className
}) => {
  return (
    <div
      className={`signal-summary-container ${className || ''}`}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '10px 12px',
        backgroundColor: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '6px',
        fontFamily: 'var(--font-sans)',
        marginBottom: '12px'
      }}
    >
      {/* SOURCE: Read-Only Telemetry */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
        <span
          style={{
            fontSize: '9px',
            textTransform: 'uppercase',
            color: '#52525b', // Muted text
            letterSpacing: '0.05em',
            fontWeight: 700,
            userSelect: 'none'
          }}
        >
          SOURCE
        </span>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '6px' }}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '12px', color: '#ededed', fontWeight: 600 }}>
              {sourceResolution}
            </span>
            {sourceDetail && (
              <span style={{ fontSize: '9px', color: '#52525b', fontFamily: 'var(--font-mono)' }}>
                {sourceDetail}
              </span>
            )}
          </div>
          {sourceFps && (
             <span style={{ fontSize: '10px', color: '#52525b', fontFamily: 'var(--font-mono)', marginBottom: sourceDetail ? '2px' : '1px' }}>
               {sourceFps}
             </span>
          )}
        </div>
      </div>

      {/* SIGNAL FLOW ARROW */}
      <div style={{ color: '#52525b', fontSize: '12px', padding: '0 8px', opacity: 0.5 }}>
        ➔
      </div>

      {/* TARGET: User Intent */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', alignItems: 'flex-end', textAlign: 'right' }}>
        <span
          style={{
            fontSize: '9px',
            textTransform: 'uppercase',
            color: 'var(--brand-primary)',
            letterSpacing: '0.05em',
            fontWeight: 700,
            userSelect: 'none'
          }}
        >
          TARGET
        </span>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '6px' }}>
          {targetFps && (
             <span style={{ fontSize: '10px', color: 'var(--brand-primary)', fontFamily: 'var(--font-mono)', marginBottom: targetDetail ? '2px' : '1px', opacity: 0.8 }}>
               {targetFps}
             </span>
          )}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
            <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: 800 }}>
              {targetResolution}
            </span>
            {targetDetail && (
              <span style={{ fontSize: '9px', color: '#71717a', fontFamily: 'var(--font-mono)' }}>
                {targetDetail}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
