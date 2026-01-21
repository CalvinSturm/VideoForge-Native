import React from 'react';
import { useJobStore } from '../Store/useJobStore';
import { invoke } from "@tauri-apps/api/core";

// --- ICONS ---
const IconSun = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5" /><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" /></svg>;
const IconMoon = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>;
const IconFolder = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" /></svg>;
// NEW: Info Icon
const IconInfo = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>;

// --- HELPERS ---
const fmtMem = (bytes: number) => {
  if (!bytes) return "0.0 G";
  return (bytes / 1024 / 1024 / 1024).toFixed(1) + "G";
};

// --- COMPONENT ---

interface StatusFooterProps {
  toggleTheme: () => void;
  darkMode: boolean;
  showTechSpecs: boolean;
  setShowTechSpecs: (show: boolean) => void;
  outputPath: string;
}

export const StatusFooter: React.FC<StatusFooterProps> = ({
  toggleTheme, darkMode, showTechSpecs, setShowTechSpecs, outputPath
}) => {
  const { isProcessing, framesProcessed, totalFrames, progressPercent, stats } = useJobStore();

  const handleOpenFolder = () => {
    if (outputPath) invoke('show_in_folder', { path: outputPath });
  };

  const statItemStyle: React.CSSProperties = {
    display: 'flex', alignItems: 'center', gap: '6px',
    padding: '0 8px', borderLeft: '1px solid rgba(255,255,255,0.08)', height: '16px'
  };

  return (
    <footer style={{
      height: '28px', // Slightly slimmer for sleekness
      backgroundColor: '#09090b',
      borderTop: '1px solid rgba(255,255,255,0.06)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 12px',
      fontSize: '10px',
      fontFamily: '"JetBrains Mono", monospace',
      color: '#71717a',
      userSelect: 'none',
      zIndex: 1000
    }}>

      {/* LEFT: Toggles (Ghost Buttons) */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <button
          onClick={() => setShowTechSpecs(!showTechSpecs)}
          title="Toggle Hardware Stats"
          style={{
            background: showTechSpecs ? 'rgba(255,255,255,0.08)' : 'transparent',
            border: 'none',
            color: showTechSpecs ? '#ededed' : '#71717a',
            cursor: 'pointer',
            padding: '4px',
            borderRadius: '4px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'all 0.1s'
          }}
          onMouseEnter={(e) => !showTechSpecs && (e.currentTarget.style.color = '#ededed')}
          onMouseLeave={(e) => !showTechSpecs && (e.currentTarget.style.color = '#71717a')}
        >
          {/* UPDATED ICON */}
          <IconInfo />
        </button>

        <button
          onClick={toggleTheme}
          title="Toggle Theme"
          style={{
            background: 'transparent',
            border: 'none',
            color: '#71717a',
            cursor: 'pointer',
            padding: '4px',
            borderRadius: '4px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'color 0.1s'
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = '#ededed')}
          onMouseLeave={(e) => (e.currentTarget.style.color = '#71717a')}
        >
          {darkMode ? <IconMoon /> : <IconSun />}
        </button>

        <div style={{ width: 1, height: 12, background: 'rgba(255,255,255,0.1)', margin: '0 6px' }} />

        <button
          onClick={handleOpenFolder}
          disabled={!outputPath}
          title="Reveal Output Folder"
          style={{
            background: 'transparent',
            border: 'none',
            color: outputPath ? '#71717a' : '#333',
            cursor: outputPath ? 'pointer' : 'default',
            padding: '4px',
            borderRadius: '4px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'color 0.1s'
          }}
          onMouseEnter={(e) => outputPath && (e.currentTarget.style.color = '#ededed')}
          onMouseLeave={(e) => outputPath && (e.currentTarget.style.color = '#71717a')}
        >
          <IconFolder />
        </button>
      </div>

      {/* CENTER: Frame Progress */}
      <div style={{ flex: 1, display: 'flex', justifyContent: 'center' }}>
        {isProcessing ? (
           <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
             <span style={{ color: '#00ff88', fontWeight: 'bold' }}>
               PROCESSING
             </span>
             <div style={{ width: '150px', height: '3px', background: '#27272a', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{
                   width: `${progressPercent}%`,
                   height: '100%', background: '#00ff88',
                   transition: 'width 0.2s linear'
                }} />
             </div>
             <span style={{ color: '#ededed' }}>
                {totalFrames > 0
                  ? `${framesProcessed} / ${totalFrames}`
                  : `${Math.round(progressPercent)}%`
                }
             </span>
           </div>
        ) : (
           <span style={{ color: '#3f3f46', cursor: 'default' }}>
             {outputPath ? "READY" : "IDLE"}
           </span>
        )}
      </div>

      {/* RIGHT: Stats (Only visible if active) */}
      {showTechSpecs ? (
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={statItemStyle}>
             <span style={{ opacity: 0.5 }}>CPU</span>
             <span>{Math.round(stats.cpu)}%</span>
          </div>
          <div style={statItemStyle}>
             <span style={{ opacity: 0.5 }}>RAM</span>
             <span style={{ color: stats.ramUsed > stats.ramTotal * 0.9 ? '#ef4444' : 'inherit' }}>
                {fmtMem(stats.ramUsed)}
             </span>
          </div>
          <div style={{ ...statItemStyle, borderRight: 'none', borderLeft: '1px solid rgba(255,255,255,0.08)' }}>
             <span style={{ color: '#00ff88', marginRight: '4px' }}>GPU</span>
             <span style={{ opacity: 0.7 }}>ACTIVE</span>
          </div>
        </div>
      ) : (
        <div style={{ width: '100px' }} />
      )}

    </footer>
  );
};
