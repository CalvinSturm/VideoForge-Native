import React from 'react';
import { useJobStore } from '../Store/useJobStore';
import { invoke } from "@tauri-apps/api/core";

// --- ICONS ---
const IconSun = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5" /><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" /></svg>;
const IconMoon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>;
const IconFolder = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" /></svg>;
const IconCpu = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" /><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" /></svg>;
const IconActivity = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>;
const IconZap = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>;

// --- HELPERS ---
const fmtMem = (bytes: number) => {
  if (!bytes) return "0.0 GB";
  return (bytes / 1024 / 1024 / 1024).toFixed(1) + " GB";
};

// --- COMPONENT ---
interface StatusFooterProps {
  toggleTheme: () => void;
  darkMode: boolean;
  showTechSpecs: boolean;
  setShowTechSpecs: (show: boolean) => void;
}

export const StatusFooter: React.FC<StatusFooterProps> = ({
  toggleTheme, darkMode, showTechSpecs, setShowTechSpecs
}) => {
  const { isProcessing, framesProcessed, totalFrames, progressPercent, stats, lastOutputPath } = useJobStore();

  const handleOpenFolder = () => {
    if (lastOutputPath) invoke('show_in_folder', { path: lastOutputPath });
  };

  // Stat pill component
  const StatPill: React.FC<{
    icon: React.ReactNode;
    label: string;
    value: string;
    accent?: string;
    warning?: boolean;
  }> = ({ icon, label, value, accent, warning }) => (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      padding: '3px 10px',
      background: 'rgba(255,255,255,0.03)',
      borderRadius: '4px',
      border: '1px solid rgba(255,255,255,0.06)'
    }}>
      <span style={{ color: accent || 'var(--text-muted)', display: 'flex' }}>{icon}</span>
      <span style={{
        fontSize: '9px',
        color: 'var(--text-muted)',
        fontWeight: 600,
        letterSpacing: '0.03em'
      }}>{label}</span>
      <span style={{
        fontSize: '10px',
        color: warning ? '#ef4444' : (accent || 'var(--text-primary)'),
        fontWeight: 600,
        fontFamily: 'var(--font-mono)'
      }}>{value}</span>
    </div>
  );

  // Footer button component
  const FooterButton: React.FC<{
    icon: React.ReactNode;
    onClick: () => void;
    title: string;
    active?: boolean;
    disabled?: boolean;
  }> = ({ icon, onClick, title, active, disabled }) => (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        width: '36px',
        height: '32px',
        background: active ? 'rgba(0,255,136,0.1)' : 'transparent',
        border: active ? '1px solid rgba(0,255,136,0.2)' : '1px solid transparent',
        borderRadius: '4px',
        color: disabled ? 'var(--text-muted)' : (active ? 'var(--brand-primary)' : 'var(--text-secondary)'),
        cursor: disabled ? 'not-allowed' : 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'all 0.15s ease',
        opacity: disabled ? 0.4 : 1
      }}
      onMouseEnter={(e) => !disabled && !active && (e.currentTarget.style.background = 'rgba(255,255,255,0.05)', e.currentTarget.style.color = 'var(--text-primary)')}
      onMouseLeave={(e) => !disabled && !active && (e.currentTarget.style.background = 'transparent', e.currentTarget.style.color = 'var(--text-secondary)')}
    >
      {icon}
    </button>
  );

  return (
    <footer style={{
      height: '36px',
      backgroundColor: 'var(--panel-bg)',
      borderTop: '1px solid var(--panel-border)',
      display: 'grid',
      gridTemplateColumns: '1fr auto 1fr',
      alignItems: 'center',
      padding: '0 12px',
      fontSize: '10px',
      fontFamily: 'var(--font-mono)',
      color: 'var(--text-secondary)',
      userSelect: 'none',
      zIndex: 1000
    }}>

      {/* LEFT: Control Buttons */}
      <div style={{ justifySelf: 'start', display: 'flex', alignItems: 'center', gap: '4px' }}>
        <FooterButton
          icon={<IconCpu />}
          onClick={() => setShowTechSpecs(!showTechSpecs)}
          title="Toggle Hardware Stats"
          active={showTechSpecs}
        />
        <FooterButton
          icon={darkMode ? <IconMoon /> : <IconSun />}
          onClick={toggleTheme}
          title={`Switch to ${darkMode ? 'Light' : 'Dark'} Mode`}
        />

        <div style={{ width: '1px', height: '16px', background: 'rgba(255,255,255,0.08)', margin: '0 6px' }} />

        <FooterButton
          icon={<IconFolder />}
          onClick={handleOpenFolder}
          title="Reveal Output in Folder"
          disabled={!lastOutputPath}
        />
      </div>

      {/* CENTER: Processing Status */}
      <div style={{ justifySelf: 'center', display: 'flex', alignItems: 'center', gap: '16px' }}>
        {isProcessing ? (
          <>
            {/* Status indicator */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '4px 12px',
              background: 'rgba(0,255,136,0.08)',
              borderRadius: '6px',
              border: '1px solid rgba(0,255,136,0.2)'
            }}>
              <div style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: 'var(--brand-primary)',
                boxShadow: '0 0 8px var(--brand-primary)',
                animation: 'pulse 1.5s infinite'
              }} />
              <span style={{
                color: 'var(--brand-primary)',
                fontWeight: 700,
                fontSize: '9px',
                letterSpacing: '0.08em'
              }}>
                PROCESSING
              </span>
            </div>

            {/* Progress bar */}
            <div style={{
              width: '140px',
              height: '6px',
              background: 'rgba(0,0,0,0.4)',
              borderRadius: '3px',
              overflow: 'hidden',
              boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.4)'
            }}>
              <div style={{
                width: `${progressPercent}%`,
                height: '100%',
                background: 'linear-gradient(90deg, var(--brand-primary), #33ff99)',
                borderRadius: '3px',
                transition: 'width 0.2s linear',
                boxShadow: '0 0 10px rgba(0,255,136,0.5)'
              }} />
            </div>

            {/* Frame count */}
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
              {totalFrames > 0
                ? `${framesProcessed} / ${totalFrames}`
                : `${Math.round(progressPercent)}%`
              }
            </span>
          </>
        ) : (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '4px 12px',
            background: lastOutputPath ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.02)',
            borderRadius: '6px',
            border: lastOutputPath ? '1px solid rgba(16,185,129,0.2)' : '1px solid rgba(255,255,255,0.04)'
          }}>
            <div style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              background: lastOutputPath ? '#10b981' : 'var(--text-muted)'
            }} />
            <span style={{
              color: lastOutputPath ? '#10b981' : 'var(--text-muted)',
              fontWeight: 600,
              fontSize: '9px',
              letterSpacing: '0.08em'
            }}>
              {lastOutputPath ? "READY" : "IDLE"}
            </span>
          </div>
        )}
      </div>

      {/* RIGHT: System Stats */}
      <div style={{ justifySelf: 'end', display: 'flex', alignItems: 'center', gap: '8px' }}>
        {showTechSpecs && (
          <>
            <StatPill
              icon={<IconCpu />}
              label="CPU"
              value={`${Math.round(stats.cpu)}%`}
              warning={stats.cpu > 90}
            />
            <StatPill
              icon={<IconActivity />}
              label="RAM"
              value={fmtMem(stats.ramUsed)}
              warning={stats.ramUsed > stats.ramTotal * 0.9}
            />
            <StatPill
              icon={<IconZap />}
              label="GPU"
              value="ACTIVE"
              accent="var(--brand-primary)"
            />
          </>
        )}
      </div>
    </footer>
  );
};
