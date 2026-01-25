import React from 'react';
import type { PanelId } from '../Store/viewLayoutStore';

// Panel icons - unique for each panel type
const IconSettings = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

const IconPreview = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const IconQueue = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="8" y1="6" x2="21" y2="6" />
    <line x1="8" y1="12" x2="21" y2="12" />
    <line x1="8" y1="18" x2="21" y2="18" />
    <line x1="3" y1="6" x2="3.01" y2="6" />
    <line x1="3" y1="12" x2="3.01" y2="12" />
    <line x1="3" y1="18" x2="3.01" y2="18" />
  </svg>
);

const IconActivity = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
  </svg>
);

// Panel configuration with colors and shortcuts
const PANEL_CONFIG: Record<PanelId, {
  icon: React.ReactNode;
  accentColor: string;
  shortcut: string;
  label: string;
}> = {
  SETTINGS: {
    icon: <IconSettings />,
    accentColor: '#3b82f6', // Blue
    shortcut: 'Ctrl+1',
    label: 'Settings'
  },
  PREVIEW: {
    icon: <IconPreview />,
    accentColor: '#00ff88', // Brand green
    shortcut: 'Ctrl+2',
    label: 'Preview'
  },
  QUEUE: {
    icon: <IconQueue />,
    accentColor: '#f59e0b', // Amber
    shortcut: 'Ctrl+3',
    label: 'Queue'
  },
  ACTIVITY: {
    icon: <IconActivity />,
    accentColor: '#a855f7', // Purple
    shortcut: 'Ctrl+4',
    label: 'Activity'
  }
};

interface PanelHeaderProps {
  title: string;
  onClose?: () => void;
  onSplit?: () => void;
  className?: string;
}

export const PanelHeader: React.FC<PanelHeaderProps> = ({
  title,
  onClose,
  onSplit,
  className
}) => {
  // Map title to PanelId
  const panelId = title as PanelId;
  const config = PANEL_CONFIG[panelId] || PANEL_CONFIG.SETTINGS;

  return (
    <div
      className={className}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        width: '100%',
        height: '100%',
        padding: '0 8px 0 12px',
        backgroundColor: 'var(--mosaic-title-bg, #000000)',
        borderBottom: '1px solid var(--mosaic-title-border, rgba(255, 255, 255, 0.06))',
        userSelect: 'none',
        position: 'relative'
      }}
    >
      {/* Accent line indicator */}
      <div style={{
        position: 'absolute',
        left: 0,
        top: 0,
        bottom: 0,
        width: '3px',
        background: `linear-gradient(180deg, ${config.accentColor}, ${config.accentColor}60)`,
        borderRadius: '0 2px 2px 0'
      }} />

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingLeft: '6px' }}>
        {/* Panel icon with accent color */}
        <div style={{
          color: config.accentColor,
          display: 'flex',
          alignItems: 'center',
          transition: 'color 0.15s'
        }}>
          {config.icon}
        </div>

        <span
          style={{
            fontFamily: 'Inter, sans-serif',
            fontSize: '10px',
            fontWeight: 700,
            color: 'var(--text-secondary)',
            textTransform: 'uppercase',
            letterSpacing: '0.05em'
          }}
        >
          {config.label}
        </span>

        {/* Keyboard shortcut badge */}
        <span style={{
          fontSize: '8px',
          fontFamily: 'var(--font-mono, monospace)',
          color: 'var(--text-muted)',
          background: 'rgba(255,255,255,0.04)',
          padding: '2px 5px',
          borderRadius: '3px',
          border: '1px solid rgba(255,255,255,0.06)',
          letterSpacing: '0.02em'
        }}>
          {config.shortcut}
        </span>
      </div>

      <div style={{ display: 'flex', gap: '2px', height: '100%', alignItems: 'center' }}>
        {onSplit && (
          <button
            onClick={onSplit}
            title="Split Panel"
            className="titlebar-btn"
            style={{
              width: '28px', height: '24px',
              display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '4px',
              color: 'var(--text-secondary)',
              transition: 'all 0.15s ease'
            }}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 2v12" />
              <rect x="2" y="2" width="12" height="12" rx="1" />
            </svg>
          </button>
        )}

        {onClose && (
          <button
            onClick={onClose}
            title={`Close Panel (${config.shortcut})`}
            className="titlebar-btn"
            style={{
              width: '28px', height: '24px',
              display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '4px',
              color: 'var(--text-secondary)',
              transition: 'all 0.15s ease'
            }}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};

// Export config for use in other components
export { PANEL_CONFIG };
export type { PanelId };
