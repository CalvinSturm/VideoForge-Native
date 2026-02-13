import React, { useState } from 'react';
import { useViewLayoutStore } from '../Store/viewLayoutStore';
import type { PanelId } from '../Store/viewLayoutStore';

// Panel icons
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

const IconGrid = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="7" height="7" />
    <rect x="14" y="3" width="7" height="7" />
    <rect x="14" y="14" width="7" height="7" />
    <rect x="3" y="14" width="7" height="7" />
  </svg>
);

const IconReset = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="1 4 1 10 7 10" />
    <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
  </svg>
);

// Panel configuration
const PANEL_CONFIG: Record<PanelId, {
  icon: React.ReactNode;
  accentColor: string;
  shortcut: string;
  label: string;
}> = {
  SETTINGS: {
    icon: <IconSettings />,
    accentColor: '#3b82f6',
    shortcut: '⌘1',
    label: 'Settings & Inputs'
  },
  PREVIEW: {
    icon: <IconPreview />,
    accentColor: '#00ff88',
    shortcut: '⌘2',
    label: 'Viewport'
  },
  QUEUE: {
    icon: <IconQueue />,
    accentColor: '#f59e0b',
    shortcut: '⌘3',
    label: 'Job Queue'
  },
  ACTIVITY: {
    icon: <IconActivity />,
    accentColor: '#a855f7',
    shortcut: '⌘4',
    label: 'Activity Log'
  }
};

// Toggle Switch Component
const ToggleSwitch: React.FC<{ checked: boolean; color: string }> = ({ checked, color }) => (
  <div style={{
    width: '28px',
    height: '16px',
    borderRadius: '8px',
    background: checked
      ? `linear-gradient(135deg, ${color}, ${color}99)`
      : 'rgba(255,255,255,0.08)',
    border: checked
      ? `1px solid ${color}60`
      : '1px solid rgba(255,255,255,0.1)',
    position: 'relative',
    transition: 'all 0.2s ease',
    boxShadow: checked ? `0 0 8px ${color}40` : 'inset 0 1px 2px rgba(0,0,0,0.3)'
  }}>
    <div style={{
      width: '12px',
      height: '12px',
      borderRadius: '50%',
      background: checked ? '#000' : '#555',
      position: 'absolute',
      top: '1px',
      left: checked ? '13px' : '1px',
      transition: 'left 0.2s ease',
      boxShadow: '0 1px 2px rgba(0,0,0,0.3)'
    }} />
  </div>
);

export const ViewMenu: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { panels, togglePanel, resetLayout, showAllPanels } = useViewLayoutStore();

  const handleToggle = (id: PanelId) => {
    togglePanel(id);
  };

  const allVisible = Object.values(panels).every(v => v);

  return (
    <div style={{ position: 'relative', display: 'inline-block', height: '100%' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          height: '100%',
          background: isOpen ? 'rgba(255,255,255,0.04)' : 'transparent',
          border: 'none',
          color: isOpen ? '#ededed' : '#71717a',
          fontSize: '11px',
          fontWeight: 500,
          cursor: 'pointer',
          padding: '0 12px',
          fontFamily: 'Inter, sans-serif',
          transition: 'color 0.1s ease',
          outline: 'none',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}
        onMouseEnter={(e) => !isOpen && (e.currentTarget.style.color = '#ededed')}
        onMouseLeave={(e) => !isOpen && (e.currentTarget.style.color = '#71717a')}
      >
        View
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{
          transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
          transition: 'transform 0.2s ease'
        }}>
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {isOpen && (
        <>
          <div
            style={{
              position: 'absolute',
              top: 'calc(100% + 4px)',
              left: 0,
              backgroundColor: 'var(--panel-bg)',
              border: '1px solid var(--panel-border)',
              borderRadius: '8px',
              boxShadow: 'var(--shadow-md)',
              minWidth: '220px',
              zIndex: 10001,
              padding: '8px 0',
              display: 'flex',
              flexDirection: 'column',
              backdropFilter: 'blur(10px)',
              animation: 'fadeIn 0.15s ease'
            }}
          >
            {/* Header */}
            <div style={{
              padding: '6px 14px 10px',
              fontSize: '9px',
              color: 'var(--text-muted)',
              fontWeight: 700,
              letterSpacing: '0.08em',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <IconGrid />
              PANELS
            </div>

            {/* Panel toggles */}
            {(Object.keys(PANEL_CONFIG) as PanelId[]).map((id) => {
              const config = PANEL_CONFIG[id];
              const isActive = panels[id];

              return (
                <button
                  key={id}
                  onClick={() => handleToggle(id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    width: '100%',
                    background: 'transparent',
                    border: 'none',
                    color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                    fontSize: '11px',
                    padding: '8px 14px',
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontFamily: 'Inter, sans-serif',
                    transition: 'all 0.12s ease',
                    gap: '10px'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'var(--button-hover-bg)';
                    e.currentTarget.style.color = 'var(--text-primary)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.color = isActive ? 'var(--text-primary)' : 'var(--text-secondary)';
                  }}
                >
                  {/* Icon */}
                  <div style={{
                    color: isActive ? config.accentColor : 'inherit',
                    display: 'flex',
                    alignItems: 'center',
                    transition: 'color 0.15s'
                  }}>
                    {config.icon}
                  </div>

                  {/* Label */}
                  <span style={{ flex: 1, fontWeight: isActive ? 600 : 400 }}>
                    {config.label}
                  </span>

                  {/* Shortcut */}
                  <span style={{
                    fontSize: '9px',
                    color: 'var(--text-muted)',
                    fontFamily: 'var(--font-mono, monospace)',
                    marginRight: '8px'
                  }}>
                    {config.shortcut}
                  </span>

                  {/* Toggle */}
                  <ToggleSwitch checked={isActive} color={config.accentColor} />
                </button>
              );
            })}

            <div style={{ height: '1px', background: 'var(--panel-border)', margin: '8px 0' }} />

            {/* Show All / Reset Actions */}
            <button
              onClick={() => { showAllPanels?.(); setIsOpen(false); }}
              disabled={allVisible}
              style={{
                display: 'flex',
                alignItems: 'center',
                width: '100%',
                background: 'transparent',
                border: 'none',
                color: allVisible ? '#3f3f46' : '#a1a1aa',
                fontSize: '11px',
                padding: '8px 14px',
                cursor: allVisible ? 'not-allowed' : 'pointer',
                textAlign: 'left',
                fontFamily: 'Inter, sans-serif',
                gap: '10px',
                transition: 'all 0.12s ease',
                opacity: allVisible ? 0.5 : 1
              }}
              onMouseEnter={(e) => !allVisible && (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.04)', e.currentTarget.style.color = '#ededed')}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent', e.currentTarget.style.color = allVisible ? '#3f3f46' : '#a1a1aa')}
            >
              <IconGrid />
              <span style={{ flex: 1 }}>Show All Panels</span>
            </button>

            <button
              onClick={() => { resetLayout(); setIsOpen(false); }}
              style={{
                display: 'flex',
                alignItems: 'center',
                width: '100%',
                background: 'transparent',
                border: 'none',
                color: '#a1a1aa',
                fontSize: '11px',
                padding: '8px 14px',
                cursor: 'pointer',
                textAlign: 'left',
                fontFamily: 'Inter, sans-serif',
                gap: '10px',
                transition: 'all 0.12s ease'
              }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.04)'; e.currentTarget.style.color = '#ededed'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; e.currentTarget.style.color = '#a1a1aa'; }}
            >
              <IconReset />
              <span style={{ flex: 1 }}>Reset Layout</span>
            </button>
          </div>

          <div
            style={{ position: 'fixed', inset: 0, zIndex: 10000 }}
            onClick={() => setIsOpen(false)}
          />
        </>
      )}
    </div>
  );
};
