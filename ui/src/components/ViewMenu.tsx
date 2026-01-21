import React, { useState } from 'react';
import { useViewLayoutStore } from '../Store/viewLayoutStore';
import type { PanelId } from '../Store/viewLayoutStore';

const PANEL_LABELS: Record<PanelId, string> = {
  SETTINGS: 'Settings & Inputs',
  PREVIEW: 'Viewport',
  QUEUE: 'Job Queue',
  ACTIVITY: 'Activity Log'
};

export const ViewMenu: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { panels, togglePanel, resetLayout } = useViewLayoutStore();

  const handleToggle = (id: PanelId) => {
    togglePanel(id);
  };

  return (
    <div style={{ position: 'relative', display: 'inline-block', height: '100%' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          height: '100%',
          background: isOpen ? 'rgba(255,255,255,0.04)' : 'transparent',
          border: 'none',
          // Muted when idle, white when open
          color: isOpen ? '#ededed' : '#71717a',
          fontSize: '11px',
          fontWeight: 500,
          cursor: 'pointer',
          padding: '0 12px',
          fontFamily: 'Inter, sans-serif',
          transition: 'color 0.1s ease',
          outline: 'none'
        }}
        onMouseEnter={(e) => !isOpen && (e.currentTarget.style.color = '#ededed')}
        onMouseLeave={(e) => !isOpen && (e.currentTarget.style.color = '#71717a')}
      >
        View
      </button>

      {isOpen && (
        <>
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              backgroundColor: '#111113',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: '0 0 4px 4px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
              minWidth: '180px',
              zIndex: 10001,
              padding: '6px 0',
              display: 'flex',
              flexDirection: 'column'
            }}
          >
            <div style={{ padding: '6px 12px', fontSize: '9px', color: '#52525b', fontWeight: 700, letterSpacing: '0.05em' }}>
              PANELS
            </div>

            {(Object.keys(PANEL_LABELS) as PanelId[]).map((id) => (
              <button
                key={id}
                onClick={() => handleToggle(id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  width: '100%',
                  background: 'transparent',
                  border: 'none',
                  color: '#ededed',
                  fontSize: '11px',
                  padding: '6px 12px',
                  cursor: 'pointer',
                  textAlign: 'left',
                  fontFamily: 'Inter, sans-serif'
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
              >
                <div style={{ width: '16px', marginRight: '8px', display: 'flex', justifyContent: 'center' }}>
                  {panels[id] && <span style={{ color: '#00ff88', fontSize: '10px' }}>●</span>}
                </div>
                {PANEL_LABELS[id]}
              </button>
            ))}

            <div style={{ height: '1px', background: 'rgba(255,255,255,0.08)', margin: '6px 0' }} />

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
                padding: '6px 12px',
                cursor: 'pointer',
                textAlign: 'left',
                fontFamily: 'Inter, sans-serif'
              }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)'; e.currentTarget.style.color = '#ededed'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; e.currentTarget.style.color = '#a1a1aa'; }}
            >
              <div style={{ width: '16px', marginRight: '8px' }}></div>
              Reset Layout
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
