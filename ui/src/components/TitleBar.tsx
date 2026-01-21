import React, { useState, useEffect } from 'react';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { ViewMenu } from './ViewMenu';

export const TitleBar: React.FC = () => {
  const [isMaximized, setIsMaximized] = useState(false);
  const appWindow = getCurrentWindow();

  useEffect(() => {
    appWindow.isMaximized().then(setIsMaximized).catch(console.error);
    const unlisten = appWindow.listen('tauri://resize', async () => {
      setIsMaximized(await appWindow.isMaximized());
    });
    return () => { unlisten.then(f => f()); };
  }, []);

  const handleMin = () => appWindow.minimize();
  const handleMax = async () => {
    await appWindow.toggleMaximize();
    setIsMaximized(await appWindow.isMaximized());
  };
  const handleClose = () => appWindow.close();

  // Larger, clearer Windows 11 style icons
  const IconMinus = () => (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1">
      <path d="M2 8h12" />
    </svg>
  );

  const IconSquare = () => (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1">
      <rect x="2.5" y="2.5" width="11" height="11" />
    </svg>
  );

  const IconRestore = () => (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1">
      <path d="M4.5 4.5v-2h9v9h-2" />
      <rect x="2.5" y="4.5" width="9" height="9" />
    </svg>
  );

  const IconClose = () => (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2">
      <path d="M3 3l10 10M13 3l-10 10" />
    </svg>
  );

  const buttonStyle: React.CSSProperties = {
    height: '100%',
    width: '48px', // Standard windows hit target width
    background: 'transparent',
    border: 'none',
    color: '#a1a1aa',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'default',
    transition: 'all 0.1s ease-in-out',
    outline: 'none'
  };

  return (
    <div style={{
      height: '32px',
      background: 'var(--bg-color)',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      borderBottom: '1px solid var(--panel-border)',
      fontFamily: 'Inter, sans-serif',
      userSelect: 'none',
      position: 'relative',
      zIndex: 10000,
      flexShrink: 0
    }}>

      {/* LEFT: Branding & Menu */}
      <div style={{ display: 'flex', alignItems: 'center', height: '100%', paddingLeft: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginRight: '16px' }}>
          <div style={{ width: '6px', height: '6px', background: 'var(--brand-primary)', borderRadius: '50%', boxShadow: '0 0 6px rgba(0, 255, 136, 0.4)' }} />
          <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.05em', color: 'var(--text-primary)' }}>
            VIDEOFORGE
          </span>
        </div>

        <ViewMenu />
      </div>

      {/* DRAG REGION */}
      <div data-tauri-drag-region style={{ flex: 1, height: '100%' }} />

      {/* RIGHT: Window Controls */}
      <div style={{ display: 'flex', height: '100%' }}>
        <button
          style={buttonStyle}
          onClick={handleMin}
          title="Minimize"
          onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.04)'; e.currentTarget.style.color = 'var(--text-primary)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; e.currentTarget.style.color = '#a1a1aa'; }}
        >
          <IconMinus />
        </button>

        <button
          style={buttonStyle}
          onClick={handleMax}
          title={isMaximized ? "Restore" : "Maximize"}
          onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.04)'; e.currentTarget.style.color = 'var(--text-primary)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; e.currentTarget.style.color = '#a1a1aa'; }}
        >
          {isMaximized ? <IconRestore /> : <IconSquare />}
        </button>

        <button
          style={buttonStyle}
          onClick={handleClose}
          title="Close"
          onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#e81123'; e.currentTarget.style.color = '#fff'; }}
          onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; e.currentTarget.style.color = '#a1a1aa'; }}
        >
          <IconClose />
        </button>
      </div>
    </div>
  );
};
