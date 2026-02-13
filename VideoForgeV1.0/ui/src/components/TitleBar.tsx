import React, { useState, useEffect } from 'react';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { ViewMenu } from './ViewMenu';

/**
 * TitleBar - VideoForge Application Header
 * 
 * UX Rationale:
 * - Height increased to 40px for better touch targets and visual weight
 * - Icons increased to 16x16 (from 14x14) for DPI scaling and legibility
 * - Button width at 46px matches Windows 11 design language
 * - Logo uses brand accent with subtle glow for identity
 * - Professional, cinematic aesthetic with restrained embellishment
 */

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

  // Improved icons - 16x16 with proper stroke weights for clarity
  const IconMinus = () => (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2">
      <path d="M3 8h10" strokeLinecap="round" />
    </svg>
  );

  const IconSquare = () => (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2">
      <rect x="3" y="3" width="10" height="10" rx="0.5" />
    </svg>
  );

  const IconRestore = () => (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2">
      <path d="M5 4.5V3h8v8h-1.5" />
      <rect x="3" y="5" width="8" height="8" rx="0.5" />
    </svg>
  );

  const IconClose = () => (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
    </svg>
  );

  // Logo icon - stylized "V" for VideoForge
  const LogoIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <defs>
        <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="var(--brand-primary)" />
          <stop offset="100%" stopColor="#00cc6a" />
        </linearGradient>
      </defs>
      <path
        d="M4 5l8 14 8-14"
        stroke="url(#logoGrad)"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <path
        d="M8 5l4 7 4-7"
        stroke="url(#logoGrad)"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
        opacity="0.5"
      />
    </svg>
  );

  const buttonStyle: React.CSSProperties = {
    height: '100%',
    width: '46px',
    background: 'transparent',
    border: 'none',
    color: 'var(--text-secondary)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'default',
    transition: 'all 0.12s ease',
    outline: 'none'
  };

  return (
    <header style={{
      height: '40px',
      background: 'linear-gradient(180deg, var(--bg-color), rgba(0,0,0,0.1))',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      borderBottom: '1px solid var(--panel-border)',
      fontFamily: 'var(--font-sans)',
      userSelect: 'none',
      position: 'relative',
      zIndex: 10000,
      flexShrink: 0
    }}>

      {/* LEFT: Branding & Menu */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        height: '100%',
        paddingLeft: '14px',
        gap: '16px'
      }}>
        {/* Logo */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            filter: 'drop-shadow(0 0 6px rgba(0, 255, 136, 0.3))'
          }}>
            <LogoIcon />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0px' }}>
            <span style={{
              fontSize: '12px',
              fontWeight: 800,
              letterSpacing: '0.08em',
              color: 'var(--text-primary)',
              lineHeight: 1
            }}>
              VIDEOFORGE
            </span>
            <span style={{
              fontSize: '8px',
              fontWeight: 500,
              letterSpacing: '0.15em',
              color: 'var(--text-muted)',
              lineHeight: 1,
              marginTop: '2px'
            }}>
              AI UPSCALER
            </span>
          </div>
        </div>

        {/* Separator */}
        <div style={{
          width: '1px',
          height: '20px',
          background: 'var(--panel-border)'
        }} />

        {/* View Menu */}
        <ViewMenu />
      </div>

      {/* CENTER: Drag Region */}
      <div data-tauri-drag-region style={{ flex: 1, height: '100%' }} />

      {/* RIGHT: Window Controls */}
      <div style={{ display: 'flex', height: '100%' }}>
        <button
          style={buttonStyle}
          onClick={handleMin}
          title="Minimize"
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)';
            e.currentTarget.style.color = 'var(--text-primary)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = 'var(--text-secondary)';
          }}
        >
          <IconMinus />
        </button>

        <button
          style={buttonStyle}
          onClick={handleMax}
          title={isMaximized ? "Restore Down" : "Maximize"}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)';
            e.currentTarget.style.color = 'var(--text-primary)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = 'var(--text-secondary)';
          }}
        >
          {isMaximized ? <IconRestore /> : <IconSquare />}
        </button>

        <button
          style={{
            ...buttonStyle,
            width: '48px' // Slightly wider for close button (Windows convention)
          }}
          onClick={handleClose}
          title="Close"
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#e81123';
            e.currentTarget.style.color = '#ffffff';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = 'var(--text-secondary)';
          }}
        >
          <IconClose />
        </button>
      </div>
    </header>
  );
};
