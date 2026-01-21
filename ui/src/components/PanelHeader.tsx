import React from 'react';

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
        userSelect: 'none'
      }}
    >
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
        {title}
      </span>

      <div style={{ display: 'flex', gap: '2px', height: '100%', alignItems: 'center' }}>
        {onSplit && (
          <button
            onClick={onSplit}
            title="Split Panel"
            className="titlebar-btn"
            style={{
              width: '28px', height: '24px',
              display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '4px',
              color: 'var(--text-secondary)'
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
            title="Collapse Panel"
            className="titlebar-btn"
            style={{
              width: '28px', height: '24px',
              display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '4px',
              color: 'var(--text-secondary)'
            }}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 6l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};
