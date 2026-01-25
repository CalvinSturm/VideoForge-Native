import React from 'react';
import { useViewLayoutStore } from '../Store/viewLayoutStore';

/**
 * EmptyState - Displayed when all panels are closed
 * 
 * UX Rationale:
 * - Intentional, non-punitive design that feels like a clean slate
 * - Clear visual hierarchy: Icon → Title → Description → Action
 * - Quick keyboard shortcuts displayed for power users
 * - One-click "Restore Workspace" to immediately recover
 * - Subtle animations make the state feel alive, not dead
 */

// Icons
const IconLayout = () => (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <line x1="9" y1="3" x2="9" y2="21" />
        <line x1="9" y1="14" x2="21" y2="14" />
    </svg>
);

const IconKeyboard = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="2" y="6" width="20" height="12" rx="2" />
        <line x1="6" y1="10" x2="6" y2="10.01" />
        <line x1="10" y1="10" x2="10" y2="10.01" />
        <line x1="14" y1="10" x2="14" y2="10.01" />
        <line x1="18" y1="10" x2="18" y2="10.01" />
        <line x1="8" y1="14" x2="16" y2="14" />
    </svg>
);

const IconGrid = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
    </svg>
);

export const EmptyState: React.FC = () => {
    const { showAllPanels } = useViewLayoutStore();

    const shortcuts = [
        { key: 'Ctrl+1', label: 'Settings' },
        { key: 'Ctrl+2', label: 'Preview' },
        { key: 'Ctrl+3', label: 'Queue' },
        { key: 'Ctrl+4', label: 'Activity' },
    ];

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            padding: '48px',
            background: 'linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.1) 100%)',
            animation: 'fadeIn 0.3s ease'
        }}>
            {/* Icon with subtle animation */}
            <div style={{
                width: '80px',
                height: '80px',
                borderRadius: '20px',
                background: 'rgba(255,255,255,0.02)',
                border: '1px dashed rgba(255,255,255,0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-muted)',
                marginBottom: '24px',
                animation: 'pulse 3s infinite'
            }}>
                <IconLayout />
            </div>

            {/* Title */}
            <h2 style={{
                margin: '0 0 8px 0',
                fontSize: '16px',
                fontWeight: 600,
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-sans)',
                letterSpacing: '-0.01em'
            }}>
                Workspace Cleared
            </h2>

            {/* Description */}
            <p style={{
                margin: '0 0 24px 0',
                fontSize: '12px',
                color: 'var(--text-secondary)',
                fontFamily: 'var(--font-sans)',
                textAlign: 'center',
                maxWidth: '280px',
                lineHeight: 1.5
            }}>
                All panels are hidden. Restore your workspace to continue editing.
            </p>

            {/* Primary Action */}
            <button
                onClick={() => showAllPanels?.()}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 20px',
                    background: 'linear-gradient(135deg, var(--brand-primary), #00cc6a)',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#000',
                    fontSize: '12px',
                    fontWeight: 700,
                    fontFamily: 'var(--font-sans)',
                    cursor: 'pointer',
                    boxShadow: '0 4px 12px rgba(0,255,136,0.3), inset 0 1px 0 rgba(255,255,255,0.2)',
                    transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 6px 20px rgba(0,255,136,0.4), inset 0 1px 0 rgba(255,255,255,0.2)';
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,255,136,0.3), inset 0 1px 0 rgba(255,255,255,0.2)';
                }}
            >
                <IconGrid />
                Restore Workspace
            </button>

            {/* Keyboard Shortcuts */}
            <div style={{
                marginTop: '32px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '12px'
            }}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontSize: '10px',
                    color: 'var(--text-muted)',
                    fontFamily: 'var(--font-sans)'
                }}>
                    <IconKeyboard />
                    <span>Quick Access</span>
                </div>

                <div style={{ display: 'flex', gap: '8px' }}>
                    {shortcuts.map((s) => (
                        <div
                            key={s.key}
                            title={`Open ${s.label}`}
                            style={{
                                padding: '6px 10px',
                                background: 'rgba(255,255,255,0.03)',
                                border: '1px solid rgba(255,255,255,0.08)',
                                borderRadius: '6px',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                gap: '4px',
                                cursor: 'default'
                            }}
                        >
                            <span style={{
                                fontSize: '10px',
                                fontFamily: 'var(--font-mono)',
                                color: 'var(--text-secondary)',
                                fontWeight: 600
                            }}>
                                {s.key}
                            </span>
                            <span style={{
                                fontSize: '8px',
                                color: 'var(--text-muted)',
                                letterSpacing: '0.03em'
                            }}>
                                {s.label}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Hint */}
            <p style={{
                marginTop: '32px',
                fontSize: '10px',
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-sans)',
                opacity: 0.7
            }}>
                Or use <strong style={{ color: 'var(--text-secondary)' }}>View</strong> menu in the header
            </p>
        </div>
    );
};
