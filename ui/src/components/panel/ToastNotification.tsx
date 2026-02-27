// VideoForge Toast Notification — Auto-dismissing toast with slide animation
// Extracted from InputOutputPanel.tsx
import React, { useState, useEffect } from "react";

export const ToastNotification: React.FC<{
    message: string;
    visible: boolean;
    onDismiss: () => void;
    duration?: number;
}> = ({ message, visible, onDismiss, duration = 3000 }) => {
    const [render, setRender] = useState(visible);

    useEffect(() => {
        if (visible) {
            setRender(true);
            const timer = setTimeout(() => onDismiss(), duration);
            return () => clearTimeout(timer);
        } else {
            const hideTimer = setTimeout(() => setRender(false), 300);
            return () => clearTimeout(hideTimer);
        }
    }, [visible, duration, onDismiss]);

    if (!render) return null;

    return (
        <div style={{
            position: 'absolute', top: '16px', right: '16px', zIndex: 2000,
            background: 'var(--toast-bg)', border: '1px solid var(--toast-border)',
            borderLeft: '3px solid var(--brand-primary)',
            borderRadius: '4px', padding: '10px 12px',
            boxShadow: 'var(--shadow-md)',
            display: 'flex', alignItems: 'center', gap: '10px',
            transform: visible ? 'translateX(0)' : 'translateX(100%)',
            opacity: visible ? 1 : 0,
            transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
            maxWidth: '260px'
        }}>
            <span style={{ fontSize: '11px', color: 'var(--toast-text)', fontFamily: 'var(--font-sans)', fontWeight: 500 }}>{message}</span>
            <button onClick={onDismiss} style={{
                background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '0 4px', fontSize: '14px'
            }}>×</button>
        </div>
    );
};
