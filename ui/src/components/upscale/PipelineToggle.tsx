/**
 * Interactive pipeline feature toggle chip.
 * Extracted from AIUpscaleNode.tsx (Task #8).
 */

import React from 'react';

export interface PipelineToggleProps {
    label: string;
    enabled: boolean;
    onToggle: () => void;
}

export const PipelineToggle: React.FC<PipelineToggleProps> = ({
    label, enabled, onToggle,
}) => (
    <button
        onClick={(e) => { e.stopPropagation(); onToggle(); }}
        style={{
            fontSize: '7px',
            fontWeight: 600,
            padding: '2px 5px',
            borderRadius: '3px',
            background: enabled ? 'rgba(0,255,136,0.1)' : 'rgba(255,255,255,0.05)',
            border: enabled ? '1px solid rgba(0,255,136,0.2)' : '1px solid rgba(255,255,255,0.08)',
            color: enabled ? 'var(--brand-primary)' : 'var(--text-muted)',
            letterSpacing: '0.03em',
            cursor: 'pointer',
            transition: 'all 0.15s ease',
        }}
    >
        {label}
    </button>
);
