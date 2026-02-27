/**
 * Scale factor toggle button.
 * Extracted from AIUpscaleNode.tsx (Task #8).
 */

import React from 'react';
import { type UpscaleScale } from '../../Store/useJobStore';
import { IconLock } from '../panel/Icons';

export interface ScaleToggleProps {
    scale: UpscaleScale;
    selected: boolean;
    available: boolean;
    onClick: () => void;
}

export const ScaleToggle: React.FC<ScaleToggleProps> = ({
    scale, selected, available, onClick,
}) => (
    <button
        onClick={onClick}
        disabled={!available}
        style={{
            flex: 1,
            height: '36px',
            borderRadius: '5px',
            border: selected
                ? '1px solid var(--brand-primary)'
                : '1px solid rgba(255,255,255,0.08)',
            background: selected
                ? 'var(--brand-dim)'
                : 'rgba(255,255,255,0.03)',
            color: selected
                ? 'var(--brand-primary)'
                : available
                    ? 'var(--text-secondary)'
                    : 'var(--text-muted)',
            cursor: available ? 'pointer' : 'not-allowed',
            opacity: available ? 1 : 0.4,
            transition: 'all 0.15s ease',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '4px',
            fontSize: '12px',
            fontWeight: 700,
        }}
    >
        {scale}×
        {!available && <IconLock />}
    </button>
);
