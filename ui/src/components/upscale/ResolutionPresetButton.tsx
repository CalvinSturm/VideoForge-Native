/**
 * Resolution preset button.
 * Extracted from AIUpscaleNode.tsx (Task #8).
 */

import React from 'react';
import { RESOLUTION_PRESETS } from '../../utils/modelClassification';

export interface ResolutionPresetButtonProps {
    preset: keyof typeof RESOLUTION_PRESETS;
    selected: boolean;
    onClick: () => void;
}

export const ResolutionPresetButton: React.FC<ResolutionPresetButtonProps> = ({
    preset, selected, onClick,
}) => {
    const info = RESOLUTION_PRESETS[preset];

    return (
        <button
            onClick={onClick}
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
                    : 'var(--text-secondary)',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '10px',
                fontWeight: 700,
                gap: '1px',
            }}
        >
            {info.shortLabel}
            <span style={{ fontSize: '7px', opacity: 0.6, fontFamily: 'var(--font-mono)' }}>
                {info.width}×{info.height}
            </span>
        </button>
    );
};
