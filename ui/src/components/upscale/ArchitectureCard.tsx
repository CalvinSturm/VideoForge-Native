/**
 * Architecture selection card.
 * Extracted from AIUpscaleNode.tsx (Task #8).
 */

import React from 'react';
import { type ArchitectureClass } from '../../Store/useJobStore';
import { getArchitectureInfo } from '../../utils/modelClassification';

export interface ArchitectureCardProps {
    archClass: ArchitectureClass;
    selected: boolean;
    onClick: () => void;
    modelCount: number;
    disabled?: boolean;
}

export const ArchitectureCard: React.FC<ArchitectureCardProps> = ({
    archClass, selected, onClick, modelCount, disabled,
}) => {
    const info = getArchitectureInfo(archClass);

    return (
        <button
            onClick={onClick}
            disabled={disabled || modelCount === 0}
            title={info.description}
            style={{
                flex: '1 1 0',
                minWidth: '60px',
                height: '52px',
                padding: '6px 10px',
                borderRadius: '6px',
                border: selected
                    ? '1px solid var(--brand-primary)'
                    : '1px solid rgba(255,255,255,0.08)',
                background: selected
                    ? 'linear-gradient(135deg, var(--brand-dim), rgba(0,255,136,0.05))'
                    : 'rgba(255,255,255,0.03)',
                cursor: disabled || modelCount === 0 ? 'not-allowed' : 'pointer',
                opacity: disabled || modelCount === 0 ? 0.4 : 1,
                transition: 'all 0.15s ease',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '2px',
            }}
        >
            <span style={{ fontSize: '16px' }}>{info.icon}</span>
            <span style={{
                fontSize: '9px',
                fontWeight: 700,
                color: selected ? 'var(--brand-primary)' : 'var(--text-secondary)',
                letterSpacing: '0.03em',
            }}>
                {info.label}
            </span>
            <span style={{
                fontSize: '7px',
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
            }}>
                {modelCount} {modelCount === 1 ? 'model' : 'models'}
            </span>
        </button>
    );
};
