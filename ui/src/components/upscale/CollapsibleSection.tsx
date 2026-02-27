/**
 * Collapsible section with header and animated content.
 * Extracted from AIUpscaleNode.tsx (Task #8).
 */

import React, { useState } from 'react';
import { IconChevronDown } from '../panel/Icons';

export interface CollapsibleSectionProps {
    title: string;
    subtitle?: string;
    icon?: React.ReactNode;
    defaultOpen?: boolean;
    badge?: React.ReactNode;
    children: React.ReactNode;
}

export const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
    title, subtitle, icon, defaultOpen = true, badge, children,
}) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
        <div style={{
            background: 'rgba(0,0,0,0.15)',
            borderRadius: '8px',
            border: '1px solid rgba(255,255,255,0.06)',
            overflow: 'hidden',
        }}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 12px',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: 'var(--text-primary)',
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {icon && <span style={{ color: 'var(--text-muted)', opacity: 0.7 }}>{icon}</span>}
                    <span style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.05em' }}>{title}</span>
                    {subtitle && <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>• {subtitle}</span>}
                    {badge}
                </div>
                <div style={{
                    color: 'var(--text-muted)',
                    transform: isOpen ? 'rotate(0deg)' : 'rotate(-90deg)',
                    transition: 'transform 0.2s ease',
                    opacity: 0.5,
                }}>
                    <IconChevronDown />
                </div>
            </button>
            <div style={{
                maxHeight: isOpen ? '1000px' : '0',
                overflow: 'hidden',
                transition: 'max-height 0.3s ease-out',
            }}>
                <div style={{ padding: '0 12px 12px' }}>
                    {children}
                </div>
            </div>
        </div>
    );
};
