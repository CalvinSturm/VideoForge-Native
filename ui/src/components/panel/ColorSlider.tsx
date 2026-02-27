// VideoForge ColorSlider — Styled range slider with accent color and value display
// Extracted from InputOutputPanel.tsx
import React, { useState } from "react";

export const ColorSlider = ({ label, value, onChange, min = -1, max = 1, step = 0.01, formatValue, icon, accentColor = 'var(--brand-primary)' }: {
    label: string;
    value: number;
    onChange: (v: number) => void;
    min?: number;
    max?: number;
    step?: number;
    formatValue?: (v: number) => string;
    icon?: React.ReactNode;
    accentColor?: string;
}) => {
    const [isDragging, setIsDragging] = useState(false);

    const defaultFormat = (v: number) => {
        if (min === -1 && max === 1) return `${v >= 0 ? '+' : ''}${Math.round(v * 100)}%`;
        return v.toFixed(2);
    };
    const displayValue = formatValue ? formatValue(value) : defaultFormat(value);
    const isDefault = min === -1 && max === 1 ? Math.abs(value) < 0.01 : Math.abs(value - 1) < 0.01;
    const percentage = ((value - min) / (max - min)) * 100;

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            padding: '10px 12px',
            background: isDragging ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.2)',
            borderRadius: '8px',
            border: isDragging ? `1px solid ${accentColor}40` : '1px solid rgba(255,255,255,0.04)',
            transition: 'all 0.15s ease'
        }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    {icon && <span style={{ color: 'var(--text-muted)', opacity: 0.6 }}>{icon}</span>}
                    <label style={{
                        fontSize: '10px',
                        color: 'var(--text-secondary)',
                        fontWeight: 600,
                        letterSpacing: '0.03em'
                    }}>
                        {label}
                    </label>
                </div>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                }}>
                    <span style={{
                        fontSize: '11px',
                        fontFamily: 'var(--font-mono)',
                        color: isDefault ? 'var(--text-muted)' : accentColor,
                        fontWeight: 600,
                        minWidth: '48px',
                        textAlign: 'right'
                    }}>
                        {displayValue}
                    </span>
                    {!isDefault && (
                        <button
                            onClick={() => onChange(min === -1 && max === 1 ? 0 : 1)}
                            style={{
                                background: 'none',
                                border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: '4px',
                                color: 'var(--text-muted)',
                                fontSize: '8px',
                                padding: '2px 5px',
                                cursor: 'pointer',
                                fontFamily: 'var(--font-mono)',
                                opacity: 0.6,
                                transition: 'opacity 0.15s'
                            }}
                            title="Reset to default"
                        >
                            RST
                        </button>
                    )}
                </div>
            </div>
            <div style={{ position: 'relative', height: '20px', display: 'flex', alignItems: 'center' }}>
                {/* Track background */}
                <div style={{
                    position: 'absolute',
                    left: 0,
                    right: 0,
                    height: '4px',
                    background: 'rgba(255,255,255,0.06)',
                    borderRadius: '2px',
                    overflow: 'hidden'
                }}>
                    {/* Filled portion */}
                    <div style={{
                        position: 'absolute',
                        left: min < 0 ? '50%' : 0,
                        width: min < 0
                            ? `${Math.abs(percentage - 50)}%`
                            : `${percentage}%`,
                        transform: min < 0 && value < 0 ? 'translateX(-100%)' : 'none',
                        height: '100%',
                        background: `linear-gradient(90deg, ${accentColor}80, ${accentColor})`,
                        borderRadius: '2px',
                        transition: isDragging ? 'none' : 'all 0.1s ease'
                    }} />
                </div>
                {/* Center mark for bipolar sliders */}
                {min < 0 && (
                    <div style={{
                        position: 'absolute',
                        left: '50%',
                        top: '50%',
                        transform: 'translate(-50%, -50%)',
                        width: '2px',
                        height: '10px',
                        background: 'rgba(255,255,255,0.15)',
                        borderRadius: '1px'
                    }} />
                )}
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value))}
                    onMouseDown={() => setIsDragging(true)}
                    onMouseUp={() => setIsDragging(false)}
                    onTouchStart={() => setIsDragging(true)}
                    onTouchEnd={() => setIsDragging(false)}
                    style={{
                        width: '100%',
                        height: '20px',
                        appearance: 'none',
                        WebkitAppearance: 'none',
                        background: 'transparent',
                        cursor: 'pointer',
                        position: 'relative',
                        zIndex: 1
                    }}
                />
            </div>
        </div>
    );
};
