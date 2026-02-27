// VideoForge ToggleGroup — Segmented toggle button group
// Extracted from InputOutputPanel.tsx
import React, { useState } from "react";

export const ToggleGroup = ({ options, value, onChange, disabled }: {
    options: { label: string; sub?: string; value: any; disabled?: boolean }[];
    value: any;
    onChange: (v: any) => void;
    disabled?: boolean;
}) => (
    <div style={{
        display: "flex",
        gap: "4px",
        background: "rgba(0,0,0,0.3)",
        padding: "4px",
        borderRadius: "8px",
        border: "1px solid rgba(255,255,255,0.06)",
        boxShadow: "inset 0 2px 4px rgba(0,0,0,0.3)",
        opacity: disabled ? 0.5 : 1,
        pointerEvents: disabled ? 'none' : 'auto'
    }}>
        {options.map((opt) => {
            const isActive = value === opt.value;
            const isOptDisabled = opt.disabled;
            return (
                <button
                    key={opt.label}
                    onClick={() => !isOptDisabled && onChange(opt.value)}
                    disabled={isOptDisabled}
                    style={{
                        flex: 1,
                        height: "36px",
                        border: "none",
                        borderRadius: "6px",
                        minWidth: 0,
                        background: isActive && !isOptDisabled
                            ? "linear-gradient(135deg, var(--brand-primary), rgba(0,255,136,0.8))"
                            : "transparent",
                        color: isActive && !isOptDisabled
                            ? "#000"
                            : isOptDisabled
                                ? "var(--text-muted)"
                                : "var(--text-secondary)",
                        fontSize: "10px",
                        fontWeight: isActive ? 800 : 600,
                        fontFamily: 'var(--font-sans)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        lineHeight: 1.2,
                        boxShadow: isActive && !isOptDisabled
                            ? "0 2px 8px rgba(0,255,136,0.4)"
                            : "none",
                        opacity: isOptDisabled ? 0.3 : 1,
                        cursor: isOptDisabled ? 'not-allowed' : 'pointer',
                        padding: '0 8px',
                        transition: "all 0.15s ease"
                    }}
                >
                    <span>{opt.label}</span>
                    {opt.sub && (
                        <span style={{
                            fontSize: '8px',
                            opacity: isActive ? 0.7 : 0.5,
                            fontFamily: 'var(--font-mono)',
                            marginTop: '1px'
                        }}>
                            {opt.sub}
                        </span>
                    )}
                </button>
            );
        })}
    </div>
);
