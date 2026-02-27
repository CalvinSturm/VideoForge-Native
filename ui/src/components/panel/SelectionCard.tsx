// VideoForge SelectionCard — Card-style selection button with checkmark indicator
// Extracted from InputOutputPanel.tsx
import React, { useState } from "react";
import { IconCheck } from "./Icons";

export const SelectionCard = ({ selected, onClick, title, subtitle, icon, disabled, badge }: {
    selected: boolean;
    onClick: () => void;
    title: string;
    subtitle: string;
    icon: React.ReactNode;
    disabled?: boolean;
    badge?: React.ReactNode;
}) => {
    const [isHovered, setIsHovered] = useState(false);

    return (
        <button
            onClick={onClick}
            disabled={disabled}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                justifyContent: "center",
                height: "60px",
                padding: "10px 12px",
                minWidth: "90px",
                background: selected && !disabled
                    ? "linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,255,136,0.05))"
                    : isHovered && !disabled
                        ? "rgba(255,255,255,0.04)"
                        : "rgba(255,255,255,0.02)",
                border: selected && !disabled
                    ? "1px solid rgba(0,255,136,0.5)"
                    : "1px solid rgba(255,255,255,0.08)",
                borderRadius: "8px",
                cursor: disabled ? "not-allowed" : "pointer",
                transition: "all 0.2s ease",
                position: "relative",
                overflow: "hidden",
                opacity: disabled ? 0.4 : 1,
                boxShadow: selected && !disabled
                    ? "0 4px 16px rgba(0,255,136,0.15), inset 0 1px 0 rgba(255,255,255,0.05)"
                    : "inset 0 1px 0 rgba(255,255,255,0.03)"
            }}
        >
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px", width: '100%' }}>
                <div style={{
                    color: selected && !disabled ? "var(--brand-primary)" : "var(--text-muted)",
                    transition: "color 0.15s"
                }}>
                    {icon}
                </div>
                <span style={{
                    fontWeight: 700,
                    fontSize: "11px",
                    color: selected && !disabled ? "var(--text-primary)" : "var(--text-secondary)",
                    fontFamily: 'var(--font-sans)',
                    letterSpacing: '0.02em',
                    flex: 1
                }}>
                    {title}
                </span>
                {badge}
            </div>
            <span style={{
                fontSize: "9px",
                color: selected && !disabled ? "var(--brand-primary)" : "var(--text-muted)",
                marginLeft: "24px",
                fontFamily: 'var(--font-mono)',
                letterSpacing: "0.05em",
                opacity: 0.8
            }}>
                {subtitle}
            </span>

            {/* Selection indicator */}
            {selected && !disabled && (
                <>
                    <div style={{
                        position: "absolute",
                        top: "6px",
                        right: "6px",
                        width: "16px",
                        height: "16px",
                        borderRadius: "50%",
                        background: "var(--brand-primary)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "#000",
                        boxShadow: "0 2px 6px rgba(0,255,136,0.3)"
                    }}>
                        <IconCheck />
                    </div>
                    {/* Glow effect */}
                    <div style={{
                        position: "absolute",
                        inset: 0,
                        background: "radial-gradient(circle at 50% 100%, rgba(0,255,136,0.1), transparent 60%)",
                        pointerEvents: "none"
                    }} />
                </>
            )}
        </button>
    );
};
