// VideoForge PipelineNode — Collapsible pipeline node card with status indicator
// Extracted from InputOutputPanel.tsx
import React, { useState } from "react";
import { IconChevronDown } from "./Icons";

/** Connection line between pipeline nodes */
export const PipelineConnector = ({ isActive = false }: { isActive?: boolean }) => (
    <div style={{
        display: "flex",
        justifyContent: "center",
        padding: "2px 0",
        position: "relative"
    }}>
        <div style={{
            width: "2px",
            height: "12px",
            background: isActive
                ? "linear-gradient(180deg, var(--brand-primary)60, var(--brand-primary)30)"
                : "linear-gradient(180deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05))",
            borderRadius: "1px"
        }} />
        {isActive && (
            <div style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                width: "6px",
                height: "6px",
                borderRadius: "50%",
                background: "var(--brand-primary)",
                boxShadow: "0 0 8px var(--brand-primary)",
                animation: "pulse 2s infinite"
            }} />
        )}
    </div>
);

/** A visual node card for the processing pipeline with status indicator */
export const PipelineNode = ({
    title,
    icon,
    children,
    defaultOpen = true,
    extra,
    badge,
    isActive = false,
    accentColor = 'var(--brand-primary)',
    nodeNumber
}: {
    title: string;
    icon: React.ReactNode;
    children: React.ReactNode;
    defaultOpen?: boolean;
    extra?: React.ReactNode;
    badge?: React.ReactNode;
    isActive?: boolean;
    accentColor?: string;
    nodeNumber?: number;
}) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    const [isHovered, setIsHovered] = useState(false);

    return (
        <div
            style={{
                marginBottom: "8px",
                background: isActive
                    ? `linear-gradient(135deg, rgba(0,255,136,0.03), transparent 60%)`
                    : "var(--node-bg)",
                border: isActive
                    ? `1px solid rgba(0,255,136,0.35)`
                    : "1px solid var(--node-border)",
                borderRadius: "10px",
                overflow: "hidden",
                flexShrink: 0,
                boxShadow: isActive
                    ? `0 4px 20px rgba(0,255,136,0.12), inset 0 1px 0 rgba(255,255,255,0.04)`
                    : "var(--shadow-md)",
                transition: "all 0.2s ease",
                position: "relative"
            }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            {/* Active indicator line */}
            {isActive && (
                <div style={{
                    position: "absolute",
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: "3px",
                    background: `linear-gradient(180deg, ${accentColor}, ${accentColor}60)`,
                    borderRadius: "3px 0 0 3px"
                }} />
            )}

            {/* Header */}
            <div
                onClick={() => setIsOpen(!isOpen)}
                style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    cursor: "pointer",
                    padding: "10px 12px",
                    paddingLeft: isActive ? "15px" : "12px",
                    background: isHovered ? "var(--node-bg-hover)" : "transparent",
                    userSelect: "none",
                    transition: "background 0.15s",
                    borderBottom: isOpen ? "1px solid rgba(255,255,255,0.04)" : "none",
                    minHeight: "40px",
                    gap: "8px"
                }}
            >
                <div style={{ display: "flex", alignItems: "center", gap: "10px", flex: 1, minWidth: 0 }}>
                    {/* Node number indicator */}
                    {nodeNumber !== undefined && (
                        <div style={{
                            width: "20px",
                            height: "20px",
                            borderRadius: "6px",
                            background: isActive ? accentColor : "rgba(255,255,255,0.08)",
                            color: isActive ? "#000" : "var(--text-muted)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: "10px",
                            fontWeight: 700,
                            fontFamily: "var(--font-mono)",
                            flexShrink: 0
                        }}>
                            {nodeNumber}
                        </div>
                    )}

                    {/* Icon */}
                    <div style={{
                        color: isActive ? accentColor : "var(--text-muted)",
                        opacity: isActive ? 1 : 0.7,
                        transition: "all 0.15s"
                    }}>
                        {icon}
                    </div>

                    {/* Title */}
                    <h3 style={{
                        margin: 0,
                        color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
                        fontSize: "11px",
                        fontWeight: 600,
                        letterSpacing: "0.02em",
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis"
                    }}>
                        {title}
                    </h3>

                    {badge}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
                    {extra}
                    <div style={{
                        color: "var(--text-muted)",
                        transform: isOpen ? "rotate(0deg)" : "rotate(-90deg)",
                        transition: 'transform 0.2s ease',
                        opacity: 0.5
                    }}>
                        <IconChevronDown />
                    </div>
                </div>
            </div>

            {/* Content */}
            <div style={{
                maxHeight: isOpen ? '2000px' : '0',
                overflow: 'hidden',
                transition: 'max-height 0.3s ease-out'
            }}>
                <div style={{ padding: "14px", display: "flex", flexDirection: "column", gap: "14px" }}>
                    {children}
                </div>
            </div>
        </div>
    );
};

/** Legacy Section wrapper for backwards compatibility */
export const Section = ({ title, children, defaultOpen = true, extra, badge }: {
    title: string;
    children: React.ReactNode;
    defaultOpen?: boolean;
    extra?: React.ReactNode;
    badge?: React.ReactNode;
}) => (
    <PipelineNode
        title={title}
        icon={null}
        defaultOpen={defaultOpen}
        extra={extra}
        badge={badge}
    >
        {children}
    </PipelineNode>
);
