// VideoForge SmartPath — Intelligent path truncation component
// Extracted from InputOutputPanel.tsx
import React from "react";

export const SmartPath: React.FC<{ path: string; placeholder?: string }> = ({ path, placeholder }) => {
    if (!path) return <bdo dir="ltr">{placeholder || ""}</bdo>;

    const formatPath = (p: string) => {
        if (p.length < 45) return p;
        const parts = p.split(/[/\\]/);
        if (parts.length < 3) return p;
        const filename = parts.pop();
        const drive = parts.shift();
        return `${drive}\\...\\${filename}`;
    };

    return (
        <span style={{ fontFamily: 'var(--font-mono)', direction: 'ltr', whiteSpace: 'nowrap' }}>
            {formatPath(path)}
        </span>
    );
};
