// VideoForge Tooltip — Portal-based tooltip with viewport boundary detection
// Extracted from InputOutputPanel.tsx
import React, { useState, useRef, useCallback } from "react";
import { createPortal } from "react-dom";

const TOOLTIP_WIDTH = 200;
const TOOLTIP_PADDING = 8;

export const Tooltip: React.FC<{ text: string; children: React.ReactNode; position?: 'top' | 'bottom' }> = ({
    text,
    children,
    position = 'top'
}) => {
    const [isVisible, setIsVisible] = useState(false);
    const [isMounted, setIsMounted] = useState(false);
    const [style, setStyle] = useState<React.CSSProperties>({});
    const [arrowStyle, setArrowStyle] = useState<React.CSSProperties>({});
    const triggerRef = useRef<HTMLDivElement>(null);
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const updatePosition = useCallback(() => {
        if (!triggerRef.current) return;

        const rect = triggerRef.current.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let x = rect.left + rect.width / 2;
        let y = position === 'top' ? rect.top - 8 : rect.bottom + 8;

        const tooltipLeft = x - TOOLTIP_WIDTH / 2;
        const tooltipRight = x + TOOLTIP_WIDTH / 2;

        let offsetX = 0;
        if (tooltipLeft < TOOLTIP_PADDING) {
            offsetX = TOOLTIP_PADDING - tooltipLeft;
        } else if (tooltipRight > viewportWidth - TOOLTIP_PADDING) {
            offsetX = (viewportWidth - TOOLTIP_PADDING) - tooltipRight;
        }

        let actualPosition = position;
        if (position === 'top' && y < 60) {
            actualPosition = 'bottom';
            y = rect.bottom + 8;
        } else if (position === 'bottom' && y + 60 > viewportHeight) {
            actualPosition = 'top';
            y = rect.top - 8;
        }

        setStyle({
            position: 'fixed',
            left: x + offsetX,
            top: y,
            transform: `translateX(-50%) translateY(${actualPosition === 'top' ? '-100%' : '0'})`,
            width: TOOLTIP_WIDTH,
            padding: '8px 12px',
            background: 'var(--panel-bg)',
            border: '1px solid var(--panel-border)',
            borderRadius: '6px',
            fontSize: '10px',
            color: 'var(--text-primary)',
            whiteSpace: 'normal' as const,
            zIndex: 99999,
            boxShadow: 'var(--shadow-md)',
            pointerEvents: 'none' as const,
            lineHeight: 1.4
        });

        setArrowStyle({
            position: 'absolute',
            [actualPosition === 'top' ? 'bottom' : 'top']: '-5px',
            left: `calc(50% - ${offsetX}px)`,
            transform: 'translateX(-50%) rotate(45deg)',
            width: '8px',
            height: '8px',
            background: 'var(--panel-bg)',
            border: '1px solid var(--panel-border)',
            borderTop: actualPosition === 'top' ? 'none' : '1px solid var(--panel-border)',
            borderLeft: actualPosition === 'top' ? 'none' : '1px solid var(--panel-border)',
            borderBottom: actualPosition === 'top' ? '1px solid var(--panel-border)' : 'none',
            borderRight: actualPosition === 'top' ? '1px solid var(--panel-border)' : 'none',
        });
    }, [position]);

    const showTooltip = () => {
        timeoutRef.current = setTimeout(() => {
            updatePosition();
            setIsMounted(true);
            requestAnimationFrame(() => setIsVisible(true));
        }, 400);
    };

    const hideTooltip = () => {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        setIsVisible(false);
        setTimeout(() => setIsMounted(false), 100);
    };

    const tooltipContent = isMounted && createPortal(
        <div style={{ ...style, opacity: isVisible ? 1 : 0, transition: 'opacity 100ms ease' }}>
            {text}
            <div style={arrowStyle} />
        </div>,
        document.body
    );

    return (
        <div
            ref={triggerRef}
            style={{ display: 'inline-flex' }}
            onMouseEnter={showTooltip}
            onMouseLeave={hideTooltip}
            onFocus={showTooltip}
            onBlur={hideTooltip}
        >
            {children}
            {tooltipContent}
        </div>
    );
};
