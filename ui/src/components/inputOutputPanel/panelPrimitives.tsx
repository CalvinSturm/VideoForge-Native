import React, { useCallback, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { IconChevronDown, IconX } from "./panelIcons";

const TOOLTIP_WIDTH = 200;
const TOOLTIP_PADDING = 8;

export const Tooltip: React.FC<{
  text: string;
  children: React.ReactNode;
  position?: "top" | "bottom";
}> = ({ text, children, position = "top" }) => {
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
    let y = position === "top" ? rect.top - 8 : rect.bottom + 8;

    const tooltipLeft = x - TOOLTIP_WIDTH / 2;
    const tooltipRight = x + TOOLTIP_WIDTH / 2;

    let offsetX = 0;
    if (tooltipLeft < TOOLTIP_PADDING) {
      offsetX = TOOLTIP_PADDING - tooltipLeft;
    } else if (tooltipRight > viewportWidth - TOOLTIP_PADDING) {
      offsetX = viewportWidth - TOOLTIP_PADDING - tooltipRight;
    }

    let actualPosition = position;
    if (position === "top" && y < 60) {
      actualPosition = "bottom";
      y = rect.bottom + 8;
    } else if (position === "bottom" && y + 60 > viewportHeight) {
      actualPosition = "top";
      y = rect.top - 8;
    }

    setStyle({
      position: "fixed",
      left: x + offsetX,
      top: y,
      transform: `translateX(-50%) translateY(${actualPosition === "top" ? "-100%" : "0"})`,
      width: TOOLTIP_WIDTH,
      padding: "8px 12px",
      background: "var(--panel-bg)",
      border: "1px solid var(--panel-border)",
      borderRadius: "6px",
      fontSize: "10px",
      color: "var(--text-primary)",
      whiteSpace: "normal",
      zIndex: 99999,
      boxShadow: "var(--shadow-md)",
      pointerEvents: "none",
      lineHeight: 1.4,
    });

    setArrowStyle({
      position: "absolute",
      [actualPosition === "top" ? "bottom" : "top"]: "-5px",
      left: `calc(50% - ${offsetX}px)`,
      transform: "translateX(-50%) rotate(45deg)",
      width: "8px",
      height: "8px",
      background: "var(--panel-bg)",
      border: "1px solid var(--panel-border)",
      borderTop: actualPosition === "top" ? "none" : "1px solid var(--panel-border)",
      borderLeft: actualPosition === "top" ? "none" : "1px solid var(--panel-border)",
      borderBottom: actualPosition === "top" ? "1px solid var(--panel-border)" : "none",
      borderRight: actualPosition === "top" ? "1px solid var(--panel-border)" : "none",
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
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
    setTimeout(() => setIsMounted(false), 100);
  };

  const tooltipContent =
    isMounted &&
    createPortal(
      <div style={{ ...style, opacity: isVisible ? 1 : 0, transition: "opacity 100ms ease" }}>
        {text}
        <div style={arrowStyle} />
      </div>,
      document.body,
    );

  return (
    <div
      ref={triggerRef}
      style={{ display: "inline-flex" }}
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

export const SmartPath: React.FC<{ path: string; placeholder?: string }> = ({ path, placeholder }) => {
  if (!path) {
    return <bdo dir="ltr">{placeholder || ""}</bdo>;
  }

  const formatPath = (value: string) => {
    if (value.length < 45) return value;
    const parts = value.split(/[/\\]/);
    if (parts.length < 3) return value;
    const filename = parts.pop();
    const drive = parts.shift();
    return `${drive}\\...\\${filename}`;
  };

  return (
    <span style={{ fontFamily: "var(--font-mono)", direction: "ltr", whiteSpace: "nowrap" }}>
      {formatPath(path)}
    </span>
  );
};

export const PipelineConnector = ({ isActive = false }: { isActive?: boolean }) => (
  <div
    style={{
      display: "flex",
      justifyContent: "center",
      padding: "2px 0",
      position: "relative",
    }}
  >
    <div
      style={{
        width: "2px",
        height: "12px",
        background: isActive
          ? "linear-gradient(180deg, var(--brand-primary)60, var(--brand-primary)30)"
          : "linear-gradient(180deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05))",
        borderRadius: "1px",
      }}
    />
    {isActive && (
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: "6px",
          height: "6px",
          borderRadius: "50%",
          background: "var(--brand-primary)",
          boxShadow: "0 0 8px var(--brand-primary)",
          animation: "pulse 2s infinite",
        }}
      />
    )}
  </div>
);

export const PipelineNode = ({
  title,
  icon,
  children,
  defaultOpen = true,
  extra,
  badge,
  isActive = false,
  accentColor = "var(--brand-primary)",
  nodeNumber,
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
        background: isActive ? "linear-gradient(135deg, rgba(0,255,136,0.03), transparent 60%)" : "var(--node-bg)",
        border: isActive ? "1px solid rgba(0,255,136,0.35)" : "1px solid var(--node-border)",
        borderRadius: "10px",
        overflow: "hidden",
        flexShrink: 0,
        boxShadow: isActive
          ? "0 4px 20px rgba(0,255,136,0.12), inset 0 1px 0 rgba(255,255,255,0.04)"
          : "var(--shadow-md)",
        transition: "all 0.2s ease",
        position: "relative",
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {isActive && (
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: "3px",
            background: `linear-gradient(180deg, ${accentColor}, ${accentColor}60)`,
            borderRadius: "3px 0 0 3px",
          }}
        />
      )}

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
          gap: "8px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px", flex: 1, minWidth: 0 }}>
          {nodeNumber !== undefined && (
            <div
              style={{
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
                flexShrink: 0,
              }}
            >
              {nodeNumber}
            </div>
          )}

          <div
            style={{
              color: isActive ? accentColor : "var(--text-muted)",
              opacity: isActive ? 1 : 0.7,
              transition: "all 0.15s",
            }}
          >
            {icon}
          </div>

          <h3
            style={{
              margin: 0,
              color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
              fontSize: "11px",
              fontWeight: 600,
              letterSpacing: "0.02em",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {title}
          </h3>

          {badge}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
          {extra}
          <div
            style={{
              color: "var(--text-muted)",
              transform: isOpen ? "rotate(0deg)" : "rotate(-90deg)",
              transition: "transform 0.2s ease",
              opacity: 0.5,
            }}
          >
            <IconChevronDown />
          </div>
        </div>
      </div>

      <div
        style={{
          maxHeight: isOpen ? "2000px" : "0",
          overflow: "hidden",
          transition: "max-height 0.3s ease-out",
        }}
      >
        <div style={{ padding: "14px", display: "flex", flexDirection: "column", gap: "14px" }}>{children}</div>
      </div>
    </div>
  );
};

export const ColorSlider = ({
  label,
  value,
  onChange,
  min = -1,
  max = 1,
  step = 0.01,
  formatValue,
  icon,
  accentColor = "var(--brand-primary)",
}: {
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

  const defaultFormat = (currentValue: number) => {
    if (min === -1 && max === 1) return `${currentValue >= 0 ? "+" : ""}${Math.round(currentValue * 100)}%`;
    return currentValue.toFixed(2);
  };

  const displayValue = formatValue ? formatValue(value) : defaultFormat(value);
  const isDefault = min === -1 && max === 1 ? Math.abs(value) < 0.01 : Math.abs(value - 1) < 0.01;
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        padding: "10px 12px",
        background: isDragging ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.2)",
        borderRadius: "8px",
        border: isDragging ? `1px solid ${accentColor}40` : "1px solid rgba(255,255,255,0.04)",
        transition: "all 0.15s ease",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          {icon && <span style={{ color: "var(--text-muted)", opacity: 0.6 }}>{icon}</span>}
          <label
            style={{
              fontSize: "10px",
              color: "var(--text-secondary)",
              fontWeight: 600,
              letterSpacing: "0.03em",
            }}
          >
            {label}
          </label>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span
            style={{
              fontSize: "11px",
              fontFamily: "var(--font-mono)",
              color: isDefault ? "var(--text-muted)" : accentColor,
              fontWeight: 600,
              minWidth: "48px",
              textAlign: "right",
            }}
          >
            {displayValue}
          </span>
          {!isDefault && (
            <button
              onClick={() => onChange(min === -1 ? 0 : 1)}
              style={{
                background: "none",
                border: "none",
                color: "var(--text-muted)",
                cursor: "pointer",
                padding: "2px",
                display: "flex",
                alignItems: "center",
                opacity: 0.6,
                transition: "opacity 0.15s",
              }}
              title="Reset to default"
            >
              <IconX />
            </button>
          )}
        </div>
      </div>

      <div style={{ position: "relative", height: "6px" }}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(255,255,255,0.08)",
            borderRadius: "3px",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              bottom: 0,
              width: `${percentage}%`,
              background: isDefault
                ? "rgba(255,255,255,0.15)"
                : `linear-gradient(90deg, ${accentColor}80, ${accentColor})`,
              borderRadius: "3px",
              transition: isDragging ? "none" : "width 0.1s ease",
            }}
          />
        </div>

        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          onMouseDown={() => setIsDragging(true)}
          onMouseUp={() => setIsDragging(false)}
          onMouseLeave={() => setIsDragging(false)}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            opacity: 0,
            cursor: "pointer",
            margin: 0,
          }}
        />

        <div
          style={{
            position: "absolute",
            left: `${percentage}%`,
            top: "50%",
            transform: "translate(-50%, -50%)",
            width: isDragging ? "14px" : "12px",
            height: isDragging ? "14px" : "12px",
            borderRadius: "50%",
            background: isDefault ? "#666" : accentColor,
            border: "2px solid #fff",
            boxShadow: isDragging
              ? `0 0 12px ${accentColor}80, 0 2px 6px rgba(0,0,0,0.4)`
              : "0 2px 4px rgba(0,0,0,0.3)",
            pointerEvents: "none",
            transition: isDragging ? "none" : "all 0.1s ease",
          }}
        />
      </div>
    </div>
  );
};

export const ToggleGroup = ({
  options,
  value,
  onChange,
  disabled,
}: {
  options: { label: string; sub?: string; value: unknown; disabled?: boolean }[];
  value: unknown;
  onChange: (value: unknown) => void;
  disabled?: boolean;
}) => (
  <div
    style={{
      display: "flex",
      gap: "4px",
      background: "rgba(0,0,0,0.3)",
      padding: "4px",
      borderRadius: "8px",
      border: "1px solid rgba(255,255,255,0.06)",
      boxShadow: "inset 0 2px 4px rgba(0,0,0,0.3)",
      opacity: disabled ? 0.5 : 1,
      pointerEvents: disabled ? "none" : "auto",
    }}
  >
    {options.map((option) => {
      const isActive = value === option.value;
      const isOptionDisabled = option.disabled;
      return (
        <button
          key={option.label}
          onClick={() => !isOptionDisabled && onChange(option.value)}
          disabled={isOptionDisabled}
          style={{
            flex: 1,
            height: "36px",
            border: "none",
            borderRadius: "6px",
            minWidth: 0,
            background: isActive && !isOptionDisabled
              ? "linear-gradient(135deg, var(--brand-primary), rgba(0,255,136,0.8))"
              : "transparent",
            color: isActive && !isOptionDisabled
              ? "#000"
              : isOptionDisabled
                ? "var(--text-muted)"
                : "var(--text-secondary)",
            fontSize: "10px",
            fontWeight: isActive ? 800 : 600,
            fontFamily: "var(--font-sans)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            lineHeight: 1.2,
            boxShadow: isActive && !isOptionDisabled ? "0 2px 8px rgba(0,255,136,0.4)" : "none",
            opacity: isOptionDisabled ? 0.3 : 1,
            cursor: isOptionDisabled ? "not-allowed" : "pointer",
            padding: "0 8px",
            transition: "all 0.15s ease",
          }}
        >
          <span>{option.label}</span>
          {option.sub && (
            <span
              style={{
                fontSize: "8px",
                opacity: isActive ? 0.7 : 0.5,
                fontFamily: "var(--font-mono)",
                marginTop: "1px",
              }}
            >
              {option.sub}
            </span>
          )}
        </button>
      );
    })}
  </div>
);
