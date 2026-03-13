import React from "react";

export interface ToggleOption {
  label: string;
  value: string | number;
  sub?: string;
  subLabel?: string;
}

interface ToggleGroupProps {
  options: ToggleOption[];
  value: string | number;
  onChange: (value: string | number) => void;
  label?: string;
}

export const ToggleGroup: React.FC<ToggleGroupProps> = ({
  options,
  value,
  onChange,
  label,
}) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px", width: "100%" }}>
      {label ? (
        <span
          style={{
            fontSize: "10px",
            color: "#71717a",
            fontWeight: 600,
            textTransform: "uppercase",
          }}
        >
          {label}
        </span>
      ) : null}

      <div
        role="radiogroup"
        style={{
          display: "flex",
          width: "100%",
          backgroundColor: "#000000",
          padding: "2px",
          borderRadius: "4px",
          border: "1px solid #27272a",
        }}
      >
        {options.map((option) => {
          const isActive = option.value === value;
          const secondary = option.subLabel ?? option.sub;

          return (
            <button
              key={String(option.value)}
              role="radio"
              aria-checked={isActive}
              onClick={() => onChange(option.value)}
              style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                background: isActive ? "#00ff88" : "transparent",
                border: "none",
                borderRadius: "2px",
                padding: "6px 0",
                cursor: "pointer",
                transition: "background-color 0.1s",
                color: isActive ? "#000000" : "#a1a1aa",
              }}
            >
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: 700,
                  fontFamily: "Inter, sans-serif",
                }}
              >
                {option.label}
              </span>
              {secondary ? (
                <span
                  style={{
                    fontSize: "9px",
                    opacity: isActive ? 0.8 : 0.5,
                    marginTop: "1px",
                  }}
                >
                  {secondary}
                </span>
              ) : null}
            </button>
          );
        })}
      </div>
    </div>
  );
};
