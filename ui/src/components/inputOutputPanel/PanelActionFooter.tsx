import React from "react";
import type { VideoState, UpscaleMode } from "../../types";
import { IconFlash, IconPlay, IconShield } from "./panelIcons";

interface PanelActionFooterProps {
  buttonStyle: React.CSSProperties;
  canRunValidate: boolean;
  isHighIntensity: boolean;
  isMainActionDisabled: boolean;
  isValidPaths: boolean;
  mainActionHandler: () => void;
  mainActionLabel: string;
  mode: UpscaleMode;
  onRunValidate: () => void;
  videoState: VideoState;
}

export const PanelActionFooter: React.FC<PanelActionFooterProps> = ({
  buttonStyle,
  canRunValidate,
  isHighIntensity,
  isMainActionDisabled,
  isValidPaths,
  mainActionHandler,
  mainActionLabel,
  mode,
  onRunValidate,
  videoState,
}) => (
  <div
    style={{
      padding: "16px",
      borderTop: "1px solid var(--panel-border)",
      background: "var(--section-bg)",
      display: "flex",
      flexDirection: "column",
      gap: "12px",
      flexShrink: 0,
    }}
  >
    {mode === "video" && (
      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "8px", alignItems: "center" }}>
        <button
          className="action-secondary"
          onClick={videoState.renderSample}
          disabled={!isValidPaths}
          style={{
            borderColor: "rgba(255,255,255,0.15)",
            color: "white",
            background: "rgba(255,255,255,0.02)",
            fontWeight: 600,
            display: "flex",
            gap: "8px",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
          }}
        >
          <IconPlay />
          PREVIEW 2s
        </button>
        <button
          className="action-secondary"
          onClick={onRunValidate}
          disabled={!isValidPaths || !canRunValidate}
          title={canRunValidate ? "Run strict policy validation in mock mode" : "Enable AI Upscale on video input to validate"}
          style={{
            borderColor: "rgba(59,130,246,0.4)",
            color: "#bfdbfe",
            background: "rgba(59,130,246,0.08)",
            fontWeight: 700,
            display: "flex",
            gap: "8px",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
          }}
        >
          <IconShield />
          VALIDATE STRICT (MOCK)
        </button>
      </div>
    )}
    <button
      className={isHighIntensity ? "action-primary" : ""}
      onClick={mainActionHandler}
      disabled={isMainActionDisabled}
      style={buttonStyle}
    >
      {isHighIntensity ? (
        mainActionLabel
      ) : (
        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 800 }}>
          <IconFlash /> {mainActionLabel}
        </div>
      )}
    </button>
  </div>
);
