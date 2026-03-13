import React from "react";
import type { UpscaleMode } from "../../types";
import { IconFile, IconFilm, IconImport, IconPlus } from "./panelIcons";
import { PipelineConnector, PipelineNode, SmartPath } from "./panelPrimitives";

interface InputSourceSectionProps {
  inputPath: string;
  mode: UpscaleMode;
  pickInput: () => void;
  sourceFps: number;
  sourceInfo: {
    label: string;
    detail: string;
  };
  sourceW: number;
}

export const InputSourceSection: React.FC<InputSourceSectionProps> = ({
  inputPath,
  mode,
  pickInput,
  sourceFps,
  sourceInfo,
  sourceW,
}) => (
  <>
    <PipelineNode
      title="Import Source"
      icon={<IconImport />}
      nodeNumber={1}
      isActive={!!inputPath}
      accentColor="#3b82f6"
    >
      <div
        onClick={pickInput}
        title={inputPath}
        style={{
          background: "linear-gradient(135deg, rgba(59,130,246,0.1), transparent)",
          border: inputPath ? "1px solid rgba(59,130,246,0.3)" : "1px dashed rgba(255,255,255,0.15)",
          borderRadius: "8px",
          height: "48px",
          display: "flex",
          alignItems: "center",
          cursor: "pointer",
          padding: "0 14px",
          gap: "12px",
          transition: "all 0.2s ease",
        }}
      >
        <div
          style={{
            width: "32px",
            height: "32px",
            borderRadius: "6px",
            background: inputPath ? "rgba(59,130,246,0.2)" : "rgba(255,255,255,0.05)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: inputPath ? "#3b82f6" : "var(--text-muted)",
          }}
        >
          {inputPath ? (mode === "video" ? <IconFilm /> : <IconFile />) : <IconPlus />}
        </div>
        <div
          style={{
            flex: 1,
            fontSize: "11px",
            color: inputPath ? "var(--text-primary)" : "var(--text-muted)",
            overflow: "hidden",
            textAlign: "left",
          }}
        >
          <SmartPath path={inputPath} placeholder="Click to select source file..." />
        </div>
        {inputPath && (
          <div
            style={{
              fontSize: "9px",
              color: "#3b82f6",
              fontWeight: 600,
              padding: "3px 8px",
              background: "rgba(59,130,246,0.15)",
              borderRadius: "4px",
            }}
          >
            LOADED
          </div>
        )}
      </div>

      {sourceW > 0 && (
        <div style={{ display: "flex", gap: "8px", marginTop: "4px" }}>
          <div
            style={{
              flex: 1,
              padding: "8px 10px",
              background: "rgba(0,0,0,0.2)",
              borderRadius: "6px",
              border: "1px solid rgba(255,255,255,0.04)",
            }}
          >
            <div style={{ fontSize: "8px", color: "var(--text-muted)", marginBottom: "2px" }}>RESOLUTION</div>
            <div style={{ fontSize: "11px", color: "var(--text-primary)", fontWeight: 600 }}>{sourceInfo.label}</div>
            <div style={{ fontSize: "9px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{sourceInfo.detail}</div>
          </div>
          {mode === "video" && (
            <div
              style={{
                flex: 1,
                padding: "8px 10px",
                background: "rgba(0,0,0,0.2)",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.04)",
              }}
            >
              <div style={{ fontSize: "8px", color: "var(--text-muted)", marginBottom: "2px" }}>FRAME RATE</div>
              <div style={{ fontSize: "11px", color: "var(--text-primary)", fontWeight: 600 }}>{sourceFps} FPS</div>
            </div>
          )}
        </div>
      )}
    </PipelineNode>

    <PipelineConnector isActive={!!inputPath} />
  </>
);
