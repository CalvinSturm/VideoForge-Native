import React from "react";
import { SignalSummary } from "../SignalSummary";
import type { EditState, UpscaleMode } from "../../types";
import {
  IconClock,
  IconExport,
  IconPalette,
  IconSave,
  IconSparkles,
} from "./panelIcons";
import {
  ColorSlider,
  PipelineConnector,
  PipelineNode,
  SmartPath,
  ToggleGroup,
} from "./panelPrimitives";

interface ResolutionInfo {
  label: string;
  detail: string;
}

interface PostProcessingSectionProps {
  activeScale: number;
  editState: EditState;
  hasColorEdits: boolean;
  hasEdits: boolean;
  hasMotionEdits: boolean;
  isAIActive: boolean;
  isCropActive: boolean;
  mode: UpscaleMode;
  modelDisplayLabel: string;
  modelFamily: string;
  outputPath: string;
  pickOutput: () => void;
  setEditState: (state: EditState) => void;
  showToast: (message: string) => void;
  sourceInfo: ResolutionInfo;
  strSourceFps: string;
  strTargetFps: string;
  targetFps: number;
  targetInfo: ResolutionInfo;
}

export const PostProcessingSection: React.FC<PostProcessingSectionProps> = ({
  activeScale,
  editState,
  hasColorEdits,
  hasEdits,
  hasMotionEdits,
  isAIActive,
  isCropActive,
  mode,
  modelDisplayLabel,
  modelFamily,
  outputPath,
  pickOutput,
  setEditState,
  showToast,
  sourceInfo,
  strSourceFps,
  strTargetFps,
  targetFps,
  targetInfo,
}) => (
  <>
    <PipelineNode
      title="Color Grading"
      icon={<IconPalette />}
      nodeNumber={isAIActive ? 6 : 5}
      isActive={hasColorEdits}
      accentColor="#a855f7"
      extra={
        <div style={{ display: "flex", gap: "6px" }}>
          <button
            onClick={(event) => {
              event.stopPropagation();
              setEditState({
                ...editState,
                color: {
                  brightness: 0.05,
                  contrast: 0.08,
                  saturation: 0.05,
                  gamma: 1.0,
                },
              });
              showToast("Auto Grade applied (conservative)");
            }}
            style={{
              height: "24px",
              fontSize: "9px",
              padding: "0 10px",
              borderRadius: "6px",
              border: "1px solid rgba(0,255,136,0.3)",
              background: "linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,255,136,0.05))",
              color: "var(--brand-primary)",
              cursor: "pointer",
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              gap: "4px",
            }}
            title="Apply automatic color grading"
          >
            <IconSparkles /> AUTO
          </button>
          {hasColorEdits && (
            <button
              onClick={(event) => {
                event.stopPropagation();
                setEditState({
                  ...editState,
                  color: { brightness: 0, contrast: 0, saturation: 0, gamma: 1.0 },
                });
              }}
              style={{
                height: "24px",
                fontSize: "9px",
                padding: "0 10px",
                borderRadius: "6px",
                border: "1px solid rgba(168,85,247,0.3)",
                background: "transparent",
                color: "#a855f7",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              RESET
            </button>
          )}
        </div>
      }
    >
      <ColorSlider
        label="BRIGHTNESS"
        value={editState.color.brightness}
        onChange={(value) => setEditState({ ...editState, color: { ...editState.color, brightness: value } })}
        accentColor="#fbbf24"
      />
      <ColorSlider
        label="CONTRAST"
        value={editState.color.contrast}
        onChange={(value) => setEditState({ ...editState, color: { ...editState.color, contrast: value } })}
        accentColor="#22d3ee"
      />
      <ColorSlider
        label="SATURATION"
        value={editState.color.saturation}
        onChange={(value) => setEditState({ ...editState, color: { ...editState.color, saturation: value } })}
        accentColor="#f472b6"
      />
      <ColorSlider
        label="GAMMA"
        value={editState.color.gamma}
        min={0.1}
        max={3.0}
        step={0.01}
        onChange={(value) => setEditState({ ...editState, color: { ...editState.color, gamma: value } })}
        formatValue={(value) => value.toFixed(2)}
        accentColor="#a78bfa"
      />
    </PipelineNode>

    <PipelineConnector isActive={hasColorEdits} />

    {mode === "video" && (
      <PipelineNode
        title="Frame Rate"
        icon={<IconClock />}
        nodeNumber={isAIActive ? 7 : 6}
        isActive={hasMotionEdits}
        accentColor="#eab308"
        extra={
          hasMotionEdits && (
            <button
              onClick={(event) => {
                event.stopPropagation();
                setEditState({ ...editState, fps: 0 });
              }}
              style={{
                height: "24px",
                fontSize: "9px",
                padding: "0 10px",
                borderRadius: "6px",
                border: "1px solid rgba(234,179,8,0.3)",
                background: "transparent",
                color: "#eab308",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              RESET
            </button>
          )
        }
      >
        <ToggleGroup
          value={editState.fps}
          onChange={(value) => setEditState({ ...editState, fps: value as number })}
          options={[
            { label: "NATIVE", sub: "SOURCE", value: 0 },
            { label: "30", sub: "FPS", value: 30 },
            { label: "60", sub: "FPS", value: 60 },
            { label: "120", sub: "FPS", value: 120 },
          ]}
        />
      </PipelineNode>
    )}

    <PipelineConnector isActive={mode === "video" ? hasMotionEdits : hasColorEdits} />

    <PipelineNode
      title="Export Output"
      icon={<IconExport />}
      nodeNumber={mode === "video" ? (isAIActive ? 8 : 7) : (isAIActive ? 7 : 6)}
      isActive={!!outputPath}
      accentColor="#10b981"
    >
      <SignalSummary
        sourceResolution={sourceInfo.label}
        sourceDetail={sourceInfo.detail}
        sourceFps={strSourceFps}
        targetResolution={targetInfo.label}
        targetDetail={targetInfo.detail}
        targetFps={strTargetFps}
        modelLabel={modelDisplayLabel || ""}
      />

      {(hasEdits || isAIActive) && (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "6px",
            padding: "10px 12px",
            background: "rgba(0,0,0,0.25)",
            borderRadius: "8px",
            border: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          <span style={{ fontSize: "9px", color: "var(--text-muted)", fontWeight: 600, width: "100%", marginBottom: "4px", letterSpacing: "0.05em" }}>
            PIPELINE STAGES:
          </span>
          {isAIActive && (
            <span
              style={{
                fontSize: "9px",
                padding: "4px 10px",
                borderRadius: "4px",
                background: "rgba(0, 255, 136, 0.12)",
                color: "var(--brand-primary)",
                fontWeight: 600,
                border: "1px solid rgba(0, 255, 136, 0.25)",
                display: "flex",
                alignItems: "center",
                gap: "4px",
              }}
            >
              <IconSparkles /> {modelFamily} {activeScale}x
            </span>
          )}
          {isCropActive && (
            <span style={{ fontSize: "9px", padding: "4px 10px", borderRadius: "4px", background: "rgba(59, 130, 246, 0.12)", color: "#60a5fa", fontWeight: 600, border: "1px solid rgba(59, 130, 246, 0.25)" }}>
              CROP
            </span>
          )}
          {(editState.rotation !== 0 || editState.flipH || editState.flipV) && (
            <span style={{ fontSize: "9px", padding: "4px 10px", borderRadius: "4px", background: "rgba(236, 72, 153, 0.12)", color: "#f472b6", fontWeight: 600, border: "1px solid rgba(236, 72, 153, 0.25)" }}>
              TRANSFORM
            </span>
          )}
          {hasColorEdits && (
            <span style={{ fontSize: "9px", padding: "4px 10px", borderRadius: "4px", background: "rgba(168, 85, 247, 0.12)", color: "#c084fc", fontWeight: 600, border: "1px solid rgba(168, 85, 247, 0.25)" }}>
              COLOR
            </span>
          )}
          {hasMotionEdits && (
            <span style={{ fontSize: "9px", padding: "4px 10px", borderRadius: "4px", background: "rgba(234, 179, 8, 0.12)", color: "#fbbf24", fontWeight: 600, border: "1px solid rgba(234, 179, 8, 0.25)" }}>
              {targetFps} FPS
            </span>
          )}
        </div>
      )}

      {!hasEdits && !isAIActive && (
        <div
          style={{
            fontSize: "10px",
            color: "var(--text-muted)",
            textAlign: "center",
            padding: "12px",
            background: "rgba(0,0,0,0.2)",
            borderRadius: "6px",
            border: "1px dashed rgba(255,255,255,0.08)",
            fontStyle: "italic",
          }}
        >
          No pipeline stages active - enable a tool above
        </div>
      )}

      <div
        onClick={pickOutput}
        title={outputPath}
        style={{
          background: "linear-gradient(135deg, rgba(16,185,129,0.1), transparent)",
          border: outputPath ? "1px solid rgba(16,185,129,0.3)" : "1px dashed rgba(255,255,255,0.15)",
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
            background: outputPath ? "rgba(16,185,129,0.2)" : "rgba(255,255,255,0.05)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: outputPath ? "#10b981" : "var(--text-muted)",
          }}
        >
          <IconSave />
        </div>
        <div
          style={{
            flex: 1,
            fontSize: "11px",
            color: outputPath ? "var(--text-primary)" : "var(--text-muted)",
            overflow: "hidden",
            textAlign: "left",
          }}
        >
          <SmartPath path={outputPath} placeholder="Click to set export location..." />
        </div>
        {outputPath && (
          <div
            style={{
              fontSize: "9px",
              color: "#10b981",
              fontWeight: 600,
              padding: "3px 8px",
              background: "rgba(16,185,129,0.15)",
              borderRadius: "4px",
            }}
          >
            SET
          </div>
        )}
      </div>
    </PipelineNode>
  </>
);
