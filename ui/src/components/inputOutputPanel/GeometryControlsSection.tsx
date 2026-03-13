import React from "react";
import type { EditState } from "../../types";
import { ASPECT_RATIOS } from "./panelHelpers";
import {
  IconCrop,
  IconFlipH,
  IconFlipV,
  IconMove,
  IconPlus,
  IconRotateCCW,
  IconRotateCW,
  IconX,
} from "./panelIcons";
import { PipelineConnector, PipelineNode } from "./panelPrimitives";

interface GeometryControlsSectionProps {
  applyAspectRatio: (value: number | null) => void;
  applyCrop: () => void;
  editState: EditState;
  isAIActive: boolean;
  isCropActive: boolean;
  isCropApplied: boolean;
  setEditState: (state: EditState) => void;
  toggleCrop: () => void;
}

export const GeometryControlsSection: React.FC<GeometryControlsSectionProps> = ({
  applyAspectRatio,
  applyCrop,
  editState,
  isAIActive,
  isCropActive,
  isCropApplied,
  setEditState,
  toggleCrop,
}) => (
  <>
    <PipelineNode
      title="Crop & Frame"
      icon={<IconCrop />}
      nodeNumber={isAIActive ? 4 : 3}
      isActive={isCropActive}
      accentColor="#3b82f6"
      extra={
        isCropActive && (
          <button
            onClick={(event) => {
              event.stopPropagation();
              applyCrop();
            }}
            style={{
              height: "24px",
              fontSize: "9px",
              padding: "0 12px",
              borderRadius: "6px",
              border: isCropApplied ? "1px solid rgba(59,130,246,0.3)" : "none",
              background: isCropApplied ? "transparent" : "linear-gradient(135deg, #3b82f6, #2563eb)",
              color: isCropApplied ? "#3b82f6" : "white",
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.15s ease",
              boxShadow: !isCropApplied ? "0 2px 8px rgba(59,130,246,0.3)" : "none",
            }}
          >
            {isCropApplied ? "EDIT" : "APPLY"}
          </button>
        )
      }
    >
      {!isCropActive ? (
        <button
          onClick={toggleCrop}
          style={{
            width: "100%",
            height: "44px",
            border: "1px dashed rgba(59,130,246,0.3)",
            color: "#3b82f6",
            fontSize: "10px",
            fontWeight: 600,
            background: "rgba(59,130,246,0.05)",
            borderRadius: "8px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px",
            transition: "all 0.15s ease",
          }}
        >
          <IconPlus /> ENABLE CROP TOOL
        </button>
      ) : (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "6px" }}>
            {ASPECT_RATIOS.map((aspectRatio) => (
              <button
                key={aspectRatio.label}
                onClick={() => applyAspectRatio(aspectRatio.value)}
                style={{
                  fontSize: "10px",
                  height: "34px",
                  borderRadius: "6px",
                  background:
                    editState.aspectRatio === aspectRatio.value
                      ? "linear-gradient(135deg, rgba(59,130,246,0.3), rgba(59,130,246,0.1))"
                      : "rgba(255,255,255,0.03)",
                  border:
                    editState.aspectRatio === aspectRatio.value
                      ? "1px solid rgba(59,130,246,0.5)"
                      : "1px solid rgba(255,255,255,0.08)",
                  color: editState.aspectRatio === aspectRatio.value ? "#60a5fa" : "var(--text-muted)",
                  fontWeight: editState.aspectRatio === aspectRatio.value ? 700 : 500,
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                {aspectRatio.label}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "4px" }}>
            <button
              onClick={toggleCrop}
              style={{
                color: "#ef4444",
                background: "rgba(239,68,68,0.1)",
                fontSize: "9px",
                border: "1px solid rgba(239,68,68,0.2)",
                borderRadius: "4px",
                padding: "4px 10px",
                cursor: "pointer",
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                gap: "4px",
              }}
            >
              <IconX /> REMOVE
            </button>
          </div>
        </>
      )}
    </PipelineNode>

    <PipelineConnector isActive={isCropActive} />

    <PipelineNode
      title="Transform"
      icon={<IconMove />}
      nodeNumber={isAIActive ? 5 : 4}
      isActive={editState.rotation !== 0 || editState.flipH || editState.flipV}
      accentColor="#ec4899"
      extra={
        (editState.rotation !== 0 || editState.flipH || editState.flipV) && (
          <button
            onClick={(event) => {
              event.stopPropagation();
              setEditState({ ...editState, rotation: 0, flipH: false, flipV: false });
            }}
            style={{
              height: "24px",
              fontSize: "9px",
              padding: "0 10px",
              borderRadius: "6px",
              border: "1px solid rgba(236,72,153,0.3)",
              background: "transparent",
              color: "#ec4899",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            RESET
          </button>
        )
      }
    >
      <div>
        <label style={{ fontSize: "10px", color: "var(--text-secondary)", fontWeight: 600, letterSpacing: "0.03em", marginBottom: "8px", display: "block" }}>
          ROTATION
        </label>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <button
            onClick={() => {
              const rotations: (0 | 90 | 180 | 270)[] = [0, 90, 180, 270];
              const idx = rotations.indexOf(editState.rotation);
              const newRotation = rotations[(idx + 3) % 4] as 0 | 90 | 180 | 270;
              setEditState({ ...editState, rotation: newRotation });
            }}
            style={{
              height: "38px",
              width: "44px",
              borderRadius: "6px",
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.08)",
              color: "var(--text-muted)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
            title="Rotate Counter-Clockwise"
          >
            <IconRotateCCW />
          </button>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "4px", flex: 1 }}>
            {([0, 90, 180, 270] as const).map((degrees) => (
              <button
                key={degrees}
                onClick={() => setEditState({ ...editState, rotation: degrees })}
                style={{
                  fontSize: "10px",
                  height: "38px",
                  borderRadius: "6px",
                  background:
                    editState.rotation === degrees
                      ? "linear-gradient(135deg, rgba(236,72,153,0.3), rgba(236,72,153,0.1))"
                      : "rgba(255,255,255,0.03)",
                  border:
                    editState.rotation === degrees
                      ? "1px solid rgba(236,72,153,0.5)"
                      : "1px solid rgba(255,255,255,0.08)",
                  color: editState.rotation === degrees ? "#f472b6" : "var(--text-muted)",
                  fontWeight: editState.rotation === degrees ? 700 : 500,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                {degrees}°
              </button>
            ))}
          </div>
          <button
            onClick={() => {
              const rotations: (0 | 90 | 180 | 270)[] = [0, 90, 180, 270];
              const idx = rotations.indexOf(editState.rotation);
              const newRotation = rotations[(idx + 1) % 4] as 0 | 90 | 180 | 270;
              setEditState({ ...editState, rotation: newRotation });
            }}
            style={{
              height: "38px",
              width: "44px",
              borderRadius: "6px",
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.08)",
              color: "var(--text-muted)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
            title="Rotate Clockwise"
          >
            <IconRotateCW />
          </button>
        </div>
      </div>
      <div>
        <label style={{ fontSize: "10px", color: "var(--text-secondary)", fontWeight: 600, letterSpacing: "0.03em", marginBottom: "8px", display: "block" }}>
          FLIP / MIRROR
        </label>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "6px" }}>
          <button
            onClick={() => setEditState({ ...editState, flipH: !editState.flipH })}
            style={{
              fontSize: "10px",
              height: "40px",
              borderRadius: "6px",
              background: editState.flipH
                ? "linear-gradient(135deg, rgba(236,72,153,0.3), rgba(236,72,153,0.1))"
                : "rgba(255,255,255,0.03)",
              border: editState.flipH
                ? "1px solid rgba(236,72,153,0.5)"
                : "1px solid rgba(255,255,255,0.08)",
              color: editState.flipH ? "#f472b6" : "var(--text-muted)",
              fontWeight: editState.flipH ? 700 : 500,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "8px",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
          >
            <IconFlipH /> HORIZONTAL
          </button>
          <button
            onClick={() => setEditState({ ...editState, flipV: !editState.flipV })}
            style={{
              fontSize: "10px",
              height: "40px",
              borderRadius: "6px",
              background: editState.flipV
                ? "linear-gradient(135deg, rgba(236,72,153,0.3), rgba(236,72,153,0.1))"
                : "rgba(255,255,255,0.03)",
              border: editState.flipV
                ? "1px solid rgba(236,72,153,0.5)"
                : "1px solid rgba(255,255,255,0.08)",
              color: editState.flipV ? "#f472b6" : "var(--text-muted)",
              fontWeight: editState.flipV ? 700 : 500,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "8px",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
          >
            <IconFlipV /> VERTICAL
          </button>
        </div>
      </div>
    </PipelineNode>

    <PipelineConnector isActive={editState.rotation !== 0 || editState.flipH || editState.flipV} />
  </>
);
