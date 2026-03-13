import React, { useEffect, useRef } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";

interface VideoEngineProps {
  src: string;
  currentTime: number;
  setCurrentTime: (time: number) => void;
  duration: number;
  setDuration: (duration: number) => void;
  onRenderSample: () => void;
}

export const VideoEngine: React.FC<VideoEngineProps> = ({
  src,
  currentTime,
  setCurrentTime,
  duration,
  setDuration,
  onRenderSample,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!videoRef.current) {
      return;
    }
    if (Math.abs(videoRef.current.currentTime - currentTime) > 0.1) {
      videoRef.current.currentTime = currentTime;
    }
  }, [currentTime]);

  if (!src) {
    return (
      <div
        className="video-empty-state"
        style={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#444",
          fontFamily: "monospace",
          background: "#020202",
        }}
      >
        [ NO_SOURCE_DETECTED ]
      </div>
    );
  }

  return (
    <div
      className="video-engine-container"
      style={{ padding: "20px", backgroundColor: "#050505", height: "100%" }}
    >
      <div
        style={{
          position: "relative",
          borderRadius: "4px",
          overflow: "hidden",
          border: "1px solid #1a1a1a",
        }}
      >
        <video
          ref={videoRef}
          src={convertFileSrc(src)}
          style={{ width: "100%", display: "block" }}
          onTimeUpdate={() => {
            if (videoRef.current) {
              setCurrentTime(videoRef.current.currentTime);
            }
          }}
          onLoadedMetadata={() => {
            if (videoRef.current) {
              setDuration(videoRef.current.duration);
            }
          }}
          controls={false}
        />

        <div
          style={{
            position: "absolute",
            top: "10px",
            right: "10px",
            padding: "4px 8px",
            background: "rgba(0,0,0,0.7)",
            color: "#00ff88",
            fontSize: "10px",
            fontFamily: "monospace",
            border: "1px solid #00ff88",
          }}
        >
          LIVE_PREVIEW: {currentTime.toFixed(3)}s
        </div>
      </div>

      <div style={{ marginTop: "15px" }}>
        <input
          type="range"
          min={0}
          max={duration || 100}
          step={0.001}
          value={currentTime}
          onChange={(e) => setCurrentTime(Number.parseFloat(e.target.value))}
          style={{ width: "100%", accentColor: "#00ff88", cursor: "pointer" }}
        />

        <div style={{ display: "flex", justifyContent: "space-between", marginTop: "10px" }}>
          <div style={{ color: "#888", fontSize: "12px", fontFamily: "monospace" }}>
            {currentTime.toFixed(2)} / {duration.toFixed(2)}s
          </div>

          <button
            onClick={onRenderSample}
            style={{
              background: "transparent",
              border: "1px solid #333",
              color: "#ccc",
              padding: "4px 12px",
              fontSize: "11px",
              cursor: "pointer",
              textTransform: "uppercase",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.borderColor = "#00ff88";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.borderColor = "#333";
            }}
          >
            Render 3s Sample
          </button>
        </div>
      </div>
    </div>
  );
};
