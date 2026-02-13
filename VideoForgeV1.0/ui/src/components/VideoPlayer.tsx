import React, { useRef } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";

/* ---------- Types ---------- */
interface VideoState {
  src: string;
  currentTime: number;
  setCurrentTime: (t: number) => void;
  duration: number;
  setDuration: (d: number) => void;
  trimStart: number;
  setTrimStart: (t: number) => void;
  trimEnd: number;
  setTrimEnd: (t: number) => void;
  crop: { x: number; y: number; width: number; height: number };
  setCrop: (c: { x: number; y: number; width: number; height: number }) => void;
  samplePreview: string | null;
  renderSample: () => void;
}

interface VideoPlayerProps {
  videoState: VideoState;
}

/* ---------- Component ---------- */
export const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoState }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  
  const { 
    src, 
    currentTime, 
    setCurrentTime, 
    duration, 
    setDuration, 
    samplePreview, 
    renderSample 
  } = videoState;

  // Format time for the UI (e.g., 00:01:30)
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="video-player-container" style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      {/* The Main Video Feed */}
      <div style={{ position: "relative", backgroundColor: "#000", borderRadius: "8px", overflow: "hidden", aspectRatio: "16/9" }}>
        {src ? (
          <video
            ref={videoRef}
            src={convertFileSrc(src)}
            onTimeUpdate={() => videoRef.current && setCurrentTime(videoRef.current.currentTime)}
            onLoadedMetadata={() => videoRef.current && setDuration(videoRef.current.duration)}
            style={{ width: "100%", height: "100%", objectFit: "contain" }}
          />
        ) : (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#666" }}>
            Waiting for video input...
          </div>
        )}
      </div>

      {/* Timeline Controls */}
      <div style={{ padding: "0 4px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "4px", fontFamily: "monospace" }}>
          <span style={{ color: "var(--accent)" }}>{formatTime(currentTime)}</span>
          <span style={{ opacity: 0.6 }}>{formatTime(duration)}</span>
        </div>
        
        <input
          type="range"
          min={0}
          max={duration || 0}
          step={0.01}
          value={currentTime}
          onChange={(e) => {
            const t = parseFloat(e.target.value);
            if (videoRef.current) videoRef.current.currentTime = t;
            setCurrentTime(t);
          }}
          style={{ width: "100%", cursor: "pointer", accentColor: "var(--accent)" }}
        />
      </div>

      {/* Sample Rendering Section */}
      <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "12px", marginTop: "4px" }}>
        <button 
          className="bp5-button bp5-small bp5-intent-warning bp5-fill" 
          onClick={renderSample}
          disabled={!src}
        >
          Render 3s Preview Sample
        </button>
        
        {samplePreview && (
          <div style={{ marginTop: "12px" }}>
            <h5 style={{ fontSize: "11px", marginBottom: "8px", opacity: 0.8 }}>AI Sample Result:</h5>
            <div style={{ borderRadius: "4px", overflow: "hidden", border: "1px solid var(--accent)" }}>
              <video 
                src={samplePreview} 
                controls 
                style={{ width: "100%", display: "block" }} 
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};