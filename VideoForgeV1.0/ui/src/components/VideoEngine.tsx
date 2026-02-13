import React, { useRef, useEffect } from 'react';
import { convertFileSrc } from "@tauri-apps/api/core";

interface VideoEngineProps {
  src: string;
  currentTime: number;
  setCurrentTime: (t: number) => void;
  duration: number;
  setDuration: (d: number) => void;
  onRenderSample: () => void;
}

export const VideoEngine = ({ src, onToast }: { src: string, onToast: any }) => {
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [trim, setTrim] = useState({ start: 0, end: 0 });

  if (!src) return <div className="p-10 text-center border-dashed border-2">NO_VIDEO_LOADED</div>;

  return (
    <div className="video-player">
      <video 
        src={convertFileSrc(src)} 
        onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        className="w-full rounded shadow-xl"
      />
      <div className="controls mt-4">
          <input type="range" max={duration} value={currentTime} onChange={(e) => setCurrentTime(Number(e.target.value))} className="w-full" />
          <p className="text-xs font-mono text-green-500">{currentTime.toFixed(2)}s / {duration.toFixed(2)}s</p>
      </div>
    </div>
  );
};

export const VideoEngine: React.FC<VideoEngineProps> = ({
  src,
  currentTime,
  setCurrentTime,
  duration,
  setDuration,
  onRenderSample
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Sync external time changes (like from a slider) to the video element
  useEffect(() => {
    if (videoRef.current && Math.abs(videoRef.current.currentTime - currentTime) > 0.1) {
      videoRef.current.currentTime = currentTime;
    }
  }, [currentTime]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  if (!src) {
    return (
      <div className="video-empty-state" style={{
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        color: '#444',
        fontFamily: 'monospace',
        background: '#020202'
      }}>
        [ NO_SOURCE_DETECTED ]
      </div>
    );
  }

  return (
    <div className="video-engine-container" style={{ padding: '20px', backgroundColor: '#050505', height: '100%' }}>
      <div style={{ position: 'relative', borderRadius: '4px', overflow: 'hidden', border: '1px solid #1a1a1a' }}>
        <video
          ref={videoRef}
          src={convertFileSrc(src)}
          style={{ width: '100%', display: 'block' }}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          controls={false} // We build our own controls for a pro look
        />
        
        {/* Overlay for Frame Info */}
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          padding: '4px 8px',
          background: 'rgba(0,0,0,0.7)',
          color: '#00ff88',
          fontSize: '10px',
          fontFamily: 'monospace',
          border: '1px solid #00ff88'
        }}>
          LIVE_PREVIEW: {currentTime.toFixed(3)}s
        </div>
      </div>

      {/* Scrubbing Bar */}
      <div style={{ marginTop: '15px' }}>
        <input
          type="range"
          min={0}
          max={duration || 100}
          step={0.001}
          value={currentTime}
          onChange={(e) => setCurrentTime(parseFloat(e.target.value))}
          style={{ width: '100%', accentColor: '#00ff88', cursor: 'pointer' }}
        />
        
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px' }}>
          <div style={{ color: '#888', fontSize: '12px', fontFamily: 'monospace' }}>
            {currentTime.toFixed(2)} / {duration.toFixed(2)}s
          </div>
          
          <button 
            onClick={onRenderSample}
            style={{
              background: 'transparent',
              border: '1px solid #333',
              color: '#ccc',
              padding: '4px 12px',
              fontSize: '11px',
              cursor: 'pointer',
              textTransform: 'uppercase'
            }}
            onMouseOver={(e) => (e.currentTarget.style.borderColor = '#00ff88')}
            onMouseOut={(e) => (e.currentTarget.style.borderColor = '#333')}
          >
            Render 3s Sample
          </button>
        </div>
      </div>
    </div>
  );
};