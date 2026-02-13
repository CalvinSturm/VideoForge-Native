import React from 'react';

interface ProgressBarProps {
  currentFrame: number;
  totalFrames?: number;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ currentFrame, totalFrames }) => {
  // Calculate percentage if totalFrames exists and is > 0
  const percent = totalFrames && totalFrames > 0 
    ? Math.min(Math.round((currentFrame / totalFrames) * 100), 100) 
    : null;

  return (
    <div style={{ width: '100%', padding: '2px 0' }}>
      {/* Top Label Row */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        marginBottom: '4px',
        fontSize: '10px',
        fontFamily: '"JetBrains Mono", monospace',
        color: '#00ff88', // RTX Neon Green
        textTransform: 'uppercase',
        letterSpacing: '0.05em'
      }}>
        <span>
          {percent !== null ? `UPSCALE_PHASE: ${percent}%` : 'UPSCALE_PHASE: ACTIVE'}
        </span>
        <span style={{ opacity: 0.8 }}>
          FRAME_{currentFrame.toString().padStart(6, '0')}
        </span>
      </div>
      
      {/* Bar Container */}
      <div style={{ 
        height: '2px', 
        width: '100%', 
        background: 'rgba(255, 255, 255, 0.05)', 
        position: 'relative',
        overflow: 'hidden',
        borderRadius: '1px'
      }}>
        {/* Static Progress (only shows if we have a percentage) */}
        {percent !== null && (
          <div style={{ 
            height: '100%', 
            width: `${percent}%`, 
            background: '#00ff88',
            transition: 'width 0.2s ease-out'
          }} />
        )}

        {/* The Animated "Scanning" Glow Line */}
        <div style={{ 
          position: 'absolute',
          top: 0,
          height: '100%',
          width: '150px',
          background: 'linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.4), transparent)',
          animation: 'scan 2s cubic-bezier(0.4, 0, 0.2, 1) infinite'
        }} />
      </div>

      <style>{`
        @keyframes scan {
          0% { transform: translateX(-150px); }
          100% { transform: translateX(600px); }
        }
      `}</style>
    </div>
  );
};