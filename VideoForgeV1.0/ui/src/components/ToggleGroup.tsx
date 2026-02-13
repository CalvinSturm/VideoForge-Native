import React from 'react';

export interface ToggleOption {
  label: string;
  value: string | number;
  subLabel?: string;
}

interface ToggleGroupProps {
  options: ToggleOption[];
  selectedValue: string | number;
  onChange: (value: any) => void;
  label?: string;
}

export const ToggleGroup: React.FC<ToggleGroupProps> = ({
  options,
  selectedValue,
  onChange,
  label
}) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100%' }}>
      {label && (
        <span style={{ fontSize: '10px', color: '#71717a', fontWeight: 600, textTransform: 'uppercase' }}>
          {label}
        </span>
      )}

      <div
        role="radiogroup"
        style={{
          display: 'flex',
          width: '100%',
          backgroundColor: '#000000',
          padding: '2px',
          borderRadius: '4px',
          border: '1px solid #27272a'
        }}
      >
        {options.map((option) => {
          const isActive = option.value === selectedValue;
          return (
            <button
              key={String(option.value)}
              role="radio"
              aria-checked={isActive}
              onClick={() => onChange(option.value)}
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                background: isActive ? '#00ff88' : 'transparent',
                border: 'none',
                borderRadius: '2px',
                padding: '6px 0',
                cursor: 'pointer',
                transition: 'background-color 0.1s',
                color: isActive ? '#000000' : '#a1a1aa'
              }}
            >
              <span style={{
                fontSize: '11px',
                fontWeight: 700,
                fontFamily: 'Inter, sans-serif'
              }}>
                {option.label}
              </span>
              {option.subLabel && (
                <span style={{
                  fontSize: '9px',
                  opacity: isActive ? 0.8 : 0.5,
                  marginTop: '1px'
                }}>
                  {option.subLabel}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};

src/ui/tools/CropTool.tsx
import React, { useState, useEffect } from 'react';

interface CropToolProps {
  isEnabled: boolean;
  isDirty: boolean;
  onToggle: (enabled: boolean) => void;
  onCommit: () => void;
  onDiscard: () => void;
}

type CropState = 'IDLE' | 'EDITING' | 'APPLIED';

export const CropTool: React.FC<CropToolProps> = ({
  isEnabled,
  isDirty,
  onToggle,
  onCommit,
  onDiscard
}) => {
  const currentState: CropState = isEnabled
    ? (isDirty ? 'EDITING' : 'APPLIED')
    : 'IDLE';

  // Handle ESC to cancel while editing
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && currentState === 'EDITING') {
        onDiscard();
      }
      if (e.key === 'Enter' && currentState === 'EDITING') {
        onCommit();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentState, onDiscard, onCommit]);

  return (
    <div style={{
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: '4px',
      padding: '12px',
      backgroundColor: 'rgba(255,255,255,0.02)'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <span style={{ fontSize: '10px', fontWeight: 600, color: '#a1a1aa', textTransform: 'uppercase' }}>
          Crop Tool
        </span>

        {/* Status Indicator */}
        {currentState !== 'IDLE' && (
          <span style={{ fontSize: '9px', color: currentState === 'EDITING' ? '#eab308' : '#00ff88', fontWeight: 700 }}>
            {currentState === 'EDITING' ? '● UNSAVED' : '● ACTIVE'}
          </span>
        )}
      </div>

      <div style={{ display: 'flex', gap: '8px' }}>
        {currentState === 'IDLE' && (
          <button
            onClick={() => onToggle(true)}
            style={{
              width: '100%',
              padding: '6px',
              background: 'transparent',
              border: '1px solid #3f3f46',
              color: '#ededed',
              borderRadius: '4px',
              fontSize: '11px',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            Enable Crop
          </button>
        )}

        {currentState === 'EDITING' && (
          <>
            <button
              onClick={onCommit}
              style={{
                flex: 2,
                padding: '6px',
                background: '#00ff88',
                border: 'none',
                color: '#000000',
                borderRadius: '4px',
                fontSize: '11px',
                fontWeight: 700,
                cursor: 'pointer'
              }}
            >
              Confirm
            </button>
            <button
              onClick={onDiscard}
              style={{
                flex: 1,
                padding: '6px',
                background: 'transparent',
                border: '1px solid #3f3f46',
                color: '#a1a1aa',
                borderRadius: '4px',
                fontSize: '11px',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
          </>
        )}

        {currentState === 'APPLIED' && (
          <>
            <button
              onClick={() => onToggle(true)} // Re-enter edit mode
              style={{
                flex: 1,
                padding: '6px',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)',
                color: '#ededed',
                borderRadius: '4px',
                fontSize: '11px',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              Edit
            </button>
            <button
              onClick={() => onToggle(false)} // Disable
              style={{
                width: '32px',
                padding: '6px',
                background: 'transparent',
                border: '1px solid #3f3f46',
                color: '#ef4444',
                borderRadius: '4px',
                fontSize: '14px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
              title="Remove Crop"
            >
              ×
            </button>
          </>
        )}
      </div>
    </div>
  );
};

src/ui/timeline/Timeline.tsx
import React, { useRef, useState, useEffect } from 'react';

interface TimelineProps {
  duration: number;
  currentTime: number;
  trimStart: number;
  trimEnd: number;
  onSeek: (time: number) => void;
  onTrim: (start: number, end: number) => void;
}

export const Timeline: React.FC<TimelineProps> = ({
  duration,
  currentTime,
  trimStart,
  trimEnd,
  onSeek,
  onTrim
}) => {
  const trackRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState<'playhead' | 'start' | 'end' | null>(null);

  const safeDuration = Math.max(duration, 0.1);

  const getProgress = (time: number) => (time / safeDuration) * 100;

  const handleInteraction = (e: React.MouseEvent | MouseEvent) => {
    if (!trackRef.current) return;
    const rect = trackRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const ratio = x / rect.width;
    const time = ratio * safeDuration;

    if (isDragging === 'playhead') {
      onSeek(Math.min(Math.max(time, trimStart), trimEnd));
    } else if (isDragging === 'start') {
      onTrim(Math.min(time, trimEnd - 0.1), trimEnd);
    } else if (isDragging === 'end') {
      onTrim(trimStart, Math.max(time, trimStart + 0.1));
    }
  };

  useEffect(() => {
    const onMove = (e: MouseEvent) => isDragging && handleInteraction(e);
    const onUp = () => setIsDragging(null);

    if (isDragging) {
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    }
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [isDragging, trimStart, trimEnd]);

  return (
    <div style={{ width: '100%', padding: '10px 0', userSelect: 'none' }}>
      <div
        ref={trackRef}
        style={{
          position: 'relative',
          height: '24px',
          background: '#18181b',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
        onMouseDown={(e) => {
          setIsDragging('playhead');
          handleInteraction(e);
        }}
      >
        {/* Playable Range Bar */}
        <div style={{
          position: 'absolute',
          left: `${getProgress(trimStart)}%`,
          right: `${100 - getProgress(trimEnd)}%`,
          top: 0, bottom: 0,
          backgroundColor: 'rgba(0, 255, 136, 0.1)',
          borderLeft: '1px solid #00ff88',
          borderRight: '1px solid #00ff88',
          pointerEvents: 'none'
        }} />

        {/* Start Handle */}
        <div
          style={{
            position: 'absolute',
            left: `${getProgress(trimStart)}%`,
            top: 0, bottom: 0,
            width: '20px', // Large hit area
            transform: 'translateX(-50%)',
            cursor: 'ew-resize',
            zIndex: 10
          }}
          onMouseDown={(e) => {
            e.stopPropagation();
            setIsDragging('start');
          }}
        >
          <div style={{
            width: '4px', height: '100%',
            backgroundColor: '#00ff88',
            borderRadius: '4px 0 0 4px',
            margin: '0 auto'
          }} />
        </div>

        {/* End Handle */}
        <div
          style={{
            position: 'absolute',
            left: `${getProgress(trimEnd)}%`,
            top: 0, bottom: 0,
            width: '20px', // Large hit area
            transform: 'translateX(-50%)',
            cursor: 'ew-resize',
            zIndex: 10
          }}
          onMouseDown={(e) => {
            e.stopPropagation();
            setIsDragging('end');
          }}
        >
          <div style={{
            width: '4px', height: '100%',
            backgroundColor: '#00ff88',
            borderRadius: '0 4px 4px 0',
            margin: '0 auto'
          }} />
        </div>

        {/* Playhead */}
        <div style={{
          position: 'absolute',
          left: `${getProgress(currentTime)}%`,
          top: -4, bottom: -4,
          width: '20px', // Large hit area
          transform: 'translateX(-50%)',
          cursor: 'grab',
          zIndex: 20,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center'
        }}
        onMouseDown={(e) => {
          e.stopPropagation();
          setIsDragging('playhead');
        }}>
          <div style={{ width: '2px', height: '100%', backgroundColor: '#ffffff' }} />
          <div style={{ width: '10px', height: '10px', backgroundColor: '#ffffff', borderRadius: '50%', marginTop: '-5px' }} />
        </div>
      </div>

      {/* Labels */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '10px', color: '#71717a', fontFamily: 'JetBrains Mono' }}>
        <span>{trimStart.toFixed(2)}s</span>
        <span>{trimEnd.toFixed(2)}s</span>
      </div>
    </div>
  );
};
