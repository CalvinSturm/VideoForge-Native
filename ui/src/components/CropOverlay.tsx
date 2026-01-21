import React, { useState, useRef, useEffect } from 'react';
import type { EditState } from '../types';

interface CropOverlayProps {
  crop: NonNullable<EditState['crop']>;
  editState: EditState;
  onUpdate: (newCrop: NonNullable<EditState['crop']>) => void;
  onApply: () => void;
  onCancel: () => void;
  onInteractionStart: () => void;
  onInteractionEnd: () => void;
  zoom: number;
  containerWidth: number;
  containerHeight: number;
}

export const CropOverlay: React.FC<CropOverlayProps> = ({
  crop, editState, onUpdate, onApply, onCancel,
  onInteractionStart, onInteractionEnd, zoom,
  containerWidth, containerHeight
}) => {
  const [dragMode, setDragMode] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const startPos = useRef({ x: 0, y: 0 });
  const startCrop = useRef(crop);

  const isRatioLocked = !!editState.aspectRatio;

  const getNormalizedDelta = (e: MouseEvent) => {
    if (containerWidth === 0 || containerHeight === 0) return { x: 0, y: 0 };

    const screenDx = e.clientX - startPos.current.x;
    const screenDy = e.clientY - startPos.current.y;

    const nx = (screenDx / zoom) / containerWidth;
    const ny = (screenDy / zoom) / containerHeight;

    let mx = nx;
    let my = ny;

    switch (editState.rotation) {
      case 90:  mx = ny; my = -nx; break;
      case 180: mx = -nx; my = -ny; break;
      case 270: mx = -ny; my = nx; break;
      default: break;
    }
    if (editState.flipH) mx = -mx;
    if (editState.flipV) my = -my;

    return { x: mx, y: my };
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragMode) return;

      const d = getNormalizedDelta(e);
      const sc = startCrop.current;
      let nc = { ...sc };

      if (dragMode === 'move') {
        nc.x = Math.max(0, Math.min(1 - nc.width, sc.x + d.x));
        nc.y = Math.max(0, Math.min(1 - nc.height, sc.y + d.y));
        onUpdate(nc);
        return;
      }

      const dirX = dragMode.includes('e') ? 1 : (dragMode.includes('w') ? -1 : 0);
      const dirY = dragMode.includes('s') ? 1 : (dragMode.includes('n') ? -1 : 0);

      let newW = sc.width + (d.x * (dragMode.includes('w') ? -1 : 1) * (dirX === 0 ? 0 : 1));
      let newH = sc.height + (d.y * (dragMode.includes('n') ? -1 : 1) * (dirY === 0 ? 0 : 1));

      if (editState.aspectRatio) {
        const normRatio = editState.aspectRatio * (containerHeight / containerWidth);
        newH = newW / normRatio;
      }

      newW = Math.max(0.05, newW);
      newH = Math.max(0.05, newH);

      const clampW = () => {
         const maxW = dirX === 1 ? (1 - sc.x) : (sc.x + sc.width);
         if (newW > maxW) {
            newW = maxW;
            if (editState.aspectRatio) {
               const normRatio = editState.aspectRatio * (containerHeight / containerWidth);
               newH = newW / normRatio;
            }
         }
      };

      const clampH = () => {
         const maxH = dirY === 1 ? (1 - sc.y) : (sc.y + sc.height);
         if (newH > maxH) {
            newH = maxH;
            if (editState.aspectRatio) {
               const normRatio = editState.aspectRatio * (containerHeight / containerWidth);
               newW = newH * normRatio;
            }
         }
      };

      if (editState.aspectRatio) {
         clampW(); clampH(); clampW();
      } else {
         clampW(); clampH();
      }

      if (dirX === 1) nc.width = newW;
      if (dirX === -1) { nc.x = (sc.x + sc.width) - newW; nc.width = newW; }

      if (dirY === 1) nc.height = newH;
      if (dirY === -1) { nc.y = (sc.y + sc.height) - newH; nc.height = newH; }

      onUpdate(nc);
    };

    const handleMouseUp = () => {
      if (dragMode) {
        setDragMode(null);
        onInteractionEnd();
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Enter') onApply();
      if (e.key === 'Escape') onCancel();
    };

    if (dragMode) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [dragMode, editState, onUpdate, onApply, onCancel, zoom, containerWidth, containerHeight]);

  const startDrag = (e: React.MouseEvent, mode: string) => {
    e.stopPropagation();
    e.preventDefault();
    setDragMode(mode);
    onInteractionStart();
    startPos.current = { x: e.clientX, y: e.clientY };
    startCrop.current = crop;
  };

  const hitBoxStyle: React.CSSProperties = {
    position: 'absolute', width: '24px', height: '24px',
    transform: `translate(-50%, -50%) scale(${1/zoom})`,
    zIndex: 30, display: 'flex', alignItems: 'center', justifyContent: 'center',
    cursor: 'pointer', pointerEvents: 'auto'
  };

  const handleStyle: React.CSSProperties = {
    width: '8px', height: '8px', background: 'var(--brand-primary)',
    border: '1px solid black', boxShadow: '0 0 2px rgba(0,0,0,0.5)'
  };

  const RenderHandle = ({ top, left, cursor, mode }: { top: string, left: string, cursor: string, mode: string }) => (
    <div style={{ ...hitBoxStyle, top, left, cursor }} onMouseDown={(e) => startDrag(e, mode)}>
      <div style={handleStyle} />
    </div>
  );

  return (
    <>
      <div style={{
        position: 'absolute', top: -30, left: '50%', transform: 'translateX(-50%)',
        background: 'var(--brand-primary)', color: 'black', padding: '4px 8px',
        borderRadius: '4px', fontSize: '10px', fontWeight: 'bold', pointerEvents: 'none',
        boxShadow: '0 2px 8px rgba(0,0,0,0.5)', whiteSpace: 'nowrap', zIndex: 100
      }}>
        ENTER: APPLY / ESC: CANCEL
      </div>

      <div
        ref={containerRef}
        style={{
          position: 'absolute',
          left: `${crop.x * 100}%`, top: `${crop.y * 100}%`,
          width: `${crop.width * 100}%`, height: `${crop.height * 100}%`,
          border: `2px solid var(--brand-primary)`,
          boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.65)',
          cursor: 'move', touchAction: 'none'
        }}
        onMouseDown={(e) => startDrag(e, 'move')}
      >
        <RenderHandle top="0%" left="0%" cursor="nw-resize" mode="nw" />
        <RenderHandle top="0%" left="100%" cursor="ne-resize" mode="ne" />
        <RenderHandle top="100%" left="0%" cursor="sw-resize" mode="sw" />
        <RenderHandle top="100%" left="100%" cursor="se-resize" mode="se" />

        {!isRatioLocked && (
          <>
            <RenderHandle top="0%" left="50%" cursor="n-resize" mode="n" />
            <RenderHandle top="100%" left="50%" cursor="s-resize" mode="s" />
            <RenderHandle top="50%" left="0%" cursor="w-resize" mode="w" />
            <RenderHandle top="50%" left="100%" cursor="e-resize" mode="e" />
          </>
        )}
      </div>
    </>
  );
};
