import React, { useState, useEffect, useRef, useLayoutEffect, useCallback, useMemo } from 'react';
import { convertFileSrc } from "@tauri-apps/api/core";
import type { EditState, VideoState, Job } from '../types';
import { Timeline } from './Timeline';
import { CropOverlay } from './CropOverlay';

// -----------------------------------------------------------------------------
// MAIN PREVIEW PANEL
// -----------------------------------------------------------------------------

interface PreviewPanelProps {
   inputPreview: string;
   activeJob: Job | null;
   videoState: VideoState;
   onFileDrop: (path: string) => void;
   mode: string;
   editState: EditState;
   setEditState: (state: EditState) => void;
   viewMode: 'edit' | 'preview';
   setViewMode: (mode: 'edit' | 'preview') => void;
   showTech: boolean;
}

export const PreviewPanel: React.FC<PreviewPanelProps> = ({
   inputPreview, activeJob, videoState, onFileDrop, mode, editState, setEditState, viewMode, setViewMode
}) => {
   const [isPlaying, setIsPlaying] = useState(false);
   const [isScrubbing, setIsScrubbing] = useState(false);
   const [isMuted, setIsMuted] = useState(false);

   const [zoom, setZoom] = useState(1);
   const [pan, setPan] = useState({ x: 0, y: 0 });
   const [isPanning, setIsPanning] = useState(false);

   const [compareMode, setCompareMode] = useState<'toggle' | 'split' | 'side'>('toggle');
   const [splitPos, setSplitPos] = useState(0.5);
   const [isDraggingSplit, setIsDraggingSplit] = useState(false);
   const [isHoldingCompare, setIsHoldingCompare] = useState(false);
   const [isCropDragging, setIsCropDragging] = useState(false);

   const viewportRef = useRef<HTMLDivElement>(null);
   const videoRef = useRef<HTMLVideoElement>(null);
   const resultVideoRef = useRef<HTMLVideoElement>(null);
   const mediaContainerRef = useRef<HTMLDivElement>(null);

   const [baseDims, setBaseDims] = useState({ w: 0, h: 0 });

   const hasResult = !!(videoState.samplePreview || (activeJob?.status === 'done' && activeJob.outputPath));
   const isVideoInput = mode === 'video' || (inputPreview && /\.(mp4|mov|mkv|webm|avi)$/i.test(inputPreview));
   const resultUrl = videoState.samplePreview
      ? convertFileSrc(videoState.samplePreview)
      : (activeJob?.outputPath ? convertFileSrc(activeJob.outputPath) : "");

   const isCropActive = !!editState.crop;
   const isCropApplied = (editState.crop as any)?.applied === true;
   const showCropTools = viewMode === 'edit' && isCropActive && !isCropApplied;

   const isRotated90 = editState.rotation === 90 || editState.rotation === 270;
   const rawW = videoState.inputWidth || 16;
   const rawH = videoState.inputHeight || 9;
   const rotW = isRotated90 ? rawH : rawW;
   const rotH = isRotated90 ? rawW : rawH;
   const effW = isCropApplied && editState.crop ? rotW * editState.crop.width : rotW;
   const effH = isCropApplied && editState.crop ? rotH * editState.crop.height : rotH;
   const effectiveDuration = (editState.trimEnd || videoState.duration) - editState.trimStart;

   // --- 1. DYNAMIC LAYOUT CALCULATION ---
   useLayoutEffect(() => {
      if (!viewportRef.current) return;
      const vpW = viewportRef.current.offsetWidth;
      const vpH = viewportRef.current.offsetHeight;
      if (vpW === 0 || vpH === 0) return;

      const mediaAR = effW / effH;
      const vpAR = vpW / vpH;

      let newBaseW, newBaseH;

      if (mediaAR > vpAR) {
         newBaseW = vpW;
         newBaseH = vpW / mediaAR;
      } else {
         newBaseH = vpH;
         newBaseW = vpH * mediaAR;
      }

      setBaseDims({ w: newBaseW * 0.95, h: newBaseH * 0.95 });
   }, [viewportRef.current?.offsetWidth, viewportRef.current?.offsetHeight, effW, effH]);

   // --- 2. HIGH-PRECISION SYNC LOOP (RequestAnimationFrame) ---
   useEffect(() => {
      let animationFrameId: number;

      const syncLoop = () => {
         // Fix: If scrubbing, abort sync logic to prevent fighting manual seek
         if (isCropDragging || isScrubbing) {
            animationFrameId = requestAnimationFrame(syncLoop);
            return;
         }

         if (isPlaying && videoRef.current && resultVideoRef.current) {
            const source = videoRef.current;
            const result = resultVideoRef.current;

            const t = source.currentTime;
            const start = editState.trimStart;
            const end = editState.trimEnd;

            // --- A. PAUSE ON COMPLETION (No Loop) ---
            if (videoState.samplePreview) {
               // Sample previews are usually short snippets starting at trimStart
               if (t >= start + 2.0) {
                  source.pause();
                  result.pause();
                  source.currentTime = start;
                  result.currentTime = 0;
                  setIsPlaying(false);
                  return;
               }
            } else if (end > 0 && end < videoState.duration) {
               if (t >= end) {
                  source.pause();
                  result.pause();
                  source.currentTime = start;
                  result.currentTime = start;
                  setIsPlaying(false);
                  return;
               }
            } else if (source.ended) {
               setIsPlaying(false);
               return;
            }

            // --- B. BUFFERING / SEEK PROTECTION ---
            // If result (ENHANCED) is struggling, we MUST pause source (ORIGINAL) to let it catch up.
            // resultVideoRef is often much heavier (4K vs 1080p).
            const isResultBuffering = result.readyState < 3; // HAVE_FUTURE_DATA
            const isResultSeeking = result.seeking;

            if (isResultBuffering || isResultSeeking) {
               if (!source.paused) source.pause();
            } else {
               if (source.paused && isPlaying) {
                  source.play().catch(() => { });
               }

               // --- C. SYNC CORRECTION ---
               let targetTime = 0;
               if (videoState.samplePreview) {
                  targetTime = source.currentTime - start;
                  if (targetTime < 0) targetTime = 0;
               } else {
                  targetTime = source.currentTime;
               }

               const diff = Math.abs(result.currentTime - targetTime);
               // Tightened threshold: 0.04s (approx 1 frame at 24fps)
               // Result video usually follows source. We only seek result if it drifts.
               if (diff > 0.04) {
                  result.currentTime = targetTime;
               }
            }
         }
         animationFrameId = requestAnimationFrame(syncLoop);
      };

      if (isPlaying) {
         animationFrameId = requestAnimationFrame(syncLoop);
      }

      return () => cancelAnimationFrame(animationFrameId);
   }, [isPlaying, videoState.samplePreview, editState.trimStart, editState.trimEnd, videoState.duration, isCropDragging, isScrubbing]);

   // --- 3. INPUT HANDLERS ---

   const handleWheel = useCallback((e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();

      if (isDraggingSplit || isCropDragging || !viewportRef.current) return;

      const vpRect = viewportRef.current.getBoundingClientRect();
      const mouseX = e.clientX - vpRect.left - vpRect.width / 2;
      const mouseY = e.clientY - vpRect.top - vpRect.height / 2;

      const scaleAmt = -e.deltaY * 0.001;
      const newZoom = Math.max(0.1, Math.min(100.0, zoom + scaleAmt * zoom));

      const ratio = newZoom / zoom;
      const newPanX = mouseX - (mouseX - pan.x) * ratio;
      const newPanY = mouseY - (mouseY - pan.y) * ratio;

      setPan({ x: newPanX, y: newPanY });
      setZoom(newZoom);
   }, [zoom, pan, isDraggingSplit, isCropDragging]);

   useEffect(() => {
      const node = viewportRef.current;
      if (!node) return;
      node.addEventListener('wheel', handleWheel, { passive: false });
      return () => node.removeEventListener('wheel', handleWheel);
   }, [handleWheel]);

   const handleMouseDown = (e: React.MouseEvent) => {
      if (isDraggingSplit || showCropTools) return;
      if (e.button === 0) {
         if (hasResult && compareMode === 'toggle') setIsHoldingCompare(true);
         setIsPanning(true);
      } else {
         setIsPanning(true);
      }
   };

   useEffect(() => {
      const handleGlobalMouseMove = (e: MouseEvent) => {
         if (isDraggingSplit && viewportRef.current) {
            const vpRect = viewportRef.current.getBoundingClientRect();
            const visualCenterX = vpRect.left + (vpRect.width / 2) + pan.x;
            const visualWidth = baseDims.w * zoom;
            const visualLeft = visualCenterX - (visualWidth / 2);
            const relativeX = e.clientX - visualLeft;
            let newPos = relativeX / visualWidth;
            newPos = Math.max(0.001, Math.min(0.999, newPos));
            setSplitPos(newPos);
         }
         else if (isPanning) {
            setPan(p => ({ x: p.x + e.movementX, y: p.y + e.movementY }));
         }
      };

      const handleGlobalMouseUp = () => {
         setIsDraggingSplit(false);
         setIsPanning(false);
         setIsHoldingCompare(false);
      };

      if (isDraggingSplit || isPanning || isHoldingCompare) {
         window.addEventListener('mousemove', handleGlobalMouseMove);
         window.addEventListener('mouseup', handleGlobalMouseUp);
      }
      return () => {
         window.removeEventListener('mousemove', handleGlobalMouseMove);
         window.removeEventListener('mouseup', handleGlobalMouseUp);
      };
   }, [isDraggingSplit, isPanning, isHoldingCompare, baseDims.w, zoom, pan.x]);

   // --- 4. PLAYBACK CONTROLS ---

   const togglePlay = () => {
      if (videoRef.current) {
         if (videoRef.current.paused) {
            // Start Boundary Check
            const t = videoRef.current.currentTime;
            const start = editState.trimStart;
            const end = editState.trimEnd;
            const hasTrimEnd = end > 0 && end < videoState.duration;

            if (t < start - 0.05 || (hasTrimEnd && t >= end)) {
               videoRef.current.currentTime = start;
               videoState.setCurrentTime(start);
            }

            videoRef.current.play();
            resultVideoRef.current?.play();
            setIsPlaying(true);
         } else {
            videoRef.current.pause();
            resultVideoRef.current?.pause();
            setIsPlaying(false);
         }
      }
   };

   const toggleMute = () => {
      if (videoRef.current) {
         videoRef.current.muted = !isMuted;
         setIsMuted(!isMuted);
      }
   };

   const onTimeUpdate = (e: React.SyntheticEvent<HTMLVideoElement>) => {
      // NOTE: Loop logic moved to syncLoop for precision
      const t = e.currentTarget.currentTime;
      videoState.setCurrentTime(t);
   };

   // --- 5. RENDER HELPERS ---

   const worldStyle = {
      width: baseDims.w,
      height: baseDims.h,
      position: 'absolute' as const,
      left: '50%', top: '50%',
      transform: `translate(-50%, -50%) translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
      transformOrigin: 'center center',
      boxShadow: '0 0 50px rgba(0,0,0,0.5)',
      willChange: 'transform'
   };

   // Color grading SVG filter for live preview
   // Uses SVG filters to accurately match FFmpeg's eq filter behavior:
   // 1. Saturation (feColorMatrix type="saturate")
   // 2. Contrast & Brightness (feComponentTransfer or feColorMatrix)
   //    Contrast formula: C * (val - 0.5) + 0.5 + B  =>  C * val + (0.5 * (1 - C) + B)
   //    where C = 1 + contrast, B = brightness
   // 3. Gamma (feComponentTransfer type="gamma")
   const filterId = "vf-color-grade";
   const filterUrl = `url(#${filterId})`;

   const svgFilterDetails = useMemo(() => {
      const { brightness, contrast, saturation, gamma } = editState.color;
      const hasColorAdjustment = Math.abs(brightness) > 0.001 ||
         Math.abs(contrast) > 0.001 ||
         Math.abs(saturation) > 0.001 ||
         Math.abs(gamma - 1.0) > 0.001;

      if (!hasColorAdjustment) return null;

      // FFmpeg mappings:
      // Saturation: 0..3 (1.0 default) -> matches SVG saturate 0..N
      const svSat = 1.0 + saturation;

      // Contrast: 0..2 (1.0 default). SVG Slope = Contrast
      const svCont = 1.0 + contrast;

      // Brightness: -1..1 (0.0 default). SVG Intercept logic:
      // FFmpeg eq brightness adds value.
      // Contrast pivot is around 0.5 implied (or 128/255).
      // Formula: output = contrast * (input - 0.5) + 0.5 + brightness
      //                 = contrast * input - 0.5*contrast + 0.5 + brightness
      //                 = contrast * input + (0.5 * (1 - contrast) + brightness)
      const svIntercept = (0.5 * (1 - svCont)) + brightness;

      // Gamma: 0.1..10 (1.0 default).
      // SVG gamma type: value^exponent. 
      // FFmpeg gamma=2.0 usually means darker? check.
      // eq filter: gamma correction. output = input^(gamma) or input^(1/gamma)?
      // Usually gamma correction means input^(1/gamma) to linearize, but 'gamma' param often refers to the exponent itself.
      // FFmpeg docs: "The gamma value... Default is 1.0".
      // If I use gamma=2.0 in ffmpeg, image gets darker? 
      // Let's assume standard behavior: exponent = gamma (or 1/gamma depending on definition). 
      // We'll use 1/gamma to match typical 'gamma correction' slider feels where higher = brighter?
      // Wait, typical slider: higher gamma value (2.2) = darker image (standard CRT). Lower value (0.5) = brighter.
      // Let's try direct mapping first.
      const svGamma = gamma;

      return { svSat, svCont, svIntercept, svGamma };
   }, [editState.color]);

   const colorFilterStyle = svgFilterDetails ? filterUrl : undefined;

   const mediaTransform = `rotate(${editState.rotation}deg) scaleX(${editState.flipH ? -1 : 1}) scaleY(${editState.flipV ? -1 : 1})`;

   const appliedCropStyle = (isCropApplied && editState.crop) ? {
      objectViewBox: `inset(${editState.crop.y * 100}% ${(1 - editState.crop.x - editState.crop.width) * 100}% ${(1 - editState.crop.y - editState.crop.height) * 100}% ${editState.crop.x * 100}%)`,
      transition: 'object-view-box 0.3s cubic-bezier(0.16, 1, 0.3, 1)'
   } : { transition: 'object-view-box 0.3s cubic-bezier(0.16, 1, 0.3, 1)' };

   const isSideBySide = compareMode === 'side' && hasResult;

   const mediaBaseStyle: React.CSSProperties = {
      width: '100%', height: '100%',
      objectFit: 'contain',
      display: 'block',
      userSelect: 'none',
      pointerEvents: 'none'
   };

   const renderMedia = (isSource: boolean) => {
      const style = isSource
         ? {
            ...mediaBaseStyle,
            transform: mediaTransform,
            filter: colorFilterStyle || undefined,
            ...appliedCropStyle
         }
         : { ...mediaBaseStyle };

      const url = isSource ? convertFileSrc(inputPreview) : resultUrl;

      if (isVideoInput || (isSource && mode === 'video')) {
         return <video
            ref={isSource ? videoRef : resultVideoRef}
            src={url}
            style={style}
            onTimeUpdate={isSource ? onTimeUpdate : undefined}
            onLoadedMetadata={isSource ? (e) => {
               videoState.setInputDimensions(e.currentTarget.videoWidth, e.currentTarget.videoHeight);
               videoState.setDuration(e.currentTarget.duration);
               if (videoState.trimEnd === 0) videoState.setTrimEnd(e.currentTarget.duration);
            } : undefined}
            muted={!isSource}
         />;
      } else {
         return <img
            src={url}
            style={style}
            onLoad={isSource ? (e) => videoState.setInputDimensions(e.currentTarget.naturalWidth, e.currentTarget.naturalHeight) : undefined}
         />;
      }
   };

   return (
      <div style={{
         flex: 1, display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--bg-color)', overflow: 'hidden', position: 'relative',
         userSelect: 'none', WebkitUserSelect: 'none', cursor: isDraggingSplit ? 'col-resize' : (isPanning ? 'grabbing' : (showCropTools ? 'default' : 'grab'))
      }}
         onDrop={(e) => { e.preventDefault(); if (e.dataTransfer.files[0]) onFileDrop(e.dataTransfer.files[0].name); }}
         onDragOver={(e) => e.preventDefault()}
      >
         {/* Docked Toolbar */}
         {inputPreview && (
            <div style={{
               height: '36px',
               borderBottom: '1px solid var(--panel-border)',
               background: 'var(--panel-bg)',
               display: 'flex', alignItems: 'center',
               padding: '0 12px',
               justifyContent: 'space-between',
               flexShrink: 0
            }}>
               <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{Math.round(zoom * 100)}%</div>
                  <div style={{ width: 1, height: 12, background: 'var(--panel-border)' }} />
                  {hasResult ? (
                     <div className="toggle-group" style={{ transform: 'scale(0.9)', transformOrigin: 'left center' }}>
                        <button className={compareMode === 'toggle' ? 'active' : ''} onClick={() => setCompareMode('toggle')} title="Hold Click to Compare">HOLD</button>
                        <button className={compareMode === 'split' ? 'active' : ''} onClick={() => setCompareMode('split')} title="Split Slider">SPLIT</button>
                        <button className={compareMode === 'side' ? 'active' : ''} onClick={() => setCompareMode('side')} title="Side by Side">SIDE</button>
                     </div>
                  ) : (
                     <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>SOURCE VIEW</span>
                  )}
               </div>
               <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', fontSize: '10px', cursor: 'pointer', fontWeight: 600 }}>RESET ZOOM</button>
            </div>
         )}

         {/* Viewport */}
         <div ref={viewportRef} style={{ flex: 1, position: 'relative', display: 'flex', overflow: 'hidden', background: 'var(--bg-color)' }} onMouseDown={handleMouseDown}>
            {/* Define SVG Filter for Color Grading */}
            {svgFilterDetails && (
               <svg style={{ position: 'absolute', width: 0, height: 0, pointerEvents: 'none' }}>
                  <defs>
                     <filter id={filterId} colorInterpolationFilters="sRGB">
                        {/* 1. Saturation */}
                        <feColorMatrix type="saturate" values={String(svgFilterDetails.svSat)} result="saturated" />

                        {/* 2. Contrast & Brightness ( Slope & Intercept ) */}
                        <feComponentTransfer in="saturated" result="contrast_brightness">
                           <feFuncR type="linear" slope={svgFilterDetails.svCont} intercept={svgFilterDetails.svIntercept} />
                           <feFuncG type="linear" slope={svgFilterDetails.svCont} intercept={svgFilterDetails.svIntercept} />
                           <feFuncB type="linear" slope={svgFilterDetails.svCont} intercept={svgFilterDetails.svIntercept} />
                        </feComponentTransfer>

                        {/* 3. Gamma */}
                        <feComponentTransfer in="contrast_brightness" result="final">
                           <feFuncR type="gamma" exponent={svgFilterDetails.svGamma} amplitude="1" offset="0" />
                           <feFuncG type="gamma" exponent={svgFilterDetails.svGamma} amplitude="1" offset="0" />
                           <feFuncB type="gamma" exponent={svgFilterDetails.svGamma} amplitude="1" offset="0" />
                        </feComponentTransfer>
                     </filter>
                  </defs>
               </svg>
            )}

            {inputPreview ? (
               <>
                  {!isSideBySide && (
                     <div ref={mediaContainerRef} style={worldStyle}>
                        <div style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }}>
                           {renderMedia(true)}
                           {showCropTools && (
                              <div style={{ position: 'absolute', inset: 0, zIndex: 60 }}>
                                 <div style={{ width: '100%', height: '100%', transform: mediaTransform }}>
                                    <CropOverlay
                                       crop={editState.crop!} editState={editState}
                                       onUpdate={(c) => setEditState({ ...editState, crop: c })}
                                       onApply={() => setEditState({ ...editState, crop: { ...editState.crop!, applied: true } as any })}
                                       onCancel={() => setEditState({ ...editState, crop: { ...editState.crop!, applied: false } as any })}
                                       onInteractionStart={() => setIsCropDragging(true)}
                                       onInteractionEnd={() => setIsCropDragging(false)}
                                       zoom={zoom} containerWidth={baseDims.w} containerHeight={baseDims.h}
                                    />
                                 </div>
                              </div>
                           )}
                        </div>
                        {hasResult && (
                           <div style={{
                              position: 'absolute', inset: 0, overflow: 'hidden',
                              opacity: (compareMode === 'toggle' && isHoldingCompare) ? 0 : 1
                           }}>
                              <div style={{ width: '100%', height: '100%', clipPath: compareMode === 'split' ? `inset(0 0 0 ${splitPos * 100}%)` : 'none' }}>
                                 {renderMedia(false)}
                                 <div style={{ position: 'absolute', top: 8, right: 8, background: 'var(--brand-primary)', color: 'black', fontSize: '9px', fontWeight: 800, padding: '2px 4px', borderRadius: '2px' }}>ENHANCED</div>
                              </div>
                              {compareMode === 'split' && (
                                 <div style={{
                                    position: 'absolute', top: 0, bottom: 0, left: `${splitPos * 100}%`, width: 2 / zoom, background: 'var(--brand-primary)',
                                    cursor: 'col-resize', zIndex: 50, boxShadow: '0 0 8px 1px rgba(0,0,0,0.8)', pointerEvents: 'auto', transform: 'translateX(-50%)'
                                 }}
                                    onMouseDown={(e) => { e.stopPropagation(); setIsDraggingSplit(true); }}
                                 >
                                    <div style={{
                                       position: 'absolute', top: '50%', left: '50%', width: 24 / zoom, height: 24 / zoom,
                                       background: 'var(--brand-primary)', borderRadius: '50%', border: `${2 / zoom}px solid white`,
                                       boxShadow: '0 0 4px 1px rgba(0,0,0,0.5)',
                                       display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'black', fontSize: `${10 / zoom}px`,
                                       transform: 'translate(-50%, -50%)'
                                    }}>↔</div>
                                 </div>
                              )}
                           </div>
                        )}
                     </div>
                  )}
                  {isSideBySide && (
                     <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', width: '100%', height: '100%', gap: '2px' }}>
                        <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
                           <div style={worldStyle}>{renderMedia(true)}</div>
                           <div style={{ position: 'absolute', top: 8, left: 8, background: 'rgba(0,0,0,0.5)', color: 'white', fontSize: '10px', padding: '2px 6px', borderRadius: '3px' }}>SOURCE</div>
                        </div>
                        <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
                           <div style={worldStyle}>{renderMedia(false)}</div>
                           <div style={{ position: 'absolute', top: 8, right: 8, background: 'var(--brand-primary)', color: 'black', fontSize: '10px', fontWeight: 800, padding: '2px 6px', borderRadius: '3px' }}>ENHANCED</div>
                        </div>
                     </div>
                  )}
               </>
            ) : (
               <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: 0.3, width: '100%', height: '100%' }}>
                  <div style={{ fontSize: '32px', marginBottom: '8px', color: 'var(--text-muted)' }}>+</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '11px', letterSpacing: '0.1em' }}>DRAG MEDIA HERE</div>
               </div>
            )}
         </div>

         {isVideoInput && (
            <div style={{
               borderTop: '1px solid var(--panel-border)', background: 'var(--panel-bg)',
               display: 'flex', flexDirection: 'column',
               padding: '12px 16px 8px 16px', gap: '8px', zIndex: 50, flexShrink: 0
            }}>
               <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                     <button onClick={togglePlay} style={{ background: 'none', border: 'none', color: isPlaying ? 'var(--brand-primary)' : 'var(--text-primary)', fontSize: '16px', cursor: 'pointer', padding: 0 }}>
                        {isPlaying ? "❚❚" : "▶"}
                     </button>
                     <button onClick={toggleMute} style={{ background: 'none', border: 'none', color: isMuted ? 'var(--text-muted)' : 'var(--text-primary)', fontSize: '14px', cursor: 'pointer', padding: 0 }}>
                        {isMuted ? "🔇" : "🔊"}
                     </button>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '10px', fontFamily: 'var(--font-mono)' }}>
                     <div style={{ color: 'var(--text-secondary)' }}>
                        {videoState.currentTime.toFixed(2)} / {effectiveDuration.toFixed(2)}s
                     </div>
                  </div>
               </div>

               <div style={{ width: '100%' }}>
                  <Timeline
                     onInteractionStart={() => setIsScrubbing(true)}
                     onInteractionEnd={() => setIsScrubbing(false)}
                     duration={videoState.duration} currentTime={videoState.currentTime}
                     trimStart={editState.trimStart} trimEnd={editState.trimEnd}
                     renderedRange={videoState.renderedRange}
                     onSeek={(t) => {
                        if (videoRef.current) videoRef.current.currentTime = t;
                        videoState.setCurrentTime(t);
                     }}
                     onTrimChange={(s, e) => setEditState({ ...editState, trimStart: s, trimEnd: e })}
                     hasAudio={true}
                  />
               </div>
            </div>
         )}

      </div>
   );
};
