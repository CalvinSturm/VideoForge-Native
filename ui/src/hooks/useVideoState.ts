import { useState, useMemo, useCallback } from "react";
import type { UpscaleMode, VideoState, EditState } from "../types";

const INITIAL_EDIT_STATE: EditState = {
    trimStart: 0, trimEnd: 0, crop: null, rotation: 0, flipH: false, flipV: false, fps: 0,
    color: { brightness: 0, contrast: 0, saturation: 0, gamma: 1.0 }
};

export function useVideoState() {
    const [mode, setMode] = useState<UpscaleMode>("image");
    const [inputPath, setInputPath] = useState("");
    const [outputPath, setOutputPath] = useState("");
    const [previewFile, setPreviewFile] = useState<string | null>(null);
    const [viewMode, setViewMode] = useState<'edit' | 'preview'>('edit');

    const [editState, setEditState] = useState<EditState>(INITIAL_EDIT_STATE);
    const [inputDims, setInputDims] = useState({ w: 0, h: 0 });

    const [videoTime, setVideoTime] = useState(0);
    const [videoDuration, setVideoDuration] = useState(0);
    const [renderedRange, setRenderedRange] = useState<{ start: number; end: number } | null>(null);

    const handleNewInput = useCallback((path: string) => {
        setInputPath(path);
        if (/\.(mp4|mkv|mov|avi|webm)$/i.test(path)) {
            setMode('video');
        } else {
            setMode('image');
        }
        setEditState(INITIAL_EDIT_STATE);
        setVideoTime(0); setVideoDuration(0); setInputDims({ w: 0, h: 0 });
        setViewMode('edit'); setOutputPath(""); setPreviewFile(null);
        setRenderedRange(null);
    }, []);

    const videoState: VideoState = useMemo(() => ({
        src: inputPath,
        currentTime: videoTime,
        setCurrentTime: setVideoTime,
        duration: videoDuration,
        setDuration: setVideoDuration,
        inputWidth: inputDims.w,
        inputHeight: inputDims.h,
        setInputDimensions: (w: number, h: number) => setInputDims({ w, h }),
        trimStart: editState.trimStart,
        trimEnd: editState.trimEnd,
        setTrimStart: (t: number) => setEditState(p => ({ ...p, trimStart: t })),
        setTrimEnd: (t: number) => setEditState(p => ({ ...p, trimEnd: t })),
        crop: { x: 0, y: 0, width: 0, height: 0 },
        setCrop: () => { },
        samplePreview: previewFile,
        renderSample: () => { },   // Overridden by App when wiring to useUpscaleJob
        clearPreview: () => setPreviewFile(null),
        renderedRange
    }), [inputPath, videoTime, videoDuration, editState, inputDims, previewFile, renderedRange]);

    const getRustEditConfig = useCallback(() => ({
        trim_start: editState.trimStart,
        trim_end: editState.trimEnd,
        crop: editState.crop,
        rotation: editState.rotation,
        flip_h: editState.flipH,
        flip_v: editState.flipV,
        fps: editState.fps,
        color: editState.color
    }), [editState]);

    return {
        mode, setMode,
        inputPath, setInputPath,
        outputPath, setOutputPath,
        previewFile, setPreviewFile,
        viewMode, setViewMode,
        editState, setEditState,
        inputDims,
        videoTime, videoDuration,
        renderedRange, setRenderedRange,
        videoState,
        handleNewInput,
        getRustEditConfig,
    };
}
