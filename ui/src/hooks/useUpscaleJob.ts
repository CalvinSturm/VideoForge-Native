import { useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
    Job,
    RavePolicy,
    RaveCommandJson,
    NativeUpscaleResultJson,
    ModelInfo
} from "../types";
import type { useVideoState } from "./useVideoState";
import type { useRaveIntegration } from "./useRaveIntegration";
import { useJobStore } from "../Store/useJobStore";
import type { PanelId } from "../Store/viewLayoutStore";

interface UseUpscaleJobOptions {
    // Video state
    video: ReturnType<typeof useVideoState>;
    // Model state
    model: string;
    modelInfoMap: Map<string, ModelInfo>;
    availableModels: string[];
    showTechSpecs: boolean;
    // RAVE integration
    rave: ReturnType<typeof useRaveIntegration>;
    // Job state
    jobs: Job[];
    setJobs: React.Dispatch<React.SetStateAction<Job[]>>;
    activeJob: Job | null;
    setActiveJob: React.Dispatch<React.SetStateAction<Job | null>>;
    // Logs / toasts
    setLogs: React.Dispatch<React.SetStateAction<string[]>>;
    addToast: (msg: string, type: string) => void;
    // Panel control
    panels: Record<PanelId, boolean>;
    openPanel: (id: PanelId) => void;
}

// Helper: Extract scale factor from model string (robust fallback)
function getScaleFromModel(modelId?: string): number {
    if (!modelId) return 4;
    const match = modelId.match(/x(\d)/);
    return match?.[1] ? parseInt(match[1], 10) : 4;
}

function inferNativePrecision(modelPath: string): "fp16" | "fp32" {
    const lower = modelPath.toLowerCase();
    if (lower.includes("_fp16") || lower.includes("-fp16") || lower.includes("half")) {
        return "fp16";
    }
    return "fp32";
}

export function useUpscaleJob(opts: UseUpscaleJobOptions) {
    const {
        video, model, modelInfoMap, showTechSpecs, rave,
        jobs, setJobs, activeJob, setActiveJob,
        setLogs, addToast, panels, openPanel,
    } = opts;

    const { setIsProcessing, setLastOutputPath, upscaleConfig } = useJobStore();

    // ── Start Upscale ────────────────────────────────────────────────────────

    const startUpscale = useCallback(async () => {
        if (!video.inputPath) return addToast("Select an input file first!", "error");
        if (!upscaleConfig.isEnabled) {
            return addToast("AI Upscale is bypassed. Enable it to upscale.", "warning");
        }

        const jobId = Date.now().toString();
        const newJob: Job = {
            id: jobId,
            command: `Upscale: ${video.inputPath.split(/[/\\]/).pop()}`,
            status: "running", progress: 0, statusMessage: "Initializing...",
            paused: false, eta: 0, startedAt: Date.now()
        };
        setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);
        if (!panels.QUEUE) openPanel('QUEUE');

        const activeScale = upscaleConfig.scaleFactor || getScaleFromModel(model);
        const upscalePayload = {
            inputPath: video.inputPath,
            outputPath: video.outputPath,
            model: upscaleConfig.primaryModelId || model,
            editConfig: video.getRustEditConfig(),
            scale: activeScale,
            architectureClass: upscaleConfig.architectureClass,
            secondaryModel: null,
            blendAlpha: 0,
            resolutionMode: upscaleConfig.resolutionMode,
            targetWidth: upscaleConfig.resolutionMode === 'target' ? upscaleConfig.targetWidth : null,
            targetHeight: upscaleConfig.resolutionMode === 'target' ? upscaleConfig.targetHeight : null,
        };

        try {
            const selectedModel = upscaleConfig.primaryModelId || model;
            const info = modelInfoMap.get(selectedModel);

            let resultPath: string;
            let policy: RavePolicy | undefined;
            let hostCopyAuditEnabled: boolean | undefined;
            let hostCopyAuditDisableReason: string | null | undefined;
            let nativeEngine: string | undefined;
            let encoderMode: string | undefined;
            let encoderDetail: string | null | undefined;
            let framesProcessed: number | undefined;
            let audioPreserved: boolean | undefined;
            const canUseNative = upscaleConfig.useNativeEngine && video.mode === 'video';

            if (canUseNative) {
                if (info?.format === "onnx") {
                    const envReady = await rave.ensureRaveEnvironmentReady();
                    if (!envReady) throw new Error("RAVE environment is not ready for native video upscale.");

                    const resolvedOutputPath = video.outputPath?.trim() ? video.outputPath : rave.defaultRaveOutputPath(video.inputPath);

                    if (showTechSpecs) {
                        try {
                            const benchmark = await invoke<RaveCommandJson>("rave_benchmark", {
                                args: rave.buildRaveBenchmarkArgs({ input: video.inputPath, modelPath: info.path, maxBatch: upscaleConfig.maxBatch, architectureClass: upscaleConfig.architectureClass }),
                                strictAudit: true, mockRun: false, uiOptIn: true
                            });
                            setLogs(prev => [...prev, `[RAVE] benchmark dry-run fps=${String((benchmark as any).fps ?? "n/a")} policy=${JSON.stringify(benchmark.policy ?? {})}`]);
                        } catch (benchErr) {
                            setLogs(prev => [...prev, `[RAVE] benchmark dry-run failed: ${benchErr}`]);
                        }
                    }

                    const nativeResult = await invoke<NativeUpscaleResultJson>("upscale_request_native", {
                        inputPath: video.inputPath,
                        outputPath: resolvedOutputPath,
                        modelPath: info.path,
                        scale: activeScale,
                        precision: inferNativePrecision(info.path),
                        audio: true,
                        maxBatch: upscaleConfig.maxBatch
                    });
                    if (!nativeResult.output_path || typeof nativeResult.output_path !== "string") throw new Error("upscale_request_native did not return a valid output path");
                    resultPath = nativeResult.output_path;
                    nativeEngine = nativeResult.engine;
                    encoderMode = nativeResult.encoder_mode;
                    encoderDetail = nativeResult.encoder_detail ?? null;
                    framesProcessed = nativeResult.frames_processed;
                    audioPreserved = nativeResult.audio_preserved;
                    setLogs(prev => [
                        ...prev,
                        `[NATIVE] engine=${nativeResult.engine} encoder_mode=${nativeResult.encoder_mode} frames=${nativeResult.frames_processed}`
                            + (nativeResult.encoder_detail ? ` detail=${nativeResult.encoder_detail}` : "")
                    ]);
                } else {
                    addToast(`Native engine requires an ONNX model. "${selectedModel}" is PyTorch-only — using Python pipeline.`, "warning");
                    resultPath = await invoke<string>("upscale_request", upscalePayload);
                }
            } else {
                resultPath = await invoke<string>("upscale_request", upscalePayload);
            }

            setLogs(prev => [...prev, `[SYSTEM] Job ${jobId} finished.`]);
            const finishedJob: Job = {
                ...newJob, status: 'done', progress: 100, outputPath: resultPath, eta: 0, completedAt: Date.now(),
                ...(policy ? { policy } : {}),
                ...(typeof hostCopyAuditEnabled === "boolean" ? { hostCopyAuditEnabled } : {}),
                ...(hostCopyAuditDisableReason !== undefined ? { hostCopyAuditDisableReason } : {}),
                ...(nativeEngine ? { nativeEngine } : {}),
                ...(encoderMode ? { encoderMode } : {}),
                ...(encoderDetail !== undefined ? { encoderDetail } : {}),
                ...(typeof framesProcessed === "number" ? { framesProcessed } : {}),
                ...(typeof audioPreserved === "boolean" ? { audioPreserved } : {})
            };
            setJobs(prev => prev.map(j => j.id === jobId ? finishedJob : j));
            setActiveJob(finishedJob);
            setLastOutputPath(resultPath);
        } catch (err) {
            const normalized = rave.parseRaveError(err);
            const msg = `[${normalized.category}] ${normalized.message}`;
            addToast(`Error: ${msg}`, "error");
            setLogs(prev => [...prev, `[ERROR][RAVE][${normalized.category}] Job ${jobId} failed: ${normalized.message}`]);
            setJobs(prev => prev.map(j => j.id === jobId ? {
                ...j, status: 'error' as const, statusMessage: msg, errorCategory: normalized.category,
                ...(normalized.nextAction ? { errorHint: normalized.nextAction } : {}),
                errorMessage: normalized.detail ? `${normalized.message} :: ${normalized.detail}` : normalized.message,
                completedAt: Date.now()
            } : j));
        } finally { setIsProcessing(false); }
    }, [video, model, modelInfoMap, showTechSpecs, rave, upscaleConfig, setJobs, setActiveJob, setIsProcessing, setLastOutputPath, setLogs, addToast, panels, openPanel]);

    // ── Export (Transcode only) ──────────────────────────────────────────────

    const onExportEdited = useCallback(async () => {
        if (!video.inputPath) return addToast("Select input first!", "error");
        const jobId = Date.now().toString();
        const newJob: Job = {
            id: jobId,
            command: `Transcode: ${video.inputPath.split(/[/\\]/).pop()}`,
            status: "running", progress: 0, statusMessage: "Encoding...",
            paused: false, eta: 0, startedAt: Date.now()
        };
        setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);
        if (!panels.QUEUE) openPanel('QUEUE');

        try {
            const resultPath = await invoke<string>("export_request", {
                inputPath: video.inputPath, outputPath: video.outputPath,
                editConfig: video.getRustEditConfig(), scale: 1
            });
            setLogs(prev => [...prev, `[SYSTEM] Export ${jobId} complete.`]);
            addToast("Export Completed", "success");
            const finishedJob: Job = { ...newJob, status: 'done', progress: 100, outputPath: resultPath, eta: 0, completedAt: Date.now() };
            setJobs(prev => prev.map(j => j.id === jobId ? finishedJob : j));
            setActiveJob(finishedJob);
            setLastOutputPath(resultPath);
        } catch (err) {
            const normalized = rave.parseRaveError(err);
            const msg = `[${normalized.category}] ${normalized.message}`;
            addToast(`Error: ${msg}`, "error");
            setLogs(prev => [...prev, `[ERROR][${normalized.category}] Export ${jobId} failed: ${normalized.message}`]);
            setJobs(prev => prev.map(j => j.id === jobId ? {
                ...j, status: 'error' as const, statusMessage: msg, errorCategory: normalized.category,
                ...(normalized.nextAction ? { errorHint: normalized.nextAction } : {}),
                errorMessage: normalized.detail ? `${normalized.message} :: ${normalized.detail}` : normalized.message,
                completedAt: Date.now()
            } : j));
        } finally { setIsProcessing(false); }
    }, [video, rave, setJobs, setActiveJob, setIsProcessing, setLastOutputPath, setLogs, addToast, panels, openPanel]);

    // ── RAVE Validate ────────────────────────────────────────────────────────

    const startRaveValidate = useCallback(async () => {
        if (!video.inputPath) return addToast("Select an input file first!", "error");
        const jobId = `validate_${Date.now().toString()}`;
        const newJob: Job = {
            id: jobId, command: "Validate: production_strict (mock)",
            status: "running", progress: 0, statusMessage: "Validating strict policy...",
            paused: false, eta: 0, startedAt: Date.now()
        };
        setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);
        if (!panels.QUEUE) openPanel('QUEUE');

        try {
            const result = await invoke<RaveCommandJson>("rave_validate", {
                fixture: null, profile: "production_strict", bestEffort: true, strictAudit: true, mockRun: true
            });
            const policy = result.policy;
            const hostCopyAuditEnabled = result.host_copy_audit_enabled;
            const hostCopyAuditDisableReason = result.host_copy_audit_disable_reason;
            const skipped = (result as any).skipped === true;

            const finishedJob: Job = {
                ...newJob, status: "done", progress: 100,
                statusMessage: skipped ? "Validation skipped" : "Validation passed",
                completedAt: Date.now(),
                ...(policy ? { policy } : {}),
                ...(typeof hostCopyAuditEnabled === "boolean" ? { hostCopyAuditEnabled } : {}),
                ...(hostCopyAuditDisableReason !== undefined ? { hostCopyAuditDisableReason } : {})
            };
            setJobs(prev => prev.map(j => j.id === jobId ? finishedJob : j));
            setActiveJob(finishedJob);
            setLogs(prev => [...prev, `[RAVE] validate ok=${String((result as any).ok ?? "unknown")} skipped=${String(skipped)} policy=${JSON.stringify(policy ?? {})}`]);
            addToast(skipped ? "Validate completed (skipped)" : "Validate passed", skipped ? "warning" : "success");
        } catch (err) {
            const normalized = rave.parseRaveError(err);
            setJobs(prev => prev.map(j => j.id === jobId ? {
                ...j, status: "error" as const,
                statusMessage: `[${normalized.category}] ${normalized.message}`,
                errorCategory: normalized.category,
                ...(normalized.nextAction ? { errorHint: normalized.nextAction } : {}),
                errorMessage: normalized.detail ? `${normalized.message} :: ${normalized.detail}` : normalized.message,
                completedAt: Date.now()
            } : j));
            setLogs(prev => [...prev, `[RAVE][${normalized.category}] validate failed: ${normalized.message}`]);
            addToast(`Validate failed: [${normalized.category}] ${normalized.message}`, "error");
        } finally { setIsProcessing(false); }
    }, [video, rave, setJobs, setActiveJob, setIsProcessing, setLogs, addToast, panels, openPanel]);

    // ── Preview Sample ───────────────────────────────────────────────────────

    const renderPreviewSample = useCallback(async () => {
        if (!video.inputPath || video.mode !== 'video') return;
        addToast("Rendering 2s Sample...", "info");
        video.setPreviewFile(null);
        const start = Math.max(0, video.videoTime);
        const safeDuration = video.videoDuration > 0 ? video.videoDuration : 1000;
        const end = Math.min(safeDuration, start + 2.0);
        const previewConfig = { ...video.getRustEditConfig(), trim_start: start, trim_end: end };
        const activeScale = upscaleConfig.scaleFactor || getScaleFromModel(model);
        const jobId = "preview_" + Date.now().toString().slice(-6);
        const newJob: Job = {
            id: jobId, command: `PREVIEW SAMPLE`,
            status: "running", progress: 0, statusMessage: "Rendering...",
            paused: false, eta: 0, startedAt: Date.now()
        };
        setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);

        const previewPayload = {
            inputPath: video.inputPath, outputPath: "",
            model: upscaleConfig.primaryModelId || model,
            editConfig: previewConfig, scale: activeScale,
            architectureClass: upscaleConfig.architectureClass,
            secondaryModel: null, blendAlpha: 0,
            resolutionMode: upscaleConfig.resolutionMode,
            targetWidth: upscaleConfig.resolutionMode === 'target' ? upscaleConfig.targetWidth : null,
            targetHeight: upscaleConfig.resolutionMode === 'target' ? upscaleConfig.targetHeight : null,
        };

        try {
            const resultPath = await invoke<string>("upscale_request", previewPayload);
            video.setPreviewFile(resultPath);
            video.setRenderedRange({ start, end });
            addToast("Sample ready.", "success");
            const finishedJob: Job = { ...newJob, status: 'done', progress: 100, outputPath: resultPath, eta: 0, completedAt: Date.now() };
            setJobs(prev => prev.map(j => j.id === jobId ? finishedJob : j));
            setActiveJob(finishedJob);
        } catch (err) {
            addToast(`Sample failed: ${err}`, "error");
            setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'error' as const, statusMessage: String(err), completedAt: Date.now() } : j));
        } finally { setIsProcessing(false); }
    }, [video, model, upscaleConfig, setJobs, setActiveJob, setIsProcessing, addToast]);

    // ── Job management ───────────────────────────────────────────────────────

    const clearCompletedJobs = useCallback(() => {
        setJobs(prev => prev.filter(j => j.status === 'running' || j.status === 'queued' || j.status === 'paused'));
    }, [setJobs]);

    const handleCancelJob = useCallback(async (id: string) => {
        const job = jobs.find(j => j.id === id);
        if (!job) return;

        if (job.status === 'running' || job.status === 'paused') {
            try {
                setJobs(prev => prev.map(j => j.id === id ? { ...j, status: 'cancelled' as const, progress: 0, eta: 0, completedAt: Date.now() } : j));
                setLogs(prev => [...prev, `[SYSTEM] Job ${id} cancelled by user.`]);
                if (activeJob?.id === id) setActiveJob(null);
                setIsProcessing(false);
            } catch {
                addToast("Failed to cancel job", "error");
            }
        } else {
            setJobs(prev => prev.filter(j => j.id !== id));
            if (activeJob?.id === id) setActiveJob(null);
        }
    }, [jobs, activeJob, setJobs, setActiveJob, setIsProcessing, setLogs, addToast]);

    return {
        startUpscale,
        onExportEdited,
        startRaveValidate,
        renderPreviewSample,
        clearCompletedJobs,
        handleCancelJob,
        getScaleFromModel,
    };
}
