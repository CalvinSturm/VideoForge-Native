import { useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
    Job,
    RavePolicy,
    ModelInfo,
    ToastType
} from "../types";
import type { useVideoState } from "./useVideoState";
import type { useRaveIntegration } from "./useRaveIntegration";
import { useJobStore } from "../Store/useJobStore";
import type { PanelId } from "../Store/viewLayoutStore";
import { getScaleFromModel, inferNativePrecision } from "../utils/modelRuntime";
import { cancelJob, completeJob, failJob, updateJobById } from "../utils/jobState";
import type {
    ExportRequestArgs,
    ExportRequestResult,
    NativeUpscaleRequestArgs,
    NativeUpscaleRequestResult,
    RaveBenchmarkArgs,
    RaveBenchmarkResult,
    RaveValidateArgs,
    RaveValidateResult,
    UpscaleRequestArgs,
    UpscaleRequestResult,
} from "../tauri/contracts";

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
    addToast: (msg: string, type: ToastType) => void;
    // Panel control
    panels: Record<PanelId, boolean>;
    openPanel: (id: PanelId) => void;
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
            eta: 0, startedAt: Date.now()
        };
        setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);
        if (!panels.QUEUE) openPanel('QUEUE');

        const activeScale = upscaleConfig.scaleFactor || getScaleFromModel(model);
        const upscalePayload: UpscaleRequestArgs = {
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
            let trtCacheEnabled: boolean | undefined;
            let trtCacheDir: string | null | undefined;
            let requestedExecutor: string | null | undefined;
            let executedExecutor: string | null | undefined;
            let directAttempted: boolean | undefined;
            let fallbackUsed: boolean | undefined;
            let fallbackReasonCode: string | null | undefined;
            let fallbackReasonMessage: string | null | undefined;
            const canUseNative = upscaleConfig.useNativeEngine && video.mode === 'video';

            if (canUseNative) {
                if (info?.format === "onnx") {
                    const envReady = await rave.ensureRaveEnvironmentReady();
                    if (!envReady) throw new Error("RAVE environment is not ready for native video upscale.");

                    const resolvedOutputPath = video.outputPath?.trim() ? video.outputPath : rave.defaultRaveOutputPath(video.inputPath);

                    if (showTechSpecs) {
                        try {
                            const benchmarkArgs: RaveBenchmarkArgs = {
                                args: rave.buildRaveBenchmarkArgs({ input: video.inputPath, modelPath: info.path, maxBatch: upscaleConfig.maxBatch, architectureClass: upscaleConfig.architectureClass }),
                                strictAudit: true, mockRun: false, uiOptIn: true
                            };
                            const benchmark = await invoke<RaveBenchmarkResult>("rave_benchmark", benchmarkArgs);
                            setLogs(prev => [...prev, `[RAVE] benchmark dry-run fps=${String(benchmark.fps ?? "n/a")} policy=${JSON.stringify(benchmark.policy ?? {})}`]);
                        } catch (benchErr) {
                            setLogs(prev => [...prev, `[RAVE] benchmark dry-run failed: ${benchErr}`]);
                        }
                    }

                    const nativeRequest: NativeUpscaleRequestArgs = {
                        inputPath: video.inputPath,
                        outputPath: resolvedOutputPath,
                        modelPath: info.path,
                        scale: activeScale,
                        precision: inferNativePrecision(info.path),
                        audio: true,
                        maxBatch: upscaleConfig.maxBatch
                    };
                    const nativeResult = await invoke<NativeUpscaleRequestResult>("upscale_request_native", nativeRequest);
                    if (!nativeResult.output_path || typeof nativeResult.output_path !== "string") throw new Error("upscale_request_native did not return a valid output path");
                    resultPath = nativeResult.output_path;
                    nativeEngine = nativeResult.engine;
                    encoderMode = nativeResult.encoder_mode;
                    encoderDetail = nativeResult.encoder_detail ?? null;
                    framesProcessed = nativeResult.frames_processed;
                    audioPreserved = nativeResult.audio_preserved;
                    trtCacheEnabled = nativeResult.trt_cache_enabled;
                    trtCacheDir = nativeResult.trt_cache_dir ?? null;
                    requestedExecutor = nativeResult.requested_executor ?? null;
                    executedExecutor = nativeResult.executed_executor ?? null;
                    directAttempted = nativeResult.direct_attempted;
                    fallbackUsed = nativeResult.fallback_used;
                    fallbackReasonCode = nativeResult.fallback_reason_code ?? null;
                    fallbackReasonMessage = nativeResult.fallback_reason_message ?? null;
                    setLogs(prev => [
                        ...prev,
                        `[NATIVE] engine=${nativeResult.engine} encoder_mode=${nativeResult.encoder_mode} frames=${nativeResult.frames_processed}`
                            + ` trt_cache=${String(nativeResult.trt_cache_enabled)}`
                            + (nativeResult.requested_executor ? ` requested=${nativeResult.requested_executor}` : "")
                            + (nativeResult.executed_executor ? ` executed=${nativeResult.executed_executor}` : "")
                            + (typeof nativeResult.fallback_used === "boolean" ? ` fallback=${String(nativeResult.fallback_used)}` : "")
                            + (nativeResult.fallback_reason_code ? ` fallback_code=${nativeResult.fallback_reason_code}` : "")
                            + (nativeResult.trt_cache_dir ? ` cache_dir=${nativeResult.trt_cache_dir}` : "")
                            + (nativeResult.encoder_detail ? ` detail=${nativeResult.encoder_detail}` : "")
                    ]);
                } else {
                    addToast(`Native engine requires an ONNX model. "${selectedModel}" is PyTorch-only — using Python pipeline.`, "warning");
                    resultPath = await invoke<UpscaleRequestResult>("upscale_request", upscalePayload);
                }
            } else {
                resultPath = await invoke<UpscaleRequestResult>("upscale_request", upscalePayload);
            }

            setLogs(prev => [...prev, `[SYSTEM] Job ${jobId} finished.`]);
            const finishedJob = completeJob(newJob, {
                outputPath: resultPath,
                ...(policy ? { policy } : {}),
                ...(typeof hostCopyAuditEnabled === "boolean" ? { hostCopyAuditEnabled } : {}),
                ...(hostCopyAuditDisableReason !== undefined ? { hostCopyAuditDisableReason } : {}),
                ...(nativeEngine ? { nativeEngine } : {}),
                ...(encoderMode ? { encoderMode } : {}),
                ...(encoderDetail !== undefined ? { encoderDetail } : {}),
                ...(typeof framesProcessed === "number" ? { framesProcessed } : {}),
                ...(typeof audioPreserved === "boolean" ? { audioPreserved } : {}),
                ...(typeof trtCacheEnabled === "boolean" ? { trtCacheEnabled } : {}),
                ...(trtCacheDir !== undefined ? { trtCacheDir } : {}),
                ...(requestedExecutor !== undefined ? { requestedExecutor } : {}),
                ...(executedExecutor !== undefined ? { executedExecutor } : {}),
                ...(typeof directAttempted === "boolean" ? { directAttempted } : {}),
                ...(typeof fallbackUsed === "boolean" ? { fallbackUsed } : {}),
                ...(fallbackReasonCode !== undefined ? { fallbackReasonCode } : {}),
                ...(fallbackReasonMessage !== undefined ? { fallbackReasonMessage } : {})
            });
            setJobs(prev => updateJobById(prev, jobId, () => finishedJob));
            setActiveJob(finishedJob);
            setLastOutputPath(resultPath);
        } catch (err) {
            const normalized = rave.parseRaveError(err);
            const msg = `[${normalized.category}] ${normalized.message}`;
            addToast(`Error: ${msg}`, "error");
            setLogs(prev => [...prev, `[ERROR][RAVE][${normalized.category}] Job ${jobId} failed: ${normalized.message}`]);
            setJobs(prev => updateJobById(prev, jobId, job => failJob(job, normalized)));
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
            eta: 0, startedAt: Date.now()
        };
        setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);
        if (!panels.QUEUE) openPanel('QUEUE');

        try {
            const exportRequest: ExportRequestArgs = {
                inputPath: video.inputPath, outputPath: video.outputPath,
                editConfig: video.getRustEditConfig(), scale: 1
            };
            const resultPath = await invoke<ExportRequestResult>("export_request", exportRequest);
            setLogs(prev => [...prev, `[SYSTEM] Export ${jobId} complete.`]);
            addToast("Export Completed", "success");
            const finishedJob = completeJob(newJob, { outputPath: resultPath });
            setJobs(prev => updateJobById(prev, jobId, () => finishedJob));
            setActiveJob(finishedJob);
            setLastOutputPath(resultPath);
        } catch (err) {
            const normalized = rave.parseRaveError(err);
            const msg = `[${normalized.category}] ${normalized.message}`;
            addToast(`Error: ${msg}`, "error");
            setLogs(prev => [...prev, `[ERROR][${normalized.category}] Export ${jobId} failed: ${normalized.message}`]);
            setJobs(prev => updateJobById(prev, jobId, job => failJob(job, normalized)));
        } finally { setIsProcessing(false); }
    }, [video, rave, setJobs, setActiveJob, setIsProcessing, setLastOutputPath, setLogs, addToast, panels, openPanel]);

    // ── RAVE Validate ────────────────────────────────────────────────────────

    const startRaveValidate = useCallback(async () => {
        if (!video.inputPath) return addToast("Select an input file first!", "error");
        const jobId = `validate_${Date.now().toString()}`;
        const newJob: Job = {
            id: jobId, command: "Validate: production_strict (mock)",
            status: "running", progress: 0, statusMessage: "Validating strict policy...",
            eta: 0, startedAt: Date.now()
        };
        setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);
        if (!panels.QUEUE) openPanel('QUEUE');

        try {
            const validateRequest: RaveValidateArgs = {
                fixture: null, profile: "production_strict", bestEffort: true, strictAudit: true, mockRun: true
            };
            const result = await invoke<RaveValidateResult>("rave_validate", validateRequest);
            const policy = result.policy;
            const hostCopyAuditEnabled = result.host_copy_audit_enabled;
            const hostCopyAuditDisableReason = result.host_copy_audit_disable_reason;
            const skipped = result.skipped === true;

            const finishedJob = completeJob(newJob, {
                statusMessage: skipped ? "Validation skipped" : "Validation passed",
                ...(policy ? { policy } : {}),
                ...(typeof hostCopyAuditEnabled === "boolean" ? { hostCopyAuditEnabled } : {}),
                ...(hostCopyAuditDisableReason !== undefined ? { hostCopyAuditDisableReason } : {})
            });
            setJobs(prev => updateJobById(prev, jobId, () => finishedJob));
            setActiveJob(finishedJob);
            setLogs(prev => [...prev, `[RAVE] validate ok=${String(result.ok ?? "unknown")} skipped=${String(skipped)} policy=${JSON.stringify(policy ?? {})}`]);
            addToast(skipped ? "Validate completed (skipped)" : "Validate passed", skipped ? "warning" : "success");
        } catch (err) {
            const normalized = rave.parseRaveError(err);
            setJobs(prev => updateJobById(prev, jobId, job => failJob(job, normalized)));
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
            eta: 0, startedAt: Date.now()
        };
        setJobs(prev => [...prev, newJob]); setActiveJob(newJob); setIsProcessing(true);

        const previewPayload: UpscaleRequestArgs = {
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
            const resultPath = await invoke<UpscaleRequestResult>("upscale_request", previewPayload);
            video.setPreviewFile(resultPath);
            video.setRenderedRange({ start, end });
            addToast("Sample ready.", "success");
            const finishedJob = completeJob(newJob, { outputPath: resultPath });
            setJobs(prev => updateJobById(prev, jobId, () => finishedJob));
            setActiveJob(finishedJob);
        } catch (err) {
            addToast(`Sample failed: ${err}`, "error");
            setJobs(prev => updateJobById(prev, jobId, job => ({
                ...job,
                status: "error",
                statusMessage: String(err),
                completedAt: Date.now()
            })));
        } finally { setIsProcessing(false); }
    }, [video, model, upscaleConfig, setJobs, setActiveJob, setIsProcessing, addToast]);

    // ── Job management ───────────────────────────────────────────────────────

    const clearCompletedJobs = useCallback(() => {
        setJobs(prev => prev.filter(j => j.status === 'running' || j.status === 'queued'));
    }, [setJobs]);

    const handleCancelJob = useCallback(async (id: string) => {
        const job = jobs.find(j => j.id === id);
        if (!job) return;

        if (job.status === 'running') {
            try {
                setJobs(prev => updateJobById(prev, id, jobToCancel => cancelJob(jobToCancel)));
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
    };
}
