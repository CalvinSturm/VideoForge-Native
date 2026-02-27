import { useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { RaveEnvironmentJson, RaveErrorPayload, ParsedRaveError } from "../types";

interface UseRaveIntegrationOptions {
    addToast: (msg: string, type: string) => void;
    setLogs: React.Dispatch<React.SetStateAction<string[]>>;
}

export function useRaveIntegration({ addToast, setLogs }: UseRaveIntegrationOptions) {
    const [raveEnvironment, setRaveEnvironment] = useState<RaveEnvironmentJson | null>(null);
    const [raveEnvironmentLoading, setRaveEnvironmentLoading] = useState(false);

    // ── Error classification ─────────────────────────────────────────────────

    const hintForCategory = (category: string): string | undefined => {
        switch (category) {
            case "policy_violation":
                return "Enable strict-profile requirements (audit/no-host-copies) or switch profile.";
            case "provider_loader_error":
                return "Check CUDA/driver/ORT/TensorRT provider installation and loader paths.";
            case "runtime_dependency_missing":
                return "Install missing runtime dependency and rerun.";
            case "input_contract_error":
                return "Fix input/CLI contract values (max_batch must be 1-8).";
            default:
                return undefined;
        }
    };

    const buildCategorizedError = (category: string, message: string): ParsedRaveError => {
        const hint = hintForCategory(category);
        return hint ? { category, message, nextAction: hint } : { category, message };
    };

    const parseRaveError = useCallback((err: unknown): ParsedRaveError => {
        const raw = String(err);
        try {
            const payload = JSON.parse(raw) as RaveErrorPayload;
            if (payload && typeof payload === "object" && payload.category && payload.message) {
                return {
                    category: payload.category,
                    message: payload.message,
                    ...(payload.detail !== undefined ? { detail: payload.detail } : {}),
                    ...(payload.next_action !== undefined ? { nextAction: payload.next_action } : {})
                };
            }
        } catch {
            // fall back to heuristic classification
        }

        const lower = raw.toLowerCase();
        if (lower.includes("strict no-host-copies") || lower.includes("host copy audit")) {
            return buildCategorizedError("policy_violation", raw);
        }
        if (lower.includes("provider") || lower.includes("onnxruntime") || lower.includes("tensorrt")) {
            return buildCategorizedError("provider_loader_error", raw);
        }
        if (lower.includes("missing") || lower.includes("not found")) {
            return buildCategorizedError("runtime_dependency_missing", raw);
        }
        if (lower.includes("max_batch")) {
            return buildCategorizedError("input_contract_error", raw);
        }
        return buildCategorizedError("runtime_error", raw);
    }, []);

    // ── CLI arg builders ─────────────────────────────────────────────────────

    const defaultRaveOutputPath = (input: string): string => {
        const ext = input.split('.').pop()?.toLowerCase();
        if (ext) return input.replace(new RegExp(`\\.${ext}$`), `_rave_upscaled.mp4`);
        return `${input}_rave_upscaled.mp4`;
    };

    const buildRaveUpscaleArgs = (params: {
        input: string; output: string; modelPath: string; maxBatch?: number;
    }): string[] => {
        const args = [
            "-i", params.input,
            "-m", params.modelPath,
            "-o", params.output,
            "--precision", "fp16",
            "--progress", "jsonl"
        ];
        if (params.maxBatch && params.maxBatch > 1) {
            args.push("--max-batch", String(params.maxBatch));
        }
        return args;
    };

    const buildRaveBenchmarkArgs = (params: {
        input: string; modelPath: string; maxBatch?: number;
    }): string[] => {
        const args = [
            "-i", params.input,
            "-m", params.modelPath,
            "--skip-encode",
            "--dry-run",
            "--progress", "jsonl"
        ];
        if (params.maxBatch && params.maxBatch > 1) {
            args.push("--max-batch", String(params.maxBatch));
        }
        return args;
    };

    // ── Environment checks ───────────────────────────────────────────────────

    const fetchRaveEnvironment = useCallback(async (showToastOnResult: boolean): Promise<RaveEnvironmentJson | null> => {
        setRaveEnvironmentLoading(true);
        try {
            const env = await invoke<RaveEnvironmentJson>("rave_environment");
            setRaveEnvironment(env);
            const ffmpegStatus = env.ffmpeg.layout_ok && env.ffmpeg.abi_ok ? "ok" : "bad";
            const cudnnStatus = env.providers.cudnn_found ? "ok" : "missing";
            const tensorrtStatus = env.providers.tensorrt_found ? "ok" : "missing";
            const tensorrtParserStatus = env.providers.tensorrt_parser_found ? "ok" : "missing";
            const tensorrtPluginStatus = env.providers.tensorrt_plugin_found ? "ok" : "missing";
            setLogs(prev => [
                ...prev,
                `[RAVE] env ready=${String(env.ready)} profile=${env.profile} rave_bin=${String(env.rave_bin.exists)} ffmpeg=${ffmpegStatus} cudnn=${cudnnStatus} tensorrt=${tensorrtStatus} trt_parser=${tensorrtParserStatus} trt_plugin=${tensorrtPluginStatus}`
            ]);
            if (showToastOnResult) {
                addToast(env.ready ? "RAVE environment ready" : "RAVE environment has issues", env.ready ? "success" : "warning");
            }
            return env;
        } catch (err) {
            const msg = String(err);
            setLogs(prev => [...prev, `[RAVE] env diagnostics failed: ${msg}`]);
            if (showToastOnResult) {
                addToast(`RAVE env check failed: ${msg}`, "error");
            }
            return null;
        } finally {
            setRaveEnvironmentLoading(false);
        }
    }, [addToast, setLogs]);

    const ensureRaveEnvironmentReady = useCallback(async (): Promise<boolean> => {
        const env = await fetchRaveEnvironment(false);
        if (!env) {
            addToast("RAVE environment check failed", "error");
            return false;
        }
        if (env.ready) return true;

        const providerHint = !env.providers.cudnn_found
            ? `Missing ${env.providers.cudnn_probe}`
            : !env.providers.tensorrt_found
                ? `Missing ${env.providers.tensorrt_probe}`
                : !env.providers.tensorrt_parser_found
                    ? `Missing ${env.providers.tensorrt_parser_probe}`
                    : !env.providers.tensorrt_plugin_found
                        ? `Missing ${env.providers.tensorrt_plugin_probe}`
                        : "RAVE runtime dependencies are incomplete";
        const next = env.hints[0] ?? "Run CHECK RAVE ENV and follow remediation hints.";
        addToast(`[RAVE] ${providerHint}`, "error");
        setLogs(prev => [...prev, `[RAVE] blocked native run: ${providerHint}. next=${next}`]);
        return false;
    }, [fetchRaveEnvironment, addToast, setLogs]);

    const runRaveDiagnostics = useCallback(async () => {
        await fetchRaveEnvironment(true);
    }, [fetchRaveEnvironment]);

    return {
        raveEnvironment,
        raveEnvironmentLoading,
        fetchRaveEnvironment,
        ensureRaveEnvironmentReady,
        runRaveDiagnostics,
        parseRaveError,
        defaultRaveOutputPath,
        buildRaveUpscaleArgs,
        buildRaveBenchmarkArgs,
    };
}
