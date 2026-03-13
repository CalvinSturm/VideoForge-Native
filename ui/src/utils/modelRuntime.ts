/**
 * Shared model/runtime helpers used by UI orchestration code.
 */

/**
 * Extract the intended upscale factor from a model identifier.
 *
 * Falls back to `4` when no scale suffix is present so legacy model IDs keep
 * the current UI behavior.
 */
export function getScaleFromModel(modelId?: string): number {
    if (!modelId) {
        return 4;
    }

    const match = modelId.match(/x(\d)/i);
    return match?.[1] ? parseInt(match[1], 10) : 4;
}

/**
 * Infer native precision from the model filename/path.
 *
 * Native ONNX exports often encode half-precision variants in the filename.
 */
export function inferNativePrecision(modelPath: string): "fp16" | "fp32" {
    const lower = modelPath.toLowerCase();
    if (lower.includes("_fp16") || lower.includes("-fp16") || lower.includes("half")) {
        return "fp16";
    }

    return "fp32";
}
