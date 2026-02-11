/**
 * Model Classification Utilities
 * 
 * This module provides functions for classifying SR models by architecture type,
 * extracting scale factors, and determining model compatibility for blending.
 * 
 * Classification rules are aligned with the Python backend (sr_settings_node.py)
 * to ensure consistent behavior across the stack.
 */

import type { ArchitectureClass } from '../Store/useJobStore';

// ═══════════════════════════════════════════════════════════════════════════════
// ARCHITECTURE CLASSIFICATION PATTERNS
// ═══════════════════════════════════════════════════════════════════════════════
// 
// Each pattern maps model name prefixes to architecture classes.
// Order matters: more specific patterns should come before general ones.
// These patterns are case-insensitive.

interface ArchitecturePattern {
    pattern: RegExp;
    class: ArchitectureClass;
    label: string;
    icon: string;
    description: string;
}

const ARCHITECTURE_PATTERNS: ArchitecturePattern[] = [
    // ─── Transformer (Attention-based) ───────────────────────────────────────
    // Best quality/speed tradeoff, global context awareness
    // IMPORTANT: Must be checked before GAN — model names like "Swin_2SR_..._BSRGAN"
    // contain GAN-family training method names but are transformer architectures.
    {
        pattern: /(swin[-_]?ir|swin[-_]?2?sr|hat|ipt|edt|dat|ffhq[-_]?dat|faceup)/i,
        class: 'Transformer',
        label: 'Transformer',
        icon: '🔀',
        description: 'Attention-based transformers. Best quality/speed tradeoff with global context.'
    },

    // ─── CNN (Convolutional Neural Networks) ─────────────────────────────────
    // Classic feedforward networks, deterministic output, fast inference
    {
        pattern: /(rcan|edsr|mdsr)/i,
        class: 'CNN',
        label: 'CNN',
        icon: '🧠',
        description: 'Classic convolutional networks. Deterministic, stable output. Best for archival work.'
    },

    // ─── GAN (Generative Adversarial Networks) ───────────────────────────────
    // Adds realistic texture detail, may introduce artifacts
    {
        pattern: /(real[-_]?esrgan|esrgan|bsrgan|spsr|star[-_]?sr[-_]?gan|nmkd|siax|animesharp|remacri|universal)/i,
        class: 'GAN',
        label: 'GAN',
        icon: '⚡',
        description: 'Generative adversarial networks. Adds texture detail but may hallucinate features.'
    },

    // ─── Diffusion (Iterative Refinement) ────────────────────────────────────
    // Highest quality, but slow and may require many steps
    {
        pattern: /(sr3|stable[-_]?sr|resshift|dit[-_]?sr|dit|diffusion)/i,
        class: 'Diffusion',
        label: 'Diffusion',
        icon: '🌀',
        description: 'Iterative diffusion models. Highest quality but slowest inference.'
    },

    // ─── Lightweight (Mobile-optimized) ──────────────────────────────────────
    // Fast inference, lower quality, good for previews
    {
        pattern: /(fsrcnn|carn[-_]?m?|lapsrn|omni[-_]?sr|nomos|mosr|janai|compact|span)/i,
        class: 'Lightweight',
        label: 'Lightweight',
        icon: '🪶',
        description: 'Mobile-optimized networks. Fastest inference, suitable for previews.'
    },
];

// Fallback for unrecognized models
const DEFAULT_ARCHITECTURE: ArchitectureClass = 'CNN';

// ═══════════════════════════════════════════════════════════════════════════════
// MODEL CAPABILITY FLAGS
// ═══════════════════════════════════════════════════════════════════════════════
// 
// Capabilities determine which features a model supports.
// These must match the Python backend's ModelCapability class.

export type ModelCapability =
    | 'temporal'      // EMA temporal stabilization
    | 'adr'           // Adaptive Detail Residual
    | 'edge_aware'    // Sobel edge-aware blending
    | 'luma_blend'    // YCbCr luminance-only blend
    | 'sharpen'       // Unsharp mask post-processing
    | 'secondary';    // Can act as secondary model for blending

// Capability sets by architecture class
const ARCHITECTURE_CAPABILITIES: Record<ArchitectureClass, Set<ModelCapability>> = {
    CNN: new Set(['temporal', 'edge_aware', 'luma_blend', 'sharpen']),
    GAN: new Set(['temporal', 'adr', 'edge_aware', 'luma_blend', 'sharpen', 'secondary']),
    Transformer: new Set(['temporal', 'adr', 'edge_aware', 'luma_blend', 'sharpen', 'secondary']),
    Diffusion: new Set(['sharpen']), // Diffusion models typically don't support temporal
    Lightweight: new Set(['sharpen']), // Minimal capabilities for speed
};

// ═══════════════════════════════════════════════════════════════════════════════
// CLASSIFICATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Classify a model ID into an architecture class.
 * 
 * @param modelId - Full model identifier (e.g., "RCAN_x4", "RealESRGAN_x4plus")
 * @returns The architecture class for this model
 * 
 * Classification uses prefix matching against known patterns.
 * Unknown models default to CNN for safety (deterministic output).
 */
export function getArchitectureClass(modelId: string): ArchitectureClass {
    const normalized = modelId.trim();

    for (const { pattern, class: archClass } of ARCHITECTURE_PATTERNS) {
        if (pattern.test(normalized)) {
            return archClass;
        }
    }

    // Log unrecognized models in development for debugging
    if (process.env.NODE_ENV === 'development') {
        console.debug(`[modelClassification] Unknown model "${modelId}", defaulting to ${DEFAULT_ARCHITECTURE}`);
    }

    return DEFAULT_ARCHITECTURE;
}

/**
 * Get metadata for an architecture class.
 * 
 * @param archClass - Architecture class to look up
 * @returns Pattern metadata including label, icon, and description
 */
export function getArchitectureInfo(archClass: ArchitectureClass): {
    label: string;
    icon: string;
    description: string;
} {
    // Find any pattern matching this class (they all have the same metadata)
    const pattern = ARCHITECTURE_PATTERNS.find(p => p.class === archClass);
    return pattern ?? {
        label: archClass,
        icon: '❓',
        description: 'Unknown architecture type'
    };
}

/**
 * Extract the scale factor from a model ID.
 * 
 * @param modelId - Model identifier (e.g., "RCAN_x4", "RealESRGAN_x2plus")
 * @returns Scale factor (2, 3, or 4) or null if not detected
 * 
 * Common patterns:
 *   - _x4, _x2, _x3 (e.g., RCAN_x4)
 *   - x4plus, x2plus (e.g., RealESRGAN_x4plus)
 *   - 4x prefix (e.g., 4xNomos2)
 */
export function extractScale(modelId: string): 2 | 3 | 4 | null {
    const patterns = [
        /_x([234])/i,           // _x4, _x2, _x3
        /x([234])plus/i,        // x4plus, x2plus
        /^([234])x/i,           // 4x prefix
        /[-_]([234])x?$/i,      // trailing -4 or _4x
    ];

    for (const pattern of patterns) {
        const match = modelId.match(pattern);
        if (match && match[1]) {
            const scale = parseInt(match[1], 10);
            if (scale === 2 || scale === 3 || scale === 4) {
                return scale;
            }
        }
    }

    return null;
}

/**
 * Extract the model family name (without scale suffix).
 * 
 * @param modelId - Full model identifier
 * @returns Family name (e.g., "RCAN", "RealESRGAN")
 */
export function extractFamily(modelId: string): string {
    // Remove common scale suffixes
    return modelId
        .replace(/_x[234].*$/i, '')      // RCAN_x4 -> RCAN
        .replace(/x[234]plus.*$/i, '')   // RealESRGAN_x4plus -> RealESRGAN
        .replace(/^([234])x/i, '')       // 4xNomos -> Nomos
        .replace(/[-_][234]x?$/i, '')    // model_4 or model_4x -> model
        .trim();
}

/**
 * Get capabilities for a model based on its architecture.
 * 
 * @param modelId - Model identifier
 * @returns Set of capability flags
 */
export function getModelCapabilities(modelId: string): Set<ModelCapability> {
    const arch = getArchitectureClass(modelId);
    return ARCHITECTURE_CAPABILITIES[arch];
}

/**
 * Check if a model supports acting as a secondary model for blending.
 * 
 * @param modelId - Model identifier
 * @returns true if the model can be used as a secondary
 */
export function supportsSecondary(modelId: string): boolean {
    const caps = getModelCapabilities(modelId);
    return caps.has('secondary');
}

/**
 * Check if two models are compatible for blending.
 * 
 * Compatibility rules:
 *   1. Secondary model must support the 'secondary' capability
 *   2. Models should have the same scale factor
 *   3. Primary and secondary should be different models
 * 
 * @param primaryId - Primary model identifier
 * @param secondaryId - Secondary model identifier
 * @returns true if models can be blended together
 */
export function areModelsCompatible(primaryId: string, secondaryId: string): boolean {
    // Can't blend with itself
    if (primaryId === secondaryId) {
        return false;
    }

    // Secondary must support blending
    if (!supportsSecondary(secondaryId)) {
        return false;
    }

    // Scales should match (or be unknown)
    const primaryScale = extractScale(primaryId);
    const secondaryScale = extractScale(secondaryId);

    if (primaryScale !== null && secondaryScale !== null && primaryScale !== secondaryScale) {
        return false;
    }

    return true;
}

/**
 * Group models by their architecture class.
 * 
 * @param models - Array of model identifiers
 * @returns Object mapping architecture class to array of models
 */
export function groupModelsByArchitecture(models: string[]): Record<ArchitectureClass, string[]> {
    const groups: Record<ArchitectureClass, string[]> = {
        CNN: [],
        GAN: [],
        Transformer: [],
        Diffusion: [],
        Lightweight: [],
    };

    for (const model of models) {
        const arch = getArchitectureClass(model);
        groups[arch].push(model);
    }

    return groups;
}

/**
 * Get available scales for a model family.
 * 
 * @param models - All available models
 * @param family - Family name to filter by
 * @returns Sorted array of available scales
 */
export function getAvailableScales(models: string[], family: string): (2 | 3 | 4)[] {
    const familyLower = family.toLowerCase();
    const scales = new Set<2 | 3 | 4>();

    for (const model of models) {
        if (extractFamily(model).toLowerCase() === familyLower) {
            const scale = extractScale(model);
            if (scale !== null) {
                scales.add(scale);
            }
        }
    }

    return Array.from(scales).sort((a, b) => a - b);
}

/**
 * Find the best matching model for a family and scale.
 * 
 * @param models - All available models
 * @param family - Desired family name
 * @param scale - Desired scale factor
 * @returns Best matching model ID or null if not found
 */
export function findModelByFamilyAndScale(
    models: string[],
    family: string,
    scale: 2 | 3 | 4
): string | null {
    const familyLower = family.toLowerCase();

    // Exact match
    const exact = models.find(m =>
        extractFamily(m).toLowerCase() === familyLower &&
        extractScale(m) === scale
    );
    if (exact) return exact;

    // Fallback: any model from this family
    const fallback = models.find(m =>
        extractFamily(m).toLowerCase() === familyLower
    );
    return fallback ?? null;
}

/**
 * Get compatible secondary models for a primary model.
 * 
 * @param models - All available models
 * @param primaryId - Primary model identifier
 * @returns Array of compatible secondary model IDs
 */
export function getCompatibleSecondaryModels(models: string[], primaryId: string): string[] {
    return models.filter(m => areModelsCompatible(primaryId, m));
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESOLUTION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Standard resolution presets.
 */
export const RESOLUTION_PRESETS = {
    HD: { width: 1280, height: 720, label: 'HD', shortLabel: '720p' },
    FHD: { width: 1920, height: 1080, label: 'Full HD', shortLabel: '1080p' },
    QHD: { width: 2560, height: 1440, label: 'QHD', shortLabel: '1440p' },
    '4K': { width: 3840, height: 2160, label: '4K UHD', shortLabel: '4K' },
} as const;

/**
 * Calculate required scale factor to reach target resolution from source.
 * 
 * @param sourceWidth - Source video width
 * @param sourceHeight - Source video height
 * @param targetWidth - Target output width
 * @param targetHeight - Target output height
 * @returns Required scale factor (may be fractional)
 */
export function calculateRequiredScale(
    sourceWidth: number,
    sourceHeight: number,
    targetWidth: number,
    targetHeight: number
): number {
    if (sourceWidth <= 0 || sourceHeight <= 0) return 1;

    const scaleW = targetWidth / sourceWidth;
    const scaleH = targetHeight / sourceHeight;

    // Use the larger scale to ensure target is met in both dimensions
    return Math.max(scaleW, scaleH);
}

/**
 * Find the best supported scale for a given target.
 * 
 * If exact scale isn't supported, returns the next higher available scale.
 * 
 * @param requiredScale - Calculated required scale
 * @param availableScales - Array of supported scales
 * @returns Best matching supported scale
 */
export function findBestScale(
    requiredScale: number,
    availableScales: (2 | 3 | 4)[]
): 2 | 3 | 4 {
    // Sort ascending
    const sorted = [...availableScales].sort((a, b) => a - b);

    // Find smallest scale that meets or exceeds requirement
    for (const scale of sorted) {
        if (scale >= requiredScale) {
            return scale;
        }
    }

    // Fallback to largest available
    return sorted[sorted.length - 1] ?? 4;
}

/**
 * Estimate VRAM usage for a given resolution and model.
 * 
 * This is a rough estimate based on typical model memory patterns.
 * Actual usage depends on specific model architecture.
 * 
 * @param width - Output width
 * @param height - Output height
 * @param archClass - Architecture class
 * @returns Estimated VRAM in GB
 */
export function estimateVRAM(
    width: number,
    height: number,
    archClass: ArchitectureClass
): number {
    const megapixels = (width * height) / 1_000_000;

    // Base VRAM per megapixel by architecture
    const vramPerMP: Record<ArchitectureClass, number> = {
        CNN: 0.5,
        GAN: 0.8,
        Transformer: 1.2,
        Diffusion: 2.0,
        Lightweight: 0.2,
    };

    // Fixed overhead (model weights, buffers)
    const overhead: Record<ArchitectureClass, number> = {
        CNN: 0.5,
        GAN: 0.8,
        Transformer: 1.0,
        Diffusion: 2.0,
        Lightweight: 0.2,
    };

    return overhead[archClass] + megapixels * vramPerMP[archClass];
}
