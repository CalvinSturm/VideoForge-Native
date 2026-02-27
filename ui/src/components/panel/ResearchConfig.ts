// VideoForge Research Configuration types and defaults
// Extracted from InputOutputPanel.tsx

export interface ResearchConfig {
    alpha_structure: number;
    alpha_texture: number;
    alpha_perceptual: number;
    alpha_diffusion: number;
    low_freq_strength: number;
    mid_freq_strength: number;
    high_freq_strength: number;
    h_sensitivity: number;
    h_blend_reduction: number;
    edge_model_bias: number;
    texture_model_bias: number;
    flat_region_suppression: number;
    hf_method: string;
    preset: string;
    freq_low_sigma: number;
    freq_mid_sigma: number;
    edge_threshold: number;
    texture_threshold: number;
    spatial_freq_mix: number;
    // SR Pipeline
    adr_enabled: boolean;
    detail_strength: number;
    luma_only: boolean;
    edge_strength: number;
    sharpen_strength: number;
    temporal_enabled: boolean;
    temporal_alpha: number;
    secondary_model: string;
    return_gpu_tensor: boolean;
}

export const RESEARCH_DEFAULTS: ResearchConfig = {
    alpha_structure: 0.5,
    alpha_texture: 0.3,
    alpha_perceptual: 0.15,
    alpha_diffusion: 0.05,
    low_freq_strength: 1.0,
    mid_freq_strength: 1.0,
    high_freq_strength: 1.0,
    h_sensitivity: 1.0,
    h_blend_reduction: 0.5,
    edge_model_bias: 0.7,
    texture_model_bias: 0.7,
    flat_region_suppression: 0.3,
    hf_method: "laplacian",
    preset: "balanced",
    freq_low_sigma: 4.0,
    freq_mid_sigma: 1.5,
    edge_threshold: 0.5,
    texture_threshold: 0.2,
    spatial_freq_mix: 0.5,
    // SR Pipeline
    adr_enabled: false,
    detail_strength: 0.5,
    luma_only: true,
    edge_strength: 0.3,
    sharpen_strength: 0.0,
    temporal_enabled: true,
    temporal_alpha: 0.9,
    secondary_model: "None",
    return_gpu_tensor: true,
};

export const RESEARCH_PRESETS: Record<string, Partial<ResearchConfig>> = {
    performance: {
        alpha_structure: 0.6, alpha_texture: 0.25, alpha_perceptual: 0.1, alpha_diffusion: 0.05,
        low_freq_strength: 1.2, mid_freq_strength: 0.8, high_freq_strength: 0.6,
        h_sensitivity: 0.5, h_blend_reduction: 0.3,
        edge_model_bias: 0.5, texture_model_bias: 0.5, flat_region_suppression: 0.5,
        preset: "performance",
    },
    balanced: {
        alpha_structure: 0.5, alpha_texture: 0.3, alpha_perceptual: 0.15, alpha_diffusion: 0.05,
        low_freq_strength: 1.0, mid_freq_strength: 1.0, high_freq_strength: 1.0,
        h_sensitivity: 1.0, h_blend_reduction: 0.5,
        edge_model_bias: 0.7, texture_model_bias: 0.7, flat_region_suppression: 0.3,
        preset: "balanced",
    },
    quality: {
        alpha_structure: 0.4, alpha_texture: 0.35, alpha_perceptual: 0.2, alpha_diffusion: 0.05,
        low_freq_strength: 0.8, mid_freq_strength: 1.2, high_freq_strength: 1.4,
        h_sensitivity: 1.5, h_blend_reduction: 0.7,
        edge_model_bias: 0.8, texture_model_bias: 0.8, flat_region_suppression: 0.2,
        preset: "quality",
    },
};

export const HF_METHODS = ["laplacian", "sobel", "highpass", "fft"] as const;
