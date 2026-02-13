/**
 * AI Upscale Node Component
 * 
 * Comprehensive upscale configuration UI with:
 *   - Architecture class selection (CNN, GAN, Transformer, Diffusion, Lightweight)
 *   - Primary model selection with dynamic scale detection
 *   - Secondary model blending for texture enhancement
 *   - Custom resolution overrides with presets
 *   - Capability badges and tooltips
 *   - VRAM usage estimation and warnings
 * 
 * Designed for 2026 UX best practices:
 *   - Collapsible sections with smooth animations
 *   - Visual hierarchy with architecture icons
 *   - Disabled states with clear reasoning
 *   - Comprehensive tooltips for all options
 */

import React, { useMemo, useState, useCallback } from 'react';
import { useJobStore, type ArchitectureClass, type UpscaleScale } from '../Store/useJobStore';
import type { VideoState } from '../types';
import {
    getArchitectureClass,
    getArchitectureInfo,
    extractScale,
    extractFamily,
    groupModelsByArchitecture,
    getAvailableScales,
    findModelByFamilyAndScale,
    RESOLUTION_PRESETS,
    calculateRequiredScale,
    findBestScale,
    estimateVRAM,
} from '../utils/modelClassification';

// ═══════════════════════════════════════════════════════════════════════════════
// ICONS
// ═══════════════════════════════════════════════════════════════════════════════

const IconSparkles = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z" />
        <path d="M5 19l1 3 1-3 3-1-3-1-1-3-1 3-3 1 3 1z" />
    </svg>
);

const IconChevronDown = () => (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="6 9 12 15 18 9" />
    </svg>
);

const IconCheck = () => (
    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
        <polyline points="20 6 9 17 4 12" />
    </svg>
);

const IconInfo = () => (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
);

const IconWarning = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
);

const IconLock = () => (
    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
);

// ═══════════════════════════════════════════════════════════════════════════════
// PROPS & TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/** Pipeline feature state from ResearchConfig */
interface PipelineFeatures {
    adr_enabled: boolean;
    temporal_enabled: boolean;
    luma_only: boolean;
    sharpen_strength: number;
}

interface AIUpscaleNodeProps {
    /** Current video state for dimension calculations */
    videoState: VideoState;
    /** All available model IDs from the backend */
    availableModels: string[];
    /** Callback when model selection changes */
    onModelChange: (modelId: string) => void;
    /** Trigger model loading on backend */
    loadModel: (modelId: string) => void;
    /** Whether a model is currently loading */
    isLoading: boolean;
    /** Current pipeline feature toggles from research config */
    pipelineFeatures?: PipelineFeatures;
    /** Callback to toggle a research param */
    onPipelineToggle?: (key: string, value: boolean | number) => void;
    /** Whether tech specs (including VRAM estimates) should be visible */
    showTech?: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUB-COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Collapsible section with header and animated content
 */
const CollapsibleSection: React.FC<{
    title: string;
    subtitle?: string;
    icon?: React.ReactNode;
    defaultOpen?: boolean;
    badge?: React.ReactNode;
    children: React.ReactNode;
}> = ({ title, subtitle, icon, defaultOpen = true, badge, children }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
        <div style={{
            background: 'rgba(0,0,0,0.15)',
            borderRadius: '8px',
            border: '1px solid rgba(255,255,255,0.06)',
            overflow: 'hidden',
        }}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 12px',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: 'var(--text-primary)',
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {icon && <span style={{ color: 'var(--text-muted)', opacity: 0.7 }}>{icon}</span>}
                    <span style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.05em' }}>{title}</span>
                    {subtitle && <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>• {subtitle}</span>}
                    {badge}
                </div>
                <div style={{
                    color: 'var(--text-muted)',
                    transform: isOpen ? 'rotate(0deg)' : 'rotate(-90deg)',
                    transition: 'transform 0.2s ease',
                    opacity: 0.5,
                }}>
                    <IconChevronDown />
                </div>
            </button>
            <div style={{
                maxHeight: isOpen ? '1000px' : '0',
                overflow: 'hidden',
                transition: 'max-height 0.3s ease-out',
            }}>
                <div style={{ padding: '0 12px 12px' }}>
                    {children}
                </div>
            </div>
        </div>
    );
};

/**
 * Architecture selection card
 */
const ArchitectureCard: React.FC<{
    archClass: ArchitectureClass;
    selected: boolean;
    onClick: () => void;
    modelCount: number;
    disabled?: boolean;
}> = ({ archClass, selected, onClick, modelCount, disabled }) => {
    const info = getArchitectureInfo(archClass);

    return (
        <button
            onClick={onClick}
            disabled={disabled || modelCount === 0}
            title={info.description}
            style={{
                flex: '1 1 0',
                minWidth: '60px',
                height: '52px',
                padding: '6px 10px',
                borderRadius: '6px',
                border: selected
                    ? '1px solid var(--brand-primary)'
                    : '1px solid rgba(255,255,255,0.08)',
                background: selected
                    ? 'linear-gradient(135deg, var(--brand-dim), rgba(0,255,136,0.05))'
                    : 'rgba(255,255,255,0.03)',
                cursor: disabled || modelCount === 0 ? 'not-allowed' : 'pointer',
                opacity: disabled || modelCount === 0 ? 0.4 : 1,
                transition: 'all 0.15s ease',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '2px',
            }}
        >
            <span style={{ fontSize: '16px' }}>{info.icon}</span>
            <span style={{
                fontSize: '9px',
                fontWeight: 700,
                color: selected ? 'var(--brand-primary)' : 'var(--text-secondary)',
                letterSpacing: '0.03em',
            }}>
                {info.label}
            </span>
            <span style={{
                fontSize: '7px',
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
            }}>
                {modelCount} {modelCount === 1 ? 'model' : 'models'}
            </span>
        </button>
    );
};

/**
 * Scale toggle button
 */
const ScaleToggle: React.FC<{
    scale: UpscaleScale;
    selected: boolean;
    available: boolean;
    onClick: () => void;
}> = ({ scale, selected, available, onClick }) => (
    <button
        onClick={onClick}
        disabled={!available}
        style={{
            flex: 1,
            height: '36px',
            borderRadius: '5px',
            border: selected
                ? '1px solid var(--brand-primary)'
                : '1px solid rgba(255,255,255,0.08)',
            background: selected
                ? 'var(--brand-dim)'
                : 'rgba(255,255,255,0.03)',
            color: selected
                ? 'var(--brand-primary)'
                : available
                    ? 'var(--text-secondary)'
                    : 'var(--text-muted)',
            cursor: available ? 'pointer' : 'not-allowed',
            opacity: available ? 1 : 0.4,
            transition: 'all 0.15s ease',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '4px',
            fontSize: '12px',
            fontWeight: 700,
        }}
    >
        {scale}×
        {!available && <IconLock />}
    </button>
);

/**
 * Interactive pipeline feature toggle chip
 */
const PipelineToggle: React.FC<{
    label: string;
    enabled: boolean;
    onToggle: () => void;
}> = ({ label, enabled, onToggle }) => (
    <button
        onClick={(e) => { e.stopPropagation(); onToggle(); }}
        style={{
            fontSize: '7px',
            fontWeight: 600,
            padding: '2px 5px',
            borderRadius: '3px',
            background: enabled ? 'rgba(0,255,136,0.1)' : 'rgba(255,255,255,0.05)',
            border: enabled ? '1px solid rgba(0,255,136,0.2)' : '1px solid rgba(255,255,255,0.08)',
            color: enabled ? 'var(--brand-primary)' : 'var(--text-muted)',
            letterSpacing: '0.03em',
            cursor: 'pointer',
            transition: 'all 0.15s ease',
        }}
    >
        {label}
    </button>
);

/**
 * Resolution preset button
 */
const ResolutionPresetButton: React.FC<{
    preset: keyof typeof RESOLUTION_PRESETS;
    selected: boolean;
    onClick: () => void;
}> = ({ preset, selected, onClick }) => {
    const info = RESOLUTION_PRESETS[preset];

    return (
        <button
            onClick={onClick}
            style={{
                flex: 1,
                height: '36px',
                borderRadius: '5px',
                border: selected
                    ? '1px solid var(--brand-primary)'
                    : '1px solid rgba(255,255,255,0.08)',
                background: selected
                    ? 'var(--brand-dim)'
                    : 'rgba(255,255,255,0.03)',
                color: selected
                    ? 'var(--brand-primary)'
                    : 'var(--text-secondary)',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '10px',
                fontWeight: 700,
                gap: '1px',
            }}
        >
            {info.shortLabel}
            <span style={{ fontSize: '7px', opacity: 0.6, fontFamily: 'var(--font-mono)' }}>
                {info.width}×{info.height}
            </span>
        </button>
    );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const AIUpscaleNode: React.FC<AIUpscaleNodeProps> = ({
    videoState,
    availableModels,
    onModelChange,
    loadModel,
    isLoading,
    pipelineFeatures,
    onPipelineToggle,
    showTech = false,
}) => {
    // ─── Store Access ──────────────────────────────────────────────────────────
    const { upscaleConfig, setUpscaleConfig } = useJobStore();

    // Destructure for convenience
    const {
        isEnabled,
        architectureClass,
        primaryModelId,
        scaleFactor,
        resolutionMode,
        targetWidth,
        targetHeight,
        resolutionPreset,
    } = upscaleConfig;

    // ─── Computed Values ───────────────────────────────────────────────────────

    // Group models by architecture
    const modelGroups = useMemo(() =>
        groupModelsByArchitecture(availableModels),
        [availableModels]);

    // Models for current architecture
    const currentArchModels = useMemo(() =>
        modelGroups[architectureClass] ?? [],
        [modelGroups, architectureClass]);

    // Unique families in current architecture
    const currentFamilies = useMemo(() => {
        const families = new Set<string>();
        for (const model of currentArchModels) {
            families.add(extractFamily(model));
        }
        return Array.from(families).sort();
    }, [currentArchModels]);

    // Current family from selected model
    const currentFamily = useMemo(() =>
        extractFamily(primaryModelId),
        [primaryModelId]);

    // Available scales for current family
    const familyScales = useMemo(() =>
        getAvailableScales(currentArchModels, currentFamily),
        [currentArchModels, currentFamily]);

    // Source dimensions
    const sourceWidth = videoState.inputWidth || 0;
    const sourceHeight = videoState.inputHeight || 0;

    // Output dimensions (computed based on mode)
    const outputDimensions = useMemo(() => {
        if (resolutionMode === 'target' && targetWidth && targetHeight) {
            return { width: targetWidth, height: targetHeight };
        }
        return {
            width: sourceWidth * scaleFactor,
            height: sourceHeight * scaleFactor,
        };
    }, [resolutionMode, targetWidth, targetHeight, sourceWidth, sourceHeight, scaleFactor]);

    // Required scale for target resolution
    const requiredScale = useMemo(() => {
        if (resolutionMode !== 'target' || !targetWidth || !targetHeight) return null;
        return calculateRequiredScale(sourceWidth, sourceHeight, targetWidth, targetHeight);
    }, [resolutionMode, sourceWidth, sourceHeight, targetWidth, targetHeight]);

    // VRAM estimation
    const estimatedVRAM = useMemo(() =>
        estimateVRAM(outputDimensions.width, outputDimensions.height, architectureClass),
        [outputDimensions, architectureClass]);

    const isHighVRAM = estimatedVRAM > 8; // Warning threshold: 8GB

    // ─── Event Handlers ────────────────────────────────────────────────────────

    const handleArchitectureChange = useCallback((arch: ArchitectureClass) => {
        const archModels = modelGroups[arch] ?? [];
        if (archModels.length === 0) return;

        // Select first model from this architecture
        const firstModel = archModels[0];
        if (!firstModel) return; // Guard against undefined

        const scale = extractScale(firstModel) ?? scaleFactor;

        setUpscaleConfig({
            architectureClass: arch,
            primaryModelId: firstModel,
            scaleFactor: scale as UpscaleScale,
        });

        onModelChange(firstModel);
        loadModel(firstModel);

        console.debug(`[AIUpscaleNode] Architecture changed to ${arch}, model: ${firstModel}`);
    }, [modelGroups, scaleFactor, setUpscaleConfig, onModelChange, loadModel]);

    const handleFamilyChange = useCallback((family: string) => {
        const model = findModelByFamilyAndScale(currentArchModels, family, scaleFactor);
        if (!model) return;

        setUpscaleConfig({ primaryModelId: model });
        onModelChange(model);
        loadModel(model);

        console.debug(`[AIUpscaleNode] Family changed to ${family}, model: ${model}`);
    }, [currentArchModels, scaleFactor, setUpscaleConfig, onModelChange, loadModel]);

    const handleScaleChange = useCallback((scale: UpscaleScale) => {
        const model = findModelByFamilyAndScale(currentArchModels, currentFamily, scale);

        setUpscaleConfig({
            scaleFactor: scale,
            ...(model && { primaryModelId: model }),
        });

        if (model) {
            onModelChange(model);
            loadModel(model);
        }

        console.debug(`[AIUpscaleNode] Scale changed to ${scale}×, model: ${model ?? 'unchanged'}`);
    }, [currentArchModels, currentFamily, setUpscaleConfig, onModelChange, loadModel]);

    const handleResolutionPreset = useCallback((preset: keyof typeof RESOLUTION_PRESETS) => {
        const info = RESOLUTION_PRESETS[preset];
        setUpscaleConfig({
            resolutionMode: 'target',
            targetWidth: info.width,
            targetHeight: info.height,
            resolutionPreset: preset,
        });
    }, [setUpscaleConfig]);

    const handleCustomResolution = useCallback((width: number | null, height: number | null) => {
        setUpscaleConfig({
            resolutionMode: width || height ? 'target' : 'scale',
            targetWidth: width,
            targetHeight: height,
            resolutionPreset: 'custom',
        });
    }, [setUpscaleConfig]);

    const handleResolutionModeChange = useCallback((mode: 'scale' | 'target') => {
        setUpscaleConfig({ resolutionMode: mode });
    }, [setUpscaleConfig]);

    const handleToggle = useCallback(() => {
        setUpscaleConfig({ isEnabled: !isEnabled });
    }, [isEnabled, setUpscaleConfig]);

    // ─── Render ────────────────────────────────────────────────────────────────

    const archInfo = getArchitectureInfo(architectureClass);

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
        }}>
            {/* ═══ SUMMARY LINE ═══ */}
            {isEnabled ? (
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 12px',
                    background: 'linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,255,136,0.02))',
                    borderRadius: '8px',
                    border: '1px solid rgba(0,255,136,0.15)',
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ fontSize: '14px' }}>{archInfo.icon}</span>
                        <div>
                            <div style={{
                                fontSize: '12px',
                                fontWeight: 700,
                                color: 'var(--brand-primary)',
                                fontFamily: 'var(--font-mono)',
                            }}>
                                {currentFamily} {scaleFactor}×
                            </div>
                            <div style={{
                                fontSize: '9px',
                                color: 'var(--text-muted)',
                                marginTop: '1px',
                            }}>
                                {archInfo.label}
                            </div>
                        </div>
                    </div>
                    <div style={{
                        fontSize: '9px',
                        color: 'var(--text-secondary)',
                        fontFamily: 'var(--font-mono)',
                        textAlign: 'right',
                    }}>
                        {sourceWidth > 0 && (
                            <>
                                {sourceWidth}×{sourceHeight} → {outputDimensions.width}×{outputDimensions.height}
                                {showTech && isHighVRAM && (
                                    <div style={{ color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '4px', justifyContent: 'flex-end', marginTop: '2px' }}>
                                        <IconWarning /> ~{estimatedVRAM.toFixed(1)}GB VRAM
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>
            ) : (
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '12px',
                    background: 'rgba(255,255,255,0.02)',
                    borderRadius: '8px',
                    border: '1px dashed rgba(255,255,255,0.1)',
                    color: 'var(--text-muted)',
                }}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: '8px' }}>
                        <circle cx="12" cy="12" r="10" />
                        <line x1="4.93" y1="4.93" x2="19.07" y2="19.07" />
                    </svg>
                    <span style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.08em' }}>
                        AI UPSCALE BYPASSED
                    </span>
                </div>
            )}

            {/* ═══ MAIN CONTROLS (disabled when bypassed) ═══ */}
            <div style={{
                opacity: isEnabled ? 1 : 0.4,
                pointerEvents: isEnabled ? 'auto' : 'none',
                transition: 'opacity 0.2s',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
            }}>

                {/* ─── ARCHITECTURE SELECTION ─────────────────────────────────────── */}
                <CollapsibleSection
                    title="ARCHITECTURE"
                    subtitle={archInfo.label}
                    defaultOpen={true}
                >
                    <div style={{
                        display: 'flex',
                        gap: '6px',
                        flexWrap: 'wrap',
                    }}>
                        {(['CNN', 'GAN', 'Transformer', 'Diffusion', 'Lightweight'] as ArchitectureClass[]).map(arch => (
                            <ArchitectureCard
                                key={arch}
                                archClass={arch}
                                selected={architectureClass === arch}
                                onClick={() => handleArchitectureChange(arch)}
                                modelCount={modelGroups[arch]?.length ?? 0}
                            />
                        ))}
                    </div>
                    <p style={{
                        fontSize: '9px',
                        color: 'var(--text-muted)',
                        margin: '8px 0 0',
                        lineHeight: '1.4',
                    }}>
                        {archInfo.description}
                    </p>
                </CollapsibleSection>

                {/* ─── PRIMARY MODEL ──────────────────────────────────────────────── */}
                {/* ─── PIPELINE FEATURES ─────────────────────────────────────── */}
                {pipelineFeatures && onPipelineToggle && (
                    <div style={{
                        display: 'flex',
                        gap: '6px',
                        padding: '8px 10px',
                        background: 'rgba(0,0,0,0.15)',
                        borderRadius: '8px',
                        border: '1px solid rgba(255,255,255,0.06)',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                    }}>
                        <span style={{
                            fontSize: '8px',
                            color: 'var(--text-muted)',
                            fontWeight: 600,
                            letterSpacing: '0.05em',
                            marginRight: '4px',
                        }}>
                            PIPELINE
                        </span>
                        <PipelineToggle
                            label="ADR"
                            enabled={pipelineFeatures.adr_enabled}
                            onToggle={() => onPipelineToggle('adr_enabled', !pipelineFeatures.adr_enabled)}
                        />
                        <PipelineToggle
                            label="TEMPORAL"
                            enabled={pipelineFeatures.temporal_enabled}
                            onToggle={() => onPipelineToggle('temporal_enabled', !pipelineFeatures.temporal_enabled)}
                        />
                        <PipelineToggle
                            label="LUMA"
                            enabled={pipelineFeatures.luma_only}
                            onToggle={() => onPipelineToggle('luma_only', !pipelineFeatures.luma_only)}
                        />
                        <PipelineToggle
                            label="SHARP"
                            enabled={pipelineFeatures.sharpen_strength > 0}
                            onToggle={() => onPipelineToggle('sharpen_strength', pipelineFeatures.sharpen_strength > 0 ? 0 : 0.3)}
                        />
                    </div>
                )}

                <CollapsibleSection
                    title="PRIMARY MODEL"
                    subtitle={currentFamily}
                    defaultOpen={true}
                >
                    {/* Family selector */}
                    <div style={{ marginBottom: '10px' }}>
                        <label style={{
                            display: 'block',
                            fontSize: '8px',
                            color: 'var(--text-muted)',
                            fontWeight: 600,
                            letterSpacing: '0.05em',
                            marginBottom: '6px',
                        }}>
                            MODEL FAMILY
                        </label>
                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                            {currentFamilies.map(family => (
                                <button
                                    key={family}
                                    onClick={() => handleFamilyChange(family)}
                                    style={{
                                        flex: currentFamilies.length <= 3 ? 1 : '0 0 auto',
                                        minWidth: '60px',
                                        height: '36px',
                                        padding: '0 12px',
                                        borderRadius: '5px',
                                        border: currentFamily === family
                                            ? '1px solid var(--brand-primary)'
                                            : '1px solid rgba(255,255,255,0.08)',
                                        background: currentFamily === family
                                            ? 'var(--brand-dim)'
                                            : 'rgba(255,255,255,0.03)',
                                        color: currentFamily === family
                                            ? 'var(--brand-primary)'
                                            : 'var(--text-secondary)',
                                        cursor: 'pointer',
                                        transition: 'all 0.15s ease',
                                        fontSize: '10px',
                                        fontWeight: 700,
                                    }}
                                >
                                    {family}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Scale selector */}
                    <div>
                        <label style={{
                            display: 'block',
                            fontSize: '8px',
                            color: 'var(--text-muted)',
                            fontWeight: 600,
                            letterSpacing: '0.05em',
                            marginBottom: '6px',
                        }}>
                            SCALE FACTOR
                        </label>
                        <div style={{ display: 'flex', gap: '6px' }}>
                            {([2, 3, 4] as UpscaleScale[]).map(scale => (
                                <ScaleToggle
                                    key={scale}
                                    scale={scale}
                                    selected={scaleFactor === scale}
                                    available={familyScales.includes(scale)}
                                    onClick={() => handleScaleChange(scale)}
                                />
                            ))}
                        </div>
                    </div>
                </CollapsibleSection>

                {/* ─── CUSTOM RESOLUTION ──────────────────────────────────────────── */}
                <CollapsibleSection
                    title="CUSTOM RESOLUTION"
                    subtitle={resolutionMode === 'target' ? `${targetWidth}×${targetHeight}` : `${scaleFactor}× scale`}
                    defaultOpen={false}
                >
                    {/* Mode toggle */}
                    <div style={{
                        display: 'flex',
                        gap: '6px',
                        marginBottom: '10px',
                    }}>
                        <button
                            onClick={() => handleResolutionModeChange('scale')}
                            style={{
                                flex: 1,
                                height: '32px',
                                borderRadius: '5px',
                                border: resolutionMode === 'scale'
                                    ? '1px solid var(--brand-primary)'
                                    : '1px solid rgba(255,255,255,0.08)',
                                background: resolutionMode === 'scale'
                                    ? 'var(--brand-dim)'
                                    : 'rgba(255,255,255,0.03)',
                                color: resolutionMode === 'scale'
                                    ? 'var(--brand-primary)'
                                    : 'var(--text-secondary)',
                                cursor: 'pointer',
                                fontSize: '9px',
                                fontWeight: 700,
                            }}
                        >
                            USE SCALE FACTOR
                        </button>
                        <button
                            onClick={() => handleResolutionModeChange('target')}
                            style={{
                                flex: 1,
                                height: '32px',
                                borderRadius: '5px',
                                border: resolutionMode === 'target'
                                    ? '1px solid var(--brand-primary)'
                                    : '1px solid rgba(255,255,255,0.08)',
                                background: resolutionMode === 'target'
                                    ? 'var(--brand-dim)'
                                    : 'rgba(255,255,255,0.03)',
                                color: resolutionMode === 'target'
                                    ? 'var(--brand-primary)'
                                    : 'var(--text-secondary)',
                                cursor: 'pointer',
                                fontSize: '9px',
                                fontWeight: 700,
                            }}
                        >
                            USE TARGET SIZE
                        </button>
                    </div>

                    {resolutionMode === 'target' && (
                        <>
                            {/* Presets */}
                            <div style={{ marginBottom: '10px' }}>
                                <label style={{
                                    display: 'block',
                                    fontSize: '8px',
                                    color: 'var(--text-muted)',
                                    fontWeight: 600,
                                    letterSpacing: '0.05em',
                                    marginBottom: '6px',
                                }}>
                                    PRESETS
                                </label>
                                <div style={{ display: 'flex', gap: '6px' }}>
                                    {(Object.keys(RESOLUTION_PRESETS) as Array<keyof typeof RESOLUTION_PRESETS>).map(preset => (
                                        <ResolutionPresetButton
                                            key={preset}
                                            preset={preset}
                                            selected={resolutionPreset === preset}
                                            onClick={() => handleResolutionPreset(preset)}
                                        />
                                    ))}
                                </div>
                            </div>

                            {/* Custom input */}
                            <div style={{
                                display: 'flex',
                                gap: '10px',
                                alignItems: 'center',
                            }}>
                                <div style={{ flex: 1 }}>
                                    <label style={{
                                        display: 'block',
                                        fontSize: '8px',
                                        color: 'var(--text-muted)',
                                        fontWeight: 600,
                                        letterSpacing: '0.05em',
                                        marginBottom: '4px',
                                    }}>
                                        WIDTH
                                    </label>
                                    <input
                                        type="number"
                                        value={targetWidth ?? ''}
                                        onChange={(e) => handleCustomResolution(
                                            e.target.value ? parseInt(e.target.value, 10) : null,
                                            targetHeight
                                        )}
                                        placeholder="Auto"
                                        style={{
                                            width: '100%',
                                            height: '32px',
                                            padding: '0 8px',
                                            borderRadius: '5px',
                                            border: '1px solid rgba(255,255,255,0.1)',
                                            background: 'rgba(0,0,0,0.3)',
                                            color: 'var(--text-primary)',
                                            fontSize: '11px',
                                            fontFamily: 'var(--font-mono)',
                                        }}
                                    />
                                </div>
                                <span style={{ color: 'var(--text-muted)', marginTop: '16px' }}>×</span>
                                <div style={{ flex: 1 }}>
                                    <label style={{
                                        display: 'block',
                                        fontSize: '8px',
                                        color: 'var(--text-muted)',
                                        fontWeight: 600,
                                        letterSpacing: '0.05em',
                                        marginBottom: '4px',
                                    }}>
                                        HEIGHT
                                    </label>
                                    <input
                                        type="number"
                                        value={targetHeight ?? ''}
                                        onChange={(e) => handleCustomResolution(
                                            targetWidth,
                                            e.target.value ? parseInt(e.target.value, 10) : null
                                        )}
                                        placeholder="Auto"
                                        style={{
                                            width: '100%',
                                            height: '32px',
                                            padding: '0 8px',
                                            borderRadius: '5px',
                                            border: '1px solid rgba(255,255,255,0.1)',
                                            background: 'rgba(0,0,0,0.3)',
                                            color: 'var(--text-primary)',
                                            fontSize: '11px',
                                            fontFamily: 'var(--font-mono)',
                                        }}
                                    />
                                </div>
                            </div>

                            {/* Computed scale info */}
                            {requiredScale !== null && (
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    marginTop: '10px',
                                    padding: '8px 10px',
                                    background: 'rgba(0,0,0,0.2)',
                                    borderRadius: '5px',
                                    border: '1px solid rgba(255,255,255,0.05)',
                                }}>
                                    <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
                                        Required Scale
                                    </span>
                                    <span style={{
                                        fontSize: '11px',
                                        fontWeight: 700,
                                        fontFamily: 'var(--font-mono)',
                                        color: requiredScale > 4 ? '#fbbf24' : 'var(--brand-primary)',
                                    }}>
                                        {requiredScale.toFixed(2)}×
                                        {requiredScale > 4 && ' (may require multiple passes)'}
                                    </span>
                                </div>
                            )}
                        </>
                    )}

                    {/* VRAM warning — only visible when tech specs are enabled */}
                    {showTech && isHighVRAM && (
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            marginTop: '10px',
                            padding: '8px 10px',
                            background: 'rgba(251,191,36,0.1)',
                            borderRadius: '5px',
                            border: '1px solid rgba(251,191,36,0.2)',
                        }}>
                            <IconWarning />
                            <span style={{ fontSize: '9px', color: '#fbbf24' }}>
                                High VRAM usage estimated (~{estimatedVRAM.toFixed(1)}GB). Consider using a smaller resolution or Lightweight architecture.
                            </span>
                        </div>
                    )}
                </CollapsibleSection>
            </div>
        </div>
    );
};

export default AIUpscaleNode;
