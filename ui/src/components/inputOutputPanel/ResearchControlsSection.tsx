import React from "react";
import { invoke } from "@tauri-apps/api/core";
import type { UpscaleMode } from "../../types";
import {
  HF_METHODS,
  RESEARCH_DEFAULTS,
  type ResearchConfig,
} from "./researchConfig";
import { truncateModelName } from "./panelHelpers";
import { IconChevronDown, IconCpu } from "./panelIcons";
import { ColorSlider, PipelineConnector, PipelineNode, Tooltip } from "./panelPrimitives";

type ResearchValue = number | string | boolean;

interface Props {
  advancedOpen: boolean;
  applyResearchPreset: (presetName: string) => void;
  availableModels: string[];
  isAIActive: boolean;
  mode: UpscaleMode;
  researchConfig: ResearchConfig;
  setAdvancedOpen: (open: boolean) => void;
  showResearchParams: boolean;
  updateResearchParam: (key: keyof ResearchConfig, value: ResearchValue) => void;
}

interface SliderDef {
  key: keyof ResearchConfig;
  label: string;
  text: string;
  min: number;
  max: number;
  step: number;
  digits?: number;
}

const disabledWrap = (disabled: boolean): React.CSSProperties => ({
  opacity: disabled ? 0.35 : 1,
  pointerEvents: disabled ? "none" : "auto",
  transition: "opacity 0.2s ease",
});

const sectionTitleStyle: React.CSSProperties = {
  fontSize: "8px",
  color: "var(--text-muted)",
  fontWeight: 700,
  letterSpacing: "0.08em",
  padding: "4px 0 6px",
  borderBottom: "1px solid rgba(255,255,255,0.04)",
  marginBottom: "4px",
};

const switchRowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "8px 12px",
  background: "rgba(0,0,0,0.2)",
  borderRadius: "8px",
  border: "1px solid rgba(255,255,255,0.04)",
  marginBottom: "4px",
};

const ResearchSwitch = ({
  checked,
  disabled,
  onChange,
}: {
  checked: boolean;
  disabled?: boolean;
  onChange: () => void;
}) => (
  <div
    role="switch"
    aria-checked={checked}
    tabIndex={disabled ? -1 : 0}
    onClick={disabled ? undefined : onChange}
    onKeyDown={disabled ? undefined : (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onChange();
      }
    }}
    style={{
      width: "32px",
      height: "18px",
      borderRadius: "9px",
      position: "relative",
      cursor: disabled ? "not-allowed" : "pointer",
      background: checked && !disabled ? "rgba(245,158,11,0.4)" : "rgba(255,255,255,0.08)",
      border: checked && !disabled ? "1px solid rgba(245,158,11,0.5)" : "1px solid rgba(255,255,255,0.1)",
      transition: "all 0.2s ease",
      outline: "none",
      opacity: disabled ? 0.5 : 1,
    }}
  >
    <div
      style={{
        width: "14px",
        height: "14px",
        borderRadius: "50%",
        background: checked && !disabled ? "#f59e0b" : "rgba(255,255,255,0.3)",
        position: "absolute",
        top: "1px",
        left: checked && !disabled ? "15px" : "1px",
        transition: "all 0.2s ease",
      }}
    />
  </div>
);

const slider = (
  researchConfig: ResearchConfig,
  updateResearchParam: Props["updateResearchParam"],
  { key, label, text, min, max, step, digits = 2 }: SliderDef,
) => (
  <Tooltip key={String(key)} text={text} position="bottom">
    <ColorSlider
      label={label}
      value={researchConfig[key] as number}
      onChange={(value) => updateResearchParam(key, value)}
      min={min}
      max={max}
      step={step}
      accentColor="#f59e0b"
      formatValue={(value) => value.toFixed(digits)}
    />
  </Tooltip>
);

export const ResearchControlsSection: React.FC<Props> = ({
  advancedOpen,
  applyResearchPreset,
  availableModels,
  isAIActive,
  mode,
  researchConfig,
  setAdvancedOpen,
  showResearchParams,
  updateResearchParam,
}) => {
  if (!isAIActive || !showResearchParams) return null;

  const isVideoMode = mode === "video";
  const hasSecondary = researchConfig.secondary_model !== "None";
  const isAdrActive = hasSecondary && researchConfig.adr_enabled;
  const isTemporalActive = researchConfig.temporal_enabled;
  const activeFeatures = [
    hasSecondary && "2ND MODEL",
    researchConfig.adr_enabled && hasSecondary && "ADR",
    researchConfig.temporal_enabled && isVideoMode && "TEMPORAL",
    researchConfig.luma_only && hasSecondary && "LUMA",
    researchConfig.sharpen_strength > 0 && "SHARP",
  ].filter(Boolean) as string[];
  const modifiedCount = (Object.keys(RESEARCH_DEFAULTS) as (keyof ResearchConfig)[]).filter((key) => {
    const currentValue = researchConfig[key];
    const defaultValue = RESEARCH_DEFAULTS[key];
    return typeof currentValue === "number" && typeof defaultValue === "number"
      ? Math.abs(currentValue - defaultValue) > 0.001
      : currentValue !== defaultValue;
  }).length;

  const weightSliders: SliderDef[] = [
    { key: "alpha_structure", label: "STRUCTURE", text: "Weight for structural fidelity (edges, geometry). Higher values preserve hard lines and shapes at the cost of softer textures. Default 0.50.", min: 0, max: 1, step: 0.01 },
    { key: "alpha_texture", label: "TEXTURE", text: "Weight for texture detail recovery. Controls how aggressively fine surface detail (fabric, skin pores, grain) is reconstructed. Default 0.30.", min: 0, max: 1, step: 0.01 },
    { key: "alpha_perceptual", label: "PERCEPTUAL", text: "Weight for perceptual similarity. Optimizes output to look natural to the human eye rather than pixel-exact. Higher values may smooth fine detail. Default 0.15.", min: 0, max: 1, step: 0.01 },
    { key: "alpha_diffusion", label: "DIFFUSION", text: "Weight for diffusion-based refinement pass. Adds subtle generative detail but can introduce hallucinated content at high values. Keep low for archival work. Default 0.05.", min: 0, max: 1, step: 0.01 },
  ];
  const frequencySliders: SliderDef[] = [
    { key: "low_freq_strength", label: "LOW FREQ", text: "Amplification of low-frequency content (smooth gradients, large shapes). Values above 1.0 boost, below 1.0 attenuate. Increase to strengthen broad tonal structure. Default 1.00.", min: 0, max: 2, step: 0.01 },
    { key: "mid_freq_strength", label: "MID FREQ", text: "Amplification of mid-frequency content (medium detail, object contours). Controls the body of visible sharpness. Boost for crisper mid-detail, reduce to soften. Default 1.00.", min: 0, max: 2, step: 0.01 },
    { key: "high_freq_strength", label: "HIGH FREQ", text: "Amplification of high-frequency content (fine edges, noise, micro-texture). Higher values sharpen fine detail but may amplify noise or ringing artifacts. Default 1.00.", min: 0, max: 2, step: 0.01 },
  ];
  const advancedSliders: SliderDef[] = [
    { key: "freq_low_sigma", label: "LOW SIGMA", text: "Gaussian blur sigma for the low-frequency band separation. Larger values capture broader structures in the low band. Increase for smoother tonal rolloff. Default 4.0.", min: 0.5, max: 10, step: 0.1, digits: 1 },
    { key: "freq_mid_sigma", label: "MID SIGMA", text: "Gaussian blur sigma for the mid-frequency band separation. Controls the cutoff between mid and high detail. Lower values shift more content into the high band. Default 1.5.", min: 0.5, max: 5, step: 0.1, digits: 1 },
    { key: "edge_threshold", label: "EDGE THRESHOLD", text: "Gradient magnitude threshold for classifying a pixel as an edge. Pixels above this threshold are routed to the edge model branch. Lower values detect more edges. Default 0.50.", min: 0, max: 1, step: 0.01 },
    { key: "texture_threshold", label: "TEXTURE THRESHOLD", text: "Local variance threshold for classifying a region as textured. Pixels above this threshold are routed to the texture model branch. Lower values classify more area as textured. Default 0.20.", min: 0, max: 1, step: 0.01 },
    { key: "spatial_freq_mix", label: "SPATIAL-FREQ MIX", text: "Blend ratio between spatial routing and frequency-band routing. At 0.0 only spatial routing is used; at 1.0 only frequency bands drive the blend. Default 0.50.", min: 0, max: 1, step: 0.01 },
  ];

  return (
    <>
      <PipelineConnector isActive />
      <PipelineNode
        title="Research Parameters"
        icon={<IconCpu />}
        nodeNumber={3}
        isActive
        accentColor="#f59e0b"
        defaultOpen={false}
        extra={
          <div style={{ display: "flex", gap: "4px" }}>
            {(["performance", "balanced", "quality"] as const).map((preset) => (
              <button
                key={preset}
                onClick={(event) => {
                  event.stopPropagation();
                  applyResearchPreset(preset);
                }}
                style={{
                  height: "22px",
                  fontSize: "8px",
                  padding: "0 8px",
                  borderRadius: "4px",
                  border: researchConfig.preset === preset ? "1px solid rgba(245,158,11,0.5)" : "1px solid rgba(255,255,255,0.1)",
                  background: researchConfig.preset === preset ? "rgba(245,158,11,0.15)" : "transparent",
                  color: researchConfig.preset === preset ? "#f59e0b" : "var(--text-muted)",
                  fontWeight: 700,
                  cursor: "pointer",
                  letterSpacing: "0.05em",
                  transition: "all 0.15s ease",
                  textTransform: "uppercase",
                }}
              >
                {preset === "performance" ? "PERF" : preset === "balanced" ? "BAL" : "QUAL"}
              </button>
            ))}
          </div>
        }
      >
        <div style={{ display: "flex", alignItems: "center", gap: "6px", padding: "8px 10px", background: "rgba(245,158,11,0.05)", borderRadius: "6px", border: "1px solid rgba(245,158,11,0.15)", marginBottom: "8px", flexWrap: "wrap" }}>
          {activeFeatures.length > 0 ? activeFeatures.map((feature) => (
            <span key={feature} style={{ fontSize: "7px", fontWeight: 700, padding: "2px 5px", borderRadius: "3px", background: "rgba(245,158,11,0.15)", border: "1px solid rgba(245,158,11,0.3)", color: "#f59e0b", letterSpacing: "0.04em" }}>{feature}</span>
          )) : <span style={{ fontSize: "8px", color: "var(--text-muted)", fontWeight: 600 }}>DEFAULT CONFIG</span>}
          {modifiedCount > 0 && <span style={{ fontSize: "8px", color: "var(--text-muted)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>{modifiedCount} modified</span>}
        </div>

        <div style={{ marginBottom: "4px" }}>
          <div style={sectionTitleStyle}>SECONDARY MODEL</div>
          <Tooltip text="Select a secondary (GAN/texture) model for dual-model blending. 'None' uses only the primary model. Enables ADR detail injection and luma blending." position="bottom">
            <div style={{ padding: "8px 12px", background: "rgba(0,0,0,0.2)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.04)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                <label style={{ fontSize: "10px", color: "var(--text-secondary)", fontWeight: 600, letterSpacing: "0.03em" }}>MODEL</label>
                <span style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "#f59e0b", fontWeight: 600 }}>{researchConfig.secondary_model}</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "4px", maxHeight: "120px", overflowY: "auto" }}>
                {["None", ...availableModels].map((availableModel) => (
                  <button
                    key={availableModel}
                    onClick={() => updateResearchParam("secondary_model", availableModel)}
                    title={availableModel}
                    style={{
                      fontSize: "8px",
                      height: "28px",
                      borderRadius: "5px",
                      background: researchConfig.secondary_model === availableModel ? "rgba(245,158,11,0.15)" : "rgba(255,255,255,0.03)",
                      border: researchConfig.secondary_model === availableModel ? "1px solid rgba(245,158,11,0.4)" : "1px solid rgba(255,255,255,0.08)",
                      color: researchConfig.secondary_model === availableModel ? "#f59e0b" : "var(--text-muted)",
                      fontWeight: researchConfig.secondary_model === availableModel ? 700 : 500,
                      cursor: "pointer",
                      transition: "all 0.15s ease",
                      letterSpacing: "0.03em",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {availableModel === "None" ? "NONE" : truncateModelName(availableModel)}
                  </button>
                ))}
              </div>
            </div>
          </Tooltip>
        </div>

        <div style={{ marginBottom: "4px" }}>
          <div style={sectionTitleStyle}>DETAIL ENHANCEMENT</div>
          <div style={disabledWrap(!hasSecondary)}>
            <Tooltip text="Enable Adaptive Detail Residual. Extracts high-frequency texture from the secondary (GAN) model and injects it into the primary (structure) output for richer surface detail." position="bottom">
              <div style={switchRowStyle}>
                <div>
                  <label style={{ fontSize: "10px", color: "var(--text-secondary)", fontWeight: 600, letterSpacing: "0.03em" }}>ADR ENABLED</label>
                  {!hasSecondary && <div style={{ fontSize: "8px", color: "rgba(245,158,11,0.6)", marginTop: "2px" }}>Requires secondary model</div>}
                </div>
                <ResearchSwitch checked={researchConfig.adr_enabled} onChange={() => updateResearchParam("adr_enabled", !researchConfig.adr_enabled)} disabled={!hasSecondary} />
              </div>
            </Tooltip>
          </div>
          <div style={disabledWrap(!isAdrActive)}>
            {slider(researchConfig, updateResearchParam, { key: "detail_strength", label: "DETAIL STRENGTH", text: "How much GAN high-frequency texture to inject into the structure output. 0 = no detail injection, 1 = full GAN residual. Requires ADR enabled with a secondary model. Default 0.50.", min: 0, max: 1, step: 0.01 })}
            <Tooltip text="Blend only the luminance (Y) channel in YCbCr space. Preserves the structure model's colour accuracy while injecting GAN brightness detail. Prevents colour shifts." position="bottom">
              <div style={switchRowStyle}>
                <label style={{ fontSize: "10px", color: "var(--text-secondary)", fontWeight: 600, letterSpacing: "0.03em" }}>LUMA ONLY</label>
                <ResearchSwitch checked={researchConfig.luma_only} onChange={() => updateResearchParam("luma_only", !researchConfig.luma_only)} disabled={!isAdrActive} />
              </div>
            </Tooltip>
          </div>
        </div>

        <div style={{ marginBottom: "4px" }}>
          <div style={sectionTitleStyle}>POST-PROCESSING</div>
          {slider(researchConfig, updateResearchParam, { key: "edge_strength", label: "EDGE STRENGTH", text: "Sobel edge mask strength for spatially-varying blend. Higher values apply stronger blending on edges, weaker on flat regions. 0 = uniform blend. Default 0.30.", min: 0, max: 1, step: 0.01 })}
          {slider(researchConfig, updateResearchParam, { key: "sharpen_strength", label: "SHARPEN", text: "GPU unsharp mask intensity applied after blending. Adds crispness to the final output. 0 = disabled, higher = sharper. Can amplify noise at high values. Default 0.00.", min: 0, max: 1, step: 0.01 })}
        </div>

        <div style={{ marginBottom: "4px" }}>
          <div style={{ ...sectionTitleStyle, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>TEMPORAL</span>
            {!isVideoMode && <span style={{ fontSize: "7px", color: "rgba(245,158,11,0.5)", fontWeight: 600, letterSpacing: "0.03em" }}>VIDEO MODE ONLY</span>}
          </div>
          <div style={disabledWrap(!isVideoMode)}>
            <Tooltip text="Enable exponential moving average (EMA) temporal stabilization across frames. Reduces inter-frame flicker in video upscaling." position="bottom">
              <div style={switchRowStyle}>
                <label style={{ fontSize: "10px", color: "var(--text-secondary)", fontWeight: 600, letterSpacing: "0.03em" }}>TEMPORAL EMA</label>
                <ResearchSwitch checked={researchConfig.temporal_enabled} onChange={() => updateResearchParam("temporal_enabled", !researchConfig.temporal_enabled)} disabled={!isVideoMode} />
              </div>
            </Tooltip>
            <div style={disabledWrap(!isTemporalActive)}>
              {slider(researchConfig, updateResearchParam, { key: "temporal_alpha", label: "TEMPORAL ALPHA", text: "EMA smoothing factor. Lower = more smoothing (more temporal averaging). Higher = faster response to new frames. Default 0.90.", min: 0, max: 1, step: 0.01 })}
              <Tooltip text="Flush all temporal EMA buffers. Use after seeking, changing clips, or when ghosting artifacts appear." position="bottom">
                <button
                  onClick={() => {
                    invoke("reset_temporal_buffer").catch(() => {});
                  }}
                  disabled={!isTemporalActive}
                  style={{ width: "100%", height: "30px", fontSize: "9px", fontWeight: 700, borderRadius: "6px", border: "1px solid rgba(245,158,11,0.3)", background: "rgba(245,158,11,0.08)", color: "#f59e0b", cursor: isTemporalActive ? "pointer" : "not-allowed", letterSpacing: "0.05em", transition: "all 0.15s ease", marginTop: "4px" }}
                >
                  RESET TEMPORAL BUFFER
                </button>
              </Tooltip>
            </div>
          </div>
        </div>

        <div style={{ marginBottom: "4px" }}>
          <div style={sectionTitleStyle}>MODEL WEIGHTS</div>
          {weightSliders.map((entry) => slider(researchConfig, updateResearchParam, entry))}
        </div>

        <div style={{ marginBottom: "4px" }}>
          <div style={sectionTitleStyle}>FREQUENCY BAND</div>
          {frequencySliders.map((entry) => slider(researchConfig, updateResearchParam, entry))}
        </div>

        <div style={{ marginBottom: "4px" }}>
          <div style={sectionTitleStyle}>HALLUCINATION</div>
          {slider(researchConfig, updateResearchParam, { key: "h_sensitivity", label: "SENSITIVITY", text: "How aggressively the detector flags AI-generated detail as hallucinated. Higher values catch more false detail but may suppress legitimate reconstruction. Default 1.00.", min: 0, max: 3, step: 0.01 })}
          {slider(researchConfig, updateResearchParam, { key: "h_blend_reduction", label: "BLEND REDUCTION", text: "Strength of blending applied to regions flagged as hallucinated. At 1.0, flagged regions are fully replaced with the source. Lower values allow partial AI detail to remain. Default 0.50.", min: 0, max: 1, step: 0.01 })}
        </div>

        <div style={{ marginBottom: "4px" }}>
          <div style={sectionTitleStyle}>SPATIAL ROUTING</div>
          {slider(researchConfig, updateResearchParam, { key: "edge_model_bias", label: "EDGE BIAS", text: "How strongly edge-detected regions prefer the structure-preserving model branch. Higher values keep hard edges sharper but may introduce stairstepping on diagonal lines. Default 0.70.", min: 0, max: 1, step: 0.01 })}
          {slider(researchConfig, updateResearchParam, { key: "texture_model_bias", label: "TEXTURE BIAS", text: "How strongly textured regions prefer the texture-recovery model branch. Increase for richer surface detail in complex areas (foliage, fabric). Default 0.70.", min: 0, max: 1, step: 0.01 })}
          {slider(researchConfig, updateResearchParam, { key: "flat_region_suppression", label: "FLAT SUPPRESSION", text: "Suppresses AI enhancement in flat, low-detail regions (sky, walls) to prevent noise amplification and false texture. Higher values apply more suppression. Default 0.30.", min: 0, max: 1, step: 0.01 })}
        </div>

        <div>
          <button
            onClick={() => setAdvancedOpen(!advancedOpen)}
            style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 0", border: "none", background: "none", cursor: "pointer", borderBottom: "1px solid rgba(255,255,255,0.04)", marginBottom: advancedOpen ? "4px" : "0" }}
          >
            <span style={{ fontSize: "8px", color: "var(--text-muted)", fontWeight: 700, letterSpacing: "0.08em" }}>ADVANCED</span>
            <span style={{ color: "var(--text-muted)", transform: advancedOpen ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.15s ease" }}>
              <IconChevronDown />
            </span>
          </button>
          {advancedOpen && (
            <div>
              {advancedSliders.map((entry) => slider(researchConfig, updateResearchParam, entry))}
              <Tooltip text="Algorithm used to extract high-frequency detail. Laplacian: second-order edges, general purpose. Sobel: first-order gradient, sharper edges. Highpass: simple subtraction, fast. FFT: spectral domain, most precise but slowest." position="bottom">
                <div style={{ display: "flex", flexDirection: "column", gap: "6px", padding: "10px 12px", background: "rgba(0,0,0,0.2)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.04)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <label style={{ fontSize: "10px", color: "var(--text-secondary)", fontWeight: 600, letterSpacing: "0.03em" }}>HF METHOD</label>
                    <span style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "#f59e0b", fontWeight: 600 }}>{researchConfig.hf_method.toUpperCase()}</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "4px" }}>
                    {HF_METHODS.map((method) => (
                      <button
                        key={method}
                        onClick={() => updateResearchParam("hf_method", method)}
                        style={{
                          fontSize: "9px",
                          height: "28px",
                          borderRadius: "5px",
                          background: researchConfig.hf_method === method ? "rgba(245,158,11,0.15)" : "rgba(255,255,255,0.03)",
                          border: researchConfig.hf_method === method ? "1px solid rgba(245,158,11,0.4)" : "1px solid rgba(255,255,255,0.08)",
                          color: researchConfig.hf_method === method ? "#f59e0b" : "var(--text-muted)",
                          fontWeight: researchConfig.hf_method === method ? 700 : 500,
                          cursor: "pointer",
                          transition: "all 0.15s ease",
                          textTransform: "uppercase",
                          letterSpacing: "0.03em",
                        }}
                      >
                        {method === "highpass" ? "HP" : method === "laplacian" ? "LAP" : method.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              </Tooltip>
            </div>
          )}
        </div>
      </PipelineNode>
    </>
  );
};
