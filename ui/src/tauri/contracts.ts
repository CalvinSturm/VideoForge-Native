import type {
    EditState,
    ModelInfo,
    NativeUpscaleResultJson,
    RaveCommandJson,
    RaveEnvironmentJson,
} from "../types";
import type { ArchitectureClass } from "../Store/useJobStore";

export type TauriEditConfig = {
    trim_start: number;
    trim_end: number;
    crop: EditState["crop"];
    rotation: EditState["rotation"];
    flip_h: boolean;
    flip_v: boolean;
    fps: number;
    color: EditState["color"];
};

export interface UpscaleRequestArgs extends Record<string, unknown> {
    inputPath: string;
    outputPath: string;
    model: string;
    editConfig: TauriEditConfig;
    scale: number;
    architectureClass?: ArchitectureClass | null;
    secondaryModel?: string | null;
    blendAlpha?: number | null;
    resolutionMode: "scale" | "target";
    targetWidth: number | null;
    targetHeight: number | null;
}

export interface ExportRequestArgs extends Record<string, unknown> {
    inputPath: string;
    outputPath: string;
    editConfig: TauriEditConfig;
    scale: number;
}

export interface NativeUpscaleRequestArgs extends Record<string, unknown> {
    inputPath: string;
    outputPath: string;
    modelPath: string;
    scale: number;
    precision: "fp16" | "fp32";
    audio: boolean;
    maxBatch: number;
}

export interface RaveValidateArgs extends Record<string, unknown> {
    fixture: string | null;
    profile: string;
    bestEffort: boolean;
    strictAudit: boolean;
    mockRun: boolean;
}

export interface RaveBenchmarkArgs extends Record<string, unknown> {
    args: string[];
    strictAudit: boolean;
    mockRun: boolean;
    uiOptIn: boolean;
}

export type CheckEngineStatusResult = boolean;
export type GetModelsResult = ModelInfo[];
export type UpscaleRequestResult = string;
export type ExportRequestResult = string;
export type RaveValidateResult = RaveCommandJson;
export type RaveBenchmarkResult = RaveCommandJson;
export type NativeUpscaleRequestResult = NativeUpscaleResultJson;
export type RaveEnvironmentResult = RaveEnvironmentJson;
