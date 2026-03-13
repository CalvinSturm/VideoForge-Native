# Capability Matrix

Purpose: explicit support matrix for what the product currently guarantees.

Interpretation rules:
- `Supported`: implemented and intended for normal use
- `Supported with caveats`: implemented but gated, limited, or route-specific
- `Not supported`: not a valid product claim
- `Needs verification`: behavior may exist, but is not yet a contract

## Execution Modes

| Capability | Python image | Python video | Native direct video | Native CLI video |
|---|---|---|---|---|
| App entrypoint | `upscale_request` | `upscale_request` | `upscale_request_native` | `upscale_request_native` |
| Default path | Supported | Supported | No | No |
| Requires ONNX model | No | No | Yes | Yes |
| Supports PyTorch model formats | Supported | Supported | Not supported | Not supported in this repo contract |
| Media type | Image | Video | Video | Video |
| Native family opt-in required | No | No | Yes | Yes |
| Compile-time feature required | No | No | Yes | Command surface exists; direct implementation feature only |
| Audio preservation contract | N/A | Not explicitly surfaced in result | Supported with caveats | Supported with caveats |
| Direct-to-CLI fallback | No | No | Supported for selected error classes | N/A |
| Run manifest support | Not documented | Optional | No equivalent canonical artifact in current code | No equivalent canonical artifact in current code |
| Stage metrics contract | Limited | Limited | Richer | Moderate, adapter-derived |
| Pause/resume | Not supported | Not supported | Not supported | Not supported |
| Cancel | UI-local only, not a true backend stop contract | UI-local only, not a true backend stop contract | UI-local only, not a true backend stop contract | UI-local only, not a true backend stop contract |
| Research/blending features | Supported with Python-path caveats | Supported with Python-path caveats | Not represented as native contract | Not represented as native contract |

## UI Capability Claims

| UI concept | Current truth | Source |
|---|---|---|
| Queue pause/resume | Not presented as a supported queue control | `ui/src/components/JobsPanel.tsx`, `ui/src/App.tsx` |
| Native engine toggle | Intent flag, not runtime proof | `ui/src/Store/useJobStore.tsx`, `ui/src/hooks/useUpscaleJob.ts` |
| Native result details | UI can display backend-reported route/result metadata after completion | `ui/src/types.ts`, `ui/src/components/JobsPanel.tsx` |

## Model Format Rules

| Model format | Python family | Native family |
|---|---|---|
| ONNX | Supported | Supported candidate |
| PyTorch weights | Supported | Not supported |

Primary files:
- `src-tauri/src/models.rs`
- `ui/src/hooks/useUpscaleJob.ts`

## Status Rules

The following must not be claimed as supported until the code and matrix both say so:
- pause/resume
- backend-enforced cancel/stop semantics
- apples-to-apples metric comparability across all routes
- full internal CLI-native runtime equivalence with direct-native

## Update Trigger

Update this matrix whenever any of the following changes:
- UI control surface
- backend command semantics
- route support
- metric exposure
- persistence behavior
- model eligibility
