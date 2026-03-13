# Metrics Trust

Purpose: define which metrics are trustworthy, where they come from, and what comparisons are valid.

## Rule

Do not compare metrics across routes unless:
- the metric name means the same thing
- the source is known
- cold-start and cache effects are disclosed
- fallback status is known

## Metric Provenance by Route

| Route | Metric surface | Provenance | Trust level | Notes |
|---|---|---|---|---|
| Python path | `JobProgress.stage_ms.total` | Runtime-emitted total wall time | Medium | Decode/ai/encode fields currently remain unset |
| Python path | `frames_decoded`, `frames_processed`, `frames_encoded` in progress | Runtime-emitted counters | Medium | Useful operationally, not a full stage-cost breakdown |
| Python benchmark | JSON `done.elapsed_ms`, `frames_encoded` | Bench wrapper + runtime report | Medium | Mostly end-to-end timing |
| Native direct | `NativePerfReport` fields | Runtime-native engine metrics plus wrapper shaping | High for route-local analysis | Best current perf surface in repo |
| Native CLI | `NativePerfReport` via adapter | JSON stdout + parsed stderr progress | Medium | Adapter-derived; semantics depend on downstream CLI contract |

## Current Truth

### Python path

- `StageTimingsMs` includes `decode`, `ai`, `encode`, `total`
- In current progress emission, only `total` is populated
- Therefore the repo does not yet support stage-comparable Python timing claims

Primary file:
- `src-tauri/src/commands/upscale.rs`

### Native direct

- Direct path reports:
  - elapsed time
  - frame counters
  - preprocess/inference/postprocess/encode averages
  - VRAM current and peak
  - route/fallback metadata

Primary files:
- `src-tauri/src/commands/native_engine.rs`
- `engine-v2/src/engine/pipeline.rs`

### Native CLI

- CLI perf is reconstructed from:
  - final JSON result
  - progress summary parsed from stderr
- This is useful, but not equivalent in trust level to native-direct internal metrics

Primary files:
- `src-tauri/src/rave_cli.rs`
- `src-tauri/src/commands/native_engine.rs`

## Comparison Rules

### Allowed now

- Native-direct run A vs native-direct run B when:
  - model
  - input
  - precision
  - batch
  - cache policy
  - route
  are held constant

- Python run A vs Python run B for end-to-end elapsed behavior under controlled inputs

### Not allowed as authoritative claims yet

- Python stage cost vs native-direct stage cost
- Native-direct vs native-cli micro-metric equivalence
- Cross-route claims that ignore cache warmup, fallback, or route identity

## Required Benchmark Disclosure

Every benchmark summary should include:
- route used
- requested executor
- executed executor
- fallback used
- model path or model key
- precision
- max batch
- TRT cache enabled/disabled
- warmup run count

## Update Trigger

Update this doc when:
- a metric field is added or renamed
- benchmark output changes
- Python stage timings become real
- native-cli metric provenance changes

