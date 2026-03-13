# Docs Index

This directory contains both the durable architecture contracts and the current native-engine planning notes.

## Start Here

- [`architecture_status_truth.md`](architecture_status_truth.md)
  - Canonical architecture and shipped-status summary.
- [`capability_matrix.md`](capability_matrix.md)
  - Canonical support matrix for routes, features, and guarantees.
- [`runtime_path_contracts.md`](runtime_path_contracts.md)
  - Canonical route, gating, and fallback contracts.
- [`state_and_persistence.md`](state_and_persistence.md)
  - Canonical operational state and persistence model.
- [`metrics_trust.md`](metrics_trust.md)
  - Canonical metric provenance and comparability rules.

## Status And Planning

- [`../implementation_plan.md`](../implementation_plan.md)
  - Native-engine execution tracker and archived status summary.
- [`native_engine_handoff_2026-03-07.md`](native_engine_handoff_2026-03-07.md)
  - Latest native handoff and resume point.

## Cross-Engine Audit And Planning Snapshots

- [`audits/video_upscaler_audit_2026-03-07.md`](audits/video_upscaler_audit_2026-03-07.md)
  - Broader cross-engine audit covering Python, native direct, and native-cli.
- [`plans/video_upscaler_patch_plan_2026-03-07.md`](plans/video_upscaler_patch_plan_2026-03-07.md)
  - PR-shaped cleanup and alignment plan derived from the audit.
- [`plans/video_upscaler_benchmark_plan_2026-03-07.md`](plans/video_upscaler_benchmark_plan_2026-03-07.md)
  - Benchmark policy and instrumentation plan derived from the audit.

## Working Rule

- Prefer the five canonical docs above for system truth.
- Prefer [`../implementation_plan.md`](../implementation_plan.md) for native workstream status.
- Prefer [`native_engine_handoff_2026-03-07.md`](native_engine_handoff_2026-03-07.md) for the latest native resume point.
- Treat audit, plan, and handoff files as supporting context unless a canonical doc explicitly delegates to them.
