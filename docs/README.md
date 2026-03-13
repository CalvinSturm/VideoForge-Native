# Docs Index

This directory contains the durable architecture contracts plus archived planning and handoff snapshots.

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

## Archived Status And Planning

- [`archive/implementation_plan.md`](archive/implementation_plan.md)
  - Archived native-engine execution tracker and status snapshot.
- [`archive/native_engine_handoff_2026-03-07.md`](archive/native_engine_handoff_2026-03-07.md)
  - Archived native handoff snapshot.
- [`archive/native_engine_implementation_plan.md`](archive/native_engine_implementation_plan.md)
  - Archived native workstream implementation tracker.

## Cross-Engine Audit And Planning Snapshots

- [`audits/video_upscaler_audit_2026-03-07.md`](audits/video_upscaler_audit_2026-03-07.md)
  - Broader cross-engine audit covering Python, native direct, and native-cli.
- [`plans/video_upscaler_patch_plan_2026-03-07.md`](plans/video_upscaler_patch_plan_2026-03-07.md)
  - PR-shaped cleanup and alignment plan derived from the audit.
- [`plans/video_upscaler_benchmark_plan_2026-03-07.md`](plans/video_upscaler_benchmark_plan_2026-03-07.md)
  - Benchmark policy and instrumentation plan derived from the audit.

## Working Rule

- Prefer the five canonical docs above for system truth.
- Treat files under [`archive/`](archive/) as historical context, not current-source-of-truth guidance.
- Treat audit, plan, and handoff files as supporting context unless a canonical doc explicitly delegates to them.
