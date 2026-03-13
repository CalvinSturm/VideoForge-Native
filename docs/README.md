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

## Active Working Docs

- [`handoff_native_direct_glitch_followup_2026-03-13.md`](handoff_native_direct_glitch_followup_2026-03-13.md)
  - Active native direct investigation handoff.
- [`release_hygiene_checklist.md`](release_hygiene_checklist.md)
  - Release and packaging metadata checklist.

## Historical Docs

- [`archive/`](archive/)
  - Historical handoffs, completed plans, prior audits, and retired root-level docs moved out of the active docs surface.
  - See [`archive/README.md`](archive/README.md) for archive usage rules.

## Working Rule

- Prefer the five canonical docs above for system truth.
- Treat files under [`archive/`](archive/) as historical context, not current-source-of-truth guidance.
- Treat audit, plan, and handoff files as supporting context unless a canonical doc explicitly delegates to them.
