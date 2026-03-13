# Next Phase Direction

Date: 2026-03-13
Status: active

## Current State

- The major direct-native corruption issue has been fixed.
- The direct native path now uses packet-aware demux/mux boundaries.
- The decode-to-preprocess lifetime race fix has been landed.
- Canonical docs, README, smoke runbook, and AGENTS are aligned with the current codebase state.
- Stale plans, handoffs, and audits have been moved to `docs/archive/`.

## What Was Just Completed

- Structural cleanup and docs-surface cleanup are complete enough to stop spending time on broad repo hygiene.
- The native direct investigation is no longer the main workstream unless the regression returns.
- The live docs surface now points primarily at current contracts rather than historical planning material.

## Next Phase

The next phase should focus on product and performance work rather than more cleanup.

Primary buckets:

- native-path optimization
  - throughput
  - batching
  - stream overlap / sync reduction
  - buffer reuse and memory behavior
  - TensorRT cache behavior and warmup experience
- UI / UX improvements
  - clearer route/status messaging
  - better native eligibility feedback
  - better job/result visibility
  - stronger user-facing error and fallback reporting
- regression guardrails
  - stable smoke coverage
  - repeatable benchmark flows
  - faster detection of native regressions

## What Not To Restart

- Do not reopen broad architecture-cleanup work unless a new concrete drift problem appears.
- Do not treat archived plans or handoffs as current authority.
- Do not restart the native direct corruption investigation from scratch unless the issue clearly reproduces again.

## Source Of Truth

For current behavior and contracts, start with:

- `docs/architecture_status_truth.md`
- `docs/capability_matrix.md`
- `docs/runtime_path_contracts.md`
- `docs/state_and_persistence.md`
- `docs/metrics_trust.md`

Use:

- `docs/README.md` as the docs entrypoint
- `docs/archive/README.md` for historical material rules
- `SMOKE_TEST.md` for the live smoke runbook
