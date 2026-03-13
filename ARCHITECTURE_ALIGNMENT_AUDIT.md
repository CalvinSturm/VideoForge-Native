# Architecture Alignment Audit

Date: 2026-03-13
Scope: current checked-in repo state after the cleanup follow-up plan was completed

Status language used in this document:
- `Current truth`: behavior directly supported by checked-in code or docs
- `Drift`: code, UI, or docs still send materially different signals
- `Closed since prior audit`: a previous architecture-alignment concern that is now materially addressed

## 1. Executive Summary

The repo is in substantially better shape than it was at the time of the earlier alignment audit.

`Closed since prior audit`:
- The durable contract docs now exist:
  - `docs/architecture_status_truth.md`
  - `docs/capability_matrix.md`
  - `docs/runtime_path_contracts.md`
  - `docs/state_and_persistence.md`
  - `docs/metrics_trust.md`
- The native command boundary is smaller and more explicit.
- The direct native path description in `src-tauri/src/commands/native_engine.rs` now matches the streamed FFmpeg demux/mux design.
- The cleanup follow-up tracker is complete and validation is green across the maintained matrix plus `cargo test --features native_engine --lib`.
- Smoke and benchmark tooling now share the same repo/runtime PATH bootstrap helper.
- Native runs now participate in the shared optional run-manifest artifact path.
- The README status section now reflects the post-cleanup current state instead of the older recovery narrative.

`Current truth`:
- The main remaining problems are not broad architecture confusion anymore.
- The largest alignment gaps called out in the earlier audit are now materially closed.

The codebase now looks like a repo that has largely solved its architecture-documentation crisis, but still has a few product-contract mismatches that can mislead contributors or users at runtime.

## 2. Current Strengths

### Durable truth sources now exist

`Current truth`:
- `README.md` points to the five canonical docs first.
- `docs/README.md` establishes an authority order between canonical docs, plans, and handoff files.
- The docs now name route IDs, gating rules, fallback semantics, persistence expectations, and metric trust boundaries.

Why this matters:
- Contributors no longer have to reconstruct the entire runtime model from plans and handoffs.

### Runtime truth is better instrumented

`Current truth`:
- Python and native paths both emit runtime snapshot and observed-metrics structures.
- Native direct-to-CLI fallback is represented explicitly in native runtime snapshot shaping.

Primary code:
- `src-tauri/src/commands/upscale.rs`
- `src-tauri/src/commands/native_routing.rs`
- `src-tauri/src/runtime_truth.rs`

Why this matters:
- The repo now has a real execution-truth layer rather than only comments and adapter assumptions.

### Native boundaries are materially clearer

`Current truth`:
- Native probing, routing, streaming IO, direct pipeline execution, and tooling are split into focused modules.
- `native_engine.rs` is now the app-facing native control plane rather than a grab bag of all native implementation details.

Primary code:
- `src-tauri/src/commands/native_engine.rs`
- `src-tauri/src/commands/native_probe.rs`
- `src-tauri/src/commands/native_routing.rs`
- `src-tauri/src/commands/native_streaming_io.rs`
- `src-tauri/src/commands/native_direct_pipeline.rs`

## 3. Remaining Findings

## 4. Recommended Next Moves

1. Use the canonical docs as the source of truth for future feature/status changes.
   The main remaining risk is drift reappearing, not a currently known contract hole.

## 5. Bottom Line

This repo no longer has an architecture-alignment emergency.

`Current truth`:
- The durable docs exist.
- Route contracts are named.
- Native routing and validation are cleaner.
- The cleanup follow-up work is complete.

That is a much healthier place to be. The next changes should be product-driven or capability-driven work, not another broad cleanup campaign.
