# Archive Index

This directory contains historical docs that are still worth keeping for reference, but are **not** the current source of truth.

## What Belongs Here

- completed plans
- superseded handoffs
- one-off audits
- retired root-level notes moved out of the active docs surface
- old implementation snapshots that still have historical value

## How To Use Archived Docs

- Use archived docs for historical context, migration history, and reasoning trails.
- Do **not** use archived docs to infer current runtime contracts, shipped behavior, or active execution rules.
- Prefer the canonical docs in the parent `docs/` directory first:
  - `architecture_status_truth.md`
  - `capability_matrix.md`
  - `runtime_path_contracts.md`
  - `state_and_persistence.md`
  - `metrics_trust.md`

## Working Rule

If an archived doc disagrees with:

- checked-in code
- the canonical docs in `docs/`
- the current top-level `README.md`

then treat the archived doc as historical only.
