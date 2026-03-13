# User-Facing Feature Direction

Date: 2026-03-13
Status: active

## Purpose

This note captures the current user-facing product shape of VideoForge and the next highest-value UX and workflow features discussed after the cleanup and native recovery phase.

## Current Strengths

VideoForge already has a stronger user-facing foundation than a simple "AI upscaler" label suggests.

Implemented or confirmed current strengths include:

- crop
- transform
- simple auto color correction
- trim before export
- interpolation
- post-export before/after preview
- hold-to-show-original comparison
- split side-by-side comparison
- linked zoom for both compared images
- vertical before/after slider
- panel layout flexibility, including closing all panels to leave a strong preview-only media viewer experience

## Product Interpretation

The current feature set means VideoForge is already three things at once:

- enhancement tool
- export tool
- inspection and review tool

That is a better market position than "just an AI upscaler."

## What Users Likely Need Next

The next highest-value user-facing work is less about adding another image adjustment knob and more about making the workflow feel faster, clearer, and more professional.

Primary needs:

- clearer project and export flow
- stronger pre-export confidence
- better model guidance
- better route and fallback transparency
- presets and repeatable workflow speed
- stronger output management
- richer comparison across multiple exports

## UX Priorities

### 1. Comparison source selection

High-value next feature:

- allow users to compare not only original vs latest export, but also different exports against each other

Recommended comparison-source model:

- source A
- source B

Supported choices:

- original
- current export
- prior export 1
- prior export 2
- arbitrary selected export

All existing compare modes should continue to work with this model:

- hold-to-toggle
- split view
- side-by-side
- slider

Why this matters:

- it makes VideoForge a stronger decision-making tool for comparing:
  - different models
  - interpolation on/off
  - color correction on/off
  - scale choices
  - different export settings

### 2. Drag-and-drop into the media view

High-value workflow feature:

- allow users to drag supported media directly into the main preview/media area

Recommended behavior:

- if no job is loaded, the dropped file becomes the active media
- if a job is already loaded, behavior should be explicit:
  - replace current media
  - add as comparison media
  - or queue/import intentionally

Important UX rule:

- do not silently overload drag-and-drop behavior in confusing ways

Why this matters:

- it strengthens the viewer/workstation feel
- it reduces intake friction
- it makes the media panel feel like a real workspace instead of only a form-driven surface

### 3. Presets and workflow reuse

Very high-value product feature:

- save and reuse common workflows

Examples:

- clean upscale
- upscale + interpolation
- archival cleanup
- social clip enhancement

Why this matters:

- it improves repeatability
- it reduces setup friction
- it helps newer users get good results faster

### 4. Better export and result management

Users likely need:

- clearer output naming
- overwrite protection
- rerun with same settings
- duplicate job
- recent outputs
- export summary

Recommended export summary fields:

- output resolution
- duration
- model
- route
- elapsed time

### 5. Route and support clarity

Important product trust feature:

- explain native eligibility before the run
- show which route actually executed
- show fallback when it happened
- explain unsupported combinations in plain language

This should feel like confidence-building product UX, not like exposing engine internals for their own sake.

## Preview Mode Direction

Preview-only mode is strategically important.

Because the app already supports a strong preview/viewer mode, future polish should treat that as part of the product identity.

High-value polish areas:

- keyboard shortcuts
- fast compare-mode switching
- cleaner HUD / chrome behavior
- remembered panel and compare layout preferences
- remembered zoom / preview preferences

## Recommended Priority Order

If the goal is user-facing impact rather than adding more low-level controls, the recommended order is:

1. export selection for comparison
2. drag-and-drop into the media view
3. presets
4. export summary and rerun / duplicate ergonomics
5. route and fallback clarity
6. project / history view

## Practical Product Rule

Do not treat internal engine or route complexity as a user feature.

Users care more about:

- confidence
- speed
- clarity
- output quality

than about the exact pipeline topology under the hood.

## Related Docs

- `docs/product_engineering_roadmap_2026-03-13.md`
- `docs/next_phase_direction_2026-03-13.md`
