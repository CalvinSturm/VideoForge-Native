# Product And Engineering Roadmap

Date: 2026-03-13
Status: active

## Purpose

This roadmap turns the current post-cleanup direction into a concrete product and engineering sequence for VideoForge.

The main goal is not to add more architectural variety. The main goal is to make VideoForge feel:

- fast
- dependable
- clear
- professionally productized

## Product North Star

VideoForge should feel like one coherent local-first video enhancement product, not a collection of internal pipelines.

That means:

- one primary fast path
- one trustworthy fallback story
- clear execution visibility
- strong export reliability
- measured performance improvements rather than intuition-only tuning

## Current Starting Point

- The major direct-native corruption issue has been fixed.
- The direct native path now uses packet-aware demux and mux boundaries.
- The decode-to-preprocess lifetime race fix has been landed.
- Canonical docs and README alignment are in place.
- RunScope integration work has started, so evaluation and regression tracking now have a realistic home.

## Phase 1: Product Consolidation

Goal:

- turn the current route complexity into one product-facing engine experience

Recommended engine policy:

- primary route:
  - native direct for eligible ONNX video jobs
- native-family fallback:
  - native CLI-backed path
- broad-compatibility fallback:
  - Python path for unsupported or native-ineligible cases

Required outcomes:

- the route policy is explicit and deterministic
- the support matrix is frozen to what is actually shippable
- the UI presents one coherent engine story instead of exposing internal route complexity by default

## Phase 2: Reliability First

Goal:

- make exports boringly dependable

Required work:

- define a small release validation matrix
- cover representative short, medium, and long clips
- cover representative ONNX-native and Python-route scenarios
- validate:
  - route selection
  - fallback behavior
  - output correctness
  - startup reliability
  - mux/output sanity

Success condition:

- reliability bugs are treated as product blockers, not just debug curiosities

## Phase 3: Performance Measurement System

Goal:

- make optimization measurement-led

Required work:

- build a stable benchmark workflow for:
  - native direct
  - native CLI-backed
  - Python path
- measure:
  - startup time
  - first-frame latency
  - steady-state throughput
  - total elapsed time
  - VRAM peak
  - route-specific stage timings

Success condition:

- every optimization claim can be tied back to repeatable benchmark evidence

## Phase 4: Native Direct Optimization

Goal:

- make the primary supported path clearly best

Highest-value likely targets:

- reduce conservative synchronization where correctness allows
- improve stage overlap between:
  - decode
  - preprocess
  - inference
  - postprocess
  - encode
- improve buffer reuse and reduce allocation churn
- improve TensorRT cache behavior and warmup
- reduce host-side orchestration overhead

Important rule:

- do not reopen risky low-level pipeline work without benchmark evidence and regression checks

## Phase 5: UX And Product Surface

Goal:

- make performance and reliability legible to users

Required work:

- improve route and status visibility
- improve model-eligibility feedback
- improve progress and result surfaces
- improve fallback messaging
- improve output/reveal/rerun ergonomics

Desired user experience:

- users choose the job they want
- VideoForge chooses the best route
- VideoForge clearly reports what happened
- users do not need to understand internal engine topology to trust the product

## Phase 6: Evaluation And Regression Discipline

Goal:

- make quality and performance regressions visible early

Required work:

- use RunScope as the eval and run-history dashboard
- ingest VideoForge run artifacts in dev, smoke, and benchmark workflows
- track:
  - throughput trends
  - startup regressions
  - route-specific reliability
  - model-specific behavior

Success condition:

- regressions become easy to spot and compare instead of living only in terminal history and ad hoc notes

## Phase 7: Ship Readiness

Goal:

- narrow the repo into a real product promise

Required work:

- publish and enforce a realistic support matrix
- finalize route defaults
- hide or demote low-confidence options from the normal user path
- keep release hygiene, docs, smoke coverage, and benchmark expectations aligned

Success condition:

- the shipped experience matches what the docs and UI imply

## Recommended Immediate Sequence

1. Finish the first RunScope dashboard slice.
2. Define the explicit VideoForge engine policy and shipped support matrix.
3. Build the benchmark and reliability matrix.
4. Optimize native direct against measured bottlenecks.
5. Improve user-facing route, status, and fallback UX.
6. Move into release-hardening work.

## What Not To Do

- Do not expose internal engine complexity as the main product story.
- Do not optimize blindly without repeatable measurements.
- Do not reopen broad cleanup work unless concrete drift appears.
- Do not add more execution-path complexity before the existing product surface feels excellent.

## Related Docs

- `docs/next_phase_direction_2026-03-13.md`
- `docs/architecture_status_truth.md`
- `docs/capability_matrix.md`
- `docs/runtime_path_contracts.md`
- `docs/state_and_persistence.md`
- `docs/metrics_trust.md`
