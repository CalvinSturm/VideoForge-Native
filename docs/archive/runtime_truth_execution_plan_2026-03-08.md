# Runtime Truth Execution Plan

Status: Approved execution plan
Created: 2026-03-08
Scope: Shared runtime-truth contracts, runtime wiring, tool pass-through, and optional artifact persistence
Use this for: Progressive implementation tracking of the runtime-truth hardening work
Do not use this for: Current shipped behavior without checking code and canonical docs

## 1. Progress Tracking
- [x] PR 1: Define shared runtime truth contracts
- [x] PR 2: Migrate Python and native to shared effective snapshots
- [x] PR 3: Add shared observed metrics mapping
- [ ] PR 4: Make bench and smoke pass through runtime truth
- [ ] PR 5: Add optional runtime truth artifacts
- [ ] PR 6: Finalize docs and remove migration compatibility

## 2. Locked Decisions
- `RuntimeConfigSnapshot` is the canonical runtime-owned representation of resolved execution truth before meaningful work begins.
- `RunObservedMetrics` is the canonical runtime-owned representation of observed execution facts during and after execution.
- Config truth and observed metrics stay separate in code, logs, tool output, and persisted artifacts.
- Shared fields are allowed only where they have the same semantics across routes.
- Route-specific truth lives in optional extension blocks, not overloaded shared fields.
- Missing metrics mean unavailable or untrustworthy, never zero.
- Bench and smoke are consumers of runtime truth, not reconstructors of it.
- CLI-native coverage is limited to adapter-visible truth and must not imply downstream `rave` internals.
- Tool JSON, logs, and persisted artifacts must serialize the same underlying structs, not parallel hand-built shapes.
- Native fallback is represented as one canonical effective snapshot plus explicit fallback metadata.
- Runtime-truth persistence is a separate artifact system, not an ownership transfer into `run_manifest.rs`.

## 3. Schema Evolution Rules
- Every shared runtime-truth schema includes `schema_version`.
- Additive field growth is allowed within a version.
- Rename, removal, or semantic reinterpretation requires a version bump.
- Route-specific extension objects must be optional and namespaced.
- Shared serializer helpers must be used by app code, smoke, bench, and persistence surfaces.
- Compatibility aliases are temporary migration aids only and must be removed after consumers are updated.

## 4. Origin-Bearing Fields
Only fields with real ambiguity between user intent and runtime resolution should carry `FieldOrigin`.

Initial origin-bearing fields:
- executor selection
- batch size
- precision when runtime-adjusted
- scale when inferred or overridden
- cache policy when env-driven or runtime-forced

Plain resolved fields without origin metadata unless proven necessary:
- input path
- output path
- run ID
- route ID
- model path or model key once resolved

## 5. PR Plan

### [x] PR 1: Define Shared Runtime Truth Contracts
**Primary goal:** Add the shared runtime-truth schema layer and lock its semantics before changing runtime behavior.

**Depends on:** None

**Likely files:**
- `src-tauri/src/runtime_truth.rs` or equivalent new module
- `src-tauri/src/lib.rs`

**In scope:**
- Define `RuntimeConfigSnapshot`
- Define `RunObservedMetrics`
- Define `FieldOrigin`
- Define route-specific extension strategy
- Define schema versioning rules in code comments and tests
- Define missing-field semantics for metrics
- Define the initial origin-bearing field set

**Out of scope:**
- Runtime wiring in Python or native paths
- Bench or smoke output changes
- On-disk artifact persistence

**Acceptance criteria:**
- [x] `RuntimeConfigSnapshot` excludes timings, throughput, counters, and post-run summaries
- [x] `RunObservedMetrics` excludes user intent, config resolution rationale, and planned-only values
- [x] Shared fields are limited to route-comparable semantics
- [x] Route-specific data is isolated behind optional namespaced extension blocks
- [x] Missing metrics are represented as absent or null according to schema rules, never zero-as-unknown
- [x] Only explicitly approved fields carry `FieldOrigin`
- [x] Serialization tests lock the required fields and version fields

**Verification:**
- [x] Unit tests for schema serialization
- [x] Unit tests for required vs optional fields
- [x] Unit tests for version field presence

**Why this PR boundary is correct:**
- This is the contract-definition PR. Later work should consume these rules, not redefine them.

### [x] PR 2: Migrate Python And Native To Shared Effective Snapshots
**Primary goal:** Replace route-specific run-start snapshot structs with one shared effective snapshot model.

**Depends on:** PR 1

**Likely files:**
- `src-tauri/src/commands/upscale.rs`
- `src-tauri/src/commands/native_engine.rs`

**In scope:**
- Replace Python route-specific snapshot structs with the shared schema
- Replace native route-specific snapshot structs with the shared schema
- Use one shared serializer/helper path across both routes
- Represent fallback as canonical effective snapshot plus explicit fallback metadata

**Out of scope:**
- Observed metrics
- Tool output changes
- Artifact persistence

**Acceptance criteria:**
- [x] Python emits the shared top-level snapshot schema
- [x] Native emits the shared top-level snapshot schema
- [x] Native fallback does not require consumers to reconcile multiple sibling snapshots to discover final truth
- [x] Effective route, requested route, and fallback metadata are explicit where applicable
- [x] Existing route-specific snapshot names are removed or deliberately compatibility-wrapped for a short migration window

**Verification:**
- [x] Unit tests for Python snapshot mapping
- [x] Unit tests for native direct snapshot mapping
- [x] Unit tests for native fallback snapshot mapping

**Why this PR boundary is correct:**
- It finishes config-truth normalization before metrics and consumer-surface changes.

### [x] PR 3: Add Shared Observed Metrics Mapping
**Primary goal:** Introduce runtime-owned observed metrics without inventing false comparability.

**Depends on:** PR 1

**Likely files:**
- `src-tauri/src/commands/upscale.rs`
- `src-tauri/src/commands/native_engine.rs`

**In scope:**
- Add `RunObservedMetrics` builders or mapping helpers
- Map Python only to currently trustworthy observed fields
- Map native metrics from existing trustworthy native perf data
- Use route-specific metric extension blocks where semantics differ

**Out of scope:**
- Tool JSON output changes
- Artifact persistence

**Acceptance criteria:**
- [x] Shared metric fields mean the same thing across Python and native
- [x] Route-specific observations live in extension blocks
- [x] Python does not emit fabricated decode, AI, or encode timings if they are not actually observed
- [x] Native mapping is lossless with respect to trustworthy `NativePerfReport` data
- [x] Unknown or unavailable metrics are absent, not zero-filled

**Verification:**
- [x] Unit tests for absent-field behavior
- [x] Unit tests for route-specific metric extension presence
- [x] Unit tests for native mapping fidelity

**Why this PR boundary is correct:**
- It separates runtime configuration truth from observed execution truth.

### [ ] PR 4: Make Bench And Smoke Pass Through Runtime Truth
**Primary goal:** Make tooling a direct consumer of runtime-owned truth.

**Depends on:** PR 2, PR 3

**Likely files:**
- `src-tauri/src/bin/videoforge_bench.rs`
- `src-tauri/src/bin/smoke.rs`
- shared serializer helpers if needed

**In scope:**
- Embed shared `RuntimeConfigSnapshot` in tool JSON output
- Embed shared `RunObservedMetrics` in tool JSON output
- Use direct serialization helpers from the shared runtime-truth layer

**Out of scope:**
- On-disk artifact persistence

**Acceptance criteria:**
- [x] Tool JSON embeds runtime-owned structs or direct serializer output
- [x] Bench does not reconstruct canonical route or config truth from CLI args
- [x] Smoke does not reconstruct canonical route or config truth from tool-local inference
- [x] Route identity, fallback state, and resolved config are visible from the embedded snapshot
- [x] Observed metrics in tool JSON come from runtime-owned metrics, not tool-local recomputation

**Verification:**
- [x] JSON shape tests where helper boundaries exist
- [ ] Manual sample-output verification for bench
- [ ] Manual sample-output verification for smoke

**Why this PR boundary is correct:**
- Tools are downstream interfaces and should not own truth logic.

### [ ] PR 5: Add Optional Runtime Truth Artifacts
**Primary goal:** Persist runtime truth as optional artifacts without overloading existing run-manifest ownership.

**Depends on:** PR 2, PR 3

**Likely files:**
- new runtime-truth artifact module
- `src-tauri/src/commands/upscale.rs`
- `src-tauri/src/commands/native_engine.rs`
- optional reference integration with `src-tauri/src/run_manifest.rs`

**Artifact model decision:**
- Separate persistence module from `run_manifest.rs`
- Two sibling files, not one overloaded manifest:
  - `runtime_config_snapshot.json`
  - `run_observed_metrics.json`

**In scope:**
- Define artifact layout and naming
- Add opt-in persistence gate
- Write shared snapshot and metrics artifacts for Python and native
- Keep schema ownership in the runtime-truth module

**Out of scope:**
- Making `run_manifest.rs` the canonical container for runtime truth
- Making artifact persistence default unless explicitly decided later

**Acceptance criteria:**
- [ ] Python and native write the same runtime-truth artifact layout when enabled
- [ ] Artifact filenames are stable and documented
- [ ] Artifact persistence is explicitly opt-in
- [ ] Runtime-truth schema ownership remains separate from `run_manifest.rs`
- [ ] Any manifest integration is referential only, not an ownership merge

**Verification:**
- [ ] Path and write tests for artifact creation
- [ ] Manual verification of artifact layout on a sample Python run
- [ ] Manual verification of artifact layout on a sample native run

**Why this PR boundary is correct:**
- Persistence is an operational concern and should not distort the contract or mapping reviews.

### [ ] PR 6: Finalize Docs And Remove Migration Compatibility
**Primary goal:** Align the canonical docs to the implemented shared schema family and remove transition shims.

**Depends on:** PR 2 through PR 5

**Likely files:**
- `docs/runtime_path_contracts.md`
- `docs/metrics_trust.md`
- `docs/architecture_status_truth.md`

**In scope:**
- Update canonical docs to the final shared runtime-truth model
- Remove temporary route-specific snapshot naming from docs
- Remove compatibility notes or aliases introduced only for migration

**Out of scope:**
- New runtime behavior

**Acceptance criteria:**
- [ ] Canonical docs describe one shared snapshot schema family
- [ ] Canonical docs describe one shared observed-metrics schema family
- [ ] Temporary route-specific snapshot naming is removed
- [ ] Docs match the actual code-level serializer and artifact shape

**Verification:**
- [ ] Manual consistency review against code
- [ ] Manual consistency review against bench and smoke output

**Why this PR boundary is correct:**
- Documentation should trail the stable implementation, not intermediate migration states.

## 6. Cross-Cutting Risks
- Shared schema pressure can over-normalize Python and native truth. Keep the common core small and use extensions aggressively.
- Migration can create temporary event/log compatibility churn. If compatibility aliases are needed, make them short-lived and explicit.
- CLI-native truth must remain adapter-scoped. Do not imply internal `rave` behavior the wrapper cannot observe.
- Python metrics will remain partial until more instrumentation exists. Partial truth is acceptable; fabricated symmetry is not.

## 7. Definition Of Done
- [ ] Every execution family emits the same shared `RuntimeConfigSnapshot` top-level schema
- [ ] Every execution family can emit `RunObservedMetrics` without inventing unsupported metrics
- [ ] Shared fields are route-comparable by definition, not by name only
- [ ] Bench and smoke pass through runtime-owned truth instead of reconstructing it
- [ ] Optional runtime-truth artifacts can be emitted with the same layout for Python and native
- [ ] Canonical docs match the final schema, artifact model, and tool output
