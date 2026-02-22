# VideoForge 2-Week Execution Plan

Start date: 2026-02-23 (Monday)  
End date: 2026-03-06 (Friday)  
Scope: Productionize `rave-*` integration in VideoForge with strict-policy guarantees, CI coverage, and operator-ready diagnostics.

## References

- `docs/WINDOWS_RAVE_RUNTIME.md` — Windows-first runtime setup, profile controls, and troubleshooting.

## Goals

1. Make `rave-cli` (`validate`, `upscale`, `benchmark`) first-class VideoForge pipeline stages.
2. Enforce profile mapping centrally (`production_strict` default in release workflows, explicit `dev` override).
3. Split CI into deterministic non-GPU coverage and GPU runtime coverage.
4. Add integration guardrails around JSON schema/policy contracts and unsupported micro-batching.
5. Improve runtime diagnostics surfaced to VideoForge operators.

## Owners

- Platform Lead: pipeline orchestration, profile policy wiring, release gating.
- Runtime Engineer: loader/provider diagnostics, GPU-path behavior, determinism behavior.
- QA/CI Engineer: CI matrix, fixtures, contract/regression tests.
- UI/UX Engineer: operator-facing error presentation/log views.

## Workstreams And Deliverables

### WS1. Pipeline Integration

Deliverables:
- VideoForge stage wrappers for `validate`, `upscale`, `benchmark`.
- Structured parser for CLI `--json` output (stdout-only contract).
- Stage-level propagation of top-level `policy` object and validate audit fields.

Acceptance criteria:
- All stage wrappers run from a single integration entrypoint.
- JSON parsing rejects non-schema assumptions and does not inspect stderr.
- One final JSON object consumed per invocation in `--json` mode.

### WS2. Profile And Strictness Enforcement

Deliverables:
- Release workflow defaults to `production_strict`.
- Local/dev workflow supports explicit `dev` profile override.
- `production_strict` gate enforces audit capability requirement (`audit-no-host-copies`).

Acceptance criteria:
- Release jobs fail fast when strict policy or audit requirements are missing.
- Dev jobs run with best-effort defaults without ad hoc policy remapping.

### WS3. CI Split (Non-GPU + GPU)

Deliverables:
- Non-GPU CI lane using `RAVE_MOCK_RUN=1` strict validate fixture coverage.
- GPU CI lane for real CUDA/TensorRT execution and regression monitoring.
- Published baseline thresholds for benchmark deltas.

Acceptance criteria:
- Non-GPU jobs do not attempt CUDA init.
- GPU jobs fail on threshold regressions and strict policy violations.
- CI artifacts include JSON outputs for post-failure analysis.

### WS4. Contract Tests

Deliverables:
- Tests for JSON stdout contract (single final object).
- Tests for top-level `policy` presence in `upscale`, `benchmark`, `validate`.
- Determinism tests for stable hash-skip reason codes.
- `max_batch > 1` fail-fast tests with actionable message assertions.

Acceptance criteria:
- Contract test suite runs in CI and is required for merge.
- Schema-stability checks block removals/renames without version bump.

### WS5. Operator Diagnostics

Deliverables:
- Normalized error mapping for loader/provider failures.
- UI/log surfaces for actionable diagnostics (what failed, candidate paths/strategy, next action).
- Distinct presentation for strict-policy violations vs environment/runtime faults.

Acceptance criteria:
- Operator can identify root cause without reading raw stderr dumps.
- Missing CUDA/provider issues show actionable remediation hints.

## 2-Week Schedule

## Week 1 (2026-02-23 to 2026-02-27)

### Day 1 (Mon)
- Kickoff and scope lock.
- Confirm crate version pins in VideoForge.
- Define owners and branch strategy.

Exit criteria:
- Integration branch created.
- Workstream tickets approved with acceptance criteria.

### Day 2 (Tue)
- Implement stage wrappers for `validate/upscale/benchmark`.
- Add JSON parsing boundary and policy field extraction.

Exit criteria:
- Local runs succeed for all 3 commands in mock mode.

### Day 3 (Wed)
- Wire profile defaults (`production_strict` release, explicit `dev` override).
- Enforce strict audit capability gate behavior.

Exit criteria:
- Release-path dry run fails correctly when strict prerequisites are absent.

### Day 4 (Thu)
- Add non-GPU CI lane with `RAVE_MOCK_RUN=1` strict validate fixture.
- Add initial contract tests for stdout JSON and policy visibility.

Exit criteria:
- Non-GPU lane green on integration branch.

### Day 5 (Fri)
- Add deterministic reason-code tests and `max_batch > 1` fail-fast tests.
- Integrate diagnostics normalization layer in runtime boundary.

Exit criteria:
- Contract suite green locally and in CI.
- Diagnostics mapped for top loader/provider failure classes.

## Week 2 (2026-03-02 to 2026-03-06)

### Day 6 (Mon)
- Stand up GPU CI lane with runtime deps and fixture set.
- Capture baseline performance metrics for benchmark command.

Exit criteria:
- GPU lane executes end-to-end at least once.

### Day 7 (Tue)
- Add regression thresholds and fail conditions for GPU benchmarks.
- Harden strict-policy failure messaging and routing.

Exit criteria:
- GPU lane enforces thresholds and strict-policy contract.

### Day 8 (Wed)
- UI/log integration for operator diagnostics.
- Distinguish policy/config failures from environment/runtime failures.

Exit criteria:
- Error surface review passed by Runtime + UI owners.

### Day 9 (Thu)
- End-to-end reliability run across representative fixtures.
- Burn down remaining integration bugs and flaky assertions.

Exit criteria:
- No P0/P1 defects open.
- CI lanes stable for 3 consecutive runs.

### Day 10 (Fri)
- Release readiness review.
- Final contract signoff and rollout checklist execution.

Exit criteria:
- Signoff document approved.
- Rollout can proceed without waivers.

## Required CI Commands (Signoff Gate)

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test
./scripts/check_deps.sh
./scripts/check_docs.sh
```

Non-GPU strict validation command:

```bash
RAVE_MOCK_RUN=1 cargo run -p rave-cli --features audit-no-host-copies --bin rave -- validate --json --best-effort --profile production_strict --fixture tests/fixtures/validate_production_strict.json
```

## Risks And Mitigations

- GPU environment drift: pin runner image and dependency versions; run daily smoke on GPU lane.
- Schema drift in consumers: gate merges on contract tests and schema version discipline.
- Strict profile bypass risk: centralize profile mapping and ban downstream remapping.
- Poor diagnosability in runtime failures: require normalized actionable messages in UI/logs.

## Definition Of Done

1. VideoForge executes all three CLI stages using schema-based JSON parsing.
2. Strict profile behavior is enforced by default in release paths.
3. Non-GPU and GPU CI lanes are both green and required.
4. Contract tests cover JSON output, policy visibility, determinism reasons, and micro-batch fail-fast.
5. Operator diagnostics are actionable and validated by QA.
