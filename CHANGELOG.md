# Changelog

This changelog starts from the clean Core package baseline.

## [Unreleased]

### Removed

- `StrategyState.pop_queued_intents` (unused; Execution Control uses per-order queue helpers).
- `fold_event_stream_entries` and root export (use `process_event_entry` in order).
- Unused telemetry models in `core/events/events.py`.
- Unused `combine_candidate_intents` helper.
- Root exports for Execution Control apply detail types and `apply_execution_control_plan` (internal Execution Control apply stage).

### Added

- Deterministic `run_core_step` and `run_core_wakeup_step` architecture.
- CoreWakeupStep final Strategy evaluation: reduce all entries, then `CoreWakeupStrategyEvaluator` once.
- Canonical Event input models and `EventStreamEntry`/`ProcessingPosition`.
- Intent pipeline candidate records with dominance/reconciliation.
- Risk Engine (policy-only) admission and Execution Control plan/apply integration.
- `CoreStepResult.dispatchable_intents` and `ControlSchedulingObligation` outputs.
- Core-only quickstart example and focused semantics test coverage.
- Root export of `PolicyIntentEvaluator` and documentation of extension points vs convenience implementations.
- Pipeline integration tests for `RiskEngine` as `policy_evaluator` in `run_core_step`.
- `FillEvent` reducer and pipeline tests.
- Runnable Risk Engine example at `examples/core_step_with_risk_engine.py`.
- Extension-point docs under `docs/` and U3 candidate list at `docs/roadmap/dead-code-cleanup-candidates.md`.
- Explicit terminal canonical Order lifecycle Events:
  `OrderCanceledEvent`, `OrderRejectedEvent`, and `OrderExpiredEvent`.
- Terminal lifecycle reduction behavior and focused semantics tests for
  deterministic working-order removal, canonical projection terminal state, and
  inflight cleanup.

### Changed

- Package metadata, exports, and docs reset for standalone Core library identity.
- Pydantic models established as contract source of truth across public API docs.
- README clarifies internally wired pipeline vs externally supplied extension points.
- Runtime/Core contract wording hardened with a normative Runtime-obligation
  matrix, explicit `CoreStepResult` output semantics, and clearer
  `ControlSchedulingObligation` vs `ControlTimeEvent` boundary language.
- Canonical `MarketEvent` contract wording now explicitly documents the current
  book-only reduction baseline; trade-shaped payloads are explicitly unsupported
  for canonical reduction in this slice.
- Public API docs/tests hardening for Core-F2: root Public API classification,
  Advanced API labeling, root-vs-non-root clarity, and explicit surface lock
  coverage for all current root exports. No exports were removed or renamed.

### Removed

- Legacy compatibility-first contracts and references not part of the clean baseline.
