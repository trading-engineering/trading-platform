# Changelog

This changelog starts from the clean Core package baseline.

## [Unreleased]

### Added

- Deterministic `run_core_step` and `run_core_wakeup_step` architecture.
- CoreWakeupStep final Strategy evaluation: reduce all entries, then `CoreWakeupStrategyEvaluator` once.
- Canonical Event input models and `EventStreamEntry`/`ProcessingPosition`.
- Intent candidate record Pipeline with dominance/reconciliation.
- Risk Engine (policy-only) admission and Execution Control plan/apply integration.
- `CoreStepResult.dispatchable_intents` and `ControlSchedulingObligation` outputs.
- Core-only quickstart example and focused semantics test coverage.

### Changed

- Package metadata, exports, and docs reset for standalone Core library identity.
- Pydantic models established as contract source of truth across public API docs.

### Removed

- Legacy compatibility-first contracts and references not part of the clean baseline.
