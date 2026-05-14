# Changelog

This changelog starts from the clean Core package baseline.

## [Unreleased]

### Added

- Deterministic `run_core_step` and `run_core_wakeup_step` architecture.
- Canonical Event input models and `EventStreamEntry`/`ProcessingPosition`.
- Intent candidate record pipeline with dominance/reconciliation.
- Policy-only risk admission and Execution Control plan/apply integration.
- `CoreStepResult.dispatchable_intents` and `ControlSchedulingObligation` outputs.
- Core-only quickstart example and focused semantics test coverage.

### Changed

- Package metadata, exports, and docs reset for standalone Core library identity.
- Pydantic models established as contract source of truth across public API docs.

### Removed

- Legacy compatibility-first contracts and references not part of the clean baseline.
