# Contributing to TradingChassis Core

Thanks for contributing to `tradingchassis_core`.

This repository is the Core semantic package. Keep changes deterministic, explicit, and scoped to
Core responsibilities.

## Repository Scope

Core owns semantic models and deterministic processing contracts, including canonical event
reduction and CoreStep/CoreWakeupStep result contracts.

Core does not own runtime I/O, venue adapters, or runtime orchestration concerns.

## Development Setup

From the `core` repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Use Python 3.11+.

## Architecture Constraints

Contributions must preserve the accepted MVP baseline and boundaries:

- Core must not depend on runtime/hftbacktest integration layers
- Core consumes canonical events and returns deterministic Core outputs
- Runtime owns external I/O, dispatch execution, and scheduling realization timing
- For migrated flag-on paths, runtime dispatches from `CoreStepResult.dispatchable_intents`
- `ControlSchedulingObligation` remains non-canonical Core output
- `GateDecision` remains compatibility-only for legacy/default-off paths

Do not introduce behavior that implies final-architecture completion.

## Testing Expectations

- Add or update Core semantic tests for any behavior change
- Keep tests deterministic and scoped to the Core package
- Runtime integration validation may require a separate runtime environment; treat missing runtime
  dependencies as environment/tooling blockers, not as Core semantic failures

## Documentation Expectations

When semantics or boundaries change:

- Update relevant files under `docs/`
- Keep MVP baseline, compatibility, and post-MVP boundaries explicit
- Avoid mixing speculative final-state claims into MVP docs

## Contribution Hygiene

- Prefer small, focused pull requests
- Avoid mixing broad refactors with documentation-only or behavior-only changes
- Do not introduce docs tooling or generated docs unless explicitly requested for the phase
