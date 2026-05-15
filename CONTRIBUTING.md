# Contributing to TradingChassis Core

Contributions should preserve TradingChassis Core as a deterministic,
Runtime-agnostic library.

> Terminology: Definitions and related terms match the [canonical
> terminology](https://tradingchassis.github.io/docs/latest/00-guides/terminology/).

## Package Scope

- Core owns canonical Events, State reduction, Strategy evaluation boundary,
  candidate reconciliation, Risk Engine (policy), Execution Control plan/apply,
  and `CoreStepResult`.
- Core does not own Runtime orchestration, Venue Adapters, dispatch lifecycle,
  or deployment/config wiring.

## Development Setup

From `core`:

```bash
python -m pip install -e ".[dev]"
```

## Validation Commands

Run before opening a PR:

```bash
python examples/core_step_quickstart.py
python -m pytest -q
python -m build
```

## Architecture Rules

- Core accepts canonical Events through `EventStreamEntry` and
  `process_event_entry` / `process_canonical_event`.
- Core returns deterministic `CoreStepResult`; Runtime dispatches.
- Do not introduce Runtime imports.
- Pydantic models are the source of truth for contract structure.

## Changing Core Behavior

### Canonical Events

- Add Event models in `tradingchassis_core/core/domain/types.py`.
- Register canonical category handling in `core/domain/event_model.py`.
- Update canonical reduction behavior in `core/domain/processing.py`.

### CoreStep/CoreWakeupStep Pipeline

- Update `core/domain/processing_step.py` for deterministic flow changes.
- Keep reconciliation/policy/apply transitions explicit and side-effect-safe.

### Risk Engine (policy) behavior

- Implement policy checks in `core/risk/` and wire through
  `evaluate_policy_intent`.
- Keep Risk Engine admission as policy-only; no dispatch/Runtime side effects.

### Execution Control behavior

- Update plan/apply stages in `core/domain/execution_control_plan.py` and
  `core/domain/execution_control_apply.py`.
- Preserve `ControlSchedulingObligation` as non-canonical output.

### Public API exports and docs

- Update `tradingchassis_core/__init__.py` for intentional public exports only.
- Sync docs in `README.md` and `docs/reference/public-api.md`.

## Pull Request Checklist

- [ ] Package remains Core-only and deterministic.
- [ ] Public API changes are intentional and tested.
- [ ] Quickstart still runs via public imports.
- [ ] `python -m pytest -q` passes.
- [ ] `python -m build` succeeds.
- [ ] README/docs/changelog updated to match behavior.
