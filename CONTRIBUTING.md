# Contributing to TradingChassis Core

Contributions should preserve TradingChassis Core as a deterministic,
runtime-agnostic library.

## Package Scope

- Core owns canonical events, state reduction, strategy evaluation boundary,
  candidate reconciliation, policy admission, execution-control plan/apply,
  and `CoreStepResult`.
- Core does not own runtime orchestration, venue adapters, dispatch lifecycle,
  `hftbacktest`, or deployment/config wiring.

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

- Core accepts canonical events through `EventStreamEntry` and
  `process_event_entry` / `process_canonical_event`.
- Core returns deterministic `CoreStepResult`; runtime dispatch happens later.
- Do not introduce runtime/backtest imports (`core_runtime`, `hftbacktest`).
- Do not restore `GateDecision`, snapshot lifecycle compatibility, or
  runtime-owned decision contracts.
- Pydantic models are the source of truth for contract structure.

## Changing Core Behavior

### Canonical events

- Add event models in `tradingchassis_core/core/domain/types.py`.
- Register canonical category handling in `core/domain/event_model.py`.
- Update canonical reduction behavior in `core/domain/processing.py`.

### CoreStep/CoreWakeupStep pipeline

- Update `core/domain/processing_step.py` for deterministic flow changes.
- Keep reconciliation/policy/apply transitions explicit and side-effect-safe.

### Policy and risk behavior

- Implement policy checks in `core/risk/` and wire through
  `evaluate_policy_intent`.
- Keep risk admission as policy-only; no dispatch/runtime side effects.

### Execution-control behavior

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

