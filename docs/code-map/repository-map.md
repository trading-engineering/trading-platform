# Repository Map

This page is a code navigation guide for the Core repository. It is not an API reference.

## High-level tree

```text
core/
  tradingchassis_core/
    __init__.py
    core/
      domain/
      execution_control/
      risk/
      events/
      ports/
    strategies/
  tests/
  docs/
```

## Area responsibilities

- `tradingchassis_core/__init__.py`
  - Curated package export surface (`__all__`) and compatibility-facing public entry points.
- `tradingchassis_core/core/domain/`
  - Core semantic models and orchestration primitives:
    - event and intent models (`types.py`)
    - processing boundaries (`processing.py`, `processing_order.py`)
    - CoreStep/CoreWakeupStep pipeline (`processing_step.py`)
    - result and decision models (`step_result.py`, `step_decision.py`)
    - candidate/combination/policy/execution-control planning and apply helpers
- `tradingchassis_core/core/risk/`
  - Policy and compatibility gate logic (`risk_policy.py`, `risk_engine.py`, `risk_config.py`).
- `tradingchassis_core/core/execution_control/`
  - Queue/rate/inflight dispatchability mechanics and internal control helper types.
- `tradingchassis_core/core/events/`
  - Event bus/sink and telemetry-style event record abstractions.
- `tradingchassis_core/core/ports/`
  - Protocol-style integration boundaries (engine context, venue adapter/policy contracts).
- `tradingchassis_core/strategies/`
  - Strategy interface and strategy-facing configuration models.
- `tests/`
  - Core semantic/regression coverage for deterministic behavior and boundaries.
- `docs/`
  - Core repository documentation (MVP baseline, concepts, flows, code map, and reference).

## Recommended reading order

1. `tradingchassis_core/__init__.py` (public exports and what is intentionally exposed)
2. `tradingchassis_core/core/domain/types.py` (event/intent models and canonical vocabulary)
3. `tradingchassis_core/core/domain/processing_step.py` (`run_core_step` and wakeup pipeline)
4. `tradingchassis_core/core/domain/processing.py` + `state.py` (reduction boundary and state updates)
5. `tradingchassis_core/core/domain/intent_combination.py` + `candidate_intent.py`
6. `tradingchassis_core/core/domain/policy_risk_decision.py` (+ compatibility mapping context)
7. `tradingchassis_core/core/domain/execution_control_apply.py` and related decision/plan modules
8. `tests/` to confirm expected behavior and migration/compatibility semantics

## Boundary note

Runtime and venue I/O internals are out of scope for this repository map. Treat runtime behavior as
an integration boundary consumed by Core outputs.
