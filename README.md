# TradingChassis Core

`tradingchassis_core` is a deterministic trading decision engine core library.

It provides a clean Core-only package baseline centered on `run_core_step` and
`run_core_wakeup_step`, with Pydantic contracts as the source of truth.

## What TradingChassis Core Is

- Canonical event models and event taxonomy for Core reduction
- Processing-order contracts: `EventStreamEntry` and `ProcessingPosition`
- Deterministic state reduction and strategy evaluator boundary
- Intent models, candidate records, and dominance/reconciliation
- Policy admission (`PolicyRiskDecision`) and execution-control plan/apply
- `CoreStepResult` outputs for runtime dispatch and control scheduling

## What TradingChassis Core Is Not

- Runtime orchestration or runtime order lifecycle management
- Venue adapters, backtest/live I/O, or dispatch implementations
- Deployment ownership or runtime-owned entrypoints

## Clean Core Pipeline

The clean deterministic pipeline is:

`EventStreamEntry`
`-> process_event_entry / process_canonical_event`
`-> strategy evaluator`
`-> generated intents`
`-> candidate intent records`
`-> dominance / reconciliation`
`-> policy admission`
`-> execution-control plan/apply`
`-> CoreStepResult.dispatchable_intents`
`-> runtime dispatches later`

## Installation

From the `core` directory:

```bash
python -m pip install -e ".[dev]"
```

## Quickstart

```bash
python examples/core_step_quickstart.py
```

The quickstart demonstrates canonical input, `run_core_step`, generated intents,
candidate intent records, and dispatchable outputs after policy/admission apply.

Minimal shape:

```python
import tradingchassis_core as tc

state = tc.StrategyState(event_bus=tc.NullEventBus())
result = tc.run_core_step(
    state,
    tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=0),
        event=tc.ControlTimeEvent(
            ts_ns_local_control=1_000,
            reason="scheduled_control_recheck",
            due_ts_ns_local=1_000,
            realized_ts_ns_local=1_000,
        ),
    ),
)
print(result.generated_intents, result.dispatchable_intents)
```

See `examples/core_step_quickstart.py` for a complete runnable example.

## Public API Overview

Main exported categories from `tradingchassis_core`:

- Canonical events: `MarketEvent`, `ControlTimeEvent`, `OrderSubmittedEvent`,
  `OrderExecutionFeedbackEvent`, `FillEvent`
- Pipeline models/APIs: `EventStreamEntry`, `ProcessingPosition`,
  `process_canonical_event`, `process_event_entry`, `run_core_step`,
  `run_core_wakeup_reduction`, `run_core_wakeup_decision`,
  `run_core_wakeup_step`
- Decision/output contracts: `CoreStepDecision`, `CoreStepResult`,
  `PolicyRiskDecision`, `ExecutionControlDecision`,
  `ControlSchedulingObligation`
- Intents and candidates: `OrderIntent`, `NewOrderIntent`,
  `CancelOrderIntent`, `ReplaceOrderIntent`, `CandidateIntentRecord`,
  `CandidateIntentOrigin`
- Core state/config helpers: `StrategyState`, `CoreConfiguration`,
  `ExecutionControl`, `RiskEngine`, `RiskConfig`, `NullEventBus`

## Testing

From the `core` directory:

```bash
python -m pytest -q
```

## Documentation

- `docs/README.md`
- `docs/reference/public-api.md`
- `docs/reference/events-reference.md`
- `docs/code-map/core-pipeline-map.md`
- `docs/code-map/repository-map.md`
- `docs/how-to/add-canonical-event.md`
- `docs/how-to/update-core-step-pipeline.md`
- `docs/how-to/update-policy-and-execution-control.md`

## Project References

- Changelog: `CHANGELOG.md`
- Contributing: `CONTRIBUTING.md`
- Security: `SECURITY.md`
