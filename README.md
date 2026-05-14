# TradingChassis Core

`tradingchassis_core` is a deterministic Core package.

It owns one architecture:

`EventStreamEntry -> run_core_step/run_core_wakeup_step -> candidate intents -> policy admission -> execution-control apply -> CoreStepResult`

## Scope

Core owns:

- canonical event models (`MarketEvent`, `ControlTimeEvent`, `OrderSubmittedEvent`, `OrderExecutionFeedbackEvent`, `FillEvent`)
- Pydantic contract models as the schema source of truth
- deterministic state reduction
- strategy evaluator protocol
- candidate intent combination + provenance
- policy admission semantics
- execution-control semantics (queue/rate/inflight/sendability)
- `CoreStepResult` outputs (`dispatchable_intents`, `control_scheduling_obligation`)

Core does not own:

- venue/backtest/live I/O
- runtime dispatch and runtime execution errors
- adapter integrations or `hftbacktest`
- runtime config file loading and deployment wiring

## Quickstart

From the `core` directory:

```bash
python -m pip install -e ".[dev]"
python examples/core_step_quickstart.py
python -m pytest -q tests/semantics/examples/test_core_step_quickstart.py
```

## Docs

- `docs/README.md`
- `docs/reference/public-api.md`
