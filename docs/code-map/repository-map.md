# Repository Map

High-level map for the standalone Core package.

## Package layout

- `tradingchassis_core/__init__.py`: public package boundary exports
- `tradingchassis_core/core/domain/`: canonical contracts and deterministic
  Pipeline orchestration
- `tradingchassis_core/core/risk/`: policy-only Risk Engine evaluator/config
- `tradingchassis_core/core/execution_control/`: Execution Control primitives
- `tradingchassis_core/core/events/`: Event bus/sink utilities (`NullEventBus`; `LoggingEventSink` for Runtime)

## Tests and examples

- `tests/semantics/`: focused contract and deterministic behavior tests
- `examples/core_step_quickstart.py`: minimal inline-policy quickstart
- `examples/core_step_with_risk_engine.py`: Risk Engine policy quickstart

## Top-level package docs and metadata

- `README.md`: package front door
- `CHANGELOG.md`: clean baseline changelog
- `CONTRIBUTING.md`: development and architecture rules
- `SECURITY.md`: vulnerability handling and scope policy
- `pyproject.toml`: build and tooling configuration

## Boundary matrix

Core owns:

- canonical Events and Processing Order contracts
- deterministic reduction and step decisions
- Intent candidate records, Risk Engine (policy), Execution Control outputs

Core does not own:

- Runtime orchestration, Venue Adapters, I/O, deployment
- dispatch lifecycle beyond `CoreStepResult` outputs
