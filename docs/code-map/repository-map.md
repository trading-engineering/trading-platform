# Repository Map

High-level map for the standalone Core package.

## Package layout

- `tradingchassis_core/__init__.py`: public package boundary exports
- `tradingchassis_core/core/domain/`: canonical contracts and deterministic
  pipeline orchestration
- `tradingchassis_core/core/risk/`: policy-only risk evaluator/config
- `tradingchassis_core/core/execution_control/`: execution-control primitives
- `tradingchassis_core/core/events/`: internal event bus/sink utilities

## Tests and examples

- `tests/semantics/`: focused contract and deterministic behavior tests
- `examples/core_step_quickstart.py`: public-import quickstart

## Top-level package docs and metadata

- `README.md`: package front door
- `CHANGELOG.md`: clean baseline changelog
- `CONTRIBUTING.md`: development and architecture rules
- `SECURITY.md`: vulnerability handling and scope policy
- `pyproject.toml`: build and tooling configuration

## Boundary matrix

Core owns:

- canonical events and processing-order contracts
- deterministic reduction and step decisions
- intent candidate, policy admission, execution-control outputs

Core does not own:

- runtime orchestration, adapters, I/O, deployment
- dispatch lifecycle beyond `CoreStepResult` outputs
