# Changelog

All notable changes to the TradingChassis Core package are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to Semantic Versioning.

## [Unreleased]

### Added

- CoreStep MVP baseline documentation index alignment in `docs/README.md`.

### Changed

- Recorded accepted and frozen CoreStep MVP baseline behavior:
  - MarketEvent CoreStep path behind `enable_core_step_market_dispatch`
  - ControlTimeEvent CoreStep path behind `enable_core_step_control_time_dispatch`
  - Mixed wakeup collapse behind `enable_core_step_wakeup_collapse`
  - rc `== 3` order/execution feedback CoreStep path behind
    `enable_core_step_order_feedback_dispatch`
- Clarified runtime dispatch contract for migrated flag-on paths:
  runtime dispatches `CoreStepResult.dispatchable_intents`.
- Clarified `OrderSubmittedEvent` dispatch-success emission boundary and ordering before
  `mark_intent_sent`.

### Compatibility

- `ControlSchedulingObligation` remains non-canonical Core output.
- Runtime continues to own pending obligation realization and `ControlTimeEvent` injection.
- `GateDecision` remains compatibility-only for legacy/default-off paths.
- Migration flags remain default `false`.

### Documentation

- Root repository documentation aligned to current MVP baseline language and boundaries.

## [0.1.0] - 2026-02-17

Initial public release of `tradingchassis_core`.
