# TradingChassis Core

Deterministic semantic core package for TradingChassis.

This repository provides `tradingchassis_core`, the reusable core library that defines canonical
event processing contracts, state reduction boundaries, and CoreStep/CoreWakeupStep APIs.

## What This Package Is

- A deterministic core processing library for canonical event-driven semantics
- The home of CoreStep/CoreWakeupStep orchestration contracts
- A package that returns runtime-facing outputs such as
  `CoreStepResult.dispatchable_intents` and compatibility bridge data where needed

## What This Package Is Not

- Not venue/runtime I/O ownership
- Not a venue adapter or venue runtime shell
- Not a complete final live trading stack by itself

## Current Status (Accepted MVP Baseline)

- The CoreStep MVP baseline is accepted and frozen
- Migrated paths exist behind flags that remain default `false`
- Runtime dispatches `CoreStepResult.dispatchable_intents` on migrated flag-on paths
- Runtime does not productively use runtime `risk.decide_intents` or `GateDecision` on migrated
  flag-on paths
- `GateDecision` remains a temporary compatibility mechanism for legacy/default-off paths
- This MVP is not the final architecture

## Quickstart

From the `core` repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Runtime integration tests may require separate runtime dependencies/environment setup. Keep core
package validation centered on `core` tests in this repository.

## Architecture Entry Points

- Docs start page: `docs/README.md`
- CoreStep MVP baseline: `docs/core-step-mvp-baseline.md`
- Core vs Runtime responsibilities: `docs/core-runtime-responsibility-model.md`
- Event model: `docs/event-model.md`
- Risk vs execution control boundary: `docs/risk-vs-execution-control.md`
- GateDecision compatibility status: `docs/gate-decision-compatibility.md`

## Minimal Public API Orientation

- Step entry points: `run_core_step`, `run_core_wakeup_step`
- Runtime-facing dispatch output: `CoreStepResult.dispatchable_intents`
- Canonical event models include `MarketEvent`, `ControlTimeEvent`, `OrderSubmittedEvent`, and
  `OrderExecutionFeedbackEvent`

## Repository Guidance

- Contributing guide: `CONTRIBUTING.md`
- Changelog: `CHANGELOG.md`
- Security policy: `SECURITY.md`
