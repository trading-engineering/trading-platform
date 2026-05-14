# Core Pipeline Map

This map captures the only supported deterministic decision pipeline for
TradingChassis Core.

## Step-by-step flow

1. `EventStreamEntry` arrives with `ProcessingPosition`.
2. `process_event_entry` forwards to `process_canonical_event`.
3. Canonical reducer mutates `StrategyState` deterministically.
4. Strategy evaluation produces generated Intents.
5. Candidate records are built and reconciled/dominated.
6. Risk Engine (policy) accepts/rejects generated candidates.
7. Execution Control plan/apply computes Queue/dispatch/scheduling outputs.
8. `CoreStepResult` returns `dispatchable_intents` and optional
   `control_scheduling_obligation`.
9. Runtime can dispatch later; Core does not dispatch.

## Core APIs

- Single-entry flow: `run_core_step`
- Wakeup reduction: `run_core_wakeup_reduction`
- Wakeup decision/apply: `run_core_wakeup_decision`
- Wakeup convenience wrapper: `run_core_wakeup_step`

## Determinism notes

- Processing Order monotonicity is enforced by `ProcessingPosition`.
- Core logic is side-effect-safe apart from deterministic state mutation.
- Runtime adapters and external dispatch concerns are outside Core.
