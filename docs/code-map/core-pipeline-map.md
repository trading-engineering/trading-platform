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
   `control_scheduling_obligation` (non-canonical; **rate-limit** deferral only
   in the current slice—see `../flows/control-time-and-scheduling.md`).
9. Runtime can dispatch later and inject further canonical Events (including
   `ControlTimeEvent` when an obligation is realized); Core does not perform
   external dispatch or mutate queues outside this pipeline.

## Core APIs

- Single-entry flow: `run_core_step`
- Wakeup reduction: `run_core_wakeup_reduction`
- Wakeup decision/apply: `run_core_wakeup_decision`
- Wakeup convenience wrapper: `run_core_wakeup_step`

## Determinism notes

- Processing Order monotonicity is enforced by `ProcessingPosition`.
- Core logic is side-effect-safe apart from deterministic state mutation.
- Runtime adapters and external dispatch concerns are outside Core.


## CoreWakeupStep batch semantics

`CoreWakeupStep` is not "parallel Event processing".
It is deterministic batch processing: the Runtime gives Core an ordered sequence of
canonical `EventStreamEntry` values, and Core reduces them in that order before making
one decision.

Wakeup flow:

1. Runtime supplies an ordered batch of `EventStreamEntry` values.
2. `run_core_wakeup_reduction` calls `process_event_entry` for each entry in order.
3. `CoreWakeupStrategyEvaluator.evaluate` runs **once** on the fully reduced state
   (`CoreWakeupStrategyContext` carries all entries).
4. `run_core_wakeup_decision` snapshots queued intents once, combines generated + queued
   once, applies dominance/reconciliation once, Policy Admission once, and
   ExecutionControl plan/apply once.
5. `CoreStepResult.dispatchable_intents` is returned; Runtime dispatches later.

`run_core_step` remains single-entry: one reduction, one step-level Strategy evaluation,
one decision pass.
