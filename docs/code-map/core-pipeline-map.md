# Core Pipeline Map

This map captures the only supported deterministic decision Pipeline for
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
   external dispatch or mutate Queues outside this Pipeline.

## Core APIs

- Single-entry flow: `run_core_step`
- Wakeup reduction: `run_core_wakeup_reduction`
- Wakeup decision/apply: `run_core_wakeup_decision`
- Wakeup convenience wrapper: `run_core_wakeup_step`

## Determinism notes

- Processing Order monotonicity is enforced by `ProcessingPosition`.
- Core logic is side-effect-safe apart from deterministic State mutation.
- Runtime adapters and external dispatch concerns are outside Core.


## CoreWakeupStep batch semantics

`CoreWakeupStep` is not "parallel Event processing".
It is deterministic batch processing: the Runtime gives Core an ordered sequence of
canonical `EventStreamEntry` values, and Core reduces them in that order before making
one decision.

Wakeup flow:

1. Runtime supplies an ordered batch of `EventStreamEntry` values.
2. `run_core_wakeup_reduction` calls `process_event_entry` for each entry in order.
3. `CoreWakeupStrategyEvaluator.evaluate` runs **once** on the fully reduced State
   (`CoreWakeupStrategyContext` carries all entries).
4. `run_core_wakeup_decision` snapshots queued Intents once, combines generated + queued
   once, applies dominance/reconciliation once, Policy Admission once, and
   Execution Control plan/apply once.
5. `CoreStepResult.dispatchable_intents` is returned; Runtime dispatches later.

`run_core_step` remains single-entry: one reduction, one step-level Strategy evaluation,
one decision pass.

## Internally wired vs externally supplied

### Internally wired

- Steps 1–3, 5, and 8 in the flow above (reduction, candidates, `CoreStepResult`)
- Policy admission **machinery** when `CorePolicyAdmissionContext` is provided
- Execution Control plan/apply **machinery** when apply context is provided

### Externally supplied extension points

- **Strategy** — `CoreStepStrategyEvaluator` or `CoreWakeupStrategyEvaluator`
- **Policy** — `PolicyIntentEvaluator` via `CorePolicyAdmissionContext`
- **Execution Control instance** — `ExecutionControl` via `CoreExecutionControlApplyContext`
- **Configuration** — optional `CoreConfiguration`
- **Event bus** — `StrategyState(event_bus=...)`; `NullEventBus` for standalone use

### Convenience implementations

- Risk Engine (`RiskEngine`) — optional built-in `PolicyIntentEvaluator` (`examples/core_step_with_risk_engine.py`)
- `ExecutionControl` — default queue/rate/inflight behavior (instance still supplied by caller)
- `NullEventBus` — no-op bus for tests and examples

See `../reference/public-api.md` and `../how-to/use-policy-evaluator.md`.
