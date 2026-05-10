# CoreStep MVP Baseline

This document records the accepted and frozen CoreStep MVP baseline.

## Accepted baseline (current behavior)

- MarketEvent CoreStep path exists behind `enable_core_step_market_dispatch`.
- ControlTimeEvent CoreStep path exists behind
  `enable_core_step_control_time_dispatch`.
- Mixed wakeup collapse exists behind `enable_core_step_wakeup_collapse`.
- rc `== 3` order/execution feedback CoreStep path exists behind
  `enable_core_step_order_feedback_dispatch`.
- Runtime dispatches `CoreStepResult.dispatchable_intents` on migrated
  flag-on paths.
- Runtime does not productively use runtime `risk.decide_intents` or
  GateDecision for migrated flag-on paths.
- Runtime emits `OrderSubmittedEvent` only after successful external `NEW`
  dispatch.
- `OrderSubmittedEvent` remains ordered before `mark_intent_sent`.
- `ControlSchedulingObligation` remains non-canonical Core output.
- Runtime owns pending `ControlSchedulingObligation` and injects
  `ControlTimeEvent` when due.
- `GateDecision` remains temporary compatibility for legacy/default-off paths.
- All migrated flags remain default `false`.

## Core API/model surface used by MVP

Core result and decision models:

- `CoreStepResult`
- `CoreStepDecision`
- `PolicyRiskDecision`
- `ExecutionControlDecision`
- `CandidateIntentOrigin`
- `CandidateIntentRecord`
- `CoreWakeupReductionResult`

Core orchestration and helper APIs:

- `run_core_step`
- `run_core_wakeup_reduction`
- `run_core_wakeup_decision`
- `run_core_wakeup_step`
- `apply_policy_to_candidate_records`
- `plan_execution_control_candidates`
- `apply_execution_control_plan`
- `combine_candidate_intent_records`

## Important boundaries in this MVP

- Runtime dispatches from `dispatchable_intents` for migrated flag-on paths.
- Runtime-owned risk/gate logic is compatibility-only for those migrated paths.
- No full order lifecycle model is part of this MVP.
- Legacy rc3 path remains available when
  `enable_core_step_order_feedback_dispatch` is `false`.
