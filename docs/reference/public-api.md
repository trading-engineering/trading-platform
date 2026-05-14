# Public API Reference

The package export boundary is `tradingchassis_core`.

## Canonical events

- `MarketEvent`
- `ControlTimeEvent`
- `OrderSubmittedEvent`
- `OrderExecutionFeedbackEvent`
- `FillEvent`

## Step APIs

- `run_core_step`
- `run_core_wakeup_reduction`
- `run_core_wakeup_decision`
- `run_core_wakeup_step`

## Step inputs/outputs

- `EventStreamEntry`
- `ProcessingPosition`
- `CorePolicyAdmissionContext`
- `CoreExecutionControlApplyContext`
- `CoreStepDecision`
- `CoreStepResult`
- `CoreWakeupReductionResult`

## Supporting deterministic models

- `CoreConfiguration`
- `StrategyState`
- `CandidateIntentRecord`
- `CandidateIntentOrigin`
- `PolicyRiskDecision`
- `ExecutionControlDecision`
- `ExecutionControl`
- `ControlSchedulingObligation`

## Utility exports

- `NullEventBus`
- `RiskEngine` (policy-only evaluator)

Compatibility bridge contracts are intentionally absent from the public API.
