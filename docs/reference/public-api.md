# Public API Reference

The public package boundary is the `tradingchassis_core` root import.

## Canonical Events

- `MarketEvent`
- `ControlTimeEvent`
- `OrderSubmittedEvent`
- `OrderExecutionFeedbackEvent`
- `FillEvent`

## Step APIs

- `process_canonical_event`
- `process_event_entry`
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

## Intents and numeric models

- `OrderIntent`
- `NewOrderIntent`
- `CancelOrderIntent`
- `ReplaceOrderIntent`
- `Price`
- `Quantity`

## Runtime-safe utilities

- `NullEventBus`
- `RiskEngine` (Risk Engine; policy-only)
- `RiskConfig`

## Publicly absent by design

- `GateDecision`
- `compat_gate_decision`
- `ControlTimeQueueReevaluationContext`
- `CoreDecisionContext`
- `OrderStateEvent`
- `DerivedFillEvent`
- `VenueAdapter` / `VenuePolicy`
