# Public API Reference

The public package boundary is the `tradingchassis_core` root import.

## Internally wired vs externally supplied

### Internally wired (when step APIs run)

These run inside Core when you call `run_core_step` / wakeup APIs (no substitute
implementation required):

- `process_event_entry` / `process_canonical_event` and canonical reducers
- Candidate combination, dominance, and reconciliation
- Policy admission **mechanism** when `CorePolicyAdmissionContext` is provided
- Execution Control plan/apply **mechanism** when policy + apply contexts are provided
- `CoreStepResult` / `CoreStepDecision` production

### Externally supplied extension points

| Symbol | Role |
| --- | --- |
| `CoreStepStrategyEvaluator` / `CoreWakeupStrategyEvaluator` | Strategy evaluation |
| `PolicyIntentEvaluator` | Policy admission (`evaluate_policy_intent`) via `CorePolicyAdmissionContext` |
| `ExecutionControl` | Queue/rate/inflight apply via `CoreExecutionControlApplyContext` |
| `CoreConfiguration` | Optional instrument metadata for positioned market reduction |
| `EventBus` / `NullEventBus` | `StrategyState` requires a bus; use `NullEventBus` for standalone Core |

### Convenience implementations (optional)

| Symbol | Role |
| --- | --- |
| Risk Engine (`RiskEngine`) | Built-in `PolicyIntentEvaluator` (not wired by default) |
| `ExecutionControl` | Default Execution Control apply implementation (you still supply an instance) |
| `NullEventBus` | Discards events for tests and examples |

**Internal (not public extension points):** `RiskPolicy`, `ExecutionConstraintsPolicy`,
and other modules under `core/risk/` except `RiskEngine` / `RiskConfig`.

Examples:

- Minimal inline policy: `examples/core_step_quickstart.py`
- Built-in Risk Engine policy: `examples/core_step_with_risk_engine.py`

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
- `run_core_wakeup_step` (ordered batch: reduce all entries, then evaluate Strategy once)

## Step inputs/outputs

- `EventStreamEntry`
- `ProcessingPosition`
- `CorePolicyAdmissionContext` (holds `PolicyIntentEvaluator`)
- `CoreExecutionControlApplyContext` (holds `ExecutionControl`)
- `CoreStepDecision`
- `CoreStepResult`
- `CoreWakeupReductionResult`
- `CoreWakeupStrategyContext`
- `CoreWakeupStrategyEvaluator`

## Policy and risk

- `PolicyIntentEvaluator` (protocol)
- `PolicyRiskDecision`
- `PolicyAdmissionResult`
- `PolicyRejectedCandidate`
- `RiskEngine` (convenience `PolicyIntentEvaluator`)
- `RiskConfig`
- `RiskConstraints` (data model; often built for Strategy via `RiskEngine.build_constraints`)

## Supporting deterministic models

- `CoreConfiguration`
- `StrategyState`
- `CandidateIntentRecord`
- `CandidateIntentOrigin`
- `ExecutionControlDecision`
- `ExecutionControl`
- `ControlSchedulingObligation` (non-canonical; **rate-limit** recheck hint in the
  current slice—see `../flows/control-time-and-scheduling.md`)

## Intents and numeric models

- `OrderIntent`
- `NewOrderIntent`
- `CancelOrderIntent`
- `ReplaceOrderIntent`
- `Price`
- `Quantity`
- `NotionalLimits`

## Runtime-safe utilities

- `NullEventBus`
- `fold_event_stream_entries` (batch reduction helper)

## Publicly absent by design

- `GateDecision`
- `compat_gate_decision`
- `ControlTimeQueueReevaluationContext`
- `CoreDecisionContext`
- `OrderStateEvent`
- `DerivedFillEvent`
- `VenueAdapter` / `VenuePolicy`
- `RiskPolicy` / `ExecutionConstraintsPolicy` (internal to `RiskEngine`)
