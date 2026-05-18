# Public API Reference

The Public API boundary is the `tradingchassis_core` root import.

This page classifies root exports into:

- Public Data Model
- Public Extension Point
- Public Orchestration API
- Public Convenience Implementation
- Advanced API

`ControlSchedulingObligation` is a non-canonical output. `ControlTimeEvent` is the
canonical Event symbol for the Control-Time Event concept.

## Public Orchestration API

Primary orchestration entrypoints:

- `run_core_step`
- `run_core_wakeup_step`
- `process_event_entry`
- `process_canonical_event`

Advanced orchestration entrypoints (split CoreWakeupStep flow):

- `run_core_wakeup_reduction`
- `run_core_wakeup_decision`

Core step orchestration is deterministic: Runtime provides canonical Event Stream
input, Core reduces canonical Events, evaluates Strategy, applies Policy Admission
and Execution Control, and returns `CoreStepResult`.

Primary integrations should start with `run_core_step` or `run_core_wakeup_step`.
Split APIs are public Advanced API for diagnostics, testing, and Runtime/Core split
flows.

## Public Data Model

Core and processing models:

- `CoreConfiguration`
- `ProcessingPosition`
- `EventStreamEntry`
- `CoreStepResult`
- `StrategyStateView`

Canonical Event models:

- `MarketEvent`
- `ControlTimeEvent`
- `OrderSubmittedEvent`
- `OrderCanceledEvent`
- `OrderRejectedEvent`
- `OrderExpiredEvent`
- `OrderExecutionFeedbackEvent`
- `FillEvent`

Order Intent and numeric models:

- `OrderIntent`
- `NewOrderIntent`
- `CancelOrderIntent`
- `ReplaceOrderIntent`
- `Price`
- `Quantity`

Risk data models:

- `RiskConfig`
- `RiskConstraints`
- `NotionalLimits`

Current baseline notes:

- canonical Event reduction for `MarketEvent` is book-only in the current Core
  baseline.
- Trade-shaped `MarketEvent` payloads are not reduced in this baseline and are
  explicitly rejected at canonical processing boundaries.
- Policy Admission rejection is not an Order Lifecycle terminal Event.

## Public Extension Point

- `CoreStepStrategyEvaluator`
- `CoreWakeupStrategyEvaluator`
- `PolicyIntentEvaluator`
- `CorePolicyAdmissionContext`

Strategy evaluators read `StrategyStateView` through context and return Order Intent
values. Policy evaluators are supplied through `CorePolicyAdmissionContext`.

## Public Convenience Implementation

- `RiskEngine`
- `ExecutionControl`
- `NullEventBus`

These remain public for convenience and compatibility. Runtime still supplies
instances and owns external dispatch and canonical Event injection.

## Advanced API

Advanced state, context, and split-step models:

- `StrategyState`
- `CoreStepStrategyContext`
- `CoreWakeupStrategyContext`
- `CoreWakeupReductionResult`
- `CoreExecutionControlApplyContext`

Advanced introspection and decision scaffolds:

- `CandidateIntentOrigin`
- `CandidateIntentRecord`
- `CoreStepDecision`
- `ExecutionControlDecision`
- `PolicyRiskDecision`
- `PolicyRejectedCandidate`
- `PolicyAdmissionResult`

Advanced runtime-facing non-canonical output:

- `ControlSchedulingObligation`

Advanced deterministic slot helpers:

- `SlotKey`
- `stable_slot_order_id`

Advanced API symbols are public for compatibility and diagnostics, split-step
integration, and advanced testing. They are not themselves Runtime dispatch
obligations. Runtime dispatch obligations remain `CoreStepResult.dispatchable_intents`
plus canonical Event injection behavior.

## State and Strategy boundary

- `StrategyState` is an advanced mutable Core state container.
- Core reducers and Execution Control own mutation of Strategy State internals.
- Strategy code should read `StrategyStateView` via Strategy contexts.
- `StrategyStateView` is the public read-only Strategy boundary model.
- Queue/inflight internals and reducer methods are not Strategy mutation APIs.

`CoreStepStrategyContext` is an advanced Strategy adapter context. `context.state` is
`StrategyStateView`, not mutable `StrategyState`.

## Root export surface

The current root export surface includes all symbols listed on this page and:

- `__version__`

## Root vs non-root API

Some APIs are public but are module-level (non-root) rather than root exports.
Example:

- `EventBus` from `tradingchassis_core.core.events.event_bus`

`EventBus` is intentionally non-root. `NullEventBus` is the root convenience export.

## Internally wired vs externally supplied

Internally wired when step APIs run:

- canonical Event reduction and candidate reconciliation
- Policy Admission mechanism when `CorePolicyAdmissionContext` is provided
- Execution Control plan/apply mechanism when execution apply context is provided
- `CoreStepResult` and advanced decision scaffolds production

Externally supplied:

- Strategy evaluator (`CoreStepStrategyEvaluator` or `CoreWakeupStrategyEvaluator`)
- Policy evaluator (`PolicyIntentEvaluator`)
- Execution Control instance (`ExecutionControl`)
- optional `CoreConfiguration`
- Event bus (`EventBus` module-level API, or root `NullEventBus`)

## Publicly absent by design

- `GateDecision`
- `compat_gate_decision`
- `ControlTimeQueueReevaluationContext`
- `CoreDecisionContext`
- `OrderStateEvent`
- `DerivedFillEvent`
- `VenueAdapter` / `VenuePolicy`
- `RiskPolicy` / `ExecutionConstraintsPolicy` (internal to `RiskEngine`)
- `fold_event_stream_entries` (removed U3; use `process_event_entry` loop)
- `apply_execution_control_plan` and apply detail record types (internal; use `CoreStepResult`)
