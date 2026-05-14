# Events and Intents Reference

TradingChassis Core accepts canonical event contracts and produces intent/decision
contracts. Pydantic models are the schema source of truth.

## Canonical Event Models

- `MarketEvent`: book/trade market data input for state reduction
- `ControlTimeEvent`: control-time wakeup and scheduling context
- `OrderSubmittedEvent`: canonical submitted-order acknowledgement
- `OrderExecutionFeedbackEvent`: canonical account/execution feedback
- `FillEvent`: canonical fill lifecycle update

Canonical ingestion boundary:

- `process_canonical_event(state, event, ...)`
- `process_event_entry(state, EventStreamEntry(...), ...)`

## Processing Order Models

- `ProcessingPosition`
- `EventStreamEntry`

These models provide deterministic ordering metadata without implementing a full
stream storage/replay subsystem.

## Intent Models

- `OrderIntent` (discriminated union)
- `NewOrderIntent`
- `CancelOrderIntent`
- `ReplaceOrderIntent`
- `Price`
- `Quantity`

## Non-canonical Output Models

- `CandidateIntentRecord` with `CandidateIntentOrigin`
- `PolicyRiskDecision`
- `ExecutionControlDecision`
- `CoreStepDecision`
- `CoreStepResult`
- `ControlSchedulingObligation`
