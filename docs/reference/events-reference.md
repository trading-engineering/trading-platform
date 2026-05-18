# Events and Intents Reference

TradingChassis Core accepts canonical Event contracts and produces Intent/decision
contracts. Pydantic models are the schema source of truth.

## Canonical Event Models

- `MarketEvent`: book market data input for Market State reduction in the current Core baseline
- `ControlTimeEvent`: canonical **control** wakeup; becomes stream history only
  after Runtime injection. Reducer updates monotone time (and processing cursor
  when positioned). Scheduling **obligations** are a separate non-canonical output;
  see `../flows/control-time-and-scheduling.md`.
- `OrderSubmittedEvent`: canonical submitted-order acknowledgement
- `OrderExecutionFeedbackEvent`: canonical account/execution feedback
- `FillEvent`: canonical fill lifecycle update

Canonical ingestion boundary:

- `process_canonical_event(state, event, ...)`
- `process_event_entry(state, EventStreamEntry(...), ...)`

### MarketEvent baseline contract

In the current Core baseline, canonical reduction supports only book-shaped
`MarketEvent` payloads (`event_type="book"` with book levels).

Trade-shaped `MarketEvent` payloads are reserved in the schema but are not part
of the supported canonical reduction contract in this baseline. If a trade-shaped
`MarketEvent` reaches canonical reduction, Core rejects it with explicit
validation error behavior.

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
- `ControlSchedulingObligation` (time-dependent **rate-limit** recheck hint; not
  emitted for **inflight-only** deferral by default—see `../flows/control-time-and-scheduling.md`)
