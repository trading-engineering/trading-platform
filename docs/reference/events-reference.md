# Events Reference

This table summarizes current event/model status for the accepted MVP baseline.

| Event/model | Canonical status | Producer | Consumer/reducer | Purpose | Notes |
| --- | --- | --- | --- | --- | --- |
| `MarketEvent` | Canonical | Runtime (normalized ingress) | Core reduction (`process_event_entry` / `run_core_step`) | Market-state update input | CoreStep migrated market path is flag-gated and default `false` |
| `ControlTimeEvent` | Canonical after runtime injection | Runtime (when due obligation realized) | Core reduction (`process_event_entry` / `run_core_step`) | Canonical control re-entry boundary | Becomes canonical only after runtime injects it |
| `OrderSubmittedEvent` | Canonical | Runtime (post successful external `NEW` dispatch) | Core reduction | Canonical submission confirmation boundary | Emitted after successful `NEW`; ordered before `mark_intent_sent` |
| `OrderExecutionFeedbackEvent` | Canonical MVP ingress event | Runtime (normalized rc3 feedback/snapshot input) | Core reduction (`run_core_step`) | Canonical rc3 feedback ingress for MVP | Migrated rc3 path is flag-gated and default `false` |
| `FillEvent` | Canonical model status | Runtime/normalization boundary (model-level) | Core reduction (model support exists) | Fill-oriented execution event model | Not used as snapshot-only rc3 feedback ingress in MVP |
| `OrderStateEvent` | Compatibility/non-canonical | Compatibility snapshot/projection flows | Compatibility handling only | Legacy snapshot/materialization record | Not a canonical Event Stream record in current MVP |
| `ControlSchedulingObligation` | Non-canonical Core output (not an event) | Core output (`CoreStepResult`) | Runtime scheduling realization | Runtime wakeup planning handoff | Must not be treated as event-stream input/persisted event |

## Notes

- `OrderExecutionFeedbackEvent` is the canonical rc3 MVP feedback ingress event.
- `FillEvent` remains canonical in model terms but is not used for snapshot-only rc3 ingress.
- `GateDecision` remains compatibility for legacy/default-off paths and is not the migrated-path
  dispatch contract.
