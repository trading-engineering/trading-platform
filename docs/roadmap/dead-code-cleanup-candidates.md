# U3 dead-code cleanup — status

Phase U3 audit and cleanup (completed). This document records what was removed,
what was kept, and what remains deferred.

## Removed in U3

| Item | Rationale |
| --- | --- |
| `StrategyState.pop_queued_intents` | No callers in the Core Pipeline; Runtime tests (`core-runtime`) only monkeypatch the name to assert it is **not** invoked |
| `fold_event_stream_entries` | Zero callers; batch reduction is `process_event_entry` in a loop |
| `core/events/events.py` telemetry models | Never emitted; only referenced by unused `TELEMETRY_EVENT_TYPES` |
| `combine_candidate_intents` | Unused wrapper around `combine_candidate_intent_records` |
| Root exports: `apply_execution_control_plan`, `ExecutionControlApplyContext`, `ExecutionControlApplyResult`, `ExecutionControlBlockedRecord`, `ExecutionControlDispatchableRecord`, `ExecutionControlHandledRecord` | No monorepo consumers outside Core; pipeline uses `CoreStepResult` / `CoreStepDecision` |

## Kept (monorepo usage)

| Item | Rationale |
| --- | --- |
| `RiskEngine.build_constraints` | Called from Runtime (`core-runtime` `strategy_runner.py`) for Strategy evaluation |
| `SlotKey`, `stable_slot_order_id` | Used by Runtime (`core-runtime` `debug_strategy.py`) |
| `sink_logging.LoggingEventSink` | Used by Runtime (`core-runtime` `strategy_runner.py`) |
| Risk Engine (`RiskEngine`), `RiskPolicy`, `ExecutionConstraintsPolicy`, `PolicyIntentEvaluator`, canonical Events, Core step APIs | Active extension points / Pipeline |

## Deferred (intentionally not removed)

| Item | Evidence needed before future removal |
| --- | --- |
| `PolicyAdmissionResult` / `PolicyRejectedCandidate` at root | Exported for Policy Admission introspection; only used inside Core today — narrow export if no external adopters |
| `SlotKey` in Core-only docs | Document as Runtime/strategy helper if Core export is ever questioned again |

## Not candidates

See U2 extension-point docs: Risk Engine, Execution Control, `FillEvent`, and Core step APIs remain public and tested.
