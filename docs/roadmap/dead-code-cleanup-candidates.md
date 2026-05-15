# U3 dead-code cleanup candidates

Phase U1/U2 identified components that may be removable in U3. **Do not delete until
evidence is confirmed** (including a `core-runtime` audit where noted).

## Candidates

### `StrategyState.pop_queued_intents`

| Field | Detail |
| --- | --- |
| **Current evidence** | Defined in `core/domain/state.py`; no callers in Core source, tests, or examples |
| **Possible purpose** | Legacy queue drain API; Execution Control uses `pop_queued_intents_for_order` and `merge_intents_into_queue` |
| **Keep if** | Runtime or external tools call it |
| **Delete in U3 if** | Monorepo search shows zero callers |
| **Recommended action** | Defer; audit `core-runtime` before removal |

### `RiskEngine.build_constraints`

| Field | Detail |
| --- | --- |
| **Current evidence** | `risk_engine.py` only; no Core/tests/examples callers |
| **Possible purpose** | Build `RiskConstraints` for Strategy evaluation from `RiskConfig` |
| **Keep if** | Documented Strategy contract; add test showing evaluator consumption |
| **Delete in U3 if** | Strategy never uses `RiskConstraints` in clean Core/Runtime |
| **Recommended action** | Defer; pair with Strategy integration example or remove method |

### `fold_event_stream_entries`

| Field | Detail |
| --- | --- |
| **Current evidence** | Root export; zero in-repo usage |
| **Possible purpose** | Ergonomic batch reduction helper for replay harnesses |
| **Keep if** | Add test + docs example |
| **Delete in U3 if** | Runtimes always loop `process_event_entry` |
| **Recommended action** | Keep as utility or demote from `__all__` |

### `SlotKey`, `stable_slot_order_id`

| Field | Detail |
| --- | --- |
| **Current evidence** | Root export; zero usage in Core/tests/examples |
| **Possible purpose** | Deterministic market-making client order IDs |
| **Keep if** | Add MM strategy example using slots |
| **Delete in U3 if** | No monorepo consumer after search |
| **Recommended action** | Remove export or move to optional helper module |

### Telemetry models in `core/events/events.py`

| Field | Detail |
| --- | --- |
| **Current evidence** | Listed in `event_model.TELEMETRY_EVENT_TYPES`; not exported; not reduced by Core |
| **Possible purpose** | Future observability bus payloads (`RiskDecisionEvent`, `DerivedPnLEvent`, etc.) |
| **Keep if** | Runtime emits via custom sinks |
| **Delete in U3 if** | No emitter/listener in monorepo |
| **Recommended action** | Remove module or mark internal-only |

### `core/events/sinks/sink_logging.py`

| Field | Detail |
| --- | --- |
| **Current evidence** | No imports in Core |
| **Possible purpose** | Example `EventSink` for debugging |
| **Keep if** | Documented in how-to for custom buses |
| **Delete in U3 if** | Unused and duplicated by Runtime logging |
| **Recommended action** | Remove or document as optional pattern |

### Exported apply detail records

(`ExecutionControlBlockedRecord`, `ExecutionControlDispatchableRecord`, `ExecutionControlHandledRecord`, `ExecutionControlApplyResult`)

| Field | Detail |
| --- | --- |
| **Current evidence** | Used by `apply_execution_control_plan`; typical consumers read `CoreStepResult` / `CoreStepDecision` |
| **Possible purpose** | Advanced introspection of the apply stage |
| **Keep if** | External tools depend on root export |
| **Delete in U3 if** | Only `CoreStepDecision` needed publicly |
| **Recommended action** | Narrow `__all__` after usage audit |

## Not candidates for U3 removal

- **Risk Engine** (`RiskEngine`) — public convenience `PolicyIntentEvaluator` (see `examples/core_step_with_risk_engine.py`)
- **`PolicyIntentEvaluator`** — root-exported extension protocol
- **`FillEvent`** — canonical reducer with tests in `tests/semantics/test_fill_event_reduction.py`
- **`RiskPolicy`**, **`ExecutionConstraintsPolicy`** — internal helpers used by `RiskEngine`; not public extension points
