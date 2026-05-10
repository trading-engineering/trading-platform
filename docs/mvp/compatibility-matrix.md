# Compatibility Matrix

This matrix reflects the accepted and frozen CoreStep MVP baseline. It separates migrated
CoreStep behavior from compatibility/legacy behavior.

| Concern / path | MVP canonical or CoreStep behavior | Compatibility / legacy behavior | Flag | Default | Post-MVP note |
| --- | --- | --- | --- | --- | --- |
| MarketEvent path | CoreStep path exists and can drive migrated processing | Legacy/default-off path remains available | `enable_core_step_market_dispatch` | `false` | Candidate for default-on later |
| ControlTimeEvent path | CoreStep path exists for runtime-injected control re-entry | Legacy/default-off control handling remains | `enable_core_step_control_time_dispatch` | `false` | Keep runtime injection boundary |
| Mixed wakeup collapse | Wakeup collapse path exists in CoreStep wakeup model | Legacy/default-off wakeup behavior remains | `enable_core_step_wakeup_collapse` | `false` | Final wakeup model is post-MVP |
| rc `== 3` order/execution feedback | Runtime normalizes into canonical `OrderExecutionFeedbackEvent` and calls `run_core_step` | Legacy rc3 path remains when disabled | `enable_core_step_order_feedback_dispatch` | `false` | Full lifecycle redesign is post-MVP |
| Runtime dispatch | Migrated flag-on paths dispatch from `CoreStepResult.dispatchable_intents` | Legacy path may dispatch from compatibility gate output | path-specific migrated flags | `false` | Dispatch boundary remains runtime-owned |
| Runtime `risk.decide_intents` | Not productively used for migrated flag-on paths | May still be used by legacy/default-off paths | path-specific migrated flags | `false` | Runtime risk/gate productive role should shrink |
| `GateDecision` | Not the migrated-path dispatch contract | Temporary compatibility decision model | path-specific migrated flags | `false` | Removal is post-MVP |
| `CoreStepResult.compat_gate_decision` | Optional bridge field only | Used by compatibility flows where needed | path-specific migrated flags | `false` | Remove with GateDecision retirement |
| `ControlSchedulingObligation` | Non-canonical Core output for runtime wakeup planning | N/A (already compatibility-shaped handoff) | N/A | N/A | May evolve when final control model lands |
| `ControlTimeEvent` | Canonical only after runtime realizes obligation and injects event | N/A | `enable_core_step_control_time_dispatch` | `false` | Canonical control model can be refined later |
| `OrderSubmittedEvent` | Canonical, emitted only after successful external `NEW` dispatch and before `mark_intent_sent` | N/A | N/A | N/A | Full lifecycle semantics remain post-MVP |
| `FillEvent` | Canonical model in event taxonomy | Not used as snapshot-only rc3 feedback ingress | N/A | N/A | Expanded fill-centric lifecycle is post-MVP |
| `OrderExecutionFeedbackEvent` | Canonical MVP rc3 feedback ingress event | Replaced by legacy snapshot path when flag off | `enable_core_step_order_feedback_dispatch` | `false` | Keep canonical feedback ingress |
| `OrderStateEvent` | Non-canonical for current MVP semantics | Compatibility/snapshot projection record | N/A | N/A | Future lifecycle work may retire this role |

## Notes

- Migrated CoreStep paths are flag-gated and all migration flags remain default `false`.
- `GateDecision` remains compatibility for legacy/default-off behavior.
- `OrderExecutionFeedbackEvent` is the canonical rc3 MVP feedback ingress event.
- `FillEvent` is not the snapshot-only rc3 feedback ingress event in this MVP.
