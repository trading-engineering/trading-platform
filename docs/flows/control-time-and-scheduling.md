# Control time and scheduling

This note is the **Core package** source of truth for how non-canonical
`ControlSchedulingObligation` relates to canonical `ControlTimeEvent` input and
to Execution Control deferral.

## Terms

- **ControlSchedulingObligation** — Non-canonical Core output: a structured hint
  that a **time-dependent** recheck may be useful. It is **not** part of the
  canonical Event Stream and does not mutate `StrategyState`.
- **ControlTimeEvent** — Canonical **control** category Event. It becomes part of
  deterministic history only after the **Runtime** injects it as
  `EventStreamEntry` input (same ingestion path as other canonical Events).
- **Inflight** — Core-side **Intent-operation** gating: a sendability / operation
  slot (for example keyed by `client_order_id`) is occupied because an earlier
  Intent operation is still awaiting **canonical execution feedback**. This is
  not the same as venue-side “order ownership”; Core models sendability for the
  decision Pipeline.
- **Rate-limit deferral** — Execution control blocks dispatch because the
  configured **token / time budget** for orders or cancels is not yet available at
  the apply clock (`now_ts_ns_local` in `CoreExecutionControlApplyContext`).
- **Inflight deferral** — Dispatch is blocked because **inflight** gating applies,
  not because a rate-limit wake time is known ahead of time.

## What Core emits today

| Deferral kind | Time-dependent? | `ControlSchedulingObligation` by default? | Expected resolution |
| --- | --- | --- | --- |
| Rate limit | Yes | **Yes** (reason such as `rate_limit`) | Runtime may realize the obligation and inject `ControlTimeEvent`; the next `run_core_step` re-runs reduction → Strategy → … → Execution Control apply. |
| Inflight | No (feedback-dependent) | **No** | Later canonical **execution / lifecycle** Events (for example `OrderSubmittedEvent`, `OrderExecutionFeedbackEvent`, or `FillEvent`, depending on lifecycle) update `StrategyState` so a subsequent step can reconsider queued work. |

**Not in scope for the current contract:** inflight timeout, wall-clock recovery,
or “synthetic” obligations for inflight-only waits.

**Not implied:** every queued Intent produces a scheduling obligation or a future
`ControlTimeEvent`. Obligations are for **rate-limit** rechecks in the current
Core slice.

## Clean Core Pipeline (unchanged)

1. `EventStreamEntry`
2. `process_event_entry` / `process_canonical_event`
3. Strategy evaluator
4. generated intents
5. candidate records + dominance / reconciliation
6. policy admission
7. Execution Control plan / apply
8. `CoreStepResult.dispatchable_intents` and optional `control_scheduling_obligation`
9. Runtime performs venue dispatch and **injects** further canonical Events (including
   any `ControlTimeEvent` realized from an obligation).

Pure planning (`plan_execution_control_candidates`) does **not** emit obligations;
they are selected only in the mutable **apply** stage (`apply_execution_control_plan`).

## Runtime ownership

- Runtimes **must not** mutate Core Queues (`StrategyState.queued_intents`, etc.)
  directly outside the normal Core step / Execution Control apply path.
- Queue flush / sendability decisions remain **ExecutionControl-owned** inside
  Core when `CoreExecutionControlApplyContext` is supplied to `run_core_step` /
  wakeup APIs.

## Further reading

- [`reference/events-reference.md`](../reference/events-reference.md)
- [`code-map/core-pipeline-map.md`](../code-map/core-pipeline-map.md)
- Tests: `tests/semantics/test_control_time_scheduling_semantics.py`
