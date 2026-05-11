# Add a Canonical Event

Use this guide when introducing a new canonical Core event in the MVP architecture.

## When to add a canonical event

- You need a durable Core event-stream input with deterministic reduction semantics.
- Runtime can normalize raw input into a stable Core model boundary.
- The event participates in Core reduction and/or CoreStep decision flow.

## When not to add a canonical event

- The data is runtime-only transport/adapter metadata.
- The artifact is compatibility-only or temporary bridge data.
- The artifact is a non-canonical Core output (for example `ControlSchedulingObligation`).

## Required design questions

- What raw input does Runtime receive, and how is it normalized?
- Is this canonical, or compatibility-only?
- Who produces it (Runtime boundary) and who reduces it (Core boundary)?
- Is it event-stream input, or a non-canonical output?
- Does it require `ProcessingPosition` ordering?
- Does it overlap with existing canonical events?

## Implementation checklist

- Add/update event model types in Core domain event/type modules.
- Update event taxonomy/boundary classification used by canonical processing.
- Add/update reducer boundary handling (`process_event_entry` / canonical processing path).
- Keep state mutation in Core reducer/state methods, not Runtime snapshots.
- Verify `run_core_step` / wakeup-step implications and outputs.
- Update manual references for public/event status docs.

## Test checklist

- Canonical taxonomy test (recognized as canonical or explicitly non-canonical).
- Canonical processing boundary test (accepted/rejected at correct boundary).
- Reducer/state transition test for deterministic effects.
- Guardrail test/search for no runtime dependency creep in Core.
- Compatibility rejection/segregation tests where relevant.

## Documentation checklist

- Update `../reference/events-reference.md`.
- Update `../concepts/event-model.md`.
- Update relevant flow docs under `../flows/`.
- Update `../mvp/compatibility-matrix.md` if flags/compat behavior changed.

## Anti-patterns

- Passing raw venue/backtest objects directly into Core event models.
- Mutating runtime snapshots as if they were Core state reducers.
- Using `FillEvent` as snapshot-only rc3 feedback ingress.
- Canonicalizing `ControlSchedulingObligation` as event-stream input.
- Adding event types without reducer/test/docs updates.

## Related docs

- [Event Model](../concepts/event-model.md)
- [Core and Runtime Responsibility Model](../concepts/core-runtime-responsibility-model.md)
- [Events Reference](../reference/events-reference.md)
- [CoreStep MVP Baseline](../mvp/core-step-mvp-baseline.md)
- [Compatibility Matrix](../mvp/compatibility-matrix.md)
