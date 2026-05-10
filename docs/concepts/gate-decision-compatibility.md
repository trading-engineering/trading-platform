# GateDecision Compatibility Status

## Current role

`GateDecision` is temporary compatibility for legacy/default-off paths.

## What remains valid today

- `CoreStepResult.compat_gate_decision` can exist as bridge data.
- Legacy/default-off paths may still rely on GateDecision behavior.

## Migrated path rule

For migrated flag-on paths, Runtime dispatches from
`CoreStepResult.dispatchable_intents`, not from `GateDecision.accepted_now`.

## Architectural status

- GateDecision is not the final architecture.
- GateDecision removal is post-MVP work.
