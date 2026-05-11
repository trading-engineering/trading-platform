# Add a CoreStep or CoreWakeupStep Test

Use this guide when adding behavior coverage for Core step orchestration.

## When to write a CoreStep test

- Single-entry deterministic behavior changes in `run_core_step`.
- Candidate generation/admission/dispatchability semantics change for one entry.
- `CoreStepResult` shape/field behavior changes.

## When to write a CoreWakeupStep test

- Multi-entry wakeup reduction/decision behavior changes.
- Deterministic ordering across multiple `EventStreamEntry` values is affected.
- Wakeup-level decision/apply behavior changes.

## Core vs Runtime test ownership

- Core tests should validate deterministic Core semantics only.
- Runtime tests should validate dispatch/integration behavior around Core outputs.

## CoreStep test checklist

- Use canonical `EventStreamEntry` input.
- Include `ProcessingPosition` ordering assumptions.
- Assert reducer/state effects.
- Assert strategy evaluator behavior (if evaluator is used).
- Assert generated and candidate intent record behavior.
- Assert policy risk decision projection behavior.
- Assert execution-control apply/decision behavior.
- Assert `dispatchable_intents` and `control_scheduling_obligation` outputs.
- Assert `compat_gate_decision` only when compatibility path is intentionally under test.

## CoreWakeupStep test checklist

- Use multiple canonical `EventStreamEntry` values.
- Assert deterministic processing order.
- Assert single wakeup-level decision/apply semantics.
- Assert no duplicated risk/dispatchability semantics within one wakeup flow.

## Runtime migrated-path guardrail checklist

- Runtime dispatches `CoreStepResult.dispatchable_intents`.
- Runtime does not productively re-decide equivalent work via runtime risk.
- Runtime does not use `GateDecision` as migrated-path final contract.
- Runtime preserves `OrderSubmittedEvent` dispatch-success behavior.
- Runtime realizes/apply scheduling obligations correctly.

## Test placement guide

- Place Core semantic tests in Core repository test suites.
- Place runtime integration/guardrail tests in runtime-owned suites.
- If runtime environment is blocked (dependencies/system), classify as tooling/environment blocker,
  not as Core semantic failure.

## Anti-patterns

- Testing runtime I/O behavior inside Core-only tests.
- Treating `GateDecision` as the final migrated-path contract.
- Pulling runtime dependencies into Core tests.
- Using broad brittle end-to-end tests for small semantic changes.

## Related docs

- [Core Pipeline Map](../code-map/core-pipeline-map.md)
- [Core and Runtime Responsibility Model](../concepts/core-runtime-responsibility-model.md)
- [GateDecision Compatibility](../concepts/gate-decision-compatibility.md)
- [CoreStep MVP Baseline](../mvp/core-step-mvp-baseline.md)
- [Compatibility Matrix](../mvp/compatibility-matrix.md)
