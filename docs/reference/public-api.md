# Public API Reference

This reference is manual and curated for the current MVP baseline. Generated API docs may be added
later.

Stability tags used here:

- **Stable MVP**: current baseline contract to rely on
- **Compatibility**: transitional bridge surface
- **Internal-shape exposed**: exported today but better treated as implementation-adjacent

## Core step APIs

Stable MVP:

- `run_core_step`
- `run_core_wakeup_reduction`
- `run_core_wakeup_decision`
- `run_core_wakeup_step`

Purpose: deterministic Core entry points for step and wakeup-level processing.

## Result and decision models

Stable MVP (with transitional fields where noted):

- `CoreStepResult` (includes compatibility bridge field `compat_gate_decision`)
- `CoreStepDecision`
- `PolicyRiskDecision`
- `ExecutionControlDecision`

Purpose: structured step outcomes and policy/execution-control projections.

## Candidate intent models

Stable MVP:

- `CandidateIntentRecord`
- `CandidateIntentOrigin`

Purpose: explicit candidate provenance and deterministic merge ordering metadata.

## Event stream models

Stable MVP:

- `ProcessingPosition`
- `EventStreamEntry`

Purpose: canonical ingestion ordering envelope for deterministic reduction.

## Canonical event models

Stable MVP:

- `MarketEvent`
- `ControlTimeEvent`
- `OrderSubmittedEvent`
- `OrderExecutionFeedbackEvent`

Canonical model with caveat:

- `FillEvent` is canonical in the model taxonomy, but it is not the snapshot-only rc3 MVP ingress.

## Strategy/config/state-facing models

Stable MVP:

- `Strategy`
- `StrategyState`
- `CoreConfiguration`
- `EngineContext`

Purpose: strategy contract and deterministic configuration/state interaction surface.

## Compatibility and transitional surfaces

Compatibility:

- `GateDecision` (legacy/default-off compatibility decision model)
- `CoreStepResult.compat_gate_decision` (bridge field for compatibility paths)

Compatibility/non-canonical model context:

- `OrderStateEvent` remains compatibility/non-canonical in current MVP docs.

## Export boundary note

`tradingchassis_core.__init__` is the intended package export boundary. Not every exported symbol
should be treated as long-term stable final architecture; compatibility surfaces are explicitly
transitional.
