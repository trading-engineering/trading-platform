# TradingChassis Core Docs

This documentation set describes the standalone clean Core package baseline.

## Contents

- `reference/public-api.md`: supported root exports and package boundary
- `reference/events-reference.md`: canonical events and intent contracts
- `code-map/core-pipeline-map.md`: deterministic pipeline walkthrough
- `code-map/repository-map.md`: package layout and ownership map
- `how-to/add-canonical-event.md`: extending canonical event contracts
- `how-to/update-core-step-pipeline.md`: changing CoreStep/CoreWakeupStep behavior
- `how-to/update-policy-and-execution-control.md`: changing policy/apply behavior

## Package Purpose

TradingChassis Core is a deterministic trading decision engine library. It owns
canonical contracts, state reduction, and step-level decision outputs.

## Clean Core Pipeline

1. `EventStreamEntry`
2. `process_event_entry` / `process_canonical_event`
3. strategy evaluator
4. generated intents
5. candidate records + dominance/reconciliation
6. policy admission
7. execution-control plan/apply
8. `CoreStepResult` outputs (`dispatchable_intents`,
   `control_scheduling_obligation`)
9. runtime dispatch happens later

## Contract source of truth

Pydantic contract models in `tradingchassis_core/core/domain/types.py` are the
source of truth for canonical event/intent schemas.

## Out of Scope

- runtime orchestration and order lifecycle ownership
- Venue Adapters, Backtesting/Live I/O, external dispatch
