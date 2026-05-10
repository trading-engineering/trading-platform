# Risk vs ExecutionControl

## Risk is policy-only

Risk checks are policy/hygiene controls such as:

- trading enabled/disabled
- max loss checks
- hard hygiene checks
- validation and limits

## ExecutionControl owns dispatchability mechanics

ExecutionControl owns venue-independent dispatchability mechanics:

- queue handling
- dominance/effective pending work
- rate constraints
- inflight/sendability checks
- dispatchable selection
- scheduling obligation derivation

## Boundary for migrated paths

- Runtime must not productively call runtime `risk.decide_intents`.
- Runtime dispatches from `CoreStepResult.dispatchable_intents`.
