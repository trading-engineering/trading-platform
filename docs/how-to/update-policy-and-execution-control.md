# How To Update Policy and Execution Control

Policy admission and execution-control are separate deterministic phases.

## Policy updates

- Policy contract entrypoint:
  `PolicyIntentEvaluator.evaluate_policy_intent(...)`
- Core integration:
  `core/domain/policy_risk_decision.py` and `run_core_step` policy phase
- Built-in policy-only evaluator:
  `core/risk/risk_engine.py`

When updating policy:

1. Keep evaluation side-effect-free.
2. Return explicit accept/reject with reason.
3. Validate behavior with semantics tests.

## Execution-control updates

- Planning model:
  `core/domain/execution_control_plan.py`
- Apply stage:
  `core/domain/execution_control_apply.py`
- Runtime-facing non-canonical output:
  `ControlSchedulingObligation`

When updating execution-control:

1. Keep queue/dispatchability decisions deterministic.
2. Preserve `CoreStepResult.dispatchable_intents` contract.
3. Use `ControlSchedulingObligation` for deferred control signals.
