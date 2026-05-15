# How To Update Risk Engine and Execution Control

The Risk Engine (policy) and Execution Control are separate deterministic phases.

## Risk Engine updates

- Policy contract entrypoint:
  `PolicyIntentEvaluator.evaluate_policy_intent(...)`
- Core integration:
  `core/domain/policy_risk_decision.py` and `run_core_step` policy phase
- Built-in policy-only evaluator:
  `core/risk/risk_engine.py` (public Risk Engine class `RiskEngine`; internal `RiskPolicy` / `ExecutionConstraintsPolicy`)
- User guide:
  `use-policy-evaluator.md`

When updating Risk Engine policy behavior:

1. Keep evaluation side-effect-free.
2. Return explicit accept/reject with reason.
3. Validate behavior with semantics tests.

## Execution Control updates

- Planning model:
  `core/domain/execution_control_plan.py`
- Apply stage:
  `core/domain/execution_control_apply.py`
- Runtime-facing non-canonical output:
  `ControlSchedulingObligation`

When updating Execution Control:

1. Keep Queue/dispatchability decisions deterministic.
2. Preserve `CoreStepResult.dispatchable_intents` contract.
3. Use `ControlSchedulingObligation` for deferred control signals.
