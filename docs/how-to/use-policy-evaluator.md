# How to use a policy evaluator

Core policy admission is optional. When you pass `CorePolicyAdmissionContext`, Core
calls your evaluator for each **generated** candidate intent (queued candidates
passthrough unchanged).

## Extension point: `PolicyIntentEvaluator`

Root export: `tradingchassis_core.PolicyIntentEvaluator`

Implement:

```python
def evaluate_policy_intent(
    self,
    *,
    intent: OrderIntent,
    state: StrategyState,
    now_ts_ns_local: int,
) -> tuple[bool, str | None]:
    ...
```

Pass the instance via `CorePolicyAdmissionContext(policy_evaluator=..., now_ts_ns_local=...)`.

Any object satisfying this contract works. Core does not require `RiskEngine`.

## Convenience implementation: Risk Engine (`RiskEngine`)

The built-in **Risk Engine** (`RiskEngine`) implements `PolicyIntentEvaluator` with
policy gates (trading enabled, max loss, normalization, hard limits). Configure
with `RiskConfig`.

Runnable example:

```bash
cd core
python examples/core_step_with_risk_engine.py
```

Minimal quickstart (`examples/core_step_quickstart.py`) uses an inline allow-all
policy to stay small. That does not mean the Risk Engine is unused.

## Execution Control apply

Policy admission alone does not mutate queues or produce dispatchables. Also pass
`CoreExecutionControlApplyContext` with a supplied `ExecutionControl` instance and
set `activate_dispatchable_outputs=True` when you want `CoreStepResult.dispatchable_intents`.

See also:

- `reference/public-api.md`
- `code-map/core-pipeline-map.md`
- `update-policy-and-execution-control.md`
