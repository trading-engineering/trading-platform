"""Core-only CoreStep quickstart example.

For ordered multi-entry wakeup batches see run_core_wakeup_step in the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tradingchassis_core as tc

INSTRUMENT = "BTC-USDC-PERP"
INTENT_ID_V1 = "quickstart-new-v1"
INTENT_ID_V2 = "quickstart-new-v2"


class OneIntentEvaluator:
    """Small evaluator that emits one deterministic new-order Intent."""

    def __init__(self, client_order_id: str) -> None:
        self._client_order_id = client_order_id

    def evaluate(self, context: object) -> list[tc.NewOrderIntent]:
        _ = context
        return [
            tc.NewOrderIntent(
                ts_ns_local=1_000,
                instrument=INSTRUMENT,
                client_order_id=self._client_order_id,
                intents_correlation_id=f"corr-{self._client_order_id}",
                side="buy",
                order_type="limit",
                intended_qty=tc.Quantity(value=1.0, unit="contracts"),
                intended_price=tc.Price(currency="USDC", value=100.0),
                time_in_force="GTC",
            )
        ]


class AllowAllPolicy:
    """Policy evaluator that admits every generated candidate Intent."""

    def evaluate_policy_intent(
        self,
        *,
        intent: tc.OrderIntent,
        state: tc.StrategyState,
        now_ts_ns_local: int,
    ) -> tuple[bool, str | None]:
        _ = (intent, state, now_ts_ns_local)
        return True, None


def _control_time_entry(*, index: int, ts_ns_local: int) -> tc.EventStreamEntry:
    # EventStreamEntry is the ordered Core input unit: a canonical Event plus
    # ProcessingPosition telling Core where this Event sits in the Event Stream.
    # ControlTimeEvent here is only a driver Event; scheduling obligations come
    # from Execution Control apply (e.g. rate-limit deferral), not from every step.
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=tc.ControlTimeEvent(
            ts_ns_local_control=ts_ns_local,
            reason="scheduled_control_recheck",
            due_ts_ns_local=ts_ns_local,
            realized_ts_ns_local=ts_ns_local,
            obligation_reason="rate_limit",
            obligation_due_ts_ns_local=ts_ns_local,
            runtime_correlation=None,
        ),
    )


def run_v1_generated_only(state: tc.StrategyState) -> tc.CoreStepResult:
    # v1 shows the minimum deterministic step: Core reduces one canonical Event
    # and Strategy evaluation emits generated Intents. No policy/apply contexts
    # are provided yet, so Core returns zero dispatchable Intents by design.
    result = tc.run_core_step(
        state,
        _control_time_entry(index=0, ts_ns_local=1_000),
        strategy_evaluator=OneIntentEvaluator(INTENT_ID_V1),
    )
    assert len(result.generated_intents) == 1
    assert result.generated_intents[0].client_order_id == INTENT_ID_V1
    assert len(result.candidate_intent_records) == 1
    assert result.candidate_intent_records[0].origin is tc.CandidateIntentOrigin.GENERATED
    assert result.dispatchable_intents == ()
    return result


def run_v2_with_policy_and_apply(state: tc.StrategyState) -> tc.CoreStepResult:
    # v2 adds policy admission and Execution Control apply. With dispatchable
    # outputs activated, Core exposes Intents that Runtime can dispatch.
    result = tc.run_core_step(
        state,
        _control_time_entry(index=1, ts_ns_local=1_001),
        strategy_evaluator=OneIntentEvaluator(INTENT_ID_V2),
        policy_admission_context=tc.CorePolicyAdmissionContext(
            policy_evaluator=AllowAllPolicy(),
            now_ts_ns_local=1_001,
        ),
        execution_control_apply_context=tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=1_001,
            activate_dispatchable_outputs=True,
        ),
    )
    assert len(result.dispatchable_intents) == 1
    assert result.dispatchable_intents[0].client_order_id == INTENT_ID_V2
    return result


def main() -> None:
    # StrategyState holds deterministic Core memory across steps
    # (market snapshots, queued Intents, monotone timestamps, etc.).
    state = tc.StrategyState(event_bus=tc.NullEventBus())

    # Core consumes canonical Events. Here we use ControlTimeEvent as a simple
    # canonical trigger Event to drive the deterministic step Pipeline.
    result_v1 = run_v1_generated_only(state)
    result_v2 = run_v2_with_policy_and_apply(state)

    print("CoreStep quickstart (Core-only deterministic engine)")
    print("v1 generated:", [Intent.client_order_id for Intent in result_v1.generated_intents])
    print(
        "v1 candidate origins:",
        [record.origin.value for record in result_v1.candidate_intent_records],
    )
    print("v1 dispatchable: [] (Core does not dispatch externally)")
    print("v2 dispatchable:", [Intent.client_order_id for Intent in result_v2.dispatchable_intents])
    print("v2 obligation:", result_v2.control_scheduling_obligation)
    print("Runtime dispatches these; Core only returns decisions/Intents.")


if __name__ == "__main__":
    main()
