"""Final CoreWakeupStep Strategy evaluation semantics (Phase WU2)."""

from __future__ import annotations

import tradingchassis_core as tc
from tradingchassis_core.core.domain.types import BookLevel, BookPayload

INSTRUMENT = "BTC-USDC-PERP"

_TEST_CONFIGURATION = tc.CoreConfiguration(
    version="test-v1",
    payload={
        "market": {
            "instruments": {
                INSTRUMENT: {
                    "tick_size": 0.01,
                    "lot_size": 0.001,
                    "contract_size": 1.0,
                }
            }
        }
    },
)



class _OneIntentEvaluator:
    def evaluate(self, context: tc.CoreWakeupStrategyContext) -> list[tc.NewOrderIntent]:
        _ = context
        return [
            tc.NewOrderIntent(
                intent_type="new",
                ts_ns_local=10,
                instrument=INSTRUMENT,
                client_order_id="wake-generated",
                intents_correlation_id="corr-wake",
                side="buy",
                order_type="limit",
                intended_qty=tc.Quantity(value=1.0, unit="contracts"),
                intended_price=tc.Price(currency="USDC", value=100.0),
                time_in_force="GTC",
            )
        ]


class _CountingWakeupEvaluator:
    def __init__(self) -> None:
        self.call_count = 0
        self.last_context: tc.CoreWakeupStrategyContext | None = None

    def evaluate(self, context: tc.CoreWakeupStrategyContext) -> list[tc.OrderIntent]:
        self.call_count += 1
        self.last_context = context
        return []


class _FinalStateAwareEvaluator:
    def evaluate(self, context: tc.CoreWakeupStrategyContext) -> list[tc.OrderIntent]:
        assert context.state.sim_ts_ns_local == 203
        market = context.state.market[INSTRUMENT]
        assert market.best_bid == 99.0
        assert market.best_ask == 101.0
        account = context.state.account[INSTRUMENT]
        assert account.position == 2.5
        assert len(context.entries) == 3
        assert isinstance(context.entries[0].event, tc.OrderExecutionFeedbackEvent)
        assert isinstance(context.entries[1].event, tc.MarketEvent)
        assert isinstance(context.entries[2].event, tc.ControlTimeEvent)
        assert context.last_position is not None
        assert context.last_position.index == 2
        return []


class _AllowAllPolicy:
    def evaluate_policy_intent(
        self,
        *,
        intent: tc.OrderIntent,
        state: tc.StrategyState,
        now_ts_ns_local: int,
    ) -> tuple[bool, str | None]:
        _ = (intent, state, now_ts_ns_local)
        return True, None


def _control_entry(index: int, ts: int) -> tc.EventStreamEntry:
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=tc.ControlTimeEvent(
            ts_ns_local_control=ts,
            reason="scheduled_control_recheck",
            due_ts_ns_local=ts,
            realized_ts_ns_local=ts,
            obligation_reason="rate_limit",
            obligation_due_ts_ns_local=ts,
            runtime_correlation=None,
        ),
    )


def _market_entry(index: int, ts: int) -> tc.EventStreamEntry:
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=tc.MarketEvent(
            ts_ns_exch=ts - 1,
            ts_ns_local=ts,
            instrument=INSTRUMENT,
            event_type="book",
            book=BookPayload(
                book_type="snapshot",
                bids=[
                    BookLevel(
                        price=tc.Price(currency="USDC", value=99.0),
                        quantity=tc.Quantity(value=1.0, unit="contracts"),
                    )
                ],
                asks=[
                    BookLevel(
                        price=tc.Price(currency="USDC", value=101.0),
                        quantity=tc.Quantity(value=2.0, unit="contracts"),
                    )
                ],
                depth=1,
            ),
        ),
    )


def _execution_feedback_entry(index: int, ts: int) -> tc.EventStreamEntry:
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=tc.OrderExecutionFeedbackEvent(
            ts_ns_local_feedback=ts,
            instrument=INSTRUMENT,
            position=2.5,
            balance=10_000.0,
            fee=0.1,
            trading_volume=1.0,
            trading_value=100.0,
            num_trades=1,
            runtime_correlation=None,
        ),
    )


def _queued_intent(client_order_id: str) -> tc.NewOrderIntent:
    return tc.NewOrderIntent(
        intent_type="new",
        ts_ns_local=50,
        instrument=INSTRUMENT,
        client_order_id=client_order_id,
        intents_correlation_id="queued-corr",
        side="sell",
        order_type="limit",
        intended_qty=tc.Quantity(value=1.0, unit="contracts"),
        intended_price=tc.Price(currency="USDC", value=99.5),
        time_in_force="GTC",
    )


def test_wakeup_reduces_all_entries_before_strategy_evaluation() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    entries = (
        _execution_feedback_entry(0, 200),
        _market_entry(1, 201),
        _control_entry(2, 203),
    )
    _ = tc.run_core_wakeup_step(
        state,
        entries,
        configuration=_TEST_CONFIGURATION,
        wakeup_strategy_evaluator=_FinalStateAwareEvaluator(),
    )


def test_wakeup_strategy_evaluator_called_exactly_once() -> None:
    entries = (_control_entry(0, 100), _control_entry(1, 101))
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    evaluator = _CountingWakeupEvaluator()
    _ = tc.run_core_wakeup_step(state, entries, wakeup_strategy_evaluator=evaluator)
    assert evaluator.call_count == 1
    assert evaluator.last_context is not None
    assert evaluator.last_context.entries == entries
    assert evaluator.last_context.state.sim_ts_ns_local == 101
    assert evaluator.last_context.last_position == entries[-1].position


def test_wakeup_generated_intents_combined_once_with_queued_intents() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    state.merge_intents_into_queue(INSTRUMENT, [_queued_intent("queued-1")])
    result = tc.run_core_wakeup_step(
        state,
        (_control_entry(0, 100),),
        wakeup_strategy_evaluator=_OneIntentEvaluator(),
    )
    assert tuple(intent.client_order_id for intent in result.generated_intents) == (
        "wake-generated",
    )
    origins = tuple(record.origin for record in result.candidate_intent_records)
    assert tc.CandidateIntentOrigin.GENERATED in origins
    assert tc.CandidateIntentOrigin.QUEUED in origins
    client_ids = {record.intent.client_order_id for record in result.candidate_intent_records}
    assert client_ids == {"wake-generated", "queued-1"}


def test_wakeup_policy_and_execution_control_apply_once() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    result = tc.run_core_wakeup_step(
        state,
        (_control_entry(0, 100), _control_entry(1, 101)),
        wakeup_strategy_evaluator=_OneIntentEvaluator(),
        policy_admission_context=tc.CorePolicyAdmissionContext(
            policy_evaluator=_AllowAllPolicy(),
            now_ts_ns_local=101,
        ),
        execution_control_apply_context=tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=101,
            activate_dispatchable_outputs=True,
        ),
    )
    assert len(result.generated_intents) == 1
    assert len(result.dispatchable_intents) == 1
    assert result.core_step_decision is not None
    policy_decision = result.core_step_decision.policy_risk_decision
    assert policy_decision is not None
    assert len(policy_decision.accepted_intents) == 1


def test_empty_wakeup_batch_is_valid_without_evaluator() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    reduction = tc.run_core_wakeup_reduction(state, ())
    assert reduction.entries == ()
    assert reduction.generated_intents == ()
    result = tc.run_core_wakeup_decision(state, reduction)
    assert result.generated_intents == ()
    assert result.candidate_intent_records == ()


def test_empty_wakeup_batch_with_evaluator_runs_once() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    evaluator = _CountingWakeupEvaluator()
    reduction = tc.run_core_wakeup_reduction(state, (), wakeup_strategy_evaluator=evaluator)
    assert evaluator.call_count == 1
    assert reduction.generated_intents == ()
    assert evaluator.last_context is not None
    assert evaluator.last_context.entries == ()
    assert evaluator.last_context.last_position is None


def test_single_entry_wakeup_evaluates_once() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    evaluator = _CountingWakeupEvaluator()
    _ = tc.run_core_wakeup_step(state, (_control_entry(0, 100),), wakeup_strategy_evaluator=evaluator)
    assert evaluator.call_count == 1


def test_run_core_wakeup_step_matches_reduction_then_decision_path() -> None:
    entries = (_control_entry(0, 100), _control_entry(1, 101))
    reduction_state = tc.StrategyState(event_bus=tc.NullEventBus())
    reduction = tc.run_core_wakeup_reduction(
        reduction_state,
        entries,
        wakeup_strategy_evaluator=_OneIntentEvaluator(),
    )
    decision_result = tc.run_core_wakeup_decision(
        reduction_state,
        reduction,
        policy_admission_context=tc.CorePolicyAdmissionContext(
            policy_evaluator=_AllowAllPolicy(),
            now_ts_ns_local=101,
        ),
        execution_control_apply_context=tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=101,
            activate_dispatchable_outputs=True,
        ),
    )

    step_state = tc.StrategyState(event_bus=tc.NullEventBus())
    step_result = tc.run_core_wakeup_step(
        step_state,
        entries,
        wakeup_strategy_evaluator=_OneIntentEvaluator(),
        policy_admission_context=tc.CorePolicyAdmissionContext(
            policy_evaluator=_AllowAllPolicy(),
            now_ts_ns_local=101,
        ),
        execution_control_apply_context=tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=101,
            activate_dispatchable_outputs=True,
        ),
    )

    assert decision_result == step_result
    assert len(step_result.generated_intents) == 1
    assert len(step_result.dispatchable_intents) == 1
