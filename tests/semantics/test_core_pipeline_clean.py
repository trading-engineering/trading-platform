"""Clean CoreStep/CoreWakeupStep pipeline tests."""

from __future__ import annotations

import tradingchassis_core as tc
from tradingchassis_core.core.domain.types import BookLevel, BookPayload


class _OneIntentEvaluator:
    def evaluate(self, context: object) -> list[tc.NewOrderIntent]:
        _ = context
        return [
            tc.NewOrderIntent(
                intent_type="new",
                ts_ns_local=10,
                instrument="BTC-USDC-PERP",
                client_order_id="intent-1",
                intents_correlation_id="corr-1",
                side="buy",
                order_type="limit",
                intended_qty=tc.Quantity(value=1.0, unit="contracts"),
                intended_price=tc.Price(currency="USDC", value=100.0),
                time_in_force="GTC",
            )
        ]


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


class _RejectAllPolicy:
    def evaluate_policy_intent(
        self,
        *,
        intent: tc.OrderIntent,
        state: tc.StrategyState,
        now_ts_ns_local: int,
    ) -> tuple[bool, str | None]:
        _ = (intent, state, now_ts_ns_local)
        return False, "blocked_for_test"


class _DuplicateIntentEvaluator:
    def evaluate(self, context: object) -> list[tc.NewOrderIntent]:
        _ = context
        first = tc.NewOrderIntent(
            intent_type="new",
            ts_ns_local=10,
            instrument="BTC-USDC-PERP",
            client_order_id="dup-intent",
            intents_correlation_id="corr-a",
            side="buy",
            order_type="limit",
            intended_qty=tc.Quantity(value=1.0, unit="contracts"),
            intended_price=tc.Price(currency="USDC", value=100.0),
            time_in_force="GTC",
        )
        second = tc.NewOrderIntent(
            intent_type="new",
            ts_ns_local=11,
            instrument="BTC-USDC-PERP",
            client_order_id="dup-intent",
            intents_correlation_id="corr-b",
            side="buy",
            order_type="limit",
            intended_qty=tc.Quantity(value=2.0, unit="contracts"),
            intended_price=tc.Price(currency="USDC", value=101.0),
            time_in_force="GTC",
        )
        return [first, second]


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


def test_run_core_step_clean_pipeline_dispatchable() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    result = tc.run_core_step(
        state,
        _control_entry(0, 100),
        strategy_evaluator=_OneIntentEvaluator(),
        policy_admission_context=tc.CorePolicyAdmissionContext(
            policy_evaluator=_AllowAllPolicy(),
            now_ts_ns_local=100,
        ),
        execution_control_apply_context=tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=100,
            activate_dispatchable_outputs=True,
        ),
    )
    assert tuple(intent.client_order_id for intent in result.generated_intents) == ("intent-1",)
    assert tuple(intent.client_order_id for intent in result.candidate_intents) == ("intent-1",)
    assert tuple(intent.client_order_id for intent in result.dispatchable_intents) == ("intent-1",)
    assert result.core_step_decision is not None


def test_run_core_step_processes_entry_before_strategy_evaluation() -> None:
    class _ChecksReducedStateEvaluator:
        def evaluate(self, context: tc.CoreStepStrategyContext) -> list[tc.OrderIntent]:
            # ControlTimeEvent reduction updates monotone timestamp before evaluation.
            assert context.state.sim_ts_ns_local == 100
            return []

    state = tc.StrategyState(event_bus=tc.NullEventBus())
    _ = tc.run_core_step(
        state,
        _control_entry(0, 100),
        strategy_evaluator=_ChecksReducedStateEvaluator(),
    )
    assert state._last_processing_position_index == 0


def test_run_core_wakeup_step_clean_pipeline_dispatchable() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    result = tc.run_core_wakeup_step(
        state,
        (_control_entry(0, 100), _control_entry(1, 101)),
        strategy_evaluator=_OneIntentEvaluator(),
        strategy_event_filter=lambda _event: True,
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
    assert len(result.generated_intents) == 2
    assert len(result.candidate_intent_records) == 1
    assert len(result.dispatchable_intents) == 1


def test_candidate_reconciliation_prefers_latest_same_key_generated_intent() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    result = tc.run_core_step(
        state,
        _control_entry(0, 100),
        strategy_evaluator=_DuplicateIntentEvaluator(),
    )
    assert len(result.generated_intents) == 2
    assert len(result.candidate_intent_records) == 1
    winner = result.candidate_intent_records[0].intent
    assert isinstance(winner, tc.NewOrderIntent)
    assert winner.client_order_id == "dup-intent"
    assert winner.intended_qty.value == 2.0


def test_policy_rejection_prevents_dispatchable_intents() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    result = tc.run_core_step(
        state,
        _control_entry(0, 100),
        strategy_evaluator=_OneIntentEvaluator(),
        policy_admission_context=tc.CorePolicyAdmissionContext(
            policy_evaluator=_RejectAllPolicy(),
            now_ts_ns_local=100,
        ),
        execution_control_apply_context=tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=100,
            activate_dispatchable_outputs=True,
        ),
    )
    assert result.dispatchable_intents == ()
    assert result.core_step_decision is not None
    assert len(result.core_step_decision.policy_rejected_intents) == 1


def test_execution_control_deferral_returns_scheduling_obligation() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    result = tc.run_core_step(
        state,
        _control_entry(0, 100),
        strategy_evaluator=_OneIntentEvaluator(),
        policy_admission_context=tc.CorePolicyAdmissionContext(
            policy_evaluator=_AllowAllPolicy(),
            now_ts_ns_local=100,
        ),
        execution_control_apply_context=tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=100,
            max_orders_per_sec=0.0,
            activate_dispatchable_outputs=True,
        ),
    )
    assert result.dispatchable_intents == ()
    assert result.control_scheduling_obligation is not None
    assert result.control_scheduling_obligation.reason == "rate_limit"


def test_process_canonical_event_reduces_market_event() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    tc.process_canonical_event(
        state,
        tc.MarketEvent(
            ts_ns_exch=200,
            ts_ns_local=201,
            instrument="BTC-USDC-PERP",
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
    assert state.sim_ts_ns_local == 201
    market = state.market["BTC-USDC-PERP"]
    assert market.best_bid == 99.0
    assert market.best_ask == 101.0
