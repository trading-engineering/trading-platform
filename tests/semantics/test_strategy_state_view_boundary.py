"""Strategy boundary tests for read-only Strategy State views."""

from __future__ import annotations

import pytest
import tradingchassis_core as tc

INSTRUMENT = "BTC-USDC-PERP"
WORKING_ORDER_ID = "working-order-1"
FILL_ORDER_ID = "fill-order-1"
QUEUED_ORDER_ID = "queued-order-1"
INFLIGHT_ORDER_ID = "inflight-order-1"


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


def _order_submitted_entry(index: int, ts: int, client_order_id: str) -> tc.EventStreamEntry:
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=tc.OrderSubmittedEvent(
            ts_ns_local_dispatch=ts,
            instrument=INSTRUMENT,
            client_order_id=client_order_id,
            side="buy",
            order_type="limit",
            intended_price=tc.Price(currency="USDC", value=100.0),
            intended_qty=tc.Quantity(value=1.0, unit="contracts"),
            time_in_force="GTC",
            intent_correlation_id=None,
            dispatch_attempt_id=None,
            runtime_correlation=None,
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


def _fill_entry(index: int, ts: int, client_order_id: str) -> tc.EventStreamEntry:
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=tc.FillEvent(
            ts_ns_exch=ts - 1,
            ts_ns_local=ts,
            instrument=INSTRUMENT,
            client_order_id=client_order_id,
            side="buy",
            filled_price=tc.Price(currency="USDC", value=100.0),
            cum_filled_qty=tc.Quantity(value=1.0, unit="contracts"),
            remaining_qty=tc.Quantity(value=0.0, unit="contracts"),
            time_in_force="GTC",
            liquidity_flag="taker",
        ),
    )


def _queued_intent() -> tc.NewOrderIntent:
    return tc.NewOrderIntent(
        intent_type="new",
        ts_ns_local=100,
        instrument=INSTRUMENT,
        client_order_id=QUEUED_ORDER_ID,
        intents_correlation_id="queued-corr",
        side="buy",
        order_type="limit",
        intended_qty=tc.Quantity(value=1.0, unit="contracts"),
        intended_price=tc.Price(currency="USDC", value=99.5),
        time_in_force="GTC",
    )


def _seed_state() -> tc.StrategyState:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    state.update_market(
        instrument=INSTRUMENT,
        best_bid=99.0,
        best_ask=101.0,
        best_bid_qty=1.0,
        best_ask_qty=1.0,
        tick_size=0.1,
        lot_size=0.01,
        contract_size=1.0,
        ts_ns_local=100,
        ts_ns_exch=99,
    )
    tc.process_event_entry(state, _execution_feedback_entry(0, 101))
    tc.process_event_entry(state, _order_submitted_entry(1, 102, WORKING_ORDER_ID))
    tc.process_event_entry(state, _fill_entry(2, 103, FILL_ORDER_ID))
    state.merge_intents_into_queue(INSTRUMENT, [_queued_intent()])
    state.mark_intent_sent(INSTRUMENT, INFLIGHT_ORDER_ID, "replace")
    return state


def test_core_step_strategy_state_is_read_only_view() -> None:
    state = _seed_state()

    class _ReadOnlyStepEvaluator:
        def evaluate(self, context: tc.CoreStepStrategyContext) -> list[tc.OrderIntent]:
            assert isinstance(context.state, tc.StrategyStateView)
            assert context.state.sim_ts_ns_local >= 103
            assert context.state.market[INSTRUMENT].best_bid == 99.0
            assert context.state.account[INSTRUMENT].position == 2.5
            assert context.state.orders[INSTRUMENT][WORKING_ORDER_ID].client_order_id == WORKING_ORDER_ID
            assert len(context.state.fills[INSTRUMENT]) == 1
            assert context.state.fill_cum_qty[INSTRUMENT][FILL_ORDER_ID] == 1.0

            with pytest.raises(TypeError):
                context.state.market[INSTRUMENT] = context.state.market[INSTRUMENT]  # type: ignore[index]
            with pytest.raises(AttributeError):
                context.state.market[INSTRUMENT].best_bid = 123.0  # type: ignore[misc]
            with pytest.raises(TypeError):
                context.state.fill_cum_qty[INSTRUMENT][FILL_ORDER_ID] = 2.0  # type: ignore[index]

            copied_fill = context.state.fills[INSTRUMENT][0]
            copied_fill.side = "sell"

            assert not hasattr(context.state, "update_market")
            assert not hasattr(context.state, "update_account")
            assert not hasattr(context.state, "apply_fill_event")
            assert not hasattr(context.state, "apply_order_submitted_event")
            assert not hasattr(context.state, "apply_order_execution_feedback_event")
            assert not hasattr(context.state, "apply_control_time_event")
            assert not hasattr(context.state, "merge_intents_into_queue")
            assert not hasattr(context.state, "pop_queued_intents_for_order")
            assert not hasattr(context.state, "mark_intent_sent")
            assert not hasattr(context.state, "_advance_processing_position")
            assert not hasattr(context.state, "queued_intents")
            assert not hasattr(context.state, "inflight")
            return []

    _ = tc.run_core_step(
        state,
        _control_entry(3, 104),
        strategy_evaluator=_ReadOnlyStepEvaluator(),
    )

    assert state.market[INSTRUMENT].best_bid == 99.0
    assert state.account[INSTRUMENT].position == 2.5
    assert state.orders[INSTRUMENT][WORKING_ORDER_ID].intended_price == 100.0
    assert state.fills[INSTRUMENT][0].side == "buy"
    assert state.fill_cum_qty[INSTRUMENT][FILL_ORDER_ID] == 1.0
    assert state.has_queued_intent(INSTRUMENT, QUEUED_ORDER_ID)
    assert state.has_inflight(INSTRUMENT, INFLIGHT_ORDER_ID)


def test_core_wakeup_strategy_state_is_read_only_view() -> None:
    state = _seed_state()

    class _ReadOnlyWakeupEvaluator:
        def evaluate(self, context: tc.CoreWakeupStrategyContext) -> list[tc.OrderIntent]:
            assert isinstance(context.state, tc.StrategyStateView)
            assert len(context.entries) == 2
            assert context.last_position is not None
            assert context.last_position.index == 5
            assert context.state.market[INSTRUMENT].best_ask == 101.0
            with pytest.raises(AttributeError):
                setattr(
                    context.state.orders[INSTRUMENT][WORKING_ORDER_ID],
                    "state",
                    "cancelled",
                )
            return []

    _ = tc.run_core_wakeup_step(
        state,
        (_control_entry(4, 105), _control_entry(5, 106)),
        wakeup_strategy_evaluator=_ReadOnlyWakeupEvaluator(),
    )
