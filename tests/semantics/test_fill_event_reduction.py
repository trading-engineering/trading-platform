"""FillEvent canonical reduction and Core step pipeline coverage."""

from __future__ import annotations

import tradingchassis_core as tc

INSTRUMENT = "BTC-USDC-PERP"
CLIENT_ORDER_ID = "fill-order-1"
FILL_TS = 200


def _fill_event(*, cum_qty: float, remaining_qty: float | None) -> tc.FillEvent:
    return tc.FillEvent(
        ts_ns_exch=FILL_TS - 1,
        ts_ns_local=FILL_TS,
        instrument=INSTRUMENT,
        client_order_id=CLIENT_ORDER_ID,
        side="buy",
        filled_price=tc.Price(currency="USDC", value=100.0),
        cum_filled_qty=tc.Quantity(value=cum_qty, unit="contracts"),
        remaining_qty=(
            None
            if remaining_qty is None
            else tc.Quantity(value=remaining_qty, unit="contracts")
        ),
        time_in_force="GTC",
        liquidity_flag="taker",
    )


def _order_submitted_entry(index: int, ts: int) -> tc.EventStreamEntry:
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=tc.OrderSubmittedEvent(
            ts_ns_local_dispatch=ts,
            instrument=INSTRUMENT,
            client_order_id=CLIENT_ORDER_ID,
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


def _fill_entry(index: int) -> tc.EventStreamEntry:
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=_fill_event(cum_qty=1.0, remaining_qty=0.0),
    )


def test_fill_event_via_process_event_entry_updates_state() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    tc.process_event_entry(state, _order_submitted_entry(0, 100))
    assert state.has_working_order(INSTRUMENT, CLIENT_ORDER_ID)

    tc.process_event_entry(state, _fill_entry(1))

    assert state.sim_ts_ns_local == FILL_TS
    fills = state.fills[INSTRUMENT]
    assert len(fills) == 1
    assert fills[0].client_order_id == CLIENT_ORDER_ID
    assert state.fill_cum_qty[INSTRUMENT][CLIENT_ORDER_ID] == 1.0
    assert not state.has_working_order(INSTRUMENT, CLIENT_ORDER_ID)


def test_run_core_step_strategy_evaluator_sees_fill_reduced_state() -> None:
    class _AssertFillReducedEvaluator:
        def evaluate(self, context: tc.CoreStepStrategyContext) -> list[tc.OrderIntent]:
            assert context.state.sim_ts_ns_local == FILL_TS
            assert INSTRUMENT in context.state.fills
            assert len(context.state.fills[INSTRUMENT]) == 1
            assert isinstance(context.event, tc.FillEvent)
            return []

    state = tc.StrategyState(event_bus=tc.NullEventBus())
    tc.process_event_entry(state, _order_submitted_entry(0, 100))

    result = tc.run_core_step(
        state,
        _fill_entry(1),
        strategy_evaluator=_AssertFillReducedEvaluator(),
    )

    assert result.generated_intents == ()
    assert result.candidate_intent_records == ()
    assert result.dispatchable_intents == ()
    assert result.core_step_decision is None
