"""Semantics tests for the minimal canonical processing boundary."""

from __future__ import annotations

import copy

import pytest

from tradingchassis_core.core.domain.configuration import CoreConfiguration
from tradingchassis_core.core.domain.event_model import is_canonical_stream_candidate_type
from tradingchassis_core.core.domain.processing import process_canonical_event, process_event_entry
from tradingchassis_core.core.domain.processing_order import EventStreamEntry, ProcessingPosition
from tradingchassis_core.core.domain.state import StrategyState
from tradingchassis_core.core.domain.types import (
    ControlTimeEvent,
    FillEvent,
    MarketEvent,
    OrderExecutionFeedbackEvent,
    OrderExecutionFeedbackSnapshot,
    OrderStateEvent,
    OrderSubmittedEvent,
    Price,
    Quantity,
)
from tradingchassis_core.core.events.event_bus import EventBus
from tradingchassis_core.core.events.events import DerivedFillEvent, RiskDecisionEvent
from tradingchassis_core.core.events.sinks.null_event_bus import NullEventBus


def _state_subset_snapshot(state: StrategyState) -> dict[str, object]:
    return {
        "market": copy.deepcopy(state.market),
        "fills": copy.deepcopy(state.fills),
        "fill_cum_qty": copy.deepcopy(state.fill_cum_qty),
    }


def _book_market_event(
    *,
    instrument: str,
    ts_ns_local: int,
    ts_ns_exch: int,
    best_bid: float = 100.0,
    best_ask: float = 101.0,
    best_bid_qty: float = 2.0,
    best_ask_qty: float = 3.0,
) -> MarketEvent:
    return MarketEvent(
        ts_ns_local=ts_ns_local,
        ts_ns_exch=ts_ns_exch,
        instrument=instrument,
        event_type="book",
        book={
            "book_type": "snapshot",
            "bids": [
                {
                    "price": {"currency": "USDC", "value": best_bid},
                    "quantity": {"unit": "contracts", "value": best_bid_qty},
                }
            ],
            "asks": [
                {
                    "price": {"currency": "USDC", "value": best_ask},
                    "quantity": {"unit": "contracts", "value": best_ask_qty},
                }
            ],
            "depth": 1,
        },
        trade=None,
    )


def _fill_event(
    *,
    instrument: str,
    client_order_id: str,
    ts_ns_local: int,
    ts_ns_exch: int,
    cum_filled_qty: float = 0.25,
) -> FillEvent:
    return FillEvent(
        ts_ns_local=ts_ns_local,
        ts_ns_exch=ts_ns_exch,
        instrument=instrument,
        client_order_id=client_order_id,
        side="buy",
        intended_price=Price(currency="USDC", value=100.0),
        filled_price=Price(currency="USDC", value=100.5),
        intended_qty=Quantity(unit="contracts", value=1.0),
        cum_filled_qty=Quantity(unit="contracts", value=cum_filled_qty),
        remaining_qty=Quantity(unit="contracts", value=max(0.0, 1.0 - cum_filled_qty)),
        time_in_force="GTC",
        liquidity_flag="maker",
        fee=None,
    )


def _order_state_event(
    *,
    instrument: str,
    client_order_id: str,
    ts_ns_local: int,
    ts_ns_exch: int,
    state_type: str = "accepted",
) -> OrderStateEvent:
    return OrderStateEvent(
        ts_ns_local=ts_ns_local,
        ts_ns_exch=ts_ns_exch,
        instrument=instrument,
        client_order_id=client_order_id,
        order_type="limit",
        state_type=state_type,
        side="buy",
        intended_price=Price(currency="USDC", value=100.0),
        filled_price=None,
        intended_qty=Quantity(unit="contracts", value=1.0),
        cum_filled_qty=None,
        remaining_qty=None,
        time_in_force="GTC",
        reason=None,
        raw={"req": 0, "source": "snapshot"},
    )


def _order_submitted_event(
    *,
    instrument: str,
    client_order_id: str,
    ts_ns_local_dispatch: int,
) -> OrderSubmittedEvent:
    return OrderSubmittedEvent(
        ts_ns_local_dispatch=ts_ns_local_dispatch,
        instrument=instrument,
        client_order_id=client_order_id,
        side="buy",
        order_type="limit",
        intended_price=Price(currency="USDC", value=100.0),
        intended_qty=Quantity(unit="contracts", value=1.0),
        time_in_force="GTC",
        intent_correlation_id="corr-1",
        dispatch_attempt_id="attempt-1",
        runtime_correlation={"engine": "backtest", "seq": 1},
    )


def _order_execution_feedback_event(
    *,
    instrument: str,
    ts_ns_local_feedback: int,
    order_id: str = "101",
) -> OrderExecutionFeedbackEvent:
    return OrderExecutionFeedbackEvent(
        ts_ns_local_feedback=ts_ns_local_feedback,
        instrument=instrument,
        position=1.25,
        balance=10_000.0,
        fee=3.5,
        trading_volume=20.0,
        trading_value=2_050.0,
        num_trades=7,
        order_snapshots=(
            OrderExecutionFeedbackSnapshot(
                order_id=order_id,
                order_type=0,
                side=1,
                time_in_force=0,
                status=1,
                req=0,
                price=100.0,
                qty=1.0,
                exec_price=100.25,
                exec_qty=0.25,
                leaves_qty=0.75,
                ts_ns_exch=ts_ns_local_feedback - 1,
                ts_ns_local=ts_ns_local_feedback,
            ),
        ),
        runtime_correlation={"source": "unit-test"},
    )


def _control_time_event(
    *,
    ts_ns_local_control: int,
    reason: str = "rate_limit_recheck",
    due_ts_ns_local: int | None = None,
    realized_ts_ns_local: int | None = None,
) -> ControlTimeEvent:
    return ControlTimeEvent(
        ts_ns_local_control=ts_ns_local_control,
        reason=reason,
        due_ts_ns_local=due_ts_ns_local,
        realized_ts_ns_local=realized_ts_ns_local,
        obligation_reason="rate_limit",
        obligation_due_ts_ns_local=due_ts_ns_local,
        runtime_correlation={"engine": "backtest", "seq": 1},
    )


def _market_configuration(
    *,
    instrument: str = "BTC-USDC-PERP",
    tick_size: float = 0.1,
    lot_size: float = 0.01,
    contract_size: float = 1.0,
) -> CoreConfiguration:
    return CoreConfiguration(
        version="v1",
        payload={
            "market": {
                "instruments": {
                    instrument: {
                        "tick_size": tick_size,
                        "lot_size": lot_size,
                        "contract_size": contract_size,
                    }
                }
            }
        },
    )


def _control_state_snapshot(state: StrategyState) -> dict[str, object]:
    return {
        "queued_intents": copy.deepcopy(state.queued_intents),
        "inflight": copy.deepcopy(state.inflight),
        "orders": copy.deepcopy(state.orders),
        "canonical_orders": copy.deepcopy(state.canonical_orders),
        "fills": copy.deepcopy(state.fills),
        "market": copy.deepcopy(state.market),
        "account": copy.deepcopy(state.account),
    }


def test_process_canonical_event_accepts_market_event() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=100, ts_ns_exch=90)

    process_canonical_event(state, event)

    market = state.market["BTC-USDC-PERP"]
    assert market.last_ts_ns_local == 100
    assert market.last_ts_ns_exch == 90
    assert market.best_bid == 100.0
    assert market.best_ask == 101.0
    assert market.best_bid_qty == 2.0
    assert market.best_ask_qty == 3.0
    assert market.mid == 100.5


def test_process_canonical_event_accepts_market_event_with_processing_position() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=100, ts_ns_exch=90)
    position = ProcessingPosition(index=5)

    process_canonical_event(state, event, position=position, configuration=_market_configuration())

    market = state.market["BTC-USDC-PERP"]
    assert market.last_ts_ns_local == 100
    assert market.last_ts_ns_exch == 90
    assert market.best_bid == 100.0
    assert market.best_ask == 101.0
    assert market.best_bid_qty == 2.0
    assert market.best_ask_qty == 3.0
    assert market.mid == 100.5
    assert state._last_processing_position_index == 5


def test_process_canonical_event_accepts_fill_event() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=200,
        ts_ns_exch=180,
    )

    process_canonical_event(state, event)

    fills = state.fills["BTC-USDC-PERP"]
    assert len(fills) == 1
    assert fills[0] == event
    assert state.fill_cum_qty["BTC-USDC-PERP"]["order-1"] == 0.25


def test_process_canonical_event_accepts_fill_event_with_processing_position() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=200,
        ts_ns_exch=180,
    )
    position = ProcessingPosition(index=12)

    process_canonical_event(state, event, position=position)

    fills = state.fills["BTC-USDC-PERP"]
    assert len(fills) == 1
    assert fills[0] == event
    assert state.fill_cum_qty["BTC-USDC-PERP"]["order-1"] == 0.25
    assert state._last_processing_position_index == 12


def test_process_canonical_event_accepts_order_execution_feedback_event() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _order_execution_feedback_event(
        instrument="BTC-USDC-PERP",
        ts_ns_local_feedback=250,
    )

    process_canonical_event(state, event)

    account = state.account["BTC-USDC-PERP"]
    assert account.position == 1.25
    assert account.balance == 10_000.0
    assert account.fee == 3.5
    assert account.trading_volume == 20.0
    assert account.trading_value == 2_050.0
    assert account.num_trades == 7
    assert state.orders["BTC-USDC-PERP"]["101"].state_type == "working"


def test_process_canonical_event_accepts_order_execution_feedback_event_with_position() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _order_execution_feedback_event(
        instrument="BTC-USDC-PERP",
        ts_ns_local_feedback=260,
        order_id="102",
    )

    process_canonical_event(
        state,
        event,
        position=ProcessingPosition(index=13),
    )

    assert state._last_processing_position_index == 13
    assert state.account["BTC-USDC-PERP"].num_trades == 7
    assert state.orders["BTC-USDC-PERP"]["102"].state_type == "working"


def test_process_canonical_event_accepts_order_submitted_event() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _order_submitted_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-submitted-1",
        ts_ns_local_dispatch=300,
    )

    process_canonical_event(state, event)

    projection = state.canonical_orders[("BTC-USDC-PERP", "order-submitted-1")]
    assert projection.state == "submitted"
    assert projection.submitted_ts_ns_local == 300
    assert projection.updated_ts_ns_local == 300


def test_process_canonical_event_accepts_order_submitted_event_with_processing_position() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _order_submitted_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-submitted-1",
        ts_ns_local_dispatch=300,
    )

    process_canonical_event(state, event, position=ProcessingPosition(index=13))

    projection = state.canonical_orders[("BTC-USDC-PERP", "order-submitted-1")]
    assert projection.state == "submitted"
    assert projection.submitted_ts_ns_local == 300
    assert projection.updated_ts_ns_local == 300
    assert state._last_processing_position_index == 13


def test_control_time_event_requires_due_or_realized_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="at least one of due_ts_ns_local or realized_ts_ns_local is required",
    ):
        _control_time_event(ts_ns_local_control=500)


def test_control_time_event_rejects_extra_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ControlTimeEvent(
            ts_ns_local_control=501,
            reason="rate_limit_recheck",
            due_ts_ns_local=600,
            extra_field="unexpected",
        )


def test_process_canonical_event_accepts_control_time_event_with_processing_position() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _control_time_event(
        ts_ns_local_control=510,
        due_ts_ns_local=520,
    )

    process_canonical_event(state, event, position=ProcessingPosition(index=14))

    assert state._last_processing_position_index == 14


def test_process_canonical_event_control_time_event_does_not_mutate_state_buckets() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _control_time_event(
        ts_ns_local_control=530,
        realized_ts_ns_local=531,
    )
    before = _control_state_snapshot(state)

    process_canonical_event(state, event, position=ProcessingPosition(index=15))

    after = _control_state_snapshot(state)
    assert after == before
    assert state._last_processing_position_index == 15


def test_control_time_event_still_obeys_global_processing_position_monotonicity() -> None:
    state = StrategyState(event_bus=NullEventBus())
    first = _control_time_event(
        ts_ns_local_control=540,
        due_ts_ns_local=550,
    )
    repeated = _control_time_event(
        ts_ns_local_control=541,
        due_ts_ns_local=551,
    )

    process_canonical_event(state, first, position=ProcessingPosition(index=16))
    with pytest.raises(ValueError, match="Non-monotonic ProcessingPosition index"):
        process_canonical_event(state, repeated, position=ProcessingPosition(index=16))

    assert state._last_processing_position_index == 16


def test_first_positioned_event_is_accepted() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=100, ts_ns_exch=90)

    process_canonical_event(
        state,
        event,
        position=ProcessingPosition(index=0),
        configuration=_market_configuration(),
    )

    assert state._last_processing_position_index == 0


def test_increasing_positions_are_accepted() -> None:
    state = StrategyState(event_bus=NullEventBus())
    first = _book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=100, ts_ns_exch=90)
    second = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=101,
        ts_ns_exch=91,
    )

    process_canonical_event(
        state,
        first,
        position=ProcessingPosition(index=10),
        configuration=_market_configuration(),
    )
    process_canonical_event(state, second, position=ProcessingPosition(index=11))

    assert state._last_processing_position_index == 11


def test_repeated_position_is_rejected_without_state_mutation() -> None:
    state = StrategyState(event_bus=NullEventBus())
    accepted = _book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=100, ts_ns_exch=90)
    rejected = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=101,
        ts_ns_exch=91,
    )

    process_canonical_event(
        state,
        accepted,
        position=ProcessingPosition(index=3),
        configuration=_market_configuration(),
    )
    before = _state_subset_snapshot(state)

    with pytest.raises(ValueError, match="Non-monotonic ProcessingPosition index"):
        process_canonical_event(state, rejected, position=ProcessingPosition(index=3))

    after = _state_subset_snapshot(state)
    assert after == before
    assert state._last_processing_position_index == 3


def test_regressing_position_is_rejected_without_state_mutation() -> None:
    state = StrategyState(event_bus=NullEventBus())
    accepted = _book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=100, ts_ns_exch=90)
    rejected = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=102,
        ts_ns_exch=92,
    )

    process_canonical_event(
        state,
        accepted,
        position=ProcessingPosition(index=8),
        configuration=_market_configuration(),
    )
    before = _state_subset_snapshot(state)

    with pytest.raises(ValueError, match="Non-monotonic ProcessingPosition index"):
        process_canonical_event(state, rejected, position=ProcessingPosition(index=7))

    after = _state_subset_snapshot(state)
    assert after == before
    assert state._last_processing_position_index == 8


def test_position_none_remains_allowed_and_does_not_advance_cursor() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=100, ts_ns_exch=90)

    process_canonical_event(state, event, position=None)

    assert state._last_processing_position_index is None

    positioned = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=101,
        ts_ns_exch=91,
    )
    process_canonical_event(state, positioned, position=ProcessingPosition(index=0))
    assert state._last_processing_position_index == 0


def test_processing_position_is_not_derived_from_event_time() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=1_000_000, ts_ns_exch=900_000)
    position = ProcessingPosition(index=1)

    process_canonical_event(state, event, position=position, configuration=_market_configuration())

    market = state.market["BTC-USDC-PERP"]
    assert market.last_ts_ns_local == event.ts_ns_local
    assert market.last_ts_ns_exch == event.ts_ns_exch


def test_event_time_out_of_order_but_position_increasing_is_accepted_at_boundary() -> None:
    state = StrategyState(event_bus=NullEventBus())
    first = _book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=200, ts_ns_exch=190)
    second = _book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=100, ts_ns_exch=95)

    configuration = _market_configuration()
    process_canonical_event(state, first, position=ProcessingPosition(index=1), configuration=configuration)
    process_canonical_event(state, second, position=ProcessingPosition(index=2), configuration=configuration)

    assert state._last_processing_position_index == 2
    # Positioned canonical market events are now ProcessingPosition-driven.
    market = state.market["BTC-USDC-PERP"]
    assert market.last_ts_ns_local == 100
    assert market.last_ts_ns_exch == 95


def test_position_out_of_order_but_event_time_increasing_is_rejected_at_boundary() -> None:
    state = StrategyState(event_bus=NullEventBus())
    first = _book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=100, ts_ns_exch=90)
    second = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=200,
        ts_ns_exch=180,
    )

    process_canonical_event(
        state,
        first,
        position=ProcessingPosition(index=5),
        configuration=_market_configuration(),
    )
    before = _state_subset_snapshot(state)

    with pytest.raises(ValueError, match="Non-monotonic ProcessingPosition index"):
        process_canonical_event(state, second, position=ProcessingPosition(index=4))

    after = _state_subset_snapshot(state)
    assert after == before
    assert state._last_processing_position_index == 5


@pytest.mark.parametrize("second_cum_filled_qty", [0.25, 0.20])
def test_positioned_fill_ordering_divergence_advances_cursor_but_keeps_fill_state_idempotent(
    second_cum_filled_qty: float,
) -> None:
    state = StrategyState(event_bus=NullEventBus())
    first = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=200,
        ts_ns_exch=180,
        cum_filled_qty=0.25,
    )
    second = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=201,
        ts_ns_exch=181,
        cum_filled_qty=second_cum_filled_qty,
    )

    process_canonical_event(state, first, position=ProcessingPosition(index=20))
    fills_before = copy.deepcopy(state.fills)
    fill_cum_before = copy.deepcopy(state.fill_cum_qty)

    process_canonical_event(state, second, position=ProcessingPosition(index=21))

    assert state._last_processing_position_index == 21
    assert state.fills == fills_before
    assert state.fill_cum_qty == fill_cum_before
    assert len(state.fills["BTC-USDC-PERP"]) == 1
    assert state.fill_cum_qty["BTC-USDC-PERP"]["order-1"] == 0.25


def test_interleaved_positioned_and_unpositioned_processing_preserves_cursor_monotonicity() -> None:
    state = StrategyState(event_bus=NullEventBus())
    positioned_10 = _book_market_event(
        instrument="BTC-USDC-PERP",
        ts_ns_local=100,
        ts_ns_exch=90,
    )
    unpositioned = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=101,
        ts_ns_exch=91,
        cum_filled_qty=0.25,
    )
    positioned_11 = _book_market_event(
        instrument="BTC-USDC-PERP",
        ts_ns_local=102,
        ts_ns_exch=92,
    )
    rejected = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=103,
        ts_ns_exch=93,
        cum_filled_qty=0.50,
    )

    configuration = _market_configuration()
    process_canonical_event(
        state,
        positioned_10,
        position=ProcessingPosition(index=10),
        configuration=configuration,
    )
    assert state._last_processing_position_index == 10

    process_canonical_event(state, unpositioned, position=None)
    assert state._last_processing_position_index == 10

    process_canonical_event(
        state,
        positioned_11,
        position=ProcessingPosition(index=11),
        configuration=configuration,
    )
    assert state._last_processing_position_index == 11

    with pytest.raises(ValueError, match="Non-monotonic ProcessingPosition index"):
        process_canonical_event(state, rejected, position=ProcessingPosition(index=10))
    with pytest.raises(ValueError, match="Non-monotonic ProcessingPosition index"):
        process_canonical_event(state, rejected, position=ProcessingPosition(index=11))

    assert state._last_processing_position_index == 11


def test_positioned_market_tiebreak_no_longer_gates_positioned_market_updates() -> None:
    state = StrategyState(event_bus=NullEventBus())
    base = _book_market_event(
        instrument="BTC-USDC-PERP",
        ts_ns_local=300,
        ts_ns_exch=200,
        best_bid=100.0,
        best_ask=101.0,
    )
    lower_exch = _book_market_event(
        instrument="BTC-USDC-PERP",
        ts_ns_local=300,
        ts_ns_exch=199,
        best_bid=80.0,
        best_ask=81.0,
    )
    higher_exch = _book_market_event(
        instrument="BTC-USDC-PERP",
        ts_ns_local=300,
        ts_ns_exch=201,
        best_bid=120.0,
        best_ask=121.0,
    )

    configuration = _market_configuration()
    process_canonical_event(state, base, position=ProcessingPosition(index=30), configuration=configuration)
    process_canonical_event(
        state,
        lower_exch,
        position=ProcessingPosition(index=31),
        configuration=configuration,
    )

    market = state.market["BTC-USDC-PERP"]
    assert state._last_processing_position_index == 31
    assert market.last_ts_ns_local == 300
    assert market.last_ts_ns_exch == 199
    assert market.best_bid == 80.0
    assert market.best_ask == 81.0

    process_canonical_event(
        state,
        higher_exch,
        position=ProcessingPosition(index=32),
        configuration=configuration,
    )

    market_after_higher = state.market["BTC-USDC-PERP"]
    assert state._last_processing_position_index == 32
    assert market_after_higher.last_ts_ns_local == 300
    assert market_after_higher.last_ts_ns_exch == 201
    assert market_after_higher.best_bid == 120.0
    assert market_after_higher.best_ask == 121.0


def test_valid_processing_position_can_authorize_boundary_order_while_reducer_noops() -> None:
    """Valid ProcessingPosition advances causal boundary while reducer may still no-op."""
    state = StrategyState(event_bus=NullEventBus())
    first = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=400,
        ts_ns_exch=390,
        cum_filled_qty=0.40,
    )
    duplicate = _fill_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-1",
        ts_ns_local=401,
        ts_ns_exch=391,
        cum_filled_qty=0.40,
    )

    process_canonical_event(state, first, position=ProcessingPosition(index=40))
    fills_before = copy.deepcopy(state.fills)
    fill_cum_before = copy.deepcopy(state.fill_cum_qty)

    process_canonical_event(state, duplicate, position=ProcessingPosition(index=41))

    assert state._last_processing_position_index == 41
    assert state.fills == fills_before
    assert state.fill_cum_qty == fill_cum_before


def test_positioned_order_submitted_duplicate_is_idempotent_while_cursor_advances() -> None:
    state = StrategyState(event_bus=NullEventBus())
    first = _order_submitted_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-submitted-dup-1",
        ts_ns_local_dispatch=700,
    )
    duplicate = _order_submitted_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-submitted-dup-1",
        ts_ns_local_dispatch=701,
    )

    process_canonical_event(state, first, position=ProcessingPosition(index=42))
    projection_before = copy.deepcopy(
        state.canonical_orders[("BTC-USDC-PERP", "order-submitted-dup-1")]
    )

    process_canonical_event(state, duplicate, position=ProcessingPosition(index=43))

    projection_after = state.canonical_orders[("BTC-USDC-PERP", "order-submitted-dup-1")]
    assert state._last_processing_position_index == 43
    assert projection_after == projection_before


def test_order_submitted_event_does_not_regress_existing_canonical_state() -> None:
    state = StrategyState(event_bus=NullEventBus())
    key = ("BTC-USDC-PERP", "order-no-regress-1")
    first = _order_submitted_event(
        instrument=key[0],
        client_order_id=key[1],
        ts_ns_local_dispatch=800,
    )
    accepted = _fill_event(
        instrument=key[0],
        client_order_id=key[1],
        ts_ns_local=810,
        ts_ns_exch=805,
        cum_filled_qty=0.25,
    )
    late_submitted = _order_submitted_event(
        instrument=key[0],
        client_order_id=key[1],
        ts_ns_local_dispatch=820,
    )

    process_canonical_event(state, first, position=ProcessingPosition(index=50))
    state.apply_order_state_event(
        _order_state_event(
            instrument=key[0],
            client_order_id=key[1],
            ts_ns_local=815,
            ts_ns_exch=815,
            state_type="accepted",
        )
    )
    process_canonical_event(state, accepted, position=ProcessingPosition(index=51))
    process_canonical_event(state, late_submitted, position=ProcessingPosition(index=52))

    projection = state.canonical_orders[key]
    assert projection.state == "accepted"
    assert projection.submitted_ts_ns_local == 800
    assert projection.updated_ts_ns_local == 815
    assert state._last_processing_position_index == 52


def test_order_submitted_event_does_not_mutate_snapshot_orders() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _order_submitted_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-snapshot-isolation-1",
        ts_ns_local_dispatch=900,
    )

    process_canonical_event(state, event, position=ProcessingPosition(index=60))

    assert state.orders == {}
    assert state.canonical_orders[("BTC-USDC-PERP", "order-snapshot-isolation-1")].state == "submitted"


def test_process_canonical_event_rejects_order_state_event() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _order_state_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-compat-1",
        ts_ns_local=300,
        ts_ns_exch=290,
    )

    with pytest.raises(TypeError, match="Unsupported non-canonical event type"):
        process_canonical_event(state, event)


def test_process_canonical_event_rejects_order_state_event_with_processing_position() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = _order_state_event(
        instrument="BTC-USDC-PERP",
        client_order_id="order-compat-1",
        ts_ns_local=300,
        ts_ns_exch=290,
    )
    position = ProcessingPosition(index=20)

    with pytest.raises(TypeError, match="Unsupported non-canonical event type"):
        process_canonical_event(state, event, position=position)


def test_process_event_entry_rejects_derived_fill_event() -> None:
    state = StrategyState(event_bus=NullEventBus())
    event = DerivedFillEvent(
        ts_ns_local=300,
        instrument="BTC-USDC-PERP",
        client_order_id="order-compat-derived-1",
        side="buy",
        delta_qty=0.25,
        cum_qty=0.25,
        price=100.0,
    )
    entry = EventStreamEntry(
        position=ProcessingPosition(index=21),
        event=event,
    )

    with pytest.raises(TypeError, match="Unsupported non-canonical event type"):
        process_event_entry(state, entry)


def test_process_canonical_event_rejects_telemetry_record() -> None:
    state = StrategyState(event_bus=NullEventBus())
    telemetry = RiskDecisionEvent(
        ts_ns_local=400,
        accepted=1,
        queued=0,
        rejected=0,
        handled=0,
        reject_reasons={},
    )

    with pytest.raises(TypeError, match="Unsupported non-canonical event type"):
        process_canonical_event(state, telemetry)


def test_event_bus_remains_non_canonical() -> None:
    assert is_canonical_stream_candidate_type(EventBus) is False


def test_processing_position_zero_index_is_valid() -> None:
    position = ProcessingPosition(index=0)
    assert position.index == 0


def test_processing_position_negative_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        ProcessingPosition(index=-1)

