"""MarketEvent canonical contract tests for book-only baseline behavior."""

from __future__ import annotations

import pytest
import tradingchassis_core as tc
from tradingchassis_core.core.domain.types import BookLevel, BookPayload, TradePayload

INSTRUMENT = "BTC-USDC-PERP"
BOOK_ONLY_ERROR = (
    "only book MarketEvent payloads are reduced; trade-shaped MarketEvent payloads "
    "are not supported"
)


def _book_event(*, ts_ns_local: int) -> tc.MarketEvent:
    return tc.MarketEvent(
        ts_ns_exch=ts_ns_local - 1,
        ts_ns_local=ts_ns_local,
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
    )


def _trade_event(*, ts_ns_local: int) -> tc.MarketEvent:
    return tc.MarketEvent(
        ts_ns_exch=ts_ns_local - 1,
        ts_ns_local=ts_ns_local,
        instrument=INSTRUMENT,
        event_type="trade",
        trade=TradePayload(
            side="buy",
            price=tc.Price(currency="USDC", value=100.0),
            quantity=tc.Quantity(value=0.5, unit="contracts"),
            trade_id="trade-1",
        ),
    )


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


def test_process_canonical_event_accepts_book_market_event() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    tc.process_canonical_event(state, _book_event(ts_ns_local=200))

    market = state.market[INSTRUMENT]
    assert market.best_bid == 99.0
    assert market.best_ask == 101.0
    assert market.mid == 100.0
    assert state.sim_ts_ns_local == 200


def test_market_event_trade_shape_can_be_constructed_but_reduction_rejects() -> None:
    trade_event = _trade_event(ts_ns_local=210)
    assert trade_event.is_trade()
    assert trade_event.trade is not None

    state = tc.StrategyState(event_bus=tc.NullEventBus())
    with pytest.raises(ValueError, match=BOOK_ONLY_ERROR):
        tc.process_canonical_event(state, trade_event)


def test_run_core_step_rejects_trade_market_event() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    trade_entry = tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=0),
        event=_trade_event(ts_ns_local=220),
    )

    with pytest.raises(ValueError, match=BOOK_ONLY_ERROR):
        tc.run_core_step(state, trade_entry)


def test_run_core_wakeup_step_rejects_trade_market_event() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    entries = (
        _control_entry(0, 300),
        tc.EventStreamEntry(
            position=tc.ProcessingPosition(index=1),
            event=_trade_event(ts_ns_local=301),
        ),
    )

    with pytest.raises(ValueError, match=BOOK_ONLY_ERROR):
        tc.run_core_wakeup_step(state, entries)
