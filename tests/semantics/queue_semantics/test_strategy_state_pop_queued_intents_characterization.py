"""
Characterization tests: StrategyState outbox queue pop semantics.

These tests pin the *current* behavior of:
- StrategyState.pop_queued_intents ordering (priority then FIFO)
- inflight filtering (skips blocked ids without dequeuing them)

This suite is intentionally explicit and should not be interpreted as desired semantics.
"""

from __future__ import annotations

from trading_framework.core.domain.state import StrategyState
from trading_framework.core.domain.types import (
    CancelOrderIntent,
    NewOrderIntent,
    Price,
    Quantity,
    ReplaceOrderIntent,
)
from trading_framework.core.events.sinks.null_event_bus import NullEventBus


def test_pop_queued_intents_orders_by_priority_then_fifo_characterization() -> None:
    instrument = "BTC-USDC-PERP"

    state = StrategyState(event_bus=NullEventBus())

    new_1 = NewOrderIntent(
        ts_ns_local=30,
        instrument=instrument,
        client_order_id="new-1",
        intents_correlation_id=None,
        side="buy",
        order_type="limit",
        intended_qty=Quantity(unit="contracts", value=1.0),
        intended_price=Price(currency="USDC", value=100.0),
        time_in_force="GTC",
    )
    new_2 = NewOrderIntent(
        ts_ns_local=10,
        instrument=instrument,
        client_order_id="new-2",
        intents_correlation_id=None,
        side="buy",
        order_type="limit",
        intended_qty=Quantity(unit="contracts", value=1.0),
        intended_price=Price(currency="USDC", value=101.0),
        time_in_force="GTC",
    )
    replace_1 = ReplaceOrderIntent(
        ts_ns_local=20,
        instrument=instrument,
        client_order_id="replace-1",
        intents_correlation_id=None,
        side="buy",
        intended_price=Price(currency="USDC", value=102.0),
        intended_qty=Quantity(unit="contracts", value=1.0),
    )
    cancel_1 = CancelOrderIntent(
        ts_ns_local=40,
        instrument=instrument,
        client_order_id="cancel-1",
        intents_correlation_id=None,
    )
    cancel_2 = CancelOrderIntent(
        ts_ns_local=5,
        instrument=instrument,
        client_order_id="cancel-2",
        intents_correlation_id=None,
    )

    state.merge_intents_into_queue(
        instrument=instrument,
        intents=[new_1, replace_1, cancel_1, new_2, cancel_2],
    )

    popped = state.pop_queued_intents(instrument)
    popped_ids = [it.client_order_id for it in popped]

    # Characterization: selection is computed by priority + queued_at_ts_ns,
    # but the returned list preserves the queue's iteration order for the selected set.
    # Since all intents are eligible here, this matches enqueue order.
    assert popped_ids == ["new-1", "replace-1", "cancel-1", "new-2", "cancel-2"]


def test_pop_queued_intents_filters_inflight_without_dequeuing_characterization() -> None:
    instrument = "BTC-USDC-PERP"

    state = StrategyState(event_bus=NullEventBus())

    blocked_new = NewOrderIntent(
        ts_ns_local=1,
        instrument=instrument,
        client_order_id="blocked",
        intents_correlation_id=None,
        side="buy",
        order_type="limit",
        intended_qty=Quantity(unit="contracts", value=1.0),
        intended_price=Price(currency="USDC", value=100.0),
        time_in_force="GTC",
    )
    allowed_cancel = CancelOrderIntent(
        ts_ns_local=2,
        instrument=instrument,
        client_order_id="allowed",
        intents_correlation_id=None,
    )

    state.merge_intents_into_queue(instrument=instrument, intents=[blocked_new, allowed_cancel])

    state.mark_intent_sent(
        instrument=instrument,
        client_order_id="blocked",
        intent_type="new",
    )

    popped_1 = state.pop_queued_intents(instrument)
    assert [it.client_order_id for it in popped_1] == ["allowed"]

    # Characterization: the inflight-blocked intent remains queued (not removed).
    assert state.has_queued_intent(instrument, "blocked")
    assert not state.has_queued_intent(instrument, "allowed")

    # After inflight clears, it becomes eligible.
    state._clear_inflight(instrument=instrument, client_order_id="blocked")
    popped_2 = state.pop_queued_intents(instrument)
    assert [it.client_order_id for it in popped_2] == ["blocked"]


def test_pop_queued_intents_respects_max_items_characterization() -> None:
    instrument = "BTC-USDC-PERP"
    state = StrategyState(event_bus=NullEventBus())

    cancel_a = CancelOrderIntent(
        ts_ns_local=1,
        instrument=instrument,
        client_order_id="a",
        intents_correlation_id=None,
    )
    cancel_b = CancelOrderIntent(
        ts_ns_local=2,
        instrument=instrument,
        client_order_id="b",
        intents_correlation_id=None,
    )
    cancel_c = CancelOrderIntent(
        ts_ns_local=3,
        instrument=instrument,
        client_order_id="c",
        intents_correlation_id=None,
    )

    state.merge_intents_into_queue(instrument=instrument, intents=[cancel_c, cancel_a, cancel_b])

    popped = state.pop_queued_intents(instrument, max_items=2)
    assert [it.client_order_id for it in popped] == ["a", "b"]
    assert state.has_queued_intent(instrument, "c")
