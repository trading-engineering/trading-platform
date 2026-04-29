"""
Characterization tests: mixed dominance sequences within queued intents.

These pin the current behavior for sequences like:
NEW queued -> REPLACE on same logical key -> CANCEL on same logical key

This suite is intentionally descriptive of current behavior (not prescriptive).
"""

from __future__ import annotations

from trading_framework.core.domain.state import StrategyState
from trading_framework.core.domain.types import (
    CancelOrderIntent,
    NewOrderIntent,
    NotionalLimits,
    OrderRateLimits,
    Price,
    Quantity,
    ReplaceOrderIntent,
)
from trading_framework.core.events.sinks.null_event_bus import NullEventBus
from trading_framework.core.risk.risk_config import RiskConfig
from trading_framework.core.risk.risk_engine import RiskEngine


def test_new_then_replace_then_cancel_on_same_key_characterization() -> None:
    instrument = "BTC-USDC-PERP"
    client_order_id = "order-1"

    state = StrategyState(event_bus=NullEventBus())

    risk_cfg = RiskConfig(
        scope="test",
        trading_enabled=True,
        notional_limits=NotionalLimits(
            currency="USDC",
            max_gross_notional=1e18,
            max_single_order_notional=1e18,
        ),
        order_rate_limits=OrderRateLimits(
            max_orders_per_second=0,
            max_cancels_per_second=0,
        ),
    )
    risk_engine = RiskEngine(risk_cfg=risk_cfg, event_bus=NullEventBus())

    # Step 1: NEW is queued due to order rate-limit.
    new_intent = NewOrderIntent(
        ts_ns_local=1,
        instrument=instrument,
        client_order_id=client_order_id,
        intents_correlation_id=None,
        side="buy",
        order_type="limit",
        intended_qty=Quantity(unit="contracts", value=1.0),
        intended_price=Price(currency="USDC", value=100.0),
        time_in_force="GTC",
    )
    d1 = risk_engine.decide_intents(raw_intents=[new_intent], state=state, now_ts_ns_local=1)
    assert d1.accepted_now == []
    assert d1.rejected == []
    assert [it.intent_type for it in d1.queued] == ["new"]
    assert state.has_queued_intent(instrument, client_order_id)

    # Step 2: REPLACE arrives while there is no working order, but a queued NEW exists.
    # Characterization: the REPLACE is handled locally and results in an updated queued NEW.
    replace_intent = ReplaceOrderIntent(
        ts_ns_local=2,
        instrument=instrument,
        client_order_id=client_order_id,
        intents_correlation_id=None,
        side="buy",
        intended_price=Price(currency="USDC", value=101.0),
        intended_qty=Quantity(unit="contracts", value=2.0),
    )
    d2 = risk_engine.decide_intents(raw_intents=[replace_intent], state=state, now_ts_ns_local=2)
    assert d2.accepted_now == []
    assert d2.rejected == []
    assert len(d2.handled_in_queue) == 1
    assert d2.handled_in_queue[0].intent_type == "replace"
    assert [it.intent_type for it in d2.queued] == ["new"]

    queued_new = state.find_queued_new_intent(instrument, client_order_id)
    assert queued_new is not None
    assert queued_new.intended_price.value == 101.0
    assert queued_new.intended_qty.value == 2.0

    # Step 3: CANCEL arrives while only queued state exists (no working order).
    # Characterization: the CANCEL clears queued intents for that key and is handled locally.
    cancel_intent = CancelOrderIntent(
        ts_ns_local=3,
        instrument=instrument,
        client_order_id=client_order_id,
        intents_correlation_id=None,
    )
    d3 = risk_engine.decide_intents(raw_intents=[cancel_intent], state=state, now_ts_ns_local=3)
    assert d3.accepted_now == []
    assert d3.rejected == []
    assert len(d3.handled_in_queue) == 1
    assert d3.handled_in_queue[0].intent_type == "cancel"
    assert d3.queued == []
    assert not state.has_queued_intent(instrument, client_order_id)
