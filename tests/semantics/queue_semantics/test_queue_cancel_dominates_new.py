"""
Semantic test: CANCEL dominates queued NEW intents.

Invariant:
If a NEW intent is already queued for an order id and a CANCEL intent
for the same order id arrives while the order is still not inflight,
the queued NEW must be removed and replaced by the CANCEL.
"""

from __future__ import annotations

from trading_platform.core.domain.state import StrategyState
from trading_platform.core.domain.types import (
    CancelOrderIntent,
    NewOrderIntent,
    NotionalLimits,
    OrderRateLimits,
    Price,
    Quantity,
)
from trading_platform.core.events.sinks.null_event_bus import NullEventBus
from trading_platform.core.risk.risk_config import RiskConfig
from trading_platform.core.risk.risk_engine import RiskEngine


def test_cancel_dominates_queued_new() -> None:
    """CANCEL must remove a queued NEW for the same order id."""

    instrument = "BTC-USDC-PERP"
    client_order_id = "order-1"

    state = StrategyState(event_bus=NullEventBus())

    # Configure rate-limit to force backpressure (queueing)
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
        ),
    )

    risk_engine = RiskEngine(risk_cfg=risk_cfg, event_bus=NullEventBus())

    # Step 1: NEW intent is queued due to rate-limit backpressure
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

    decision_1 = risk_engine.decide_intents(
        raw_intents=[new_intent],
        state=state,
        now_ts_ns_local=1,
    )

    assert decision_1.accepted_now == []
    assert decision_1.rejected == []
    assert len(decision_1.queued) == 1

    # Step 2: CANCEL arrives for the same order id
    cancel_intent = CancelOrderIntent(
        ts_ns_local=2,
        instrument=instrument,
        client_order_id=client_order_id,
        intents_correlation_id=None,
    )

    decision_2 = risk_engine.decide_intents(
        raw_intents=[cancel_intent],
        state=state,
        now_ts_ns_local=2,
    )

    # --- Queue mutation assertions ---
    assert decision_2.accepted_now == []
    assert decision_2.rejected == []
    assert len(decision_2.handled_in_queue) == 1
    assert decision_2.queued == []

    # Queue must contain only CANCEL
    assert not state.has_queued_intent(instrument, client_order_id)
