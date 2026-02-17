"""
Semantic test: NEW intents are queued under active rate-limit backpressure.

Invariant:
Queue is a backpressure buffer. NEW intents must be queued (not accepted_now)
if a temporary send blocker exists (e.g. rate limit exhausted).
"""

from __future__ import annotations

from trading_platform.core.domain.state import StrategyState
from trading_platform.core.domain.types import (
    NewOrderIntent,
    NotionalLimits,
    OrderRateLimits,
    Price,
    Quantity,
)
from trading_platform.core.events.sinks.null_event_bus import NullEventBus
from trading_platform.core.risk.risk_config import RiskConfig
from trading_platform.core.risk.risk_engine import RiskEngine


def test_new_is_queued_when_rate_limit_blocks() -> None:
    """NEW intent must be queued when order rate-limit blocks sending."""

    instrument = "BTC-USDC-PERP"
    client_order_id = "order-1"

    state = StrategyState(event_bus=NullEventBus())

    # Configure strict order rate-limit to force backpressure
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

    decision = risk_engine.decide_intents(
        raw_intents=[new_intent],
        state=state,
        now_ts_ns_local=1,
    )

    assert decision.accepted_now == []
    assert decision.rejected == []
    assert len(decision.queued) == 1
