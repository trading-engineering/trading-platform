"""
Semantic test: REPLACE intent with no effective change must be handled as no-op.

Invariant:
A REPLACE that does not change price or quantity after normalization
must not be sent, queued, or rejected. It must be handled as no-op.
"""

from __future__ import annotations

from trading_platform.core.domain.state import StrategyState
from trading_platform.core.domain.types import (
    NotionalLimits,
    OrderStateEvent,
    Price,
    Quantity,
    ReplaceOrderIntent,
)
from trading_platform.core.events.sinks.null_event_bus import NullEventBus
from trading_platform.core.risk.risk_config import RiskConfig
from trading_platform.core.risk.risk_engine import RiskEngine


def test_replace_without_effective_change_is_handled_noop() -> None:
    """REPLACE with identical price/qty must be handled and dropped."""

    instrument = "BTC-USDC-PERP"
    client_order_id = "order-1"

    state = StrategyState(event_bus=NullEventBus())

    # Existing working order
    existing_order = OrderStateEvent(
        ts_ns_exch=1,
        ts_ns_local=1,
        instrument=instrument,
        client_order_id=client_order_id,
        order_type="limit",
        state_type="working",
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
    state.apply_order_state_event(existing_order)

    risk_cfg = RiskConfig(
        scope="test",
        trading_enabled=True,
        notional_limits=NotionalLimits(
            currency="USDC",
            max_gross_notional=1e18,
            max_single_order_notional=1e18,
        ),
        extra={},
    )
    risk_engine = RiskEngine(risk_cfg=risk_cfg, event_bus=NullEventBus())

    replace_noop = ReplaceOrderIntent(
        ts_ns_local=2,
        instrument=instrument,
        client_order_id=client_order_id,
        intents_correlation_id=None,
        side="buy",
        intended_price=Price(currency="USDC", value=100.0),
        intended_qty=Quantity(unit="contracts", value=1.0),
    )

    decision = risk_engine.decide_intents(
        raw_intents=[replace_noop],
        state=state,
        now_ts_ns_local=2,
    )

    # ---------- assert decision ----------
    assert decision.accepted_now == []
    assert decision.queued == []
    assert decision.rejected == []
    assert len(decision.handled_in_queue) == 1

    # ---------- assert no side effects ----------
    assert state.has_working_order(instrument, client_order_id)
    assert not state.has_queued_intent(instrument, client_order_id)
