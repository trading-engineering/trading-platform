"""
Semantic test: Duplicate NEW intent must be rejected.

Invariant:
(client_order_id, instrument) must be unique while an order id is busy
(working ∪ queued). A NEW with the same id must be rejected.
"""

from __future__ import annotations

from trading_platform.core.domain.reject_reasons import RejectReason
from trading_platform.core.domain.state import StrategyState
from trading_platform.core.domain.types import (
    NewOrderIntent,
    NotionalLimits,
    OrderStateEvent,
    Price,
    Quantity,
)
from trading_platform.core.events.sinks.null_event_bus import NullEventBus
from trading_platform.core.risk.risk_config import RiskConfig
from trading_platform.core.risk.risk_engine import RiskEngine


def test_duplicate_new_is_rejected_when_working_order_exists() -> None:
    """Reject NEW intent when a working order with same (instrument, client_order_id) exists."""

    instrument = "BTC-USDC-PERP"
    client_order_id = "order-1"

    state = StrategyState(event_bus=NullEventBus())

    # Create an existing WORKING order in state via canonical snapshot ingestion.
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

    # Minimal risk config (notional_limits is required by validation).
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

    duplicate_new = NewOrderIntent(
        ts_ns_local=2,
        instrument=instrument,
        client_order_id=client_order_id,
        intents_correlation_id=None,
        side="buy",
        order_type="limit",
        intended_qty=Quantity(unit="contracts", value=1.0),
        intended_price=Price(currency="USDC", value=101.0),
        time_in_force="GTC",
    )

    decision = risk_engine.decide_intents(
        raw_intents=[duplicate_new],
        state=state,
        now_ts_ns_local=2,
    )

    # Must be rejected with DUPLICATE_ID.
    assert decision.accepted_now == []
    assert decision.queued == []
    assert decision.handled_in_queue == []
    assert len(decision.rejected) == 1
    assert decision.rejected[0].reason == RejectReason.DUPLICATE_ID

    # State must remain: working still exists; no queued intent for same id.
    assert state.has_working_order(instrument, client_order_id)
    assert not state.has_queued_intent(instrument, client_order_id)
