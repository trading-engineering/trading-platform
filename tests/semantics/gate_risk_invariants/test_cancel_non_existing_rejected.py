"""
Semantic test: CANCEL intent for a non-existing order must be rejected.

Invariant:
A CANCEL intent requires an existing order with the same
(instrument, client_order_id). Otherwise it must be rejected.
"""

from __future__ import annotations

from trading_platform.core.domain.reject_reasons import RejectReason
from trading_platform.core.domain.state import StrategyState
from trading_platform.core.domain.types import (
    CancelOrderIntent,
    NotionalLimits,
)
from trading_platform.core.events.sinks.null_event_bus import NullEventBus
from trading_platform.core.risk.risk_config import RiskConfig
from trading_platform.core.risk.risk_engine import RiskEngine


def test_cancel_for_non_existing_order_is_rejected() -> None:
    """Reject CANCEL intent when no order with the given id exists."""

    instrument = "BTC-USDC-PERP"
    client_order_id = "missing-order"

    state = StrategyState(event_bus=NullEventBus())

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

    cancel_intent = CancelOrderIntent(
        ts_ns_local=1,
        instrument=instrument,
        client_order_id=client_order_id,
        intents_correlation_id=None,
    )

    decision = risk_engine.decide_intents(
        raw_intents=[cancel_intent],
        state=state,
        now_ts_ns_local=1,
    )

    # ---------- assert decision ----------
    assert decision.accepted_now == []
    assert decision.queued == []
    assert decision.handled_in_queue == []
    assert len(decision.rejected) == 1
    assert decision.rejected[0].reason == RejectReason.ORDER_NOT_FOUND

    # ---------- assert no side effects ----------
    assert not state.has_working_order(instrument, client_order_id)
    assert not state.has_queued_intent(instrument, client_order_id)
