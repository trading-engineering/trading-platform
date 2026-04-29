"""
Characterization test: rejected/denied intents must not enter the queue.

This pins current behavior that hard rejects do not mutate StrategyState.queued_intents.
"""

from __future__ import annotations

from trading_framework.core.domain.reject_reasons import RejectReason
from trading_framework.core.domain.state import StrategyState
from trading_framework.core.domain.types import (
    NewOrderIntent,
    NotionalLimits,
    Price,
    Quantity,
)
from trading_framework.core.events.sinks.null_event_bus import NullEventBus
from trading_framework.core.risk.risk_config import RiskConfig
from trading_framework.core.risk.risk_engine import RiskEngine


def test_trading_disabled_rejects_new_without_queue_side_effects_characterization() -> None:
    instrument = "BTC-USDC-PERP"
    client_order_id = "order-1"

    state = StrategyState(event_bus=NullEventBus())

    risk_cfg = RiskConfig(
        scope="test",
        trading_enabled=False,
        notional_limits=NotionalLimits(
            currency="USDC",
            max_gross_notional=1e18,
            max_single_order_notional=1e18,
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
    assert decision.queued == []
    assert decision.handled_in_queue == []
    assert len(decision.rejected) == 1
    assert decision.rejected[0].reason == RejectReason.TRADING_DISABLED

    assert not state.has_queued_intent(instrument, client_order_id)
