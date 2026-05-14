"""RiskEngine policy-only behavior tests."""

from __future__ import annotations

import tradingchassis_core as tc
from tradingchassis_core.core.domain.types import NotionalLimits


def _risk_config() -> tc.RiskConfig:
    return tc.RiskConfig(
        scope="test",
        trading_enabled=True,
        notional_limits=NotionalLimits(
            currency="USDC",
            max_gross_notional=1_000_000.0,
            max_single_order_notional=1_000_000.0,
        ),
        position_limits=None,
        quote_limits=None,
        order_rate_limits=None,
        max_loss=None,
    )


def _intent() -> tc.NewOrderIntent:
    return tc.NewOrderIntent(
        intent_type="new",
        ts_ns_local=10,
        instrument="BTC-USDC-PERP",
        client_order_id="risk-intent-1",
        intents_correlation_id="corr-1",
        side="buy",
        order_type="limit",
        intended_qty=tc.Quantity(value=1.0, unit="contracts"),
        intended_price=tc.Price(currency="USDC", value=100.0),
        time_in_force="GTC",
    )


def test_risk_engine_evaluate_policy_intent_accepts_valid_intent() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    state.update_market(
        instrument="BTC-USDC-PERP",
        best_bid=99.0,
        best_ask=101.0,
        best_bid_qty=1.0,
        best_ask_qty=1.0,
        tick_size=0.1,
        lot_size=0.01,
        contract_size=1.0,
        ts_ns_local=10,
        ts_ns_exch=9,
    )
    engine = tc.RiskEngine(_risk_config())
    accepted, reason = engine.evaluate_policy_intent(
        intent=_intent(),
        state=state,
        now_ts_ns_local=10,
    )
    assert accepted is True
    assert reason is None
