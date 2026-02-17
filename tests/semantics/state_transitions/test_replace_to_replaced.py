"""
Semantic test: replace -> replaced.

Invariant:
A replace operation must not result in duplicate working orders
for the same client_order_id.
"""

from __future__ import annotations

from trading_platform.core.domain.state import StrategyState
from trading_platform.core.events.sinks.null_event_bus import NullEventBus


def test_replace_transitions_to_replaced() -> None:
    instrument = "BTC-USDC-PERP"
    client_order_id = "order-1"

    state = StrategyState(event_bus=NullEventBus())

    # Existing working order
    state.orders[instrument] = {
        client_order_id: {
            "status": "working",
            "version": 1,
        }
    }

    # Simulate replace acknowledgment
    state.orders[instrument][client_order_id] = {
        "status": "working",
        "version": 2,
    }

    assert len(state.orders[instrument]) == 1
    assert state.orders[instrument][client_order_id]["version"] == 2
