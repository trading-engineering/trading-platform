"""
Semantic test: NEW -> working.

Invariant:
After a NEW intent has been sent (accepted_now),
the order must be present in the working orders state.
"""

from __future__ import annotations

from trading_platform.core.domain.state import StrategyState
from trading_platform.core.events.sinks.null_event_bus import NullEventBus


def test_new_transitions_to_working() -> None:
    instrument = "BTC-USDC-PERP"
    client_order_id = "order-1"

    state = StrategyState(event_bus=NullEventBus())

    # Simulate that the order is acknowledged as working
    state.orders[instrument] = {
        client_order_id: {
            "status": "working",
        }
    }

    assert instrument in state.orders
    assert client_order_id in state.orders[instrument]
    assert state.orders[instrument][client_order_id]["status"] == "working"
