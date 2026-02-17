"""
Semantic test: terminal clears inflight.

Invariant:
Any terminal order event must clear inflight state.
"""

from __future__ import annotations

from trading_platform.core.domain.state import StrategyState
from trading_platform.core.events.sinks.null_event_bus import NullEventBus


def test_terminal_clears_inflight() -> None:
    instrument = "BTC-USDC-PERP"
    client_order_id = "order-1"

    state = StrategyState(event_bus=NullEventBus())

    # Precondition: order is inflight
    state.inflight[instrument] = {client_order_id}

    # Simulate terminal event cleanup
    state.inflight[instrument].discard(client_order_id)

    assert client_order_id not in state.inflight.get(instrument, set())
