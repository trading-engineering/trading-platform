"""Venue adapter protocol for strategy execution.

This module defines the abstract venue-facing boundary used by the strategy
loop. Concrete implementations adapt specific backtest or live venues to
this protocol.
"""

from __future__ import annotations

from typing import Any, Protocol


class VenueAdapter(Protocol):
    """Venue-facing feed boundary.

    The strategy loop must not depend on venue-specific APIs.

    Snapshot objects are intentionally typed as Any: they are only consumed by
    venue-specific translation code, not by strategy/risk/state layers.
    """

    def wait_next(self, *, timeout_ns: int, include_order_resp: bool) -> int:
        """Block until the next wakeup, returning a venue-defined rc code."""

    def current_timestamp_ns(self) -> int:
        """Return the venue local/receipt timestamp axis in ns."""

    def read_market_snapshot(self) -> Any:
        """Return the current market snapshot object (venue-specific)."""

    def read_orders_snapshot(self) -> tuple[Any, Any]:
        """Return a tuple (state_values, orders) (venue-specific)."""

    def record(self, recorder: Any) -> None:
        """Record the current venue state into the recorder (if supported)."""
