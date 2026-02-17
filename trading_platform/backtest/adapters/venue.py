"""Venue adapter implementation for hftbacktest backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hftbacktest import ROIVectorMarketDepthBacktest

from trading_platform.core.ports.venue_adapter import VenueAdapter


@dataclass(frozen=True)
class HftBacktestVenueAdapter(VenueAdapter):
    """VenueAdapter implementation for hftbacktest.

    This adapter is the only place where the strategy loop is allowed to depend
    on hftbacktest APIs.
    """

    hbt: ROIVectorMarketDepthBacktest
    asset_no: int

    def wait_next(self, *, timeout_ns: int, include_order_resp: bool) -> int:
        """Wait for the next venue event and return its type."""
        # hftbacktest backends are frequently Numba jitclass objects.
        # Those methods often do not support keyword arguments.
        return self.hbt.wait_next_feed(include_order_resp, timeout_ns)

    def current_timestamp_ns(self) -> int:
        """Return the current venue timestamp in nanoseconds."""
        return self.hbt.current_timestamp

    def read_market_snapshot(self) -> Any:
        """Return the current market depth snapshot."""
        return self.hbt.depth(self.asset_no)

    def read_orders_snapshot(self) -> tuple[Any, Any]:
        """Return the current orders and state snapshot."""
        return (
            self.hbt.state_values(self.asset_no),
            self.hbt.orders(self.asset_no),
        )

    def record(self, recorder: Any) -> None:
        """Record the current backtest state using the given recorder."""
        # hftbacktest recorder is a thin wrapper exposing .recorder.record(hbt).
        recorder.recorder.record(self.hbt)
