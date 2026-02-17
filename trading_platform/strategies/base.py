"""Base strategy interface.

This module defines the Strategy protocol used by the backtest and
live execution engines. Concrete strategies implement this interface and are
driven exclusively by venue wakeups and risk engine decisions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_platform.core.domain.state import StrategyState
    from trading_platform.core.domain.types import MarketEvent, OrderIntent, RiskConstraints
    from trading_platform.core.ports.engine_context import EngineContext
    from trading_platform.core.risk.risk_engine import GateDecision


class Strategy(ABC):
    """Strategy protocol implemented by all concrete strategies.

    The strategy is triggered by two event sources:
    - Feed events (rc=2): market data changes such as book/trade updates.
    - Order updates (rc=3): order responses / fills / cancels reflected in state snapshots.

    The strategy must NOT assume that created intents are already live in the market.
    Live state must be derived from StrategyState order snapshots.
    """

    @abstractmethod
    def on_feed(
        self,
        state: StrategyState,
        event: MarketEvent,
        engine_cfg: EngineContext,
        constraints: RiskConstraints,
    ) -> list[OrderIntent]:
        """Handle a feed wakeup (rc=2) and produce zero or more raw OrderIntents."""

    @abstractmethod
    def on_order_update(
        self,
        state: StrategyState,
        engine_cfg: EngineContext,
        constraints: RiskConstraints,
    ) -> list[OrderIntent]:
        """Handle an order update wakeup (rc=3) and produce zero or more raw OrderIntents.

        This hook is used for live-like behavior, e.g. reacting to fills, rejects,
        or cancels without waiting for the next market data tick.
        """

    @abstractmethod
    def on_risk_decision(self, decision: GateDecision) -> None:
        """Receive GateDecision feedback (accepted, queued, rejected with reasons)."""
