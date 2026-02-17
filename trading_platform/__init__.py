"""Public API for the trading_platform package.

Only symbols imported here are considered part of the stable,
supported external interface.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# ----------------------------------------------------------------------
# Backtest Engine API
# ----------------------------------------------------------------------
from trading_platform.backtest.engine.engine_base import BacktestResult
from trading_platform.backtest.engine.hft_engine import (
    HftBacktestConfig,
    HftBacktestEngine,
    HftEngineConfig,
)
from trading_platform.core.domain.slots import (
    SlotKey,
    stable_slot_order_id,
)

# ----------------------------------------------------------------------
# Domain Types (used by strategies)
# ----------------------------------------------------------------------
from trading_platform.core.domain.state import StrategyState
from trading_platform.core.domain.types import (
    MarketEvent,
    NewOrderIntent,
    OrderIntent,
    Price,
    Quantity,
    ReplaceOrderIntent,
    RiskConstraints,
)
from trading_platform.core.ports.engine_context import EngineContext

# ----------------------------------------------------------------------
# Config API (used by consumers)
# ----------------------------------------------------------------------
from trading_platform.core.risk.risk_config import RiskConfig
from trading_platform.core.risk.risk_engine import GateDecision

# ----------------------------------------------------------------------
# Strategy Interface
# ----------------------------------------------------------------------
from trading_platform.strategies.base import Strategy
from trading_platform.strategies.strategy_config import StrategyConfig

# ----------------------------------------------------------------------
# Public API definition
# ----------------------------------------------------------------------

__all__ = [
    # Engine
    "HftBacktestEngine",
    "HftBacktestConfig",
    "HftEngineConfig",
    "BacktestResult",

    # Config
    "RiskConfig",
    "StrategyConfig",

    # Strategy interface
    "Strategy",

    # Strategy-facing domain API
    "StrategyState",
    "MarketEvent",
    "RiskConstraints",
    "OrderIntent",
    "NewOrderIntent",
    "ReplaceOrderIntent",
    "Price",
    "Quantity",
    "SlotKey",
    "stable_slot_order_id",
    "EngineContext",
    "GateDecision",

    # Version
    "__version__",
]

# ----------------------------------------------------------------------
# Package version
# ----------------------------------------------------------------------

try:
    __version__ = version("trading-platform")
except PackageNotFoundError:
    __version__ = "0.0.0"
