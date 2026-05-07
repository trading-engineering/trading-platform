"""Public API for the tradingchassis_core package.

Only symbols imported here are considered part of the stable,
supported external interface.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from tradingchassis_core.core.domain.configuration import CoreConfiguration
from tradingchassis_core.core.domain.processing import (
    fold_event_stream_entries,
    process_event_entry,
)
from tradingchassis_core.core.domain.processing_order import (
    EventStreamEntry,
    ProcessingPosition,
)
from tradingchassis_core.core.domain.processing_step import (
    ControlTimeQueueReevaluationContext,
    run_core_step,
)

# ----------------------------------------------------------------------
# Backtest Engine API
# ----------------------------------------------------------------------
#
# Backtest engine/runtime code is runtime-owned and has moved to the
# Core Runtime repository (import from `core_runtime.backtest.*`).
#
# This semantic-core package must remain importable without the runtime layer.
from tradingchassis_core.core.domain.slots import (
    SlotKey,
    stable_slot_order_id,
)

# ----------------------------------------------------------------------
# Domain Types (used by strategies)
# ----------------------------------------------------------------------
from tradingchassis_core.core.domain.state import StrategyState
from tradingchassis_core.core.domain.step_decision import CoreStepDecision
from tradingchassis_core.core.domain.step_result import CoreStepResult
from tradingchassis_core.core.domain.types import (
    MarketEvent,
    NewOrderIntent,
    OrderIntent,
    Price,
    Quantity,
    ReplaceOrderIntent,
    RiskConstraints,
)
from tradingchassis_core.core.ports.engine_context import EngineContext

# ----------------------------------------------------------------------
# Config API (used by consumers)
# ----------------------------------------------------------------------
from tradingchassis_core.core.risk.risk_config import RiskConfig
from tradingchassis_core.core.risk.risk_engine import GateDecision

# ----------------------------------------------------------------------
# Strategy Interface
# ----------------------------------------------------------------------
from tradingchassis_core.strategies.base import Strategy
from tradingchassis_core.strategies.strategy_config import StrategyConfig

# ----------------------------------------------------------------------
# Public API definition
# ----------------------------------------------------------------------

__all__ = [
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
    "CoreConfiguration",
    "ProcessingPosition",
    "EventStreamEntry",
    "process_event_entry",
    "run_core_step",
    "ControlTimeQueueReevaluationContext",
    "CoreStepDecision",
    "fold_event_stream_entries",
    "CoreStepResult",

    # Version
    "__version__",
]

# ----------------------------------------------------------------------
# Package version
# ----------------------------------------------------------------------

try:
    __version__ = version("tradingchassis-core")
except PackageNotFoundError:
    __version__ = "0.0.0"
