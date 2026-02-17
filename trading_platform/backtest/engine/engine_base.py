from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BacktestConfig:
    """Generic backtest configuration.

    Engine configs should subclass this
    and add engine-specific fields.
    """
    id: str
    description: str


@dataclass
class BacktestResult:
    """Lightweight container for backtest outputs.

    For now we only track the stats file path. 
    Can be extended with PnL curves, summary metrics, etc.
    """
    id: str
    stats_file: str | None = None
    extra_metadata: dict[str, Any] | None = None


class BacktestEngine:
    """Abstract base class for all backtest engines."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(self) -> BacktestResult:
        """Run the backtest and return a result object.

        Subclass engines must implement this method.
        """
        raise NotImplementedError("run() must be implemented by subclasses")
