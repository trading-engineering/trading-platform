"""HFT backtest engine implementation based on hftbacktest."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hftbacktest import (
    BacktestAsset,
    Recorder,
    ROIVectorMarketDepthBacktest,
)

if TYPE_CHECKING:
    from trading_platform.core.risk.risk_config import RiskConfig

from trading_platform.backtest.adapters.execution import HftBacktestExecutionAdapter
from trading_platform.backtest.adapters.venue import HftBacktestVenueAdapter
from trading_platform.backtest.engine.engine_base import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)
from trading_platform.backtest.engine.strategy_runner import HftStrategyRunner
from trading_platform.strategies.base import Strategy
from trading_platform.strategies.strategy_config import StrategyConfig


# pylint: disable=too-many-instance-attributes
@dataclass
class HftEngineConfig:
    """Configuration for the HFT backtest engine."""

    # Data wiring
    initial_snapshot: str | None
    data_files: list[str]

    # Contract / microstructure parameters
    instrument: str
    tick_size: float
    lot_size: float
    contract_size: float

    # Simple fee model: maker / taker in rate on trading value
    maker_fee_rate: float
    taker_fee_rate: float

    # Latency model (constant latency)
    entry_latency_ns: int
    response_latency_ns: int

    # Queue model / venue model toggles
    use_risk_adverse_queue_model: bool
    partial_fill_venue: bool

    # Strategy loop timing
    max_steps: int

    last_trades_capacity: int
    max_price_tick_levels: int

    roi_lb: int
    roi_ub: int

    # Output
    stats_npz_path: str
    event_bus_path: str


@dataclass
class HftBacktestConfig(BacktestConfig):
    """Backtest configuration for the HFT engine."""

    engine_cfg: HftEngineConfig
    strategy_cfg: StrategyConfig
    risk_cfg: RiskConfig


def _build_backtester(engine_cfg: HftEngineConfig) -> ROIVectorMarketDepthBacktest:
    """Create an ROIVectorMarketDepthBacktest from the engine configuration."""
    asset = BacktestAsset()

    # For now we assume file paths. Later this can be replaced with an S3 resolver.
    asset = asset.data(engine_cfg.data_files)

    if engine_cfg.initial_snapshot is not None:
        asset = asset.initial_snapshot(engine_cfg.initial_snapshot)

    asset = (
        asset
        .linear_asset(engine_cfg.contract_size)
        .constant_latency(engine_cfg.entry_latency_ns, engine_cfg.response_latency_ns)
        .tick_size(engine_cfg.tick_size)
        .lot_size(engine_cfg.lot_size)
        .trading_value_fee_model(engine_cfg.maker_fee_rate, engine_cfg.taker_fee_rate)
        .last_trades_capacity(engine_cfg.last_trades_capacity)
        .roi_lb(engine_cfg.roi_lb)
        .roi_ub(engine_cfg.roi_ub)
    )

    if engine_cfg.use_risk_adverse_queue_model:
        asset = asset.risk_adverse_queue_model()

    if engine_cfg.partial_fill_venue:
        asset = asset.partial_fill_exchange()
    else:
        asset = asset.no_partial_fill_exchange()

    return ROIVectorMarketDepthBacktest([asset])


class HftBacktestEngine(BacktestEngine):
    """Backtest engine that uses hftbacktest internally."""

    def __init__(self, config: HftBacktestConfig) -> None:
        # pylint: disable=useless-super-delegation
        super().__init__(config)

    def _load_strategy_class(self, class_path: str) -> type[Strategy]:
        """Dynamically load a Strategy class from a module path."""
        module_path, class_name = class_path.split(":")
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        if not issubclass(cls, Strategy):
            raise TypeError(
                f"Loaded class {class_name} is not a subclass of Strategy."
            )

        return cls

    def _build_strategy(self, strategy_cfg: StrategyConfig) -> Strategy:
        """Instantiate the strategy specified in the configuration."""
        cls = self._load_strategy_class(strategy_cfg.class_path)
        return cls(**strategy_cfg.to_engine_params())

    def run(self) -> BacktestResult:
        """Run the backtest and return the aggregated result."""
        cfg: HftBacktestConfig = self.config
        engine_cfg: HftEngineConfig = cfg.engine_cfg
        strategy_cfg: StrategyConfig = cfg.strategy_cfg
        risk_cfg: RiskConfig = cfg.risk_cfg

        # 1) Build hftbacktest backtester from engine config
        hbt = _build_backtester(engine_cfg)

        # 2) Prepare recorder (single asset, record every step)
        recorder = Recorder(1, engine_cfg.max_steps)

        # 3) Build strategy and runner
        strategy = self._build_strategy(strategy_cfg)
        runner = HftStrategyRunner(
            engine_cfg=engine_cfg,
            strategy=strategy,
            risk_cfg=risk_cfg,
        )

        # 4) Backtest-only venue and execution adapters
        asset_no = 0
        venue = HftBacktestVenueAdapter(hbt=hbt, asset_no=asset_no)
        execution = HftBacktestExecutionAdapter(hbt=hbt, asset_no=asset_no)

        # 5) Run strategy loop (venue-agnostic)
        runner.run(venue, execution, recorder)

        # 6) Close backtester and persist statistics
        _ = hbt.close()
        recorder.to_npz(engine_cfg.stats_npz_path)

        return BacktestResult(
            id=cfg.id,
            stats_file=engine_cfg.stats_npz_path,
            extra_metadata={
                "engine": "hftbacktest",
                "instrument": engine_cfg.instrument,
                "strategy_name": strategy_cfg.class_path,
                "strategy_params": strategy_cfg.params,
                "risk_scope": risk_cfg.scope,
                "risk_params": risk_cfg.params,
            },
        )
