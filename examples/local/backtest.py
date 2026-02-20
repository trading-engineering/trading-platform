"""Command-line interface for running backtests in devcontainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Enable importing plugin-style modules outside the core package (e.g. examples/)
if __name__ == "__main__" or True:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from trading_platform import BacktestResult

from trading_platform import (
    HftBacktestConfig,
    HftBacktestEngine,
    HftEngineConfig,
    RiskConfig,
    StrategyConfig,
)


def load_config(path: str) -> HftBacktestConfig:
    """Load a backtest configuration from a JSON file."""
    config_path = Path(path)
    raw_json = json.loads(config_path.read_text(encoding="utf-8"))

    try:
        engine_raw = raw_json["engine"]
        strategy_raw = raw_json["strategy"]
        risk_raw = raw_json["risk"]
    except KeyError as exc:
        raise ValueError(
            f"Missing top-level section in {config_path}: {exc}"
        ) from exc

    engine_cfg = HftEngineConfig(**engine_raw)
    strategy_cfg = StrategyConfig(**strategy_raw)
    risk_cfg = RiskConfig(**risk_raw)

    return HftBacktestConfig(
        id=raw_json["id"],
        description=raw_json.get("description", ""),
        engine_cfg=engine_cfg,
        strategy_cfg=strategy_cfg,
        risk_cfg=risk_cfg,
    )


def main() -> None:
    """Entry point for the backtest command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run a strategy-based hftbacktest backtest."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file (HftBacktestConfig).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = HftBacktestEngine(cfg)

    print("Backtest started.")
    result: BacktestResult = engine.run()

    print("Backtest finished.")
    print(f"  id:          {result.id}")
    print(f"  stats_npz:   {result.stats_file}")
    if result.extra_metadata is not None:
        print("  metadata:")
        for key, value in result.extra_metadata.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
