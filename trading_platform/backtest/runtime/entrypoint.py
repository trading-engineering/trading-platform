from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trading_platform.backtest.orchestrator.planner_models import ExperimentPlan

from trading_platform.backtest.orchestrator.planner import plan_experiment
from trading_platform.backtest.orchestrator.s3_manifest import S3DatasetManifest
from trading_platform.backtest.orchestrator.summary import (
    print_experiment_summary,
    summarize_experiment,
)
from trading_platform.backtest.orchestrator.sweeps import RangeSpec
from trading_platform.backtest.runtime.context import SweepContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_sweep_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Same semantics as your planner CLI:
    dict -> RangeSpec or explicit lists
    """
    parsed: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            parsed[key] = RangeSpec(
                start=value["start"],
                stop=value["stop"],
                step=value["step"],
            )
        else:
            parsed[key] = value
    return parsed


def _emit_sweep_context(
    *,
    plan: ExperimentPlan,
    base_cfg: dict[str, Any],
    scratch_root: Path,
    results_root: Path,
    out_dir: Path,
) -> None:
    """
    Emit one SweepContext JSON per sweep.
    These JSON files are what Argo consumes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment: dict = base_cfg["experiment"]
    stage: str = experiment.get("stage", "derived")
    venue: str = experiment["venue"]
    datatype: str = experiment["datatype"]
    symbol: str = experiment["symbol"]

    for segment in plan.segments:
        for sweep in segment.sweeps:
            ctx = SweepContext(
                experiment_id=plan.experiment_id,
                segment_id=segment.segment_id,
                sweep_id=sweep.sweep_id,
                stage=stage,
                venue=venue,
                datatype=datatype,
                symbol=symbol,
                file_keys=tuple(segment.files),
                parameters={
                    # pass through full engine/strategy/risk blocks
                    "engine": base_cfg["engine"],
                    "strategy": base_cfg["strategy"],
                    "risk": base_cfg["risk"],
                    # plus sweep-specific parameters
                    "sweep": sweep.parameters,
                },
                scratch_root=scratch_root,
                results_root=results_root,
            )

            out_path = out_dir / f"{segment.segment_id}__{sweep.sweep_id}.json"
            out_path.write_text(
                json.dumps(asdict(ctx), indent=2, default=str),
                encoding="utf-8",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest entrypoint (plan or run via sweep fan-out)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment JSON config (inside image or mounted).",
    )

    parser.add_argument(
        "--plan",
        action="store_true",
        help="Plan experiment and print summary (no execution).",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Plan experiment and emit sweep contexts for execution.",
    )

    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/mnt/scratch"),
        help="Root directory for scratch volume.",
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/results"),
        help="Logical results root (used for context only).",
    )

    parser.add_argument(
        "--emit-dir",
        type=Path,
        default=Path("/mnt/scratch/sweeps"),
        help="Directory where SweepContext JSONs are emitted.",
    )

    args = parser.parse_args()

    if not args.plan and not args.run:
        print("Error: one of --plan or --run must be specified.", file=sys.stderr)
        sys.exit(2)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------

    cfg = _load_json(args.config)

    experiment_id: str = cfg["id"]
    experiment_cfg = cfg["experiment"]

    start_ts_ns: int = experiment_cfg["start_ts_ns"]
    end_ts_ns: int = experiment_cfg["end_ts_ns"]
    symbol: str = experiment_cfg["symbol"]
    venue: str = experiment_cfg["venue"]
    datatype: str = experiment_cfg["datatype"]

    segmentation: dict = experiment_cfg.get("segmentation", {})
    max_segment_gb: float = segmentation.get("max_segment_gb", 100)
    max_segment_bytes = max_segment_gb * 1024**3

    sweep_spec = _parse_sweep_spec(experiment_cfg.get("sweeps", {}))

    manifest = S3DatasetManifest(
        bucket="data",
        stage=experiment_cfg.get("stage", "derived"),
    )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    plan = plan_experiment(
        experiment_id=experiment_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        symbol=symbol,
        venue=venue,
        datatype=datatype,
        sweep_spec=sweep_spec,
        manifest=manifest,
        max_segment_bytes=max_segment_bytes,
    )

    summary = summarize_experiment(
        plan=plan,
        max_segment_bytes=max_segment_bytes,
    )

    # Always show the plan (this is what you want in Argo logs)
    print_experiment_summary(summary)

    if args.plan and not args.run:
        # Plan-only mode: exit after printing
        return

    # ------------------------------------------------------------------
    # Run preparation (emit sweep contexts)
    # ------------------------------------------------------------------

    index: list[str] = []
    segments_index: list[dict[str, object]] = []

    out_dir: Path = args.emit_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "experiment_id.txt").write_text(
        plan.experiment_id,
        encoding="utf-8",
    )

    expected_segments = len(plan.segments)

    (out_dir / "expected_segments.txt").write_text(
        str(expected_segments),
        encoding="utf-8",
    )

    _emit_sweep_context(
        plan=plan,
        base_cfg=cfg,
        scratch_root=args.scratch_root,
        results_root=args.results_root,
        out_dir=out_dir,
    )

    for segment in plan.segments:
        segments_index.append(
            {
                "segment_id": segment.segment_id,
                "expected_sweeps": len(segment.sweeps),
            }
        )

    (out_dir / "segments.json").write_text(
        json.dumps(segments_index, indent=2),
        encoding="utf-8",
    )

    for segment in plan.segments:
        for sweep in segment.sweeps:
            out_path = out_dir / f"{segment.segment_id}__{sweep.sweep_id}.json"
            index.append(str(out_path))

    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Emitted sweep contexts to: {args.emit_dir}")
    print("Each JSON represents exactly one sweep (one Pod).")


if __name__ == "__main__":
    main()
