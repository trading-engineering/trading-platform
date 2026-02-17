from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from trading_platform.backtest.io.s3_adapter import OCIObjectStorageS3Shim
from trading_platform.backtest.runtime.context import ExperimentContext
from trading_platform.backtest.runtime.prometheus_metrics import PrometheusMetricsClient

LOGGER = logging.getLogger(__name__)


class ExperimentFinalizer:
    """
    Finalizes an experiment after all segments have completed.

    Responsibilities:
    - write experiment_metadata.json
    - write _DONE marker
    """

    def finalize(self, *, ctx: ExperimentContext) -> None:
        finished_at = datetime.now(timezone.utc)

        status = "success"
        if ctx.failed_segments > 0:
            status = "failed"

        metadata = {
            "schema_version": "1.0",
            "identity": {
                "experiment_id": ctx.experiment_id,
            },
            "lifecycle": {
                "status": status,
                "started_at": ctx.experiment_started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_seconds": (
                    finished_at - ctx.experiment_started_at
                ).total_seconds(),
            },
            "segments": {
                "expected": ctx.expected_segments,
                "completed": ctx.completed_segments,
                "failed": ctx.failed_segments,
            },
        }

        experiment_dir = ctx.scratch_experiment_dir
        experiment_dir.mkdir(parents=True, exist_ok=True)

        (experiment_dir / "experiment_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        (experiment_dir / "_DONE").write_text(
            finished_at.isoformat(),
            encoding="utf-8",
        )

        # --- Prometheus metrics (side-effect only) ---
        metrics = PrometheusMetricsClient()

        if metrics.is_enabled():
            try:
                labels = {
                    "experiment_id": ctx.experiment_id,
                    "status": status,
                }

                metrics.push_gauge(
                    name="backtest_experiment_duration_seconds",
                    value=metadata["lifecycle"]["duration_seconds"],
                    labels=labels,
                )

                metrics.push_gauge(
                    name="backtest_experiment_completed_segments",
                    value=float(ctx.completed_segments),
                    labels=labels,
                )

                metrics.push_gauge(
                    name="backtest_experiment_failed_segments",
                    value=float(ctx.failed_segments),
                    labels=labels,
                )

                metrics.push_all(job="backtest_experiment")

            except Exception:
                LOGGER.exception("Prometheus push failed")


class ExperimentMetadataPersister:
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "backtests",
    ) -> None:
        self._s3 = OCIObjectStorageS3Shim(region="eu-frankfurt-1")
        self._bucket = bucket
        self._prefix = prefix

    def persist(
        self,
        *,
        experiment_id: str,
        experiment_dir: Path,
    ) -> None:
        prefix = f"{self._prefix}/{experiment_id}"

        for name in ("experiment_metadata.json", "_DONE"):
            path = experiment_dir / name
            if not path.exists():
                continue

            with path.open("rb") as fh:
                self._s3.put_object(
                    bucket=self._bucket,
                    key=f"{prefix}/{name}",
                    body=fh,
                )


def _cleanup_scratch(*, experiment_id: str, scratch_root: Path) -> None:
    """
    Remove all scratch data for this workflow + experiment.
    This is safe to call ONLY after successful finalization.
    """

    workflow_uid = os.environ.get("ARGO_WORKFLOW_UID")
    if not workflow_uid:
        raise RuntimeError("ARGO_WORKFLOW_UID is not set")

    sweeps_dir = scratch_root / "sweeps" / workflow_uid
    experiment_dir = scratch_root / experiment_id

    if sweeps_dir.exists():
        shutil.rmtree(sweeps_dir)

    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)


def main() -> None:
    parser = argparse.ArgumentParser("finalize experiment")

    parser.add_argument("--experiment-id", type=str, required=True)

    parser.add_argument("--expected-segments", type=int, required=True)
    parser.add_argument("--completed-segments", type=int, required=True)
    parser.add_argument("--failed-segments", type=int, required=True)

    parser.add_argument(
        "--experiment-started-at",
        type=str,
        required=True,
        help="ISO-8601 timestamp (UTC)",
    )

    parser.add_argument("--scratch-root", type=Path, required=True)

    args = parser.parse_args()

    ctx = ExperimentContext(
        experiment_id=args.experiment_id,
        expected_segments=args.expected_segments,
        completed_segments=args.completed_segments,
        failed_segments=args.failed_segments,
        experiment_started_at=datetime.fromisoformat(args.experiment_started_at),
        scratch_root=args.scratch_root,
    )

    ExperimentFinalizer().finalize(ctx=ctx)

    persister = ExperimentMetadataPersister(bucket="data")
    persister.persist(
        experiment_id=ctx.experiment_id,
        experiment_dir=ctx.scratch_experiment_dir,
    )

    _cleanup_scratch(
        experiment_id=ctx.experiment_id,
        scratch_root=ctx.scratch_root,
    )


if __name__ == "__main__":
    main()
