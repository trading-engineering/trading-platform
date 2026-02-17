from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from trading_platform.backtest.io.s3_adapter import OCIObjectStorageS3Shim
from trading_platform.backtest.runtime.context import SegmentContext
from trading_platform.backtest.runtime.mlflow_segment_logger import MlflowSegmentLogger
from trading_platform.backtest.runtime.prometheus_metrics import PrometheusMetricsClient

LOGGER = logging.getLogger(__name__)


class SegmentFinalizer:
    """
    Finalizes a segment after all sweeps have completed.

    Responsibilities:
    - write segment_metadata.json
    - write _DONE marker
    """

    def finalize(
        self,
        *,
        ctx: SegmentContext,
    ) -> None:
        finished_at = datetime.now(timezone.utc)

        status = "success"
        if ctx.failed_sweeps > 0:
            status = "failed"

        metadata = {
            "schema_version": "1.0",
            "identity": {
                "experiment_id": ctx.experiment_id,
                "segment_id": ctx.segment_id,
            },
            "lifecycle": {
                "status": status,
                "started_at": ctx.segment_started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_seconds": (
                    finished_at - ctx.segment_started_at
                ).total_seconds(),
            },
            "sweeps": {
                "expected": ctx.expected_sweeps,
                "completed": ctx.completed_sweeps,
                "failed": ctx.failed_sweeps,
            },
        }

        segment_dir = ctx.scratch_segment_dir
        segment_dir.mkdir(parents=True, exist_ok=True)

        (segment_dir / "segment_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        (segment_dir / "_DONE").write_text(
            finished_at.isoformat(),
            encoding="utf-8",
        )

        # --- MLflow logging (side-effect only) ---
        try:
            MlflowSegmentLogger().log(
                ctx=ctx,
                duration_seconds=metadata["lifecycle"]["duration_seconds"],
                status=status,
            )
        except Exception:
            LOGGER.exception("MLflow logging failed")

        # --- Prometheus metrics (side-effect only) ---
        metrics = PrometheusMetricsClient()

        if metrics.is_enabled():
            try:
                labels = {
                    "experiment_id": ctx.experiment_id,
                    "segment_id": ctx.segment_id,
                    "status": status,
                }

                metrics.push_gauge(
                    name="backtest_segment_duration_seconds",
                    value=metadata["lifecycle"]["duration_seconds"],
                    labels=labels,
                )

                metrics.push_gauge(
                    name="backtest_segment_completed_sweeps",
                    value=float(ctx.completed_sweeps),
                    labels=labels,
                )

                metrics.push_gauge(
                    name="backtest_segment_failed_sweeps",
                    value=float(ctx.failed_sweeps),
                    labels=labels,
                )

                metrics.push_all(job="backtest_segment")

            except Exception:
                LOGGER.exception("Prometheus push failed")


class SegmentMetadataPersister:
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
        segment_id: str,
        segment_dir: Path,
    ) -> None:
        prefix = f"{self._prefix}/{experiment_id}/{segment_id}"

        for name in ("segment_metadata.json", "_DONE"):
            path = segment_dir / name
            if not path.exists():
                continue

            with path.open("rb") as fh:
                self._s3.put_object(
                    bucket=self._bucket,
                    key=f"{prefix}/{name}",
                    body=fh,
                )


def main() -> None:
    parser = argparse.ArgumentParser("finalize segment")

    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--segment-id", type=str, required=True)

    parser.add_argument("--expected-sweeps", type=int, required=True)
    parser.add_argument("--completed-sweeps", type=int, required=True)
    parser.add_argument("--failed-sweeps", type=int, required=True)

    parser.add_argument(
        "--segment-started-at",
        type=str,
        required=True,
        help="ISO-8601 timestamp (UTC)",
    )

    parser.add_argument("--scratch-root", type=Path, required=True)

    args = parser.parse_args()

    ctx = SegmentContext(
        experiment_id=args.experiment_id,
        segment_id=args.segment_id,
        expected_sweeps=args.expected_sweeps,
        completed_sweeps=args.completed_sweeps,
        failed_sweeps=args.failed_sweeps,
        segment_started_at=datetime.fromisoformat(args.segment_started_at),
        scratch_root=args.scratch_root,
    )

    finalizer = SegmentFinalizer()
    finalizer.finalize(
        ctx=ctx,
    )

    persister = SegmentMetadataPersister(bucket="data")
    persister.persist(
        experiment_id=ctx.experiment_id,
        segment_id=ctx.segment_id,
        segment_dir=ctx.scratch_segment_dir,
    )


if __name__ == "__main__":
    main()
