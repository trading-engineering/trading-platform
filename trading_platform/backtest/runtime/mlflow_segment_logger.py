from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from trading_platform.backtest.runtime.context import SegmentContext

LOGGER = logging.getLogger(__name__)


class MlflowSegmentLogger:
    """Logs segment-level health & progress information to MLflow.

    Tracking is configured via environment variables (recommended for Kubernetes):
    - MLFLOW_TRACKING_URI: HTTP(S) address of the MLflow tracking server.
      Example: http://mlflow.ml.svc.cluster.local:5000

    This logger is best-effort. Callers should catch exceptions and continue.
    """

    def __init__(self) -> None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def log(
        self,
        *,
        ctx: SegmentContext,
        duration_seconds: float,
        status: str,
    ) -> None:
        """Log segment metadata as MLflow parameters/metrics/tags."""

        # mlflow.set_experiment creates the experiment if it does not exist and
        # avoids an explicit get/create race.
        mlflow.set_experiment(ctx.experiment_id)

        with mlflow.start_run(run_name=ctx.segment_id):
            # Parameters (stable, comparable)
            mlflow.log_param("expected_sweeps", ctx.expected_sweeps)
            mlflow.log_param("completed_sweeps", ctx.completed_sweeps)
            mlflow.log_param("failed_sweeps", ctx.failed_sweeps)

            # Metrics
            mlflow.log_metric("duration_seconds", duration_seconds)

            # Tags (UI / filtering)
            mlflow.set_tag("status", status)
            mlflow.set_tag("experiment_id", ctx.experiment_id)
            mlflow.set_tag("segment_id", ctx.segment_id)

        LOGGER.info(
            "MLflow segment log submitted",
            extra={
                "experiment_id": ctx.experiment_id,
                "segment_id": ctx.segment_id,
                "status": status,
            },
        )