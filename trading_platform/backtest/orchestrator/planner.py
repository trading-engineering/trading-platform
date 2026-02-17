from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trading_platform.backtest.orchestrator.manifest import DataFileMeta, DatasetManifest

from trading_platform.backtest.orchestrator.planner_models import (
    ExperimentPlan,
    SegmentPlan,
)
from trading_platform.backtest.orchestrator.segmenter import segment_files
from trading_platform.backtest.orchestrator.sweeps import (
    expand_parameter_grid,
    expand_ranges,
)


def plan_experiment(
    *,
    experiment_id: str,
    start_ts_ns: int,
    end_ts_ns: int,
    symbol: str,
    venue: str,
    datatype: str,
    sweep_spec: dict[str, Any],
    manifest: DatasetManifest,
    max_segment_bytes: int,
) -> ExperimentPlan:
    """
    Build a deterministic execution plan for an experiment.

    This function performs *planning only*.
    It does not access S3 directly, does not allocate scratch space,
    and does not execute any backtests.

    Responsibilities:
    - resolve relevant data files via the manifest
    - segment data according to scratch size limits
    - expand parameter sweeps
    - produce a pure ExperimentPlan

    Parameters
    ----------
    experiment_id:
        Stable identifier for the experiment.

    start_ts_ns / end_ts_ns:
        Experiment time range (unix timestamp, nanoseconds).

    symbol:
        Instrument included in the experiment.

    sweep_spec:
        User-facing sweep specification. May contain:
        - explicit values
        - iterables
        - RangeSpec instances

    manifest:
        Dataset manifest used to resolve physical data files.

    max_segment_bytes:
        Maximum total size (bytes) allowed per segment.

    Returns
    -------
    ExperimentPlan
        Fully expanded execution plan.
    """

    if start_ts_ns >= end_ts_ns:
        raise ValueError("start_ts_ns must be < end_ts_ns")

    if max_segment_bytes <= 0:
        raise ValueError("max_segment_bytes must be > 0")

    # ------------------------------------------------------------------
    # 1. Resolve all relevant data files
    # ------------------------------------------------------------------

    files: list[DataFileMeta] = manifest.iter_files(
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        symbol=symbol,
        venue=venue,
        datatype=datatype,
    )

    if not files:
        raise RuntimeError("No data files found for given experiment range")

    # ------------------------------------------------------------------
    # 2. Segment files according to scratch constraints
    # ------------------------------------------------------------------

    file_segments: list[list[DataFileMeta]] = segment_files(
        files=files,
        max_bytes=max_segment_bytes,
    )

    if not file_segments:
        raise RuntimeError("Segmenter produced no segments")

    # ------------------------------------------------------------------
    # 3. Expand parameter sweeps
    # ------------------------------------------------------------------

    normalized_grid = expand_ranges(sweep_spec)
    sweep_plans = expand_parameter_grid(normalized_grid)

    # ------------------------------------------------------------------
    # 4. Build SegmentPlans
    # ------------------------------------------------------------------

    segments: list[SegmentPlan] = []

    for index, segment in enumerate(file_segments):
        segment_id = f"segment_{index:04d}"

        segment_start = min(f.start_ts_ns for f in segment)
        segment_end = max(f.end_ts_ns for f in segment)

        estimated_bytes = sum(f.size_bytes for f in segment)

        if estimated_bytes > max_segment_bytes:
            raise RuntimeError(
                f"Segment {segment_id} exceeds max_segment_bytes "
                f"({estimated_bytes} > {max_segment_bytes})"
            )

        segments.append(
            SegmentPlan(
                segment_id=segment_id,
                start_ts_ns=segment_start,
                end_ts_ns=segment_end,
                estimated_bytes=estimated_bytes,
                files=[f.object_key for f in segment],
                sweeps=sweep_plans,
            )
        )

    # ------------------------------------------------------------------
    # 5. Return final experiment plan
    # ------------------------------------------------------------------

    return ExperimentPlan(
        experiment_id=experiment_id,
        segments=segments,
    )
