from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from trading_platform.backtest.orchestrator.planner_models import ExperimentPlan


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SegmentSummary:
    segment_id: str
    start_ts_ns: int
    end_ts_ns: int
    estimated_bytes: int
    file_count: int
    sweep_count: int
    scratch_utilization: float  # 0.0 - 1.0


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    experiment_id: str
    segment_count: int
    sweeps_per_segment: int
    total_backtests: int
    max_segment_bytes: int
    segments: List[SegmentSummary]
    warnings: List[str]


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def summarize_experiment(
    *,
    plan: ExperimentPlan,
    max_segment_bytes: int,
) -> ExperimentSummary:
    warnings: list[str] = []
    segments: list[SegmentSummary] = []

    if not plan.segments:
        warnings.append("Experiment contains no segments")

    sweeps_per_segment = (
        len(plan.segments[0].sweeps) if plan.segments else 0
    )

    total_backtests = len(plan.segments) * sweeps_per_segment

    if sweeps_per_segment == 0:
        warnings.append("No sweeps defined (0 backtests will run)")

    if total_backtests > 500:
        warnings.append(
            f"High number of backtests ({total_backtests}); runtime may be long"
        )

    if len(plan.segments) > 50:
        warnings.append(
            f"High number of segments ({len(plan.segments)})"
        )

    for segment in plan.segments:
        utilization = segment.estimated_bytes / max_segment_bytes

        if utilization > 1.0:
            warnings.append(
                f"{segment.segment_id} exceeds scratch size "
                f"({utilization:.0%})"
            )
        elif utilization > 0.9:
            warnings.append(
                f"{segment.segment_id} uses {utilization:.0%} of scratch size"
            )

        if segment.estimated_bytes < max_segment_bytes * 0.1:
            warnings.append(
                f"{segment.segment_id} is very small "
                f"({utilization:.0%} of scratch)"
            )

        segments.append(
            SegmentSummary(
                segment_id=segment.segment_id,
                start_ts_ns=segment.start_ts_ns,
                end_ts_ns=segment.end_ts_ns,
                estimated_bytes=segment.estimated_bytes,
                file_count=len(segment.files),
                sweep_count=len(segment.sweeps),
                scratch_utilization=utilization,
            )
        )

    return ExperimentSummary(
        experiment_id=plan.experiment_id,
        segment_count=len(plan.segments),
        sweeps_per_segment=sweeps_per_segment,
        total_backtests=total_backtests,
        max_segment_bytes=max_segment_bytes,
        segments=segments,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_experiment_summary(summary: ExperimentSummary) -> None:
    max_gb = summary.max_segment_bytes / 1024**3

    print(f"Experiment: {summary.experiment_id}")
    print(f"Segments: {summary.segment_count}")
    print(f"Sweeps per segment: {summary.sweeps_per_segment}")
    print(f"Total backtests: {summary.total_backtests}")
    print(f"Max segment size: {max_gb:.2f} GB")
    print()

    if summary.warnings:
        print("Warnings:")
        for w in summary.warnings:
            print(f"  - {w}")
        print()

    print("Segments:")
    for s in summary.segments:
        used_gb = s.estimated_bytes / 1024**3
        print(
            f"  - {s.segment_id}: "
            f"{s.file_count} files | "
            f"{used_gb:.2f} / {max_gb:.2f} GB | "
            f"{s.sweep_count} sweeps | "
            f"{s.scratch_utilization:.0%} scratch"
        )
