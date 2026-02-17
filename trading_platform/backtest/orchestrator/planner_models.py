"""
Planning model definitions.

This module contains immutable planning structures used to describe
experiments, segments, and sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_platform.backtest.orchestrator.sweeps import SweepPlan


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """
    Execution plan for a single segment of data.
    """

    segment_id: str
    start_ts_ns: int
    end_ts_ns: int
    estimated_bytes: int
    files: list[str]
    sweeps: list[SweepPlan]


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """
    High-level execution plan for an experiment.
    """

    experiment_id: str
    segments: list[SegmentPlan]
