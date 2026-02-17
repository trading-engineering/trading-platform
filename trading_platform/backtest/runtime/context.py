from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    experiment_id: str

    expected_segments: int
    completed_segments: int
    failed_segments: int

    experiment_started_at: datetime

    scratch_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "scratch_root", Path(self.scratch_root))

    @property
    def scratch_experiment_dir(self) -> Path:
        return self.scratch_root / self.experiment_id


@dataclass(frozen=True, slots=True)
class SegmentContext:
    experiment_id: str
    segment_id: str

    expected_sweeps: int
    completed_sweeps: int
    failed_sweeps: int

    segment_started_at: datetime

    scratch_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "scratch_root", Path(self.scratch_root))

    @property
    def scratch_segment_dir(self) -> Path:
        return (
            self.scratch_root
            / self.experiment_id
            / self.segment_id
        )


@dataclass(frozen=True, slots=True)
class SweepContext:
    """
    Immutable runtime context for a single backtest sweep.

    One SweepContext == one Pod == one backtest execution.
    """

    # Identity
    experiment_id: str
    segment_id: str
    sweep_id: str

    # Data
    stage: str
    venue: str
    datatype: str
    symbol: str
    file_keys: tuple[str, ...]

    # Parameters
    parameters: Mapping[str, object]

    # Runtime paths
    scratch_root: Path
    results_root: Path

    def __post_init__(self) -> None:
        """
        Normalize runtime paths after JSON deserialization.

        JSON has no Path type, so scratch_root / results_root
        may arrive as strings in worker pods.
        """
        object.__setattr__(self, "scratch_root", Path(self.scratch_root))
        object.__setattr__(self, "results_root", Path(self.results_root))

    @property
    def scratch_segment_dir(self) -> Path:
        return (
            self.scratch_root
            / self.experiment_id
            / self.segment_id
        )

    @property
    def scratch_data_dir(self) -> Path:
        return self.scratch_segment_dir / "data"

    @property
    def scratch_results_dir(self) -> Path:
        return self.scratch_segment_dir / "results" / self.sweep_id
