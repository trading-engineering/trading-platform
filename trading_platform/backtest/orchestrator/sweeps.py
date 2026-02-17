"""
Parameter sweep utilities.

This module provides helpers to expand parameter specifications into
concrete sweep plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class RangeSpec:
    """
    Numeric range specification used for parameter sweeps.
    """

    start: float
    stop: float
    step: float


@dataclass(frozen=True, slots=True)
class SweepPlan:
    """
    Concrete parameter sweep configuration.
    """

    sweep_id: str
    parameters: dict[str, Any]


def expand_ranges(spec: dict[str, Any]) -> dict[str, list[Any]]:
    """
    Expand range and iterable specifications into explicit value lists.
    """

    expanded: dict[str, list[Any]] = {}

    for key, value in spec.items():
        if isinstance(value, RangeSpec):
            values: list[Any] = []
            current = value.start

            # Add small epsilon to avoid floating point termination issues
            while current <= value.stop + 1e-12:
                values.append(round(current, 10))
                current += value.step

            expanded[key] = values
            continue

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            expanded[key] = list(value)
            continue

        expanded[key] = [value]

    return expanded


def expand_parameter_grid(grid: dict[str, list[Any]]) -> list[SweepPlan]:
    """
    Generate all parameter combinations from a parameter grid.
    """

    if not grid:
        return [SweepPlan("sweep_0000", {})]

    keys = sorted(grid.keys())
    values = [grid[key] for key in keys]

    sweeps: list[SweepPlan] = []

    for index, combination in enumerate(product(*values)):
        parameters = dict(zip(keys, combination, strict=True))
        sweep_id = f"sweep_{index:04d}"
        sweeps.append(SweepPlan(sweep_id, parameters))

    return sweeps
