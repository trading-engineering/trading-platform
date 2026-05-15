"""Processing-order semantic value objects.

This module introduces minimal Processing Order metadata for the canonical
processing boundary without implementing Event Stream or replay mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProcessingPosition:
    """Boundary metadata representing a position in Processing Order."""

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("ProcessingPosition.index must be non-negative")


@dataclass(frozen=True, slots=True)
class EventStreamEntry:
    """Minimal envelope for canonical Event processing-order input.

    This value object intentionally carries only:
    - the causal processing-order position; and
    - the event payload consumed by canonical processing boundaries.
    """

    position: ProcessingPosition
    event: object

    def __post_init__(self) -> None:
        if not isinstance(self.position, ProcessingPosition):
            raise TypeError("EventStreamEntry.position must be a ProcessingPosition")
        if self.event is None:
            raise ValueError("EventStreamEntry.event must be provided")
