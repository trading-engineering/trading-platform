"""Canonical Event taxonomy markers for core."""

from __future__ import annotations

from enum import Enum

from tradingchassis_core.core.domain.types import (
    ControlTimeEvent,
    FillEvent,
    MarketEvent,
    OrderExecutionFeedbackEvent,
    OrderSubmittedEvent,
)
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation


class CanonicalEventCategory(str, Enum):
    """Canonical Event Stream categories."""

    MARKET = "market"
    INTENT_RELATED = "intent_related"
    EXECUTION = "execution"
    CONTROL = "control"


CANONICAL_EVENT_CATEGORY_NAMES: tuple[str, ...] = tuple(
    category.value for category in CanonicalEventCategory
)

CANONICAL_STREAM_CANDIDATE_CATEGORY_BY_TYPE: dict[type[object], CanonicalEventCategory] = {
    MarketEvent: CanonicalEventCategory.MARKET,
    OrderSubmittedEvent: CanonicalEventCategory.INTENT_RELATED,
    FillEvent: CanonicalEventCategory.EXECUTION,
    OrderExecutionFeedbackEvent: CanonicalEventCategory.EXECUTION,
    ControlTimeEvent: CanonicalEventCategory.CONTROL,
}

NON_CANONICAL_CONTROL_HELPER_TYPES: frozenset[type[object]] = frozenset(
    {ControlSchedulingObligation}
)


def canonical_category_for_type(record_type: type[object]) -> CanonicalEventCategory | None:
    """Return canonical category for recognized canonical stream candidates."""
    return CANONICAL_STREAM_CANDIDATE_CATEGORY_BY_TYPE.get(record_type)


def is_canonical_stream_candidate_type(record_type: type[object]) -> bool:
    """Return True when the type is marked as a canonical Event candidate."""
    return record_type in CANONICAL_STREAM_CANDIDATE_CATEGORY_BY_TYPE
