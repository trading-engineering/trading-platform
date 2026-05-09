"""Docs-aligned event taxonomy markers for core.

This module is intentionally lightweight. It defines semantic markers used to
disambiguate canonical Event Stream candidates from non-canonical artifacts in
the current core codebase.

It does not implement Event Stream append semantics, Processing Order, replay,
or transport behavior.
"""

from __future__ import annotations

from enum import Enum

from tradingchassis_core.core.domain.types import (
    ControlTimeEvent,
    FillEvent,
    MarketEvent,
    OrderExecutionFeedbackEvent,
    OrderStateEvent,
    OrderSubmittedEvent,
)
from tradingchassis_core.core.events.events import (
    DerivedFillEvent,
    DerivedPnLEvent,
    ExposureDerivedEvent,
    OrderStateTransitionEvent,
    RiskDecisionEvent,
)
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation


class CanonicalEventCategory(str, Enum):
    """Canonical Event Stream categories from docs."""

    MARKET = "market"
    INTENT_RELATED = "intent_related"
    EXECUTION = "execution"
    CONTROL = "control"


CANONICAL_EVENT_CATEGORY_NAMES: tuple[str, ...] = tuple(
    category.value for category in CanonicalEventCategory
)


# Canonical Event Stream candidates recognized in this slice.
# Note: FillEvent is tracked as a canonical execution-event candidate, but
# candidate status does not imply it is newly wired into runtime flow.
CANONICAL_STREAM_CANDIDATE_CATEGORY_BY_TYPE: dict[type[object], CanonicalEventCategory] = {
    MarketEvent: CanonicalEventCategory.MARKET,
    OrderSubmittedEvent: CanonicalEventCategory.INTENT_RELATED,
    FillEvent: CanonicalEventCategory.EXECUTION,
    OrderExecutionFeedbackEvent: CanonicalEventCategory.EXECUTION,
    ControlTimeEvent: CanonicalEventCategory.CONTROL,
}


# Non-canonical telemetry / observability records.
TELEMETRY_EVENT_TYPES: frozenset[type[object]] = frozenset(
    {
        RiskDecisionEvent,
        DerivedPnLEvent,
        ExposureDerivedEvent,
        OrderStateTransitionEvent,
    }
)


# Compatibility projection records (kept for current snapshot-driven flow).
COMPATIBILITY_PROJECTION_TYPES: frozenset[type[object]] = frozenset(
    {
        OrderStateEvent,
        DerivedFillEvent,
    }
)


# Non-canonical runtime-facing control helper. This is intentionally not an Event.
NON_CANONICAL_CONTROL_HELPER_TYPES: frozenset[type[object]] = frozenset(
    {ControlSchedulingObligation}
)


def canonical_category_for_type(record_type: type[object]) -> CanonicalEventCategory | None:
    """Return canonical category for recognized canonical stream candidates."""

    return CANONICAL_STREAM_CANDIDATE_CATEGORY_BY_TYPE.get(record_type)


def is_canonical_stream_candidate_type(record_type: type[object]) -> bool:
    """Return True when the type is marked as a canonical Event candidate."""

    return record_type in CANONICAL_STREAM_CANDIDATE_CATEGORY_BY_TYPE

