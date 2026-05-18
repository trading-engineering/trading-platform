"""Core-owned non-canonical candidate Intent provenance models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tradingchassis_core.core.domain.types import OrderIntent


class CandidateIntentOrigin(str, Enum):
    """Origin marker for candidate Intents in one Core step."""

    GENERATED = "generated"
    QUEUED = "queued"


@dataclass(frozen=True, slots=True)
class CandidateIntentRecord:
    """Non-canonical Core-step candidate record with explicit provenance."""

    intent: OrderIntent
    origin: CandidateIntentOrigin
    logical_key: str
    merge_index: int
    priority: int
