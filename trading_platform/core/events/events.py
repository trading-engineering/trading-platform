"""
Domain event models.

These events represent immutable facts observed during execution.
They are consumed by loggers, recorders, and monitoring pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OrderStateTransitionEvent:
    ts_ns_local: int
    instrument: str
    client_order_id: str
    prev_state: str | None
    next_state: str


@dataclass(slots=True)
class DerivedFillEvent:
    ts_ns_local: int
    instrument: str
    client_order_id: str

    side: str

    delta_qty: float
    cum_qty: float

    price: float | None


@dataclass(slots=True)
class DerivedPnLEvent:
    ts_ns_local: int
    instrument: str

    delta_pnl: float
    cum_realized_pnl: float


@dataclass(slots=True)
class ExposureDerivedEvent:
    ts_ns_local: int
    instrument: str

    exposure: float
    delta_exposure: float


@dataclass(slots=True)
class RiskDecisionEvent:
    ts_ns_local: int

    accepted: int
    queued: int
    rejected: int
    handled: int

    reject_reasons: dict[str, int]
