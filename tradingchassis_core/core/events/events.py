"""Non-canonical telemetry records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OrderStateTransitionEvent:
    """Observability payload for unexpected Order-State transitions."""

    ts_ns_local: int
    instrument: str
    client_order_id: str
    prev_state: str | None
    next_state: str


@dataclass(slots=True)
class DerivedPnLEvent:
    """Observability payload for derived realized-PnL changes."""

    ts_ns_local: int
    instrument: str
    delta_pnl: float
    cum_realized_pnl: float


@dataclass(slots=True)
class ExposureDerivedEvent:
    """Observability payload for derived exposure changes."""

    ts_ns_local: int
    instrument: str
    exposure: float
    delta_exposure: float


@dataclass(slots=True)
class RiskDecisionEvent:
    """Observability payload summarizing policy-risk outcomes."""

    ts_ns_local: int
    accepted: int
    rejected: int
    reject_reasons: dict[str, int]
