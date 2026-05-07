"""Core-owned policy-risk decision scaffold and compatibility projection helpers."""

from __future__ import annotations

from dataclasses import dataclass

from tradingchassis_core.core.domain.types import OrderIntent
from tradingchassis_core.core.risk.risk_engine import GateDecision


@dataclass(frozen=True, slots=True)
class PolicyRiskDecision:
    """Immutable non-canonical policy admissibility projection."""

    accepted_intents: tuple[OrderIntent, ...] = ()
    rejected_intents: tuple[OrderIntent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.accepted_intents, tuple):
            object.__setattr__(
                self,
                "accepted_intents",
                tuple(self.accepted_intents),
            )
        if not isinstance(self.rejected_intents, tuple):
            object.__setattr__(
                self,
                "rejected_intents",
                tuple(self.rejected_intents),
            )


def map_compat_gate_decision_to_policy_risk_decision(
    decision: GateDecision,
) -> PolicyRiskDecision:
    """Project compatibility GateDecision into policy-only scaffold fields.

    Notes:
    - ``accepted_intents`` currently maps from ``accepted_now`` because the
      compatibility gate does not expose a strict pre-execution-control
      policy-accepted set.
    - ``rejected_intents`` maps from the explicit rejected intent records.
    """

    return PolicyRiskDecision(
        accepted_intents=tuple(decision.accepted_now),
        rejected_intents=tuple(rejected.intent for rejected in decision.rejected),
    )
