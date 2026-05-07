"""Core-owned policy-risk decision scaffold and policy admission helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Sequence

from tradingchassis_core.core.domain.candidate_intent import (
    CandidateIntentOrigin,
    CandidateIntentRecord,
)
from tradingchassis_core.core.domain.types import OrderIntent
from tradingchassis_core.core.risk.risk_engine import GateDecision

if TYPE_CHECKING:
    from tradingchassis_core.core.domain.state import StrategyState


class PolicyIntentEvaluator(Protocol):
    """Side-effect-safe policy evaluator contract for one candidate intent."""

    def evaluate_policy_intent(
        self,
        *,
        intent: OrderIntent,
        state: StrategyState,
        now_ts_ns_local: int,
    ) -> tuple[bool, str | None]:
        """Return (accepted, reason_if_rejected)."""


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


@dataclass(frozen=True, slots=True)
class PolicyRejectedCandidate:
    """Generated-origin candidate denied by policy with preserved reason."""

    record: CandidateIntentRecord
    reason: str


@dataclass(frozen=True, slots=True)
class PolicyAdmissionResult:
    """Result of side-effect-safe policy admission over candidate records."""

    accepted_generated: tuple[CandidateIntentRecord, ...] = ()
    rejected_generated: tuple[PolicyRejectedCandidate, ...] = ()
    passthrough_queued: tuple[CandidateIntentRecord, ...] = ()
    policy_risk_decision: PolicyRiskDecision = field(default_factory=PolicyRiskDecision)

    def __post_init__(self) -> None:
        if not isinstance(self.accepted_generated, tuple):
            object.__setattr__(
                self,
                "accepted_generated",
                tuple(self.accepted_generated),
            )
        if not isinstance(self.rejected_generated, tuple):
            object.__setattr__(
                self,
                "rejected_generated",
                tuple(self.rejected_generated),
            )
        if not isinstance(self.passthrough_queued, tuple):
            object.__setattr__(
                self,
                "passthrough_queued",
                tuple(self.passthrough_queued),
            )


def apply_policy_to_candidate_records(
    candidate_records: Sequence[CandidateIntentRecord],
    *,
    state: StrategyState,
    now_ts_ns_local: int,
    policy_evaluator: PolicyIntentEvaluator,
) -> PolicyAdmissionResult:
    """Apply policy admission to generated-origin candidates only.

    Side-effect contract:
    - does not mutate candidate records;
    - does not mutate queue/rate/inflight state by itself;
    - does not emit events by itself.
    """

    accepted_generated: list[CandidateIntentRecord] = []
    rejected_generated: list[PolicyRejectedCandidate] = []
    passthrough_queued: list[CandidateIntentRecord] = []

    accepted_intents: list[OrderIntent] = []
    rejected_intents: list[OrderIntent] = []

    for record in candidate_records:
        if record.origin == CandidateIntentOrigin.QUEUED:
            passthrough_queued.append(record)
            continue
        if record.origin != CandidateIntentOrigin.GENERATED:
            raise ValueError(f"Unsupported CandidateIntentOrigin: {record.origin!r}")

        accepted, reason = policy_evaluator.evaluate_policy_intent(
            intent=record.intent,
            state=state,
            now_ts_ns_local=now_ts_ns_local,
        )
        if accepted:
            accepted_generated.append(record)
            accepted_intents.append(record.intent)
            continue

        rejected_generated.append(
            PolicyRejectedCandidate(
                record=record,
                reason=reason or "policy_rejected",
            )
        )
        rejected_intents.append(record.intent)

    return PolicyAdmissionResult(
        accepted_generated=tuple(accepted_generated),
        rejected_generated=tuple(rejected_generated),
        passthrough_queued=tuple(passthrough_queued),
        policy_risk_decision=PolicyRiskDecision(
            accepted_intents=tuple(accepted_intents),
            rejected_intents=tuple(rejected_intents),
        ),
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
