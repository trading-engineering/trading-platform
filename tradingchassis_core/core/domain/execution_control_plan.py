"""Pure, non-canonical Execution Control candidate planning scaffolds."""

from __future__ import annotations

from dataclasses import dataclass, field

from tradingchassis_core.core.domain.candidate_intent import CandidateIntentRecord
from tradingchassis_core.core.domain.execution_control_decision import (
    ExecutionControlDecision,
)


@dataclass(frozen=True, slots=True)
class ExecutionControlCandidateInput:
    """Policy-admitted candidate records for capture-only Execution Control planning."""

    accepted_generated: tuple[CandidateIntentRecord, ...] = ()
    passthrough_queued: tuple[CandidateIntentRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.accepted_generated, tuple):
            object.__setattr__(
                self,
                "accepted_generated",
                tuple(self.accepted_generated),
            )
        if not isinstance(self.passthrough_queued, tuple):
            object.__setattr__(
                self,
                "passthrough_queued",
                tuple(self.passthrough_queued),
            )


@dataclass(frozen=True, slots=True)
class ExecutionControlPlan:
    """Capture-only Execution Control candidate planning result."""

    active_records: tuple[CandidateIntentRecord, ...] = ()
    queued_effective_records: tuple[CandidateIntentRecord, ...] = ()
    dispatchable_records: tuple[CandidateIntentRecord, ...] = ()
    execution_handled_records: tuple[CandidateIntentRecord, ...] = ()
    execution_control_decision: ExecutionControlDecision = field(
        default_factory=ExecutionControlDecision
    )

    def __post_init__(self) -> None:
        if not isinstance(self.active_records, tuple):
            object.__setattr__(self, "active_records", tuple(self.active_records))
        if not isinstance(self.queued_effective_records, tuple):
            object.__setattr__(
                self,
                "queued_effective_records",
                tuple(self.queued_effective_records),
            )
        if not isinstance(self.dispatchable_records, tuple):
            object.__setattr__(
                self,
                "dispatchable_records",
                tuple(self.dispatchable_records),
            )
        if not isinstance(self.execution_handled_records, tuple):
            object.__setattr__(
                self,
                "execution_handled_records",
                tuple(self.execution_handled_records),
            )


def plan_execution_control_candidates(
    planning_input: ExecutionControlCandidateInput,
) -> ExecutionControlPlan:
    """Build a deterministic, side-effect-free Execution Control plan projection."""

    active_records = (
        tuple(planning_input.accepted_generated)
        + tuple(planning_input.passthrough_queued)
    )
    queued_effective_records = active_records
    dispatchable_records: tuple[CandidateIntentRecord, ...] = ()
    execution_handled_records: tuple[CandidateIntentRecord, ...] = ()

    return ExecutionControlPlan(
        active_records=active_records,
        queued_effective_records=queued_effective_records,
        dispatchable_records=dispatchable_records,
        execution_handled_records=execution_handled_records,
        execution_control_decision=ExecutionControlDecision(
            queued_effective_intents=tuple(
                record.intent for record in queued_effective_records
            ),
            dispatchable_intents=(),
            execution_handled_intents=(),
            control_scheduling_obligation=None,
        ),
    )
