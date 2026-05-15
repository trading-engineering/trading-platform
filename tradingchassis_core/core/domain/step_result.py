"""Core step result contract model."""

from __future__ import annotations

from dataclasses import dataclass

from tradingchassis_core.core.domain.candidate_intent import CandidateIntentRecord
from tradingchassis_core.core.domain.step_decision import CoreStepDecision
from tradingchassis_core.core.domain.types import OrderIntent
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation


@dataclass(frozen=True, slots=True)
class CoreStepResult:
    """Immutable result object for deterministic Core step APIs.

    ``control_scheduling_obligation`` is set only when Execution Control apply
    defers for **rate limits** (time-dependent). It is ``None`` for inflight-only
    deferral and other cases without a Core-derived wake time. Only injected
    ``ControlTimeEvent`` values are canonical stream input for control time.
    """

    generated_intents: tuple[OrderIntent, ...] = ()
    candidate_intent_records: tuple[CandidateIntentRecord, ...] = ()
    candidate_intents: tuple[OrderIntent, ...] = ()
    dispatchable_intents: tuple[OrderIntent, ...] = ()
    control_scheduling_obligation: ControlSchedulingObligation | None = None
    core_step_decision: CoreStepDecision | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.generated_intents, tuple):
            object.__setattr__(
                self,
                "generated_intents",
                tuple(self.generated_intents),
            )
        if not isinstance(self.candidate_intent_records, tuple):
            object.__setattr__(
                self,
                "candidate_intent_records",
                tuple(self.candidate_intent_records),
            )
        if not isinstance(self.candidate_intents, tuple):
            object.__setattr__(
                self,
                "candidate_intents",
                tuple(self.candidate_intents),
            )
        if self.candidate_intent_records:
            object.__setattr__(
                self,
                "candidate_intents",
                tuple(record.intent for record in self.candidate_intent_records),
            )
        if not isinstance(self.dispatchable_intents, tuple):
            object.__setattr__(
                self,
                "dispatchable_intents",
                tuple(self.dispatchable_intents),
            )
