"""Core-owned non-canonical decision scaffold for run_core_step results."""

from __future__ import annotations

from dataclasses import dataclass

from tradingchassis_core.core.domain.types import OrderIntent
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation


@dataclass(frozen=True, slots=True)
class CoreStepDecision:
    """Immutable scaffold decision model for integrated Core-step semantics."""

    policy_rejected_intents: tuple[OrderIntent, ...] = ()
    queued_effective_intents: tuple[OrderIntent, ...] = ()
    dispatchable_intents: tuple[OrderIntent, ...] = ()
    execution_handled_intents: tuple[OrderIntent, ...] = ()
    control_scheduling_obligation: ControlSchedulingObligation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy_rejected_intents, tuple):
            object.__setattr__(
                self,
                "policy_rejected_intents",
                tuple(self.policy_rejected_intents),
            )
        if not isinstance(self.queued_effective_intents, tuple):
            object.__setattr__(
                self,
                "queued_effective_intents",
                tuple(self.queued_effective_intents),
            )
        if not isinstance(self.dispatchable_intents, tuple):
            object.__setattr__(
                self,
                "dispatchable_intents",
                tuple(self.dispatchable_intents),
            )
        if not isinstance(self.execution_handled_intents, tuple):
            object.__setattr__(
                self,
                "execution_handled_intents",
                tuple(self.execution_handled_intents),
            )
