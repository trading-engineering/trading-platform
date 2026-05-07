"""Higher-level Core step API skeleton.

This module defines a transitional deterministic step entrypoint above the
canonical reducer boundary. In this phase, it delegates to process_event_entry
and returns an empty CoreStepResult contract value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

from tradingchassis_core.core.domain.configuration import CoreConfiguration
from tradingchassis_core.core.domain.intent_combination import combine_candidate_intents
from tradingchassis_core.core.domain.processing import process_event_entry
from tradingchassis_core.core.domain.processing_order import EventStreamEntry, ProcessingPosition
from tradingchassis_core.core.domain.state import StrategyState
from tradingchassis_core.core.domain.step_decision import CoreStepDecision
from tradingchassis_core.core.domain.step_result import CoreStepResult
from tradingchassis_core.core.domain.types import ControlTimeEvent, OrderIntent
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation

if TYPE_CHECKING:
    from tradingchassis_core.core.risk.risk_engine import GateDecision, RiskEngine


@dataclass(frozen=True, slots=True)
class CoreStepStrategyContext:
    """Deterministic strategy-evaluation context for one Core step.

    ``state`` is currently passed by reference for compatibility. Strategy
    evaluators must treat it as read-only by contract in this scaffold slice.
    """

    state: StrategyState
    event: object
    position: ProcessingPosition
    configuration: CoreConfiguration | None = None


class CoreStepStrategyEvaluator(Protocol):
    """Core-owned strategy evaluation protocol for unified step semantics."""

    def evaluate(self, context: CoreStepStrategyContext) -> Sequence[OrderIntent]:
        """Evaluate strategy once for the provided step context."""


@dataclass(frozen=True, slots=True)
class ControlTimeQueueReevaluationContext:
    """Deterministic context for control-time queue re-evaluation in Core."""

    risk_engine: RiskEngine
    instrument: str
    now_ts_ns_local: int


def _select_effective_control_scheduling_obligation(
    decision: GateDecision,
) -> ControlSchedulingObligation | None:
    obligations = decision.control_scheduling_obligations
    if not obligations:
        return None
    return min(
        obligations,
        key=lambda obligation: (
            obligation.due_ts_ns_local,
            obligation.obligation_key,
        ),
    )


def _resolve_candidate_instrument(
    *,
    entry: EventStreamEntry,
    control_time_queue_context: ControlTimeQueueReevaluationContext | None,
) -> str | None:
    event_instrument = getattr(entry.event, "instrument", None)
    if isinstance(event_instrument, str):
        return event_instrument
    if control_time_queue_context is not None:
        return control_time_queue_context.instrument
    return None


def _map_compat_gate_decision_to_core_step_decision(
    *,
    decision: GateDecision,
    control_scheduling_obligation: ControlSchedulingObligation | None,
) -> CoreStepDecision:
    return CoreStepDecision(
        policy_rejected_intents=tuple(rejected.intent for rejected in decision.rejected),
        queued_effective_intents=tuple(decision.queued),
        dispatchable_intents=tuple(decision.accepted_now),
        execution_handled_intents=tuple(decision.handled_in_queue),
        control_scheduling_obligation=control_scheduling_obligation,
    )


def run_core_step(
    state: StrategyState,
    entry: EventStreamEntry,
    *,
    configuration: CoreConfiguration | None = None,
    control_time_queue_context: ControlTimeQueueReevaluationContext | None = None,
    strategy_evaluator: CoreStepStrategyEvaluator | None = None,
) -> CoreStepResult:
    """Run one transitional Core step.

    Behavior in this phase:
    - delegates event processing to the canonical boundary via process_event_entry;
    - propagates reducer/boundary exceptions unchanged;
    - returns an empty CoreStepResult for future deterministic effects.
    """
    process_event_entry(state, entry, configuration=configuration)

    generated_intents: tuple[OrderIntent, ...] = ()
    if strategy_evaluator is not None:
        strategy_context = CoreStepStrategyContext(
            state=state,
            event=entry.event,
            position=entry.position,
            configuration=configuration,
        )
        generated_intents = tuple(strategy_evaluator.evaluate(strategy_context))

    snapshot_instrument = _resolve_candidate_instrument(
        entry=entry,
        control_time_queue_context=control_time_queue_context,
    )
    queued_snapshot = state.queued_intents_snapshot(snapshot_instrument)
    candidate_intents = combine_candidate_intents(
        generated_intents=generated_intents,
        queued_intents=queued_snapshot,
    )

    if not isinstance(entry.event, ControlTimeEvent):
        return CoreStepResult(
            generated_intents=generated_intents,
            candidate_intents=candidate_intents,
        )

    if control_time_queue_context is None:
        return CoreStepResult(
            generated_intents=generated_intents,
            candidate_intents=candidate_intents,
        )

    popped_intents = state.pop_queued_intents(control_time_queue_context.instrument)
    if not popped_intents:
        return CoreStepResult(
            generated_intents=generated_intents,
            candidate_intents=candidate_intents,
        )

    decision = control_time_queue_context.risk_engine.decide_intents(
        raw_intents=popped_intents,
        state=state,
        now_ts_ns_local=control_time_queue_context.now_ts_ns_local,
    )
    selected_obligation = _select_effective_control_scheduling_obligation(decision)
    core_step_decision = _map_compat_gate_decision_to_core_step_decision(
        decision=decision,
        control_scheduling_obligation=selected_obligation,
    )
    return CoreStepResult(
        generated_intents=generated_intents,
        candidate_intents=candidate_intents,
        dispatchable_intents=tuple(decision.accepted_now),
        control_scheduling_obligation=selected_obligation,
        core_step_decision=core_step_decision,
        compat_gate_decision=decision,
    )
