"""Deterministic Core step orchestration over canonical reducer inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

from tradingchassis_core.core.domain.configuration import CoreConfiguration
from tradingchassis_core.core.domain.execution_control_apply import (
    ExecutionControlApplyContext,
    apply_execution_control_plan,
)
from tradingchassis_core.core.domain.execution_control_decision import (
    ExecutionControlDecision,
)
from tradingchassis_core.core.domain.execution_control_plan import (
    ExecutionControlCandidateInput,
    plan_execution_control_candidates,
)
from tradingchassis_core.core.domain.intent_combination import (
    combine_candidate_intent_records,
)
from tradingchassis_core.core.domain.policy_risk_decision import (
    PolicyAdmissionResult,
    PolicyIntentEvaluator,
    apply_policy_to_candidate_records,
)
from tradingchassis_core.core.domain.processing import process_event_entry
from tradingchassis_core.core.domain.processing_order import EventStreamEntry, ProcessingPosition
from tradingchassis_core.core.domain.state import StrategyState
from tradingchassis_core.core.domain.step_decision import CoreStepDecision
from tradingchassis_core.core.domain.step_result import CoreStepResult
from tradingchassis_core.core.domain.types import OrderIntent

if TYPE_CHECKING:
    from tradingchassis_core.core.execution_control.execution_control import ExecutionControl


@dataclass(frozen=True, slots=True)
class CoreStepStrategyContext:
    """Deterministic Strategy-evaluation context for one Core step."""

    state: StrategyState
    event: object
    position: ProcessingPosition
    configuration: CoreConfiguration | None = None


class CoreStepStrategyEvaluator(Protocol):
    """Core-owned Strategy evaluation protocol for unified step semantics."""

    def evaluate(self, context: CoreStepStrategyContext) -> Sequence[OrderIntent]:
        """Evaluate Strategy once for the provided step context."""


@dataclass(frozen=True, slots=True)
class CoreWakeupStrategyContext:
    """Deterministic Strategy-evaluation context for one Core wakeup batch."""

    state: StrategyState
    entries: tuple[EventStreamEntry, ...]
    configuration: CoreConfiguration | None = None
    last_position: ProcessingPosition | None = None


class CoreWakeupStrategyEvaluator(Protocol):
    """Core-owned Strategy evaluation protocol for one wakeup batch."""

    def evaluate(self, context: CoreWakeupStrategyContext) -> Sequence[OrderIntent]:
        """Evaluate Strategy once after all wakeup entries are reduced."""


@dataclass(frozen=True, slots=True)
class CorePolicyAdmissionContext:
    """Optional side-effect-safe policy admission capture context."""

    policy_evaluator: PolicyIntentEvaluator
    now_ts_ns_local: int


@dataclass(frozen=True, slots=True)
class CoreExecutionControlApplyContext:
    """Optional mutable Execution Control apply context for one Core step."""

    execution_control: ExecutionControl
    now_ts_ns_local: int
    max_orders_per_sec: float | None = None
    max_cancels_per_sec: float | None = None
    activate_dispatchable_outputs: bool = False


@dataclass(frozen=True, slots=True)
class CoreWakeupReductionResult:
    """Non-canonical reduction-phase output for one runtime wakeup."""

    entries: tuple[EventStreamEntry, ...] = ()
    generated_intents: tuple[OrderIntent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple):
            object.__setattr__(self, "entries", tuple(self.entries))
        if not isinstance(self.generated_intents, tuple):
            object.__setattr__(self, "generated_intents", tuple(self.generated_intents))


def _resolve_candidate_instrument(*, entry: EventStreamEntry) -> str | None:
    event_instrument = getattr(entry.event, "instrument", None)
    if isinstance(event_instrument, str):
        return event_instrument
    return None


def _to_core_step_decision(
    *,
    policy_result: PolicyAdmissionResult,
    execution_control_decision: ExecutionControlDecision,
) -> CoreStepDecision:
    return CoreStepDecision(
        policy_rejected_intents=tuple(
            rejected.record.intent for rejected in policy_result.rejected_generated
        ),
        policy_risk_decision=policy_result.policy_risk_decision,
        execution_control_decision=execution_control_decision,
        queued_effective_intents=execution_control_decision.queued_effective_intents,
        dispatchable_intents=execution_control_decision.dispatchable_intents,
        execution_handled_intents=execution_control_decision.execution_handled_intents,
        control_scheduling_obligation=execution_control_decision.control_scheduling_obligation,
    )


def run_core_step(
    state: StrategyState,
    entry: EventStreamEntry,
    *,
    configuration: CoreConfiguration | None = None,
    policy_admission_context: CorePolicyAdmissionContext | None = None,
    execution_control_apply_context: CoreExecutionControlApplyContext | None = None,
    strategy_evaluator: CoreStepStrategyEvaluator | None = None,
) -> CoreStepResult:
    """Run one deterministic Core step."""
    if execution_control_apply_context is not None and policy_admission_context is None:
        raise ValueError(
            "execution_control_apply_context requires policy_admission_context"
        )

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

    queued_instrument = _resolve_candidate_instrument(entry=entry)
    queued_snapshot = state.queued_intents_snapshot(queued_instrument)
    candidate_intent_records = combine_candidate_intent_records(
        generated_intents=generated_intents,
        queued_intents=queued_snapshot,
    )
    candidate_intents = tuple(record.intent for record in candidate_intent_records)

    if policy_admission_context is None:
        return CoreStepResult(
            generated_intents=generated_intents,
            candidate_intent_records=candidate_intent_records,
            candidate_intents=candidate_intents,
        )

    policy_result = apply_policy_to_candidate_records(
        candidate_intent_records,
        state=state,
        now_ts_ns_local=policy_admission_context.now_ts_ns_local,
        policy_evaluator=policy_admission_context.policy_evaluator,
    )
    execution_control_plan = plan_execution_control_candidates(
        ExecutionControlCandidateInput(
            accepted_generated=policy_result.accepted_generated,
            passthrough_queued=policy_result.passthrough_queued,
        )
    )

    apply_result = None
    if execution_control_apply_context is not None:
        apply_result = apply_execution_control_plan(
            execution_control_plan,
            ExecutionControlApplyContext(
                state=state,
                execution_control=execution_control_apply_context.execution_control,
                now_ts_ns_local=execution_control_apply_context.now_ts_ns_local,
                max_orders_per_sec=execution_control_apply_context.max_orders_per_sec,
                max_cancels_per_sec=execution_control_apply_context.max_cancels_per_sec,
            ),
        )

    effective_execution_control_decision = (
        execution_control_plan.execution_control_decision
        if apply_result is None
        else apply_result.execution_control_decision
    )
    core_step_decision = _to_core_step_decision(
        policy_result=policy_result,
        execution_control_decision=effective_execution_control_decision,
    )

    dispatchable_intents: tuple[OrderIntent, ...] = ()
    control_scheduling_obligation = None
    if apply_result is not None:
        control_scheduling_obligation = apply_result.control_scheduling_obligation
        if (
            execution_control_apply_context is not None
            and execution_control_apply_context.activate_dispatchable_outputs
        ):
            dispatchable_intents = tuple(
                record.record.intent for record in apply_result.dispatchable_records
            )

    return CoreStepResult(
        generated_intents=generated_intents,
        candidate_intent_records=candidate_intent_records,
        candidate_intents=candidate_intents,
        dispatchable_intents=dispatchable_intents,
        control_scheduling_obligation=control_scheduling_obligation,
        core_step_decision=core_step_decision,
    )


def run_core_wakeup_reduction(
    state: StrategyState,
    entries: Sequence[EventStreamEntry],
    *,
    configuration: CoreConfiguration | None = None,
    wakeup_strategy_evaluator: CoreWakeupStrategyEvaluator | None = None,
) -> CoreWakeupReductionResult:
    """Reduce multiple canonical entries in order for one runtime wakeup."""
    entries_tuple = tuple(entries)
    for entry in entries_tuple:
        process_event_entry(state, entry, configuration=configuration)

    generated_intents: tuple[OrderIntent, ...] = ()
    if wakeup_strategy_evaluator is not None:
        last_position = entries_tuple[-1].position if entries_tuple else None
        wakeup_context = CoreWakeupStrategyContext(
            state=state,
            entries=entries_tuple,
            configuration=configuration,
            last_position=last_position,
        )
        generated_intents = tuple(wakeup_strategy_evaluator.evaluate(wakeup_context))

    return CoreWakeupReductionResult(
        entries=entries_tuple,
        generated_intents=generated_intents,
    )


def run_core_wakeup_decision(
    state: StrategyState,
    reduction: CoreWakeupReductionResult,
    *,
    queued_instrument: str | None = None,
    policy_admission_context: CorePolicyAdmissionContext | None = None,
    execution_control_apply_context: CoreExecutionControlApplyContext | None = None,
) -> CoreStepResult:
    """Run one wakeup-level candidate/policy/Execution Control decision phase."""

    if execution_control_apply_context is not None and policy_admission_context is None:
        raise ValueError(
            "execution_control_apply_context requires policy_admission_context"
        )

    queued_snapshot = state.queued_intents_snapshot(queued_instrument)
    candidate_intent_records = combine_candidate_intent_records(
        generated_intents=reduction.generated_intents,
        queued_intents=queued_snapshot,
    )
    candidate_intents = tuple(record.intent for record in candidate_intent_records)

    if policy_admission_context is None:
        return CoreStepResult(
            generated_intents=reduction.generated_intents,
            candidate_intent_records=candidate_intent_records,
            candidate_intents=candidate_intents,
        )

    policy_result = apply_policy_to_candidate_records(
        candidate_intent_records,
        state=state,
        now_ts_ns_local=policy_admission_context.now_ts_ns_local,
        policy_evaluator=policy_admission_context.policy_evaluator,
    )
    execution_control_plan = plan_execution_control_candidates(
        ExecutionControlCandidateInput(
            accepted_generated=policy_result.accepted_generated,
            passthrough_queued=policy_result.passthrough_queued,
        )
    )
    apply_result = None
    if execution_control_apply_context is not None:
        apply_result = apply_execution_control_plan(
            execution_control_plan,
            ExecutionControlApplyContext(
                state=state,
                execution_control=execution_control_apply_context.execution_control,
                now_ts_ns_local=execution_control_apply_context.now_ts_ns_local,
                max_orders_per_sec=execution_control_apply_context.max_orders_per_sec,
                max_cancels_per_sec=execution_control_apply_context.max_cancels_per_sec,
            ),
        )
    effective_execution_control_decision = (
        execution_control_plan.execution_control_decision
        if apply_result is None
        else apply_result.execution_control_decision
    )
    core_step_decision = _to_core_step_decision(
        policy_result=policy_result,
        execution_control_decision=effective_execution_control_decision,
    )
    dispatchable_intents: tuple[OrderIntent, ...] = ()
    control_scheduling_obligation = None
    if apply_result is not None:
        control_scheduling_obligation = apply_result.control_scheduling_obligation
        if (
            execution_control_apply_context is not None
            and execution_control_apply_context.activate_dispatchable_outputs
        ):
            dispatchable_intents = tuple(
                record.record.intent for record in apply_result.dispatchable_records
            )
    return CoreStepResult(
        generated_intents=reduction.generated_intents,
        candidate_intent_records=candidate_intent_records,
        candidate_intents=candidate_intents,
        dispatchable_intents=dispatchable_intents,
        control_scheduling_obligation=control_scheduling_obligation,
        core_step_decision=core_step_decision,
    )


def run_core_wakeup_step(
    state: StrategyState,
    entries: Sequence[EventStreamEntry],
    *,
    configuration: CoreConfiguration | None = None,
    wakeup_strategy_evaluator: CoreWakeupStrategyEvaluator | None = None,
    queued_instrument: str | None = None,
    policy_admission_context: CorePolicyAdmissionContext | None = None,
    execution_control_apply_context: CoreExecutionControlApplyContext | None = None,
) -> CoreStepResult:
    """Convenience wrapper for reduction + wakeup-level decision/apply."""

    reduction = run_core_wakeup_reduction(
        state,
        entries,
        configuration=configuration,
        wakeup_strategy_evaluator=wakeup_strategy_evaluator,
    )
    return run_core_wakeup_decision(
        state,
        reduction,
        queued_instrument=queued_instrument,
        policy_admission_context=policy_admission_context,
        execution_control_apply_context=execution_control_apply_context,
    )
