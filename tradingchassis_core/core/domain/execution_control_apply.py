"""Mutable execution-control apply stage over a pure execution-control plan."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from tradingchassis_core.core.domain.candidate_intent import (
    CandidateIntentOrigin,
    CandidateIntentRecord,
)
from tradingchassis_core.core.domain.execution_control_decision import (
    ExecutionControlDecision,
)
from tradingchassis_core.core.domain.execution_control_plan import (
    ExecutionControlPlan,
)
from tradingchassis_core.core.domain.state import StrategyState
from tradingchassis_core.core.execution_control.execution_control import ExecutionControl
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation


@dataclass(frozen=True, slots=True)
class ExecutionControlApplyContext:
    """Mutable apply inputs for one deterministic apply operation."""

    state: StrategyState
    execution_control: ExecutionControl
    now_ts_ns_local: int
    max_orders_per_sec: float | None = None
    max_cancels_per_sec: float | None = None


@dataclass(frozen=True, slots=True)
class ExecutionControlDispatchableRecord:
    """Candidate record selected as dispatchable in this apply pass."""

    record: CandidateIntentRecord


@dataclass(frozen=True, slots=True)
class ExecutionControlBlockedRecord:
    """Candidate record blocked from immediate dispatch."""

    record: CandidateIntentRecord
    reason: str
    scheduling_obligation: ControlSchedulingObligation | None = None


@dataclass(frozen=True, slots=True)
class ExecutionControlHandledRecord:
    """Candidate record fully handled by queue-local semantics."""

    record: CandidateIntentRecord
    reason: str


@dataclass(frozen=True, slots=True)
class ExecutionControlApplyResult:
    """Result of mutable execution-control apply over one plan state."""

    queued_effective_records: tuple[CandidateIntentRecord, ...] = ()
    dispatchable_records: tuple[ExecutionControlDispatchableRecord, ...] = ()
    execution_handled_records: tuple[ExecutionControlHandledRecord, ...] = ()
    blocked_records: tuple[ExecutionControlBlockedRecord, ...] = ()
    control_scheduling_obligation: ControlSchedulingObligation | None = None
    execution_control_decision: ExecutionControlDecision = field(
        default_factory=ExecutionControlDecision
    )

    def __post_init__(self) -> None:
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
        if not isinstance(self.blocked_records, tuple):
            object.__setattr__(
                self,
                "blocked_records",
                tuple(self.blocked_records),
            )


def _float_equal(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-12


def _select_effective_control_scheduling_obligation(
    obligations: list[ControlSchedulingObligation],
) -> ControlSchedulingObligation | None:
    if not obligations:
        return None
    return min(
        obligations,
        key=lambda obligation: (
            obligation.due_ts_ns_local,
            obligation.obligation_key,
        ),
    )


def _record_is_currently_queued(
    state: StrategyState,
    record: CandidateIntentRecord,
) -> bool:
    queue = state.queued_intents.get(record.intent.instrument)
    if queue is None:
        return False
    return any(
        queued.intent == record.intent and queued.logical_key == record.logical_key
        for queued in queue
    )


def apply_execution_control_plan(
    plan: ExecutionControlPlan,
    context: ExecutionControlApplyContext,
) -> ExecutionControlApplyResult:
    """Apply mutable execution-control semantics over planned active records.

    This function mutates only StrategyState queue data and ExecutionControl
    rate state. It does not perform venue dispatch and does not emit canonical
    events.

    ``control_scheduling_obligation`` is selected only from **rate-limit**
    deferrals (time-dependent). **Inflight** gating queues or blocks work without
    adding a scheduling obligation; that case is resolved when later canonical
    events update sendability (not via a Core-derived wake time in this slice).
    """

    state = context.state
    execution_control = context.execution_control

    dispatchable_records: list[ExecutionControlDispatchableRecord] = []
    execution_handled_records: list[ExecutionControlHandledRecord] = []
    blocked_records: list[ExecutionControlBlockedRecord] = []
    obligations: list[ControlSchedulingObligation] = []

    processed_keys: set[str] = set()

    to_queue_by_instr: defaultdict[str, list] = defaultdict(list)
    replaced_in_queue: list[tuple] = []
    dropped_in_queue: list = []
    queued: list = []
    handled_in_queue: list = []

    for record in plan.active_records:
        if record.logical_key in processed_keys:
            execution_handled_records.append(
                ExecutionControlHandledRecord(
                    record=record,
                    reason="duplicate_candidate_record",
                )
            )
            continue
        processed_keys.add(record.logical_key)

        intent = record.intent
        instrument = intent.instrument

        if record.origin == CandidateIntentOrigin.GENERATED:
            to_queue_before = len(to_queue_by_instr[instrument])
            handled_before = len(handled_in_queue)
            continue_to_sendability, reject_reason = (
                execution_control.route_pre_submission_lifecycle_and_inflight(
                    intent,
                    state=state,
                    to_queue_by_instr=to_queue_by_instr,
                    replaced_in_queue=replaced_in_queue,
                    dropped_in_queue=dropped_in_queue,
                    queued=queued,
                    handled_in_queue=handled_in_queue,
                    float_equal=_float_equal,
                )
            )
            if not continue_to_sendability:
                to_queue_after = len(to_queue_by_instr[instrument])
                handled_after = len(handled_in_queue)
                if reject_reason is not None:
                    blocked_records.append(
                        ExecutionControlBlockedRecord(
                            record=record,
                            reason=reject_reason,
                        )
                    )
                    continue
                if to_queue_after > to_queue_before:
                    blocked_records.append(
                        ExecutionControlBlockedRecord(
                            record=record,
                            reason="inflight",
                        )
                    )
                    continue
                if handled_after > handled_before:
                    execution_handled_records.append(
                        ExecutionControlHandledRecord(
                            record=record,
                            reason="queue_local_handled",
                        )
                    )
                    continue
                execution_handled_records.append(
                    ExecutionControlHandledRecord(
                        record=record,
                        reason="handled",
                    )
                )
                continue

            rate_result = execution_control.route_after_policy_rate_limit(
                intent,
                now_ts_ns_local=context.now_ts_ns_local,
                max_orders_per_sec=context.max_orders_per_sec,
                max_cancels_per_sec=context.max_cancels_per_sec,
            )
            if rate_result.stage_to_queue:
                to_queue_by_instr[instrument].append(intent)
                blocked_records.append(
                    ExecutionControlBlockedRecord(
                        record=record,
                        reason="rate_limit",
                        scheduling_obligation=rate_result.scheduling_obligation,
                    )
                )
                if rate_result.scheduling_obligation is not None:
                    obligations.append(rate_result.scheduling_obligation)
                continue

            dispatchable_records.append(ExecutionControlDispatchableRecord(record=record))
            continue

        detached = state.pop_queued_intents_for_order(
            intent.instrument,
            intent.client_order_id,
        )
        detached_intents = [queued_item.intent for queued_item in detached]
        if not detached_intents:
            execution_handled_records.append(
                ExecutionControlHandledRecord(
                    record=record,
                    reason="queued_record_missing",
                )
            )
            continue

        if intent.intent_type in ("new", "replace") and state.has_inflight(
            intent.instrument, intent.client_order_id
        ):
            state.merge_intents_into_queue(
                instrument=intent.instrument,
                intents=detached_intents,
            )
            blocked_records.append(
                ExecutionControlBlockedRecord(
                    record=record,
                    reason="inflight",
                )
            )
            continue

        rate_result = execution_control.route_after_policy_rate_limit(
            intent,
            now_ts_ns_local=context.now_ts_ns_local,
            max_orders_per_sec=context.max_orders_per_sec,
            max_cancels_per_sec=context.max_cancels_per_sec,
        )
        if rate_result.stage_to_queue:
            state.merge_intents_into_queue(
                instrument=intent.instrument,
                intents=detached_intents,
            )
            blocked_records.append(
                ExecutionControlBlockedRecord(
                    record=record,
                    reason="rate_limit",
                    scheduling_obligation=rate_result.scheduling_obligation,
                )
            )
            if rate_result.scheduling_obligation is not None:
                obligations.append(rate_result.scheduling_obligation)
            continue

        dispatchable_records.append(ExecutionControlDispatchableRecord(record=record))

    execution_control.merge_to_queue_per_instrument(
        state=state,
        to_queue_by_instr=to_queue_by_instr,
        queued=queued,
        replaced_in_queue=replaced_in_queue,
        dropped_in_queue=dropped_in_queue,
    )

    queued_effective_records = tuple(
        record
        for record in plan.active_records
        if _record_is_currently_queued(state, record)
    )
    control_scheduling_obligation = _select_effective_control_scheduling_obligation(
        obligations
    )
    decision = ExecutionControlDecision(
        queued_effective_intents=tuple(
            record.intent for record in queued_effective_records
        ),
        dispatchable_intents=tuple(
            item.record.intent for item in dispatchable_records
        ),
        execution_handled_intents=tuple(
            item.record.intent for item in execution_handled_records
        ),
        control_scheduling_obligation=control_scheduling_obligation,
    )

    return ExecutionControlApplyResult(
        queued_effective_records=queued_effective_records,
        dispatchable_records=tuple(dispatchable_records),
        execution_handled_records=tuple(execution_handled_records),
        blocked_records=tuple(blocked_records),
        control_scheduling_obligation=control_scheduling_obligation,
        execution_control_decision=decision,
    )
