"""Semantics tests for the transitional Core step API skeleton."""

from __future__ import annotations

import copy

import pytest

import tradingchassis_core as tc
import tradingchassis_core.core.domain.processing_step as processing_step_module
from tradingchassis_core.core.domain import run_core_step as domain_run_core_step
from tradingchassis_core.core.domain.candidate_intent import CandidateIntentOrigin
from tradingchassis_core.core.domain.event_model import (
    canonical_category_for_type,
    is_canonical_stream_candidate_type,
)
from tradingchassis_core.core.domain.execution_control_apply import (
    ExecutionControlApplyResult,
    ExecutionControlDispatchableRecord,
)
from tradingchassis_core.core.domain.execution_control_decision import (
    ExecutionControlDecision,
)
from tradingchassis_core.core.domain.policy_risk_decision import PolicyRiskDecision
from tradingchassis_core.core.domain.processing import process_event_entry
from tradingchassis_core.core.domain.processing_order import EventStreamEntry, ProcessingPosition
from tradingchassis_core.core.domain.processing_step import (
    ControlTimeQueueReevaluationContext,
    CoreDecisionContext,
    CoreExecutionControlApplyContext,
    CorePolicyAdmissionContext,
    CoreStepStrategyContext,
    run_core_step,
)
from tradingchassis_core.core.domain.state import StrategyState
from tradingchassis_core.core.domain.step_decision import CoreStepDecision
from tradingchassis_core.core.domain.step_result import CoreStepResult
from tradingchassis_core.core.domain.types import (
    CancelOrderIntent,
    ControlTimeEvent,
    FillEvent,
    MarketEvent,
    NewOrderIntent,
    NotionalLimits,
    OrderIntent,
    OrderRateLimits,
    OrderStateEvent,
    Price,
    Quantity,
)
from tradingchassis_core.core.events.event_bus import EventBus
from tradingchassis_core.core.events.events import RiskDecisionEvent
from tradingchassis_core.core.events.sinks.null_event_bus import NullEventBus
from tradingchassis_core.core.execution_control.execution_control import ExecutionControl
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation
from tradingchassis_core.core.risk.risk_config import RiskConfig
from tradingchassis_core.core.risk.risk_engine import GateDecision, RejectedIntent, RiskEngine


def _book_market_event(*, instrument: str, ts_ns_local: int, ts_ns_exch: int) -> MarketEvent:
    return MarketEvent(
        ts_ns_local=ts_ns_local,
        ts_ns_exch=ts_ns_exch,
        instrument=instrument,
        event_type="book",
        book={
            "book_type": "snapshot",
            "bids": [
                {
                    "price": {"currency": "USDC", "value": 100.0},
                    "quantity": {"unit": "contracts", "value": 2.0},
                }
            ],
            "asks": [
                {
                    "price": {"currency": "USDC", "value": 101.0},
                    "quantity": {"unit": "contracts", "value": 3.0},
                }
            ],
            "depth": 1,
        },
        trade=None,
    )


def _fill_event(
    *,
    instrument: str,
    client_order_id: str,
    ts_ns_local: int,
    ts_ns_exch: int,
    cum_filled_qty: float = 0.25,
) -> FillEvent:
    return FillEvent(
        ts_ns_local=ts_ns_local,
        ts_ns_exch=ts_ns_exch,
        instrument=instrument,
        client_order_id=client_order_id,
        side="buy",
        intended_price=Price(currency="USDC", value=100.0),
        filled_price=Price(currency="USDC", value=100.5),
        intended_qty=Quantity(unit="contracts", value=1.0),
        cum_filled_qty=Quantity(unit="contracts", value=cum_filled_qty),
        remaining_qty=Quantity(unit="contracts", value=max(0.0, 1.0 - cum_filled_qty)),
        time_in_force="GTC",
        liquidity_flag="maker",
        fee=None,
    )


def _order_state_event(*, instrument: str, client_order_id: str) -> OrderStateEvent:
    return OrderStateEvent(
        ts_ns_local=300,
        ts_ns_exch=290,
        instrument=instrument,
        client_order_id=client_order_id,
        order_type="limit",
        state_type="accepted",
        side="buy",
        intended_price=Price(currency="USDC", value=100.0),
        filled_price=None,
        intended_qty=Quantity(unit="contracts", value=1.0),
        cum_filled_qty=None,
        remaining_qty=None,
        time_in_force="GTC",
        reason=None,
        raw={"req": 0, "source": "snapshot"},
    )


def _market_configuration(*, instrument: str = "BTC-USDC-PERP") -> tc.CoreConfiguration:
    return tc.CoreConfiguration(
        version="v1",
        payload={
            "market": {
                "instruments": {
                    instrument: {
                        "tick_size": 0.1,
                        "lot_size": 0.01,
                        "contract_size": 1.0,
                    }
                }
            }
        },
    )


def _control_time_event(
    *,
    due_ts_ns_local: int,
    realized_ts_ns_local: int,
) -> ControlTimeEvent:
    return ControlTimeEvent(
        ts_ns_local_control=realized_ts_ns_local,
        reason="scheduled_control_recheck",
        due_ts_ns_local=due_ts_ns_local,
        realized_ts_ns_local=realized_ts_ns_local,
        obligation_reason="rate_limit",
        obligation_due_ts_ns_local=due_ts_ns_local,
        runtime_correlation=None,
    )


def _new_intent(*, client_order_id: str) -> NewOrderIntent:
    return NewOrderIntent(
        ts_ns_local=1,
        instrument="BTC-USDC-PERP",
        client_order_id=client_order_id,
        intents_correlation_id="corr-1",
        side="buy",
        order_type="limit",
        intended_qty=Quantity(value=1.0, unit="contracts"),
        intended_price=Price(currency="USDC", value=100.0),
        time_in_force="GTC",
    )


def _state_subset_snapshot(state: StrategyState) -> dict[str, object]:
    return {
        "market": copy.deepcopy(state.market),
        "fills": copy.deepcopy(state.fills),
        "fill_cum_qty": copy.deepcopy(state.fill_cum_qty),
        "last_processing_position_index": state._last_processing_position_index,
    }


def test_run_core_step_public_exports_identity() -> None:
    assert domain_run_core_step is run_core_step
    assert hasattr(tc, "run_core_step")
    assert tc.run_core_step is run_core_step
    assert hasattr(tc, "CoreExecutionControlApplyContext")


def test_run_core_step_delegates_and_returns_default_core_step_result() -> None:
    baseline_state = StrategyState(event_bus=NullEventBus())
    skeleton_state = StrategyState(event_bus=NullEventBus())
    entry = EventStreamEntry(
        position=ProcessingPosition(index=5),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-1",
            ts_ns_local=200,
            ts_ns_exch=180,
        ),
    )

    process_event_entry(baseline_state, entry)
    result = run_core_step(skeleton_state, entry)

    assert isinstance(result, CoreStepResult)
    assert result.generated_intents == ()
    assert result.candidate_intent_records == ()
    assert result.candidate_intents == ()
    assert result.dispatchable_intents == ()
    assert result.control_scheduling_obligation is None
    assert result.core_step_decision is None
    assert result.compat_gate_decision is None
    assert _state_subset_snapshot(skeleton_state) == _state_subset_snapshot(baseline_state)


def test_run_core_step_omitting_strategy_evaluator_preserves_existing_behavior() -> None:
    baseline_state = StrategyState(event_bus=NullEventBus())
    no_strategy_state = StrategyState(event_bus=NullEventBus())
    entry = EventStreamEntry(
        position=ProcessingPosition(index=6),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-omitted-evaluator",
            ts_ns_local=300,
            ts_ns_exch=280,
        ),
    )

    process_event_entry(baseline_state, entry)
    result = run_core_step(no_strategy_state, entry)

    assert result.generated_intents == ()
    assert result.candidate_intent_records == ()
    assert result.candidate_intents == ()
    assert result.core_step_decision is None
    assert result == CoreStepResult()
    assert _state_subset_snapshot(no_strategy_state) == _state_subset_snapshot(baseline_state)


def test_run_core_step_propagates_non_canonical_rejection() -> None:
    state = StrategyState(event_bus=NullEventBus())
    entry = EventStreamEntry(
        position=ProcessingPosition(index=1),
        event=_order_state_event(
            instrument="BTC-USDC-PERP",
            client_order_id="order-compat-1",
        ),
    )

    with pytest.raises(TypeError, match="Unsupported non-canonical event type"):
        run_core_step(state, entry)


def test_run_core_step_propagates_non_monotonic_position_and_preserves_state() -> None:
    state = StrategyState(event_bus=NullEventBus())
    first = EventStreamEntry(
        position=ProcessingPosition(index=10),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-1",
            ts_ns_local=100,
            ts_ns_exch=90,
        ),
    )
    second = EventStreamEntry(
        position=ProcessingPosition(index=11),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-1",
            ts_ns_local=101,
            ts_ns_exch=91,
            cum_filled_qty=0.5,
        ),
    )
    repeated = EventStreamEntry(
        position=ProcessingPosition(index=11),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-1",
            ts_ns_local=102,
            ts_ns_exch=92,
            cum_filled_qty=0.75,
        ),
    )

    run_core_step(state, first)
    run_core_step(state, second)
    before = _state_subset_snapshot(state)

    with pytest.raises(ValueError, match="Non-monotonic ProcessingPosition index"):
        run_core_step(state, repeated)

    assert _state_subset_snapshot(state) == before


def test_run_core_step_positioned_market_requires_configuration() -> None:
    state = StrategyState(event_bus=NullEventBus())
    entry = EventStreamEntry(
        position=ProcessingPosition(index=0),
        event=_book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=100, ts_ns_exch=90),
    )

    with pytest.raises(
        ValueError,
        match="CoreConfiguration is required for positioned canonical MarketEvent processing",
    ):
        run_core_step(state, entry, configuration=None)


def test_run_core_step_passes_configuration_through_to_market_processing() -> None:
    state = StrategyState(event_bus=NullEventBus())
    entry = EventStreamEntry(
        position=ProcessingPosition(index=0),
        event=_book_market_event(instrument="BTC-USDC-PERP", ts_ns_local=100, ts_ns_exch=90),
    )

    result = run_core_step(state, entry, configuration=_market_configuration())

    market = state.market["BTC-USDC-PERP"]
    assert isinstance(result, CoreStepResult)
    assert state._last_processing_position_index == 0
    assert market.best_bid == 100.0
    assert market.best_ask == 101.0


def test_run_core_step_calls_strategy_evaluator_once_with_post_reducer_context() -> None:
    state = StrategyState(event_bus=NullEventBus())
    configuration = _market_configuration(instrument="BTC-USDC-PERP")
    entry = EventStreamEntry(
        position=ProcessingPosition(index=12),
        event=_book_market_event(
            instrument="BTC-USDC-PERP",
            ts_ns_local=1_200,
            ts_ns_exch=1_100,
        ),
    )
    generated_intent = _new_intent(client_order_id="generated-not-dispatchable-yet")
    captured_contexts: list[CoreStepStrategyContext] = []

    class _EvaluatorSpy:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            captured_contexts.append(context)
            return [generated_intent]

    result = run_core_step(
        state,
        entry,
        configuration=configuration,
        strategy_evaluator=_EvaluatorSpy(),
    )

    assert len(captured_contexts) == 1
    context = captured_contexts[0]
    assert context.event is entry.event
    assert context.position == entry.position
    assert context.configuration is configuration
    assert context.state is state
    assert context.state._last_processing_position_index == 12
    assert context.state.market["BTC-USDC-PERP"].best_bid == 100.0
    assert result.generated_intents == (generated_intent,)
    assert tuple(record.intent for record in result.candidate_intent_records) == (generated_intent,)
    assert tuple(record.origin for record in result.candidate_intent_records) == (
        CandidateIntentOrigin.GENERATED,
    )
    assert result.candidate_intents == (generated_intent,)
    assert result.dispatchable_intents == ()
    assert result.core_step_decision is None
    assert result.compat_gate_decision is None


def test_run_core_step_boundary_remains_non_canonical_for_compatibility_artifacts() -> None:
    assert is_canonical_stream_candidate_type(CoreStepResult) is False
    assert canonical_category_for_type(CoreStepResult) is None
    assert is_canonical_stream_candidate_type(ControlSchedulingObligation) is False
    assert canonical_category_for_type(ControlSchedulingObligation) is None
    assert is_canonical_stream_candidate_type(GateDecision) is False
    assert canonical_category_for_type(GateDecision) is None

    state = StrategyState(event_bus=NullEventBus())
    entries = (
        EventStreamEntry(position=ProcessingPosition(index=1), event=CoreStepResult()),
        EventStreamEntry(
            position=ProcessingPosition(index=2),
            event=ControlSchedulingObligation(
                due_ts_ns_local=1_000_000_000,
                reason="rate_limit",
                scope_key="instrument:BTC-USDC-PERP",
                source="execution_control_rate_limit",
            ),
        ),
        EventStreamEntry(
            position=ProcessingPosition(index=3),
            event=GateDecision(
                ts_ns_local=123,
                accepted_now=[_new_intent(client_order_id="accepted-now")],
                queued=[],
                rejected=[],
                replaced_in_queue=[],
                dropped_in_queue=[],
                handled_in_queue=[],
                execution_rejected=[],
                next_send_ts_ns_local=None,
                control_scheduling_obligations=(),
            ),
        ),
    )

    for entry in entries:
        with pytest.raises(TypeError, match="Unsupported non-canonical event type"):
            run_core_step(state, entry)


def test_run_core_step_control_time_with_context_processes_canonical_then_queue_and_risk() -> None:
    state = StrategyState(event_bus=NullEventBus())
    instrument = "BTC-USDC-PERP"
    queued_intent = _new_intent(client_order_id="queued-1")
    state.merge_intents_into_queue(instrument, [queued_intent])

    calls: list[str] = []
    popped_raw_intents: list[list[NewOrderIntent]] = []

    original_pop = state.pop_queued_intents

    def _spy_pop_queued_intents(target_instrument: str) -> list[NewOrderIntent]:
        assert target_instrument == instrument
        # Canonical processing runs first and advances the positioned cursor.
        assert state._last_processing_position_index == 7
        calls.append("pop")
        return original_pop(target_instrument)  # type: ignore[return-value]

    state.pop_queued_intents = _spy_pop_queued_intents  # type: ignore[method-assign]

    accepted_now = _new_intent(client_order_id="accepted-now")
    obligation_a = ControlSchedulingObligation(
        due_ts_ns_local=42,
        reason="rate_limit",
        scope_key=f"instrument:{instrument}",
        source="execution_control_rate_limit",
        obligation_key="z-key",
    )
    obligation_b = ControlSchedulingObligation(
        due_ts_ns_local=42,
        reason="rate_limit",
        scope_key=f"instrument:{instrument}",
        source="execution_control_rate_limit",
        obligation_key="a-key",
    )
    obligation_c = ControlSchedulingObligation(
        due_ts_ns_local=17,
        reason="rate_limit",
        scope_key=f"instrument:{instrument}",
        source="execution_control_rate_limit",
        obligation_key="x-key",
    )

    class _RiskSpy:
        def decide_intents(
            self,
            *,
            raw_intents: list[NewOrderIntent],
            state: StrategyState,
            now_ts_ns_local: int,
        ) -> GateDecision:
            assert state is not None
            assert now_ts_ns_local == 1_000
            calls.append("risk")
            popped_raw_intents.append(list(raw_intents))
            return GateDecision(
                ts_ns_local=now_ts_ns_local,
                accepted_now=[accepted_now],
                queued=[],
                rejected=[],
                replaced_in_queue=[],
                dropped_in_queue=[],
                handled_in_queue=[],
                execution_rejected=[],
                next_send_ts_ns_local=17,
                control_scheduling_obligations=(obligation_a, obligation_b, obligation_c),
            )

    entry = EventStreamEntry(
        position=ProcessingPosition(index=7),
        event=_control_time_event(due_ts_ns_local=999, realized_ts_ns_local=1_000),
    )
    result = run_core_step(
        state,
        entry,
        control_time_queue_context=ControlTimeQueueReevaluationContext(
            risk_engine=_RiskSpy(),  # type: ignore[arg-type]
            instrument=instrument,
            now_ts_ns_local=1_000,
        ),
    )

    assert calls == ["pop", "risk"]
    assert len(popped_raw_intents) == 1
    assert [it.client_order_id for it in popped_raw_intents[0]] == [queued_intent.client_order_id]
    assert tuple(it.client_order_id for it in result.dispatchable_intents) == ("accepted-now",)
    assert tuple(record.intent.client_order_id for record in result.candidate_intent_records) == (
        "queued-1",
    )
    assert tuple(record.origin for record in result.candidate_intent_records) == (
        CandidateIntentOrigin.QUEUED,
    )
    assert tuple(it.client_order_id for it in result.candidate_intents) == ("queued-1",)
    assert isinstance(result.core_step_decision, CoreStepDecision)
    assert tuple(
        it.client_order_id for it in result.core_step_decision.dispatchable_intents
    ) == ("accepted-now",)
    assert result.core_step_decision.control_scheduling_obligation is not None
    assert result.core_step_decision.control_scheduling_obligation.due_ts_ns_local == 17
    assert isinstance(
        result.core_step_decision.execution_control_decision,
        ExecutionControlDecision,
    )
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.dispatchable_intents
    ) == ("accepted-now",)
    assert (
        result.core_step_decision.execution_control_decision.control_scheduling_obligation
        == result.control_scheduling_obligation
    )
    assert isinstance(result.core_step_decision.policy_risk_decision, PolicyRiskDecision)
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.policy_risk_decision.accepted_intents
    ) == ("accepted-now",)
    assert result.core_step_decision.policy_risk_decision.rejected_intents == ()
    assert result.core_step_decision.queued_effective_intents == ()
    assert result.core_step_decision.policy_rejected_intents == ()
    assert result.core_step_decision.execution_handled_intents == ()
    assert result.compat_gate_decision is not None
    assert result.control_scheduling_obligation is not None
    assert result.control_scheduling_obligation.due_ts_ns_local == 17
    assert result.control_scheduling_obligation.obligation_key == "x-key"
    assert (
        result.core_step_decision.control_scheduling_obligation
        == result.control_scheduling_obligation
    )


def test_run_core_step_non_control_time_ignores_control_time_context() -> None:
    state = StrategyState(event_bus=NullEventBus())

    class _RiskMustNotRun:
        def decide_intents(self, **_: object) -> GateDecision:
            raise AssertionError("risk must not run for non-control events")

    def _pop_must_not_run(_: str) -> list[NewOrderIntent]:
        raise AssertionError("queue pop must not run for non-control events")

    state.pop_queued_intents = _pop_must_not_run  # type: ignore[method-assign]

    entry = EventStreamEntry(
        position=ProcessingPosition(index=5),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-no-control",
            ts_ns_local=5,
            ts_ns_exch=4,
        ),
    )
    result = run_core_step(
        state,
        entry,
        control_time_queue_context=ControlTimeQueueReevaluationContext(
            risk_engine=_RiskMustNotRun(),  # type: ignore[arg-type]
            instrument="BTC-USDC-PERP",
            now_ts_ns_local=5,
        ),
    )

    assert result == CoreStepResult()
    assert result.core_step_decision is None


def test_run_core_step_non_control_candidate_context_disabled_keeps_scaffold_behavior() -> None:
    state = StrategyState(event_bus=NullEventBus())
    generated_intent = _new_intent(client_order_id="generated-disabled-context")

    class _Evaluator:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 40
            return [generated_intent]

    class _RiskMustNotRun:
        def decide_intents(self, **_: object) -> GateDecision:
            raise AssertionError("risk must not run when candidate decision context is disabled")

    entry = EventStreamEntry(
        position=ProcessingPosition(index=40),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-disabled-context",
            ts_ns_local=40,
            ts_ns_exch=39,
        ),
    )
    result = run_core_step(
        state,
        entry,
        strategy_evaluator=_Evaluator(),
        core_decision_context=CoreDecisionContext(
            risk_engine=_RiskMustNotRun(),  # type: ignore[arg-type]
            now_ts_ns_local=40,
            enable_candidate_intent_decision=False,
            capture_only=True,
        ),
    )

    assert result.generated_intents == (generated_intent,)
    assert tuple(record.intent for record in result.candidate_intent_records) == (generated_intent,)
    assert tuple(record.origin for record in result.candidate_intent_records) == (
        CandidateIntentOrigin.GENERATED,
    )
    assert result.candidate_intents == (generated_intent,)
    assert result.core_step_decision is None
    assert result.compat_gate_decision is None
    assert result.dispatchable_intents == ()
    assert result.control_scheduling_obligation is None


def test_run_core_step_non_control_candidate_context_enabled_capture_only_maps_decision() -> None:
    state = StrategyState(event_bus=NullEventBus())
    generated_intent = _new_intent(client_order_id="generated-candidate-risk")
    accepted_now = _new_intent(client_order_id="accepted-now-candidate")
    queued_effective = _new_intent(client_order_id="queued-effective-candidate")
    rejected_intent = _new_intent(client_order_id="rejected-candidate")
    handled_intent = CancelOrderIntent(
        ts_ns_local=41,
        instrument="BTC-USDC-PERP",
        client_order_id="handled-candidate",
        intents_correlation_id="corr-handled-candidate",
    )
    obligation = ControlSchedulingObligation(
        due_ts_ns_local=77,
        reason="rate_limit",
        scope_key="instrument:BTC-USDC-PERP",
        source="execution_control_rate_limit",
    )
    captured_raw_intents: list[list[NewOrderIntent]] = []

    class _Evaluator:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 41
            return [generated_intent]

    class _RiskSpy:
        def decide_intents(
            self,
            *,
            raw_intents: list[NewOrderIntent],
            state: StrategyState,
            now_ts_ns_local: int,
        ) -> GateDecision:
            assert state is not None
            assert now_ts_ns_local == 41
            captured_raw_intents.append(list(raw_intents))
            return GateDecision(
                ts_ns_local=now_ts_ns_local,
                accepted_now=[accepted_now],
                queued=[queued_effective],
                rejected=[RejectedIntent(intent=rejected_intent, reason="policy_reject")],
                replaced_in_queue=[],
                dropped_in_queue=[],
                handled_in_queue=[handled_intent],
                execution_rejected=[],
                next_send_ts_ns_local=77,
                control_scheduling_obligations=(obligation,),
            )

    entry = EventStreamEntry(
        position=ProcessingPosition(index=41),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-candidate-risk",
            ts_ns_local=41,
            ts_ns_exch=40,
        ),
    )
    result = run_core_step(
        state,
        entry,
        strategy_evaluator=_Evaluator(),
        core_decision_context=CoreDecisionContext(
            risk_engine=_RiskSpy(),  # type: ignore[arg-type]
            now_ts_ns_local=41,
            enable_candidate_intent_decision=True,
            capture_only=True,
        ),
    )

    assert len(captured_raw_intents) == 1
    assert tuple(it.client_order_id for it in captured_raw_intents[0]) == (
        "generated-candidate-risk",
    )
    assert result.generated_intents == (generated_intent,)
    assert tuple(record.intent for record in result.candidate_intent_records) == (generated_intent,)
    assert tuple(record.origin for record in result.candidate_intent_records) == (
        CandidateIntentOrigin.GENERATED,
    )
    assert result.candidate_intents == (generated_intent,)
    assert result.compat_gate_decision is not None
    assert result.core_step_decision is not None
    assert tuple(
        it.client_order_id for it in result.core_step_decision.dispatchable_intents
    ) == ("accepted-now-candidate",)
    assert tuple(
        it.client_order_id for it in result.core_step_decision.queued_effective_intents
    ) == ("queued-effective-candidate",)
    assert tuple(
        it.client_order_id for it in result.core_step_decision.policy_rejected_intents
    ) == ("rejected-candidate",)
    assert tuple(
        it.client_order_id for it in result.core_step_decision.execution_handled_intents
    ) == ("handled-candidate",)
    assert result.core_step_decision.policy_risk_decision is not None
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.policy_risk_decision.accepted_intents
    ) == ("accepted-now-candidate",)
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.policy_risk_decision.rejected_intents
    ) == ("rejected-candidate",)
    assert result.core_step_decision.execution_control_decision is not None
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.dispatchable_intents
    ) == ("accepted-now-candidate",)
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.queued_effective_intents
    ) == ("queued-effective-candidate",)
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.execution_handled_intents
    ) == ("handled-candidate",)
    assert result.dispatchable_intents == ()
    assert result.control_scheduling_obligation is None


def test_run_core_step_non_control_candidate_context_enabled_empty_candidates_skips_risk() -> None:
    state = StrategyState(event_bus=NullEventBus())
    calls = {"risk": 0}

    class _RiskSpy:
        def decide_intents(self, **_: object) -> GateDecision:
            calls["risk"] += 1
            raise AssertionError("risk must not run when candidate intents are empty")

    entry = EventStreamEntry(
        position=ProcessingPosition(index=42),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-empty-candidates",
            ts_ns_local=42,
            ts_ns_exch=41,
        ),
    )
    result = run_core_step(
        state,
        entry,
        core_decision_context=CoreDecisionContext(
            risk_engine=_RiskSpy(),  # type: ignore[arg-type]
            now_ts_ns_local=42,
            enable_candidate_intent_decision=True,
            capture_only=True,
        ),
    )

    assert calls == {"risk": 0}
    assert result.generated_intents == ()
    assert result.candidate_intent_records == ()
    assert result.candidate_intents == ()
    assert result.core_step_decision is None
    assert result.compat_gate_decision is None
    assert result.dispatchable_intents == ()


def test_run_core_step_non_control_candidate_context_capture_only_false_not_supported() -> None:
    state = StrategyState(event_bus=NullEventBus())
    generated_intent = _new_intent(client_order_id="generated-capture-false")

    class _Evaluator:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 43
            return [generated_intent]

    class _RiskMustNotRun:
        def decide_intents(self, **_: object) -> GateDecision:
            raise AssertionError("risk must not run when capture_only=False is unsupported")

    entry = EventStreamEntry(
        position=ProcessingPosition(index=43),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-capture-false",
            ts_ns_local=43,
            ts_ns_exch=42,
        ),
    )
    with pytest.raises(NotImplementedError, match="capture_only=False is not supported yet"):
        run_core_step(
            state,
            entry,
            strategy_evaluator=_Evaluator(),
            core_decision_context=CoreDecisionContext(
                risk_engine=_RiskMustNotRun(),  # type: ignore[arg-type]
                now_ts_ns_local=43,
                enable_candidate_intent_decision=True,
                capture_only=False,
            ),
        )


def test_run_core_step_control_time_maps_compat_fields_to_core_step_decision() -> None:
    state = StrategyState(event_bus=NullEventBus())
    instrument = "BTC-USDC-PERP"
    queued_intent = _new_intent(client_order_id="queued-pop-source")
    state.merge_intents_into_queue(instrument, [queued_intent])

    accepted_now = _new_intent(client_order_id="accepted-now-mapped")
    queued_effective = _new_intent(client_order_id="queued-effective")
    rejected_intent = _new_intent(client_order_id="rejected-policy")
    handled_intent = CancelOrderIntent(
        ts_ns_local=33,
        instrument=instrument,
        client_order_id="handled-in-queue",
        intents_correlation_id="corr-handled",
    )

    class _RiskSpy:
        def decide_intents(
            self,
            *,
            raw_intents: list[NewOrderIntent],
            state: StrategyState,
            now_ts_ns_local: int,
        ) -> GateDecision:
            assert state is not None
            assert now_ts_ns_local == 33
            assert [it.client_order_id for it in raw_intents] == [queued_intent.client_order_id]
            return GateDecision(
                ts_ns_local=now_ts_ns_local,
                accepted_now=[accepted_now],
                queued=[queued_effective],
                rejected=[RejectedIntent(intent=rejected_intent, reason="policy_reject")],
                replaced_in_queue=[],
                dropped_in_queue=[],
                handled_in_queue=[handled_intent],
                execution_rejected=[],
                next_send_ts_ns_local=None,
                control_scheduling_obligations=(),
            )

    entry = EventStreamEntry(
        position=ProcessingPosition(index=33),
        event=_control_time_event(due_ts_ns_local=33, realized_ts_ns_local=33),
    )
    result = run_core_step(
        state,
        entry,
        control_time_queue_context=ControlTimeQueueReevaluationContext(
            risk_engine=_RiskSpy(),  # type: ignore[arg-type]
            instrument=instrument,
            now_ts_ns_local=33,
        ),
    )

    assert result.core_step_decision is not None
    assert tuple(
        it.client_order_id for it in result.core_step_decision.dispatchable_intents
    ) == ("accepted-now-mapped",)
    assert tuple(
        it.client_order_id for it in result.core_step_decision.queued_effective_intents
    ) == ("queued-effective",)
    assert tuple(
        it.client_order_id for it in result.core_step_decision.policy_rejected_intents
    ) == ("rejected-policy",)
    assert tuple(
        it.client_order_id for it in result.core_step_decision.execution_handled_intents
    ) == ("handled-in-queue",)
    assert isinstance(
        result.core_step_decision.execution_control_decision,
        ExecutionControlDecision,
    )
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.queued_effective_intents
    ) == ("queued-effective",)
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.dispatchable_intents
    ) == ("accepted-now-mapped",)
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.execution_handled_intents
    ) == ("handled-in-queue",)
    assert isinstance(result.core_step_decision.policy_risk_decision, PolicyRiskDecision)
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.policy_risk_decision.accepted_intents
    ) == ("accepted-now-mapped",)
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.policy_risk_decision.rejected_intents
    ) == ("rejected-policy",)


def test_run_core_step_includes_queued_snapshot_in_candidate_intents_without_mutation() -> None:
    state = StrategyState(event_bus=NullEventBus())
    instrument = "BTC-USDC-PERP"
    queued_intent = _new_intent(client_order_id="queued-candidate")
    state.merge_intents_into_queue(instrument, [queued_intent])

    entry = EventStreamEntry(
        position=ProcessingPosition(index=21),
        event=_fill_event(
            instrument=instrument,
            client_order_id="fill-with-queued-candidate",
            ts_ns_local=21,
            ts_ns_exch=20,
        ),
    )
    result = run_core_step(state, entry)

    assert tuple(it.client_order_id for it in result.generated_intents) == ()
    assert tuple(record.intent.client_order_id for record in result.candidate_intent_records) == (
        "queued-candidate",
    )
    assert tuple(record.origin for record in result.candidate_intent_records) == (
        CandidateIntentOrigin.QUEUED,
    )
    assert tuple(it.client_order_id for it in result.candidate_intents) == ("queued-candidate",)
    assert result.dispatchable_intents == ()
    assert state.has_queued_intent(instrument, "queued-candidate")


def test_run_core_step_candidate_intents_apply_generated_vs_queued_dominance_without_queue_mutation() -> None:
    state = StrategyState(event_bus=NullEventBus())
    instrument = "BTC-USDC-PERP"
    key = "same-key"
    queued_new = _new_intent(client_order_id=key)
    state.merge_intents_into_queue(instrument, [queued_new])

    generated_cancel = CancelOrderIntent(
        ts_ns_local=22,
        instrument=instrument,
        client_order_id=key,
        intents_correlation_id="corr-cancel",
    )

    class _Evaluator:
        def evaluate(self, context: CoreStepStrategyContext) -> list[CancelOrderIntent]:
            assert context.state._last_processing_position_index == 22
            return [generated_cancel]

    entry = EventStreamEntry(
        position=ProcessingPosition(index=22),
        event=_fill_event(
            instrument=instrument,
            client_order_id="fill-generated-vs-queued",
            ts_ns_local=22,
            ts_ns_exch=21,
        ),
    )
    result = run_core_step(state, entry, strategy_evaluator=_Evaluator())

    assert tuple(it.intent_type for it in result.generated_intents) == ("cancel",)
    assert tuple(record.intent.intent_type for record in result.candidate_intent_records) == ("cancel",)
    assert tuple(record.origin for record in result.candidate_intent_records) == (
        CandidateIntentOrigin.GENERATED,
    )
    assert tuple(it.intent_type for it in result.candidate_intents) == ("cancel",)
    assert result.dispatchable_intents == ()
    assert state.has_queued_intent(instrument, key)


def test_run_core_step_does_not_call_strategy_evaluator_when_process_event_entry_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = StrategyState(event_bus=NullEventBus())
    called = {"evaluate": 0}
    combine_called = {"value": 0}

    class _EvaluatorSpy:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            _ = context
            called["evaluate"] += 1
            return []

    def _boom(*_: object, **__: object) -> None:
        raise RuntimeError("process boundary failed")

    monkeypatch.setattr(processing_step_module, "process_event_entry", _boom)
    monkeypatch.setattr(
        processing_step_module,
        "combine_candidate_intent_records",
        lambda **_: combine_called.__setitem__("value", combine_called["value"] + 1),
    )

    entry = EventStreamEntry(
        position=ProcessingPosition(index=10),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-failure-evaluator",
            ts_ns_local=10,
            ts_ns_exch=9,
        ),
    )

    with pytest.raises(RuntimeError, match="process boundary failed"):
        run_core_step(state, entry, strategy_evaluator=_EvaluatorSpy())

    assert called == {"evaluate": 0}
    assert combine_called == {"value": 0}


def test_run_core_step_candidate_decision_context_not_reached_when_process_event_entry_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = StrategyState(event_bus=NullEventBus())
    calls = {"risk": 0}

    class _RiskSpy:
        def decide_intents(self, **_: object) -> GateDecision:
            calls["risk"] += 1
            return GateDecision(
                ts_ns_local=99,
                accepted_now=[],
                queued=[],
                rejected=[],
                replaced_in_queue=[],
                dropped_in_queue=[],
                handled_in_queue=[],
                execution_rejected=[],
                next_send_ts_ns_local=None,
                control_scheduling_obligations=(),
            )

    def _boom(*_: object, **__: object) -> None:
        raise RuntimeError("process boundary failed")

    monkeypatch.setattr(processing_step_module, "process_event_entry", _boom)

    entry = EventStreamEntry(
        position=ProcessingPosition(index=44),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-process-fail-context",
            ts_ns_local=44,
            ts_ns_exch=43,
        ),
    )
    with pytest.raises(RuntimeError, match="process boundary failed"):
        run_core_step(
            state,
            entry,
            core_decision_context=CoreDecisionContext(
                risk_engine=_RiskSpy(),  # type: ignore[arg-type]
                now_ts_ns_local=44,
                enable_candidate_intent_decision=True,
                capture_only=True,
            ),
        )
    assert calls == {"risk": 0}


def test_run_core_step_with_strategy_and_control_time_context_orders_calls_deterministically() -> None:
    state = StrategyState(event_bus=NullEventBus())
    instrument = "BTC-USDC-PERP"
    queued_intent = _new_intent(client_order_id="queued-with-strategy")
    state.merge_intents_into_queue(instrument, [queued_intent])

    calls: list[str] = []

    class _EvaluatorSpy:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 20
            calls.append("evaluate")
            return [_new_intent(client_order_id="generated-captured")]

    original_pop = state.pop_queued_intents

    def _spy_pop_queued_intents(target_instrument: str) -> list[NewOrderIntent]:
        assert target_instrument == instrument
        calls.append("pop")
        return original_pop(target_instrument)  # type: ignore[return-value]

    state.pop_queued_intents = _spy_pop_queued_intents  # type: ignore[method-assign]

    accepted_now = _new_intent(client_order_id="accepted-control-time")

    class _RiskSpy:
        def decide_intents(
            self,
            *,
            raw_intents: list[NewOrderIntent],
            state: StrategyState,
            now_ts_ns_local: int,
        ) -> GateDecision:
            assert state is not None
            assert now_ts_ns_local == 2_000
            assert [it.client_order_id for it in raw_intents] == [queued_intent.client_order_id]
            calls.append("risk")
            return GateDecision(
                ts_ns_local=now_ts_ns_local,
                accepted_now=[accepted_now],
                queued=[],
                rejected=[],
                replaced_in_queue=[],
                dropped_in_queue=[],
                handled_in_queue=[],
                execution_rejected=[],
                next_send_ts_ns_local=None,
                control_scheduling_obligations=(),
            )

    entry = EventStreamEntry(
        position=ProcessingPosition(index=20),
        event=_control_time_event(due_ts_ns_local=2_000, realized_ts_ns_local=2_000),
    )
    result = run_core_step(
        state,
        entry,
        strategy_evaluator=_EvaluatorSpy(),
        control_time_queue_context=ControlTimeQueueReevaluationContext(
            risk_engine=_RiskSpy(),  # type: ignore[arg-type]
            instrument=instrument,
            now_ts_ns_local=2_000,
        ),
    )

    assert calls == ["evaluate", "pop", "risk"]
    assert tuple(it.client_order_id for it in result.generated_intents) == (
        "generated-captured",
    )
    assert tuple(record.intent.client_order_id for record in result.candidate_intent_records) == (
        "queued-with-strategy",
        "generated-captured",
    )
    assert tuple(record.origin for record in result.candidate_intent_records) == (
        CandidateIntentOrigin.QUEUED,
        CandidateIntentOrigin.GENERATED,
    )
    assert tuple(it.client_order_id for it in result.candidate_intents) == (
        "queued-with-strategy",
        "generated-captured",
    )
    assert tuple(it.client_order_id for it in result.dispatchable_intents) == (
        "accepted-control-time",
    )
    assert isinstance(result.core_step_decision, CoreStepDecision)
    assert tuple(
        it.client_order_id for it in result.core_step_decision.dispatchable_intents
    ) == ("accepted-control-time",)
    assert isinstance(
        result.core_step_decision.execution_control_decision,
        ExecutionControlDecision,
    )
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.dispatchable_intents
    ) == ("accepted-control-time",)
    assert isinstance(result.core_step_decision.policy_risk_decision, PolicyRiskDecision)
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.policy_risk_decision.accepted_intents
    ) == ("accepted-control-time",)


def test_run_core_step_strategy_evaluator_exception_propagates_and_skips_control_time_queue_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = StrategyState(event_bus=NullEventBus())
    instrument = "BTC-USDC-PERP"
    state.merge_intents_into_queue(instrument, [_new_intent(client_order_id="queued-before-failure")])
    calls = {"pop": 0, "risk": 0}
    combine_called = {"value": 0}

    def _pop_spy(_: str) -> list[NewOrderIntent]:
        calls["pop"] += 1
        return []

    state.pop_queued_intents = _pop_spy  # type: ignore[method-assign]

    class _EvaluatorBoom:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 30
            raise RuntimeError("strategy evaluator failed")

    class _RiskSpy:
        def decide_intents(self, **_: object) -> GateDecision:
            calls["risk"] += 1
            raise AssertionError("risk should not run after strategy evaluator failure")

    monkeypatch.setattr(
        processing_step_module,
        "combine_candidate_intent_records",
        lambda **_: combine_called.__setitem__("value", combine_called["value"] + 1),
    )

    entry = EventStreamEntry(
        position=ProcessingPosition(index=30),
        event=_control_time_event(due_ts_ns_local=30, realized_ts_ns_local=30),
    )

    with pytest.raises(RuntimeError, match="strategy evaluator failed"):
        run_core_step(
            state,
            entry,
            strategy_evaluator=_EvaluatorBoom(),
            control_time_queue_context=ControlTimeQueueReevaluationContext(
                risk_engine=_RiskSpy(),  # type: ignore[arg-type]
                instrument=instrument,
                now_ts_ns_local=30,
            ),
        )

    assert calls == {"pop": 0, "risk": 0}
    assert combine_called == {"value": 0}


def test_run_core_step_candidate_decision_context_not_reached_when_strategy_fails() -> None:
    state = StrategyState(event_bus=NullEventBus())
    calls = {"risk": 0}

    class _EvaluatorBoom:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 45
            raise RuntimeError("strategy evaluator failed")

    class _RiskSpy:
        def decide_intents(self, **_: object) -> GateDecision:
            calls["risk"] += 1
            return GateDecision(
                ts_ns_local=45,
                accepted_now=[],
                queued=[],
                rejected=[],
                replaced_in_queue=[],
                dropped_in_queue=[],
                handled_in_queue=[],
                execution_rejected=[],
                next_send_ts_ns_local=None,
                control_scheduling_obligations=(),
            )

    entry = EventStreamEntry(
        position=ProcessingPosition(index=45),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-evaluator-fail-context",
            ts_ns_local=45,
            ts_ns_exch=44,
        ),
    )
    with pytest.raises(RuntimeError, match="strategy evaluator failed"):
        run_core_step(
            state,
            entry,
            strategy_evaluator=_EvaluatorBoom(),
            core_decision_context=CoreDecisionContext(
                risk_engine=_RiskSpy(),  # type: ignore[arg-type]
                now_ts_ns_local=45,
                enable_candidate_intent_decision=True,
                capture_only=True,
            ),
        )
    assert calls == {"risk": 0}


def test_run_core_step_candidate_decision_context_propagates_risk_failure() -> None:
    state = StrategyState(event_bus=NullEventBus())
    generated_intent = _new_intent(client_order_id="generated-risk-failure")

    class _Evaluator:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 46
            return [generated_intent]

    class _RiskBoom:
        def decide_intents(self, **_: object) -> GateDecision:
            raise RuntimeError("risk engine failed in candidate capture path")

    entry = EventStreamEntry(
        position=ProcessingPosition(index=46),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-risk-failure-context",
            ts_ns_local=46,
            ts_ns_exch=45,
        ),
    )
    with pytest.raises(RuntimeError, match="risk engine failed in candidate capture path"):
        run_core_step(
            state,
            entry,
            strategy_evaluator=_Evaluator(),
            core_decision_context=CoreDecisionContext(
                risk_engine=_RiskBoom(),  # type: ignore[arg-type]
                now_ts_ns_local=46,
                enable_candidate_intent_decision=True,
                capture_only=True,
            ),
        )


def test_run_core_step_candidate_decision_context_side_effects_are_opt_in_characterization() -> None:
    instrument = "BTC-USDC-PERP"
    client_order_id = "opt-in-side-effect-order"
    risk_cfg = RiskConfig(
        scope="test",
        trading_enabled=True,
        notional_limits=NotionalLimits(
            currency="USDC",
            max_gross_notional=1e18,
            max_single_order_notional=1e18,
        ),
        order_rate_limits=OrderRateLimits(max_orders_per_second=0),
    )

    baseline_state = StrategyState(event_bus=NullEventBus())
    enabled_state = StrategyState(event_bus=NullEventBus())
    enabled_risk = RiskEngine(risk_cfg=risk_cfg, event_bus=NullEventBus())

    class _Evaluator:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index in (47, 48)
            return [
                NewOrderIntent(
                    ts_ns_local=context.position.index,
                    instrument=instrument,
                    client_order_id=client_order_id,
                    intents_correlation_id="corr-opt-in",
                    side="buy",
                    order_type="limit",
                    intended_qty=Quantity(value=1.0, unit="contracts"),
                    intended_price=Price(currency="USDC", value=100.0),
                    time_in_force="GTC",
                )
            ]

    baseline_entry = EventStreamEntry(
        position=ProcessingPosition(index=47),
        event=_fill_event(
            instrument=instrument,
            client_order_id="fill-side-effect-baseline",
            ts_ns_local=47,
            ts_ns_exch=46,
        ),
    )
    enabled_entry = EventStreamEntry(
        position=ProcessingPosition(index=48),
        event=_fill_event(
            instrument=instrument,
            client_order_id="fill-side-effect-enabled",
            ts_ns_local=48,
            ts_ns_exch=47,
        ),
    )

    baseline_result = run_core_step(
        baseline_state,
        baseline_entry,
        strategy_evaluator=_Evaluator(),
    )
    enabled_result = run_core_step(
        enabled_state,
        enabled_entry,
        strategy_evaluator=_Evaluator(),
        core_decision_context=CoreDecisionContext(
            risk_engine=enabled_risk,
            now_ts_ns_local=48,
            enable_candidate_intent_decision=True,
            capture_only=True,
        ),
    )
    assert baseline_result.core_step_decision is None
    assert baseline_result.compat_gate_decision is None
    assert baseline_result.dispatchable_intents == ()
    assert not baseline_state.has_queued_intent(instrument, client_order_id)

    assert enabled_result.core_step_decision is not None
    assert enabled_result.compat_gate_decision is not None
    assert enabled_result.dispatchable_intents == ()
    assert enabled_state.has_queued_intent(instrument, client_order_id)


def test_run_core_step_control_time_with_both_contexts_preserves_existing_control_time_path() -> None:
    state = StrategyState(event_bus=NullEventBus())
    instrument = "BTC-USDC-PERP"
    queued_intent = _new_intent(client_order_id="queued-control-both-contexts")
    state.merge_intents_into_queue(instrument, [queued_intent])

    calls = {"control_risk": 0, "candidate_risk": 0}
    accepted_now = _new_intent(client_order_id="accepted-control-both-contexts")

    class _ControlRiskSpy:
        def decide_intents(
            self,
            *,
            raw_intents: list[NewOrderIntent],
            state: StrategyState,
            now_ts_ns_local: int,
        ) -> GateDecision:
            calls["control_risk"] += 1
            assert state is not None
            assert now_ts_ns_local == 49
            assert [it.client_order_id for it in raw_intents] == (
                [queued_intent.client_order_id]
            )
            return GateDecision(
                ts_ns_local=now_ts_ns_local,
                accepted_now=[accepted_now],
                queued=[],
                rejected=[],
                replaced_in_queue=[],
                dropped_in_queue=[],
                handled_in_queue=[],
                execution_rejected=[],
                next_send_ts_ns_local=None,
                control_scheduling_obligations=(),
            )

    class _CandidateRiskMustNotRun:
        def decide_intents(self, **_: object) -> GateDecision:
            calls["candidate_risk"] += 1
            raise AssertionError(
                "candidate decision path must not run on control-time compatibility path"
            )

    entry = EventStreamEntry(
        position=ProcessingPosition(index=49),
        event=_control_time_event(due_ts_ns_local=49, realized_ts_ns_local=49),
    )
    result = run_core_step(
        state,
        entry,
        control_time_queue_context=ControlTimeQueueReevaluationContext(
            risk_engine=_ControlRiskSpy(),  # type: ignore[arg-type]
            instrument=instrument,
            now_ts_ns_local=49,
        ),
        core_decision_context=CoreDecisionContext(
            risk_engine=_CandidateRiskMustNotRun(),  # type: ignore[arg-type]
            now_ts_ns_local=49,
            enable_candidate_intent_decision=True,
            capture_only=True,
        ),
    )

    assert calls == {"control_risk": 1, "candidate_risk": 0}
    assert tuple(it.client_order_id for it in result.dispatchable_intents) == (
        "accepted-control-both-contexts",
    )
    assert result.compat_gate_decision is not None
    assert result.core_step_decision is not None


def test_run_core_step_does_not_pop_or_gate_when_process_event_entry_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = StrategyState(event_bus=NullEventBus())
    calls = {"pop": 0, "risk": 0}

    def _pop_spy(_: str) -> list[NewOrderIntent]:
        calls["pop"] += 1
        return []

    state.pop_queued_intents = _pop_spy  # type: ignore[method-assign]

    class _RiskSpy:
        def decide_intents(self, **_: object) -> GateDecision:
            calls["risk"] += 1
            raise AssertionError("risk should not run when boundary fails first")

    def _boom(*_: object, **__: object) -> None:
        raise RuntimeError("process boundary failed")

    monkeypatch.setattr(processing_step_module, "process_event_entry", _boom)

    entry = EventStreamEntry(
        position=ProcessingPosition(index=9),
        event=_control_time_event(due_ts_ns_local=9, realized_ts_ns_local=9),
    )

    with pytest.raises(RuntimeError, match="process boundary failed"):
        run_core_step(
            state,
            entry,
            control_time_queue_context=ControlTimeQueueReevaluationContext(
                risk_engine=_RiskSpy(),  # type: ignore[arg-type]
                instrument="BTC-USDC-PERP",
                now_ts_ns_local=9,
            ),
        )

    assert calls == {"pop": 0, "risk": 0}


def test_run_core_step_policy_admission_context_populates_policy_decision_only() -> None:
    state = StrategyState(event_bus=NullEventBus())
    instrument = "BTC-USDC-PERP"
    queued_intent = _new_intent(client_order_id="queued-passthrough")
    state.merge_intents_into_queue(instrument, [queued_intent])

    generated_new = _new_intent(client_order_id="generated-new-rejected")
    generated_cancel = CancelOrderIntent(
        ts_ns_local=50,
        instrument=instrument,
        client_order_id="generated-cancel-accepted",
        intents_correlation_id="corr-generated-cancel",
    )
    risk_cfg = RiskConfig(
        scope="test",
        trading_enabled=False,
        notional_limits=NotionalLimits(
            currency="USDC",
            max_gross_notional=1e18,
            max_single_order_notional=1e18,
        ),
    )
    risk_engine = RiskEngine(risk_cfg=risk_cfg, event_bus=NullEventBus())

    class _Evaluator:
        def evaluate(self, context: CoreStepStrategyContext) -> list[OrderIntent]:
            assert context.state._last_processing_position_index == 50
            return [generated_new, generated_cancel]

    entry = EventStreamEntry(
        position=ProcessingPosition(index=50),
        event=_fill_event(
            instrument=instrument,
            client_order_id="fill-policy-context",
            ts_ns_local=50,
            ts_ns_exch=49,
        ),
    )
    result = run_core_step(
        state,
        entry,
        strategy_evaluator=_Evaluator(),
        policy_admission_context=CorePolicyAdmissionContext(
            policy_evaluator=risk_engine,
            now_ts_ns_local=50,
        ),
    )

    assert tuple(record.intent.client_order_id for record in result.candidate_intent_records) == (
        "generated-cancel-accepted",
        "queued-passthrough",
        "generated-new-rejected",
    )
    assert result.core_step_decision is not None
    assert result.core_step_decision.execution_control_decision is not None
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.queued_effective_intents
    ) == (
        "generated-cancel-accepted",
        "queued-passthrough",
    )
    assert (
        result.core_step_decision.execution_control_decision.dispatchable_intents
        == ()
    )
    assert (
        result.core_step_decision.execution_control_decision.execution_handled_intents
        == ()
    )
    assert (
        result.core_step_decision.execution_control_decision.control_scheduling_obligation
        is None
    )
    assert tuple(it.client_order_id for it in result.core_step_decision.policy_rejected_intents) == (
        "generated-new-rejected",
    )
    assert result.core_step_decision.policy_risk_decision is not None
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.policy_risk_decision.accepted_intents
    ) == ("generated-cancel-accepted",)
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.policy_risk_decision.rejected_intents
    ) == ("generated-new-rejected",)
    assert result.core_step_decision.queued_effective_intents == ()
    assert result.core_step_decision.dispatchable_intents == ()
    assert result.core_step_decision.execution_handled_intents == ()
    assert result.core_step_decision.control_scheduling_obligation is None
    assert result.core_step_decision.dispatchable_intents == ()
    assert result.dispatchable_intents == ()
    assert result.control_scheduling_obligation is None
    assert result.compat_gate_decision is None


def test_run_core_step_policy_admission_context_queued_only_skips_policy_evaluation() -> None:
    state = StrategyState(event_bus=NullEventBus())
    instrument = "BTC-USDC-PERP"
    state.merge_intents_into_queue(
        instrument,
        [_new_intent(client_order_id="queued-only-record")],
    )
    calls = {"evaluate": 0}

    class _Evaluator:
        def evaluate_policy_intent(self, **_: object) -> tuple[bool, str | None]:
            calls["evaluate"] += 1
            return True, None

    entry = EventStreamEntry(
        position=ProcessingPosition(index=51),
        event=_fill_event(
            instrument=instrument,
            client_order_id="fill-queued-only",
            ts_ns_local=51,
            ts_ns_exch=50,
        ),
    )
    result = run_core_step(
        state,
        entry,
        policy_admission_context=CorePolicyAdmissionContext(
            policy_evaluator=_Evaluator(),  # type: ignore[arg-type]
            now_ts_ns_local=51,
        ),
    )

    assert calls == {"evaluate": 0}
    assert tuple(record.origin for record in result.candidate_intent_records) == (
        CandidateIntentOrigin.QUEUED,
    )
    assert result.core_step_decision is not None
    assert result.core_step_decision.policy_risk_decision == PolicyRiskDecision()
    assert result.core_step_decision.policy_rejected_intents == ()
    assert result.core_step_decision.execution_control_decision is not None
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.queued_effective_intents
    ) == ("queued-only-record",)
    assert result.core_step_decision.execution_control_decision.dispatchable_intents == ()
    assert (
        result.core_step_decision.execution_control_decision.execution_handled_intents
        == ()
    )
    assert (
        result.core_step_decision.execution_control_decision.control_scheduling_obligation
        is None
    )
    assert result.dispatchable_intents == ()


def test_run_core_step_policy_admission_context_not_reached_when_process_event_entry_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = StrategyState(event_bus=NullEventBus())
    calls = {"evaluate": 0}

    class _Evaluator:
        def evaluate_policy_intent(self, **_: object) -> tuple[bool, str | None]:
            calls["evaluate"] += 1
            return True, None

    def _boom(*_: object, **__: object) -> None:
        raise RuntimeError("process boundary failed")

    monkeypatch.setattr(processing_step_module, "process_event_entry", _boom)

    entry = EventStreamEntry(
        position=ProcessingPosition(index=52),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-policy-process-fail",
            ts_ns_local=52,
            ts_ns_exch=51,
        ),
    )
    with pytest.raises(RuntimeError, match="process boundary failed"):
        run_core_step(
            state,
            entry,
            policy_admission_context=CorePolicyAdmissionContext(
                policy_evaluator=_Evaluator(),  # type: ignore[arg-type]
                now_ts_ns_local=52,
            ),
        )
    assert calls == {"evaluate": 0}


def test_run_core_step_policy_planner_not_reached_when_process_event_entry_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = StrategyState(event_bus=NullEventBus())
    calls = {"planner": 0}

    def _boom(*_: object, **__: object) -> None:
        raise RuntimeError("process boundary failed")

    def _planner_spy(*_: object, **__: object) -> object:
        calls["planner"] += 1
        raise AssertionError("planner must not run when boundary fails")

    monkeypatch.setattr(processing_step_module, "process_event_entry", _boom)
    monkeypatch.setattr(
        processing_step_module,
        "plan_execution_control_candidates",
        _planner_spy,
    )

    entry = EventStreamEntry(
        position=ProcessingPosition(index=55),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-policy-planner-process-fail",
            ts_ns_local=55,
            ts_ns_exch=54,
        ),
    )
    with pytest.raises(RuntimeError, match="process boundary failed"):
        run_core_step(
            state,
            entry,
            policy_admission_context=CorePolicyAdmissionContext(
                policy_evaluator=object(),  # type: ignore[arg-type]
                now_ts_ns_local=55,
            ),
        )
    assert calls == {"planner": 0}


def test_run_core_step_policy_admission_context_not_reached_when_strategy_fails() -> None:
    state = StrategyState(event_bus=NullEventBus())
    calls = {"evaluate": 0}

    class _EvaluatorBoom:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 53
            raise RuntimeError("strategy evaluator failed")

    class _PolicyEvaluator:
        def evaluate_policy_intent(self, **_: object) -> tuple[bool, str | None]:
            calls["evaluate"] += 1
            return True, None

    entry = EventStreamEntry(
        position=ProcessingPosition(index=53),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-policy-strategy-fail",
            ts_ns_local=53,
            ts_ns_exch=52,
        ),
    )
    with pytest.raises(RuntimeError, match="strategy evaluator failed"):
        run_core_step(
            state,
            entry,
            strategy_evaluator=_EvaluatorBoom(),
            policy_admission_context=CorePolicyAdmissionContext(
                policy_evaluator=_PolicyEvaluator(),  # type: ignore[arg-type]
                now_ts_ns_local=53,
            ),
        )
    assert calls == {"evaluate": 0}


def test_run_core_step_policy_planner_not_reached_when_strategy_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = StrategyState(event_bus=NullEventBus())
    calls = {"planner": 0}

    class _EvaluatorBoom:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 56
            raise RuntimeError("strategy evaluator failed")

    def _planner_spy(*_: object, **__: object) -> object:
        calls["planner"] += 1
        raise AssertionError("planner must not run when strategy evaluation fails")

    monkeypatch.setattr(
        processing_step_module,
        "plan_execution_control_candidates",
        _planner_spy,
    )

    entry = EventStreamEntry(
        position=ProcessingPosition(index=56),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-policy-planner-strategy-fail",
            ts_ns_local=56,
            ts_ns_exch=55,
        ),
    )
    with pytest.raises(RuntimeError, match="strategy evaluator failed"):
        run_core_step(
            state,
            entry,
            strategy_evaluator=_EvaluatorBoom(),
            policy_admission_context=CorePolicyAdmissionContext(
                policy_evaluator=object(),  # type: ignore[arg-type]
                now_ts_ns_local=56,
            ),
        )
    assert calls == {"planner": 0}


def test_run_core_step_policy_admission_context_is_side_effect_safe_characterization() -> None:
    class _CaptureSink:
        def __init__(self) -> None:
            self.events: list[object] = []

        def on_event(self, event: object) -> None:
            self.events.append(event)

    sink = _CaptureSink()
    event_bus = EventBus(sinks=[sink])
    state = StrategyState(event_bus=event_bus)
    risk_cfg = RiskConfig(
        scope="test",
        trading_enabled=True,
        notional_limits=NotionalLimits(
            currency="USDC",
            max_gross_notional=1e18,
            max_single_order_notional=1e18,
        ),
        order_rate_limits=OrderRateLimits(max_orders_per_second=0),
    )
    risk_engine = RiskEngine(risk_cfg=risk_cfg, event_bus=event_bus)

    generated_intent = _new_intent(client_order_id="side-effect-safe-generated")

    class _Evaluator:
        def evaluate(self, context: CoreStepStrategyContext) -> list[OrderIntent]:
            assert context.state._last_processing_position_index == 54
            return [generated_intent]

    before_rate_state = copy.deepcopy(risk_engine._execution_control._rate_state)
    before_queue = state.queued_intents_snapshot("BTC-USDC-PERP")

    entry = EventStreamEntry(
        position=ProcessingPosition(index=54),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-policy-side-effect-safe",
            ts_ns_local=54,
            ts_ns_exch=53,
        ),
    )
    result = run_core_step(
        state,
        entry,
        strategy_evaluator=_Evaluator(),
        policy_admission_context=CorePolicyAdmissionContext(
            policy_evaluator=risk_engine,
            now_ts_ns_local=54,
        ),
    )

    assert state.queued_intents_snapshot("BTC-USDC-PERP") == before_queue
    assert risk_engine._execution_control._rate_state == before_rate_state
    assert all(not isinstance(event, RiskDecisionEvent) for event in sink.events)
    assert result.dispatchable_intents == ()
    assert result.compat_gate_decision is None


def test_run_core_step_apply_context_requires_policy_admission_context() -> None:
    state = StrategyState(event_bus=NullEventBus())
    entry = EventStreamEntry(
        position=ProcessingPosition(index=57),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-apply-without-policy",
            ts_ns_local=57,
            ts_ns_exch=56,
        ),
    )

    with pytest.raises(
        ValueError,
        match="execution_control_apply_context requires policy_admission_context",
    ):
        run_core_step(
            state,
            entry,
            execution_control_apply_context=CoreExecutionControlApplyContext(
                execution_control=ExecutionControl(),
                now_ts_ns_local=57,
            ),
        )


def test_run_core_step_apply_context_rejects_control_time_event_path() -> None:
    state = StrategyState(event_bus=NullEventBus())
    entry = EventStreamEntry(
        position=ProcessingPosition(index=58),
        event=_control_time_event(due_ts_ns_local=58, realized_ts_ns_local=58),
    )

    class _ControlTimeRiskMustNotRun:
        def decide_intents(self, **_: object) -> GateDecision:
            raise AssertionError("control-time compatibility risk must not run")

    class _PolicyEvaluator:
        def evaluate_policy_intent(self, **_: object) -> tuple[bool, str | None]:
            return True, None

    with pytest.raises(
        ValueError,
        match="execution_control_apply_context is not supported for ControlTimeEvent",
    ):
        run_core_step(
            state,
            entry,
            control_time_queue_context=ControlTimeQueueReevaluationContext(
                risk_engine=_ControlTimeRiskMustNotRun(),  # type: ignore[arg-type]
                instrument="BTC-USDC-PERP",
                now_ts_ns_local=58,
            ),
            policy_admission_context=CorePolicyAdmissionContext(
                policy_evaluator=_PolicyEvaluator(),  # type: ignore[arg-type]
                now_ts_ns_local=58,
            ),
            execution_control_apply_context=CoreExecutionControlApplyContext(
                execution_control=ExecutionControl(),
                now_ts_ns_local=58,
            ),
        )


def test_run_core_step_apply_integration_orders_policy_plan_apply_and_maps_result() -> None:
    state = StrategyState(event_bus=NullEventBus())
    instrument = "BTC-USDC-PERP"
    queued_intent = _new_intent(client_order_id="queued-passthrough")
    state.merge_intents_into_queue(instrument, [queued_intent])
    generated_new = _new_intent(client_order_id="generated-new-rejected")
    generated_cancel = CancelOrderIntent(
        ts_ns_local=59,
        instrument=instrument,
        client_order_id="generated-cancel-accepted",
        intents_correlation_id="corr-generated-cancel-accepted",
    )
    calls: list[str] = []
    observed_apply_active_ids: list[tuple[str, ...]] = []

    class _Evaluator:
        def evaluate(self, context: CoreStepStrategyContext) -> list[OrderIntent]:
            assert context.state._last_processing_position_index == 59
            return [generated_new, generated_cancel]

    class _PolicyEvaluator:
        def evaluate_policy_intent(
            self,
            *,
            intent: OrderIntent,
            state: StrategyState,
            now_ts_ns_local: int,
        ) -> tuple[bool, str | None]:
            assert state is not None
            assert now_ts_ns_local == 59
            return intent.client_order_id == "generated-cancel-accepted", "policy_rejected"

    original_policy = processing_step_module.apply_policy_to_candidate_records
    original_plan = processing_step_module.plan_execution_control_candidates
    obligation = ControlSchedulingObligation(
        due_ts_ns_local=88,
        reason="rate_limit",
        scope_key=f"instrument:{instrument}",
        source="execution_control_rate_limit",
    )

    def _policy_spy(*args: object, **kwargs: object) -> object:
        calls.append("policy")
        return original_policy(*args, **kwargs)

    def _plan_spy(*args: object, **kwargs: object) -> object:
        calls.append("plan")
        return original_plan(*args, **kwargs)

    def _apply_spy(*args: object, **kwargs: object) -> ExecutionControlApplyResult:
        calls.append("apply")
        plan = args[0]
        context = args[1]
        assert context.state is state
        observed_apply_active_ids.append(
            tuple(record.intent.client_order_id for record in plan.active_records)
        )
        dispatchable = (
            ExecutionControlDispatchableRecord(record=plan.active_records[0]),
        )
        decision = ExecutionControlDecision(
            queued_effective_intents=tuple(record.intent for record in plan.active_records),
            dispatchable_intents=tuple(item.record.intent for item in dispatchable),
            execution_handled_intents=(),
            control_scheduling_obligation=obligation,
        )
        return ExecutionControlApplyResult(
            queued_effective_records=tuple(plan.active_records),
            dispatchable_records=dispatchable,
            execution_handled_records=(),
            blocked_records=(),
            control_scheduling_obligation=obligation,
            execution_control_decision=decision,
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        processing_step_module,
        "apply_policy_to_candidate_records",
        _policy_spy,
    )
    monkeypatch.setattr(
        processing_step_module,
        "plan_execution_control_candidates",
        _plan_spy,
    )
    monkeypatch.setattr(
        processing_step_module,
        "apply_execution_control_plan",
        _apply_spy,
    )
    try:
        entry = EventStreamEntry(
            position=ProcessingPosition(index=59),
            event=_fill_event(
                instrument=instrument,
                client_order_id="fill-apply-ordering",
                ts_ns_local=59,
                ts_ns_exch=58,
            ),
        )
        result = run_core_step(
            state,
            entry,
            strategy_evaluator=_Evaluator(),
            policy_admission_context=CorePolicyAdmissionContext(
                policy_evaluator=_PolicyEvaluator(),  # type: ignore[arg-type]
                now_ts_ns_local=59,
            ),
            execution_control_apply_context=CoreExecutionControlApplyContext(
                execution_control=ExecutionControl(),
                now_ts_ns_local=59,
                activate_dispatchable_outputs=False,
            ),
        )
    finally:
        monkeypatch.undo()

    assert calls == ["policy", "plan", "apply"]
    assert observed_apply_active_ids == [
        ("generated-cancel-accepted", "queued-passthrough"),
    ]
    assert result.core_step_decision is not None
    assert result.core_step_decision.execution_control_decision is not None
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.dispatchable_intents
    ) == ("generated-cancel-accepted",)
    assert result.core_step_decision.control_scheduling_obligation == obligation
    assert result.control_scheduling_obligation == obligation
    assert result.dispatchable_intents == ()
    assert result.compat_gate_decision is None


def test_run_core_step_apply_integration_can_activate_top_level_dispatchables() -> None:
    state = StrategyState(event_bus=NullEventBus())
    generated_intent = _new_intent(client_order_id="generated-dispatchable-activated")

    class _Evaluator:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 60
            return [generated_intent]

    class _PolicyEvaluator:
        def evaluate_policy_intent(self, **_: object) -> tuple[bool, str | None]:
            return True, None

    entry = EventStreamEntry(
        position=ProcessingPosition(index=60),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-apply-dispatchable-activation",
            ts_ns_local=60,
            ts_ns_exch=59,
        ),
    )
    result = run_core_step(
        state,
        entry,
        strategy_evaluator=_Evaluator(),
        policy_admission_context=CorePolicyAdmissionContext(
            policy_evaluator=_PolicyEvaluator(),  # type: ignore[arg-type]
            now_ts_ns_local=60,
        ),
        execution_control_apply_context=CoreExecutionControlApplyContext(
            execution_control=ExecutionControl(),
            now_ts_ns_local=60,
            activate_dispatchable_outputs=True,
        ),
    )

    assert tuple(it.client_order_id for it in result.dispatchable_intents) == (
        "generated-dispatchable-activated",
    )
    assert result.core_step_decision is not None
    assert result.core_step_decision.execution_control_decision is not None
    assert tuple(
        it.client_order_id
        for it in result.core_step_decision.execution_control_decision.dispatchable_intents
    ) == ("generated-dispatchable-activated",)
    assert result.compat_gate_decision is None


def test_run_core_step_apply_context_not_reached_when_process_event_entry_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = StrategyState(event_bus=NullEventBus())
    calls = {"apply": 0}

    class _PolicyEvaluator:
        def evaluate_policy_intent(self, **_: object) -> tuple[bool, str | None]:
            return True, None

    def _boom(*_: object, **__: object) -> None:
        raise RuntimeError("process boundary failed")

    def _apply_spy(*_: object, **__: object) -> object:
        calls["apply"] += 1
        raise AssertionError("apply must not run when boundary fails")

    monkeypatch.setattr(processing_step_module, "process_event_entry", _boom)
    monkeypatch.setattr(
        processing_step_module,
        "apply_execution_control_plan",
        _apply_spy,
    )

    entry = EventStreamEntry(
        position=ProcessingPosition(index=61),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-apply-process-fail",
            ts_ns_local=61,
            ts_ns_exch=60,
        ),
    )
    with pytest.raises(RuntimeError, match="process boundary failed"):
        run_core_step(
            state,
            entry,
            policy_admission_context=CorePolicyAdmissionContext(
                policy_evaluator=_PolicyEvaluator(),  # type: ignore[arg-type]
                now_ts_ns_local=61,
            ),
            execution_control_apply_context=CoreExecutionControlApplyContext(
                execution_control=ExecutionControl(),
                now_ts_ns_local=61,
            ),
        )
    assert calls == {"apply": 0}


def test_run_core_step_apply_context_not_reached_when_strategy_fails() -> None:
    state = StrategyState(event_bus=NullEventBus())
    calls = {"apply": 0}

    class _EvaluatorBoom:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 62
            raise RuntimeError("strategy evaluator failed")

    class _PolicyEvaluator:
        def evaluate_policy_intent(self, **_: object) -> tuple[bool, str | None]:
            return True, None

    def _apply_spy(*_: object, **__: object) -> object:
        calls["apply"] += 1
        raise AssertionError("apply must not run when strategy fails")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        processing_step_module,
        "apply_execution_control_plan",
        _apply_spy,
    )
    try:
        entry = EventStreamEntry(
            position=ProcessingPosition(index=62),
            event=_fill_event(
                instrument="BTC-USDC-PERP",
                client_order_id="fill-apply-strategy-fail",
                ts_ns_local=62,
                ts_ns_exch=61,
            ),
        )
        with pytest.raises(RuntimeError, match="strategy evaluator failed"):
            run_core_step(
                state,
                entry,
                strategy_evaluator=_EvaluatorBoom(),
                policy_admission_context=CorePolicyAdmissionContext(
                    policy_evaluator=_PolicyEvaluator(),  # type: ignore[arg-type]
                    now_ts_ns_local=62,
                ),
                execution_control_apply_context=CoreExecutionControlApplyContext(
                    execution_control=ExecutionControl(),
                    now_ts_ns_local=62,
                ),
            )
    finally:
        monkeypatch.undo()

    assert calls == {"apply": 0}


def test_run_core_step_apply_path_does_not_call_risk_decide_intents_or_emit_risk_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CaptureSink:
        def __init__(self) -> None:
            self.events: list[object] = []

        def on_event(self, event: object) -> None:
            self.events.append(event)

    sink = _CaptureSink()
    state = StrategyState(event_bus=EventBus(sinks=[sink]))
    generated_intent = _new_intent(client_order_id="apply-no-risk-decide")

    class _Evaluator:
        def evaluate(self, context: CoreStepStrategyContext) -> list[NewOrderIntent]:
            assert context.state._last_processing_position_index == 63
            return [generated_intent]

    class _PolicyEvaluator:
        def evaluate_policy_intent(self, **_: object) -> tuple[bool, str | None]:
            return True, None

    def _boom(*_: object, **__: object) -> object:
        raise AssertionError("RiskEngine.decide_intents must not run in apply path")

    monkeypatch.setattr(RiskEngine, "decide_intents", _boom)

    entry = EventStreamEntry(
        position=ProcessingPosition(index=63),
        event=_fill_event(
            instrument="BTC-USDC-PERP",
            client_order_id="fill-apply-no-risk-decide",
            ts_ns_local=63,
            ts_ns_exch=62,
        ),
    )
    result = run_core_step(
        state,
        entry,
        strategy_evaluator=_Evaluator(),
        policy_admission_context=CorePolicyAdmissionContext(
            policy_evaluator=_PolicyEvaluator(),  # type: ignore[arg-type]
            now_ts_ns_local=63,
        ),
        execution_control_apply_context=CoreExecutionControlApplyContext(
            execution_control=ExecutionControl(),
            now_ts_ns_local=63,
        ),
    )

    assert result.core_step_decision is not None
    assert result.compat_gate_decision is None
    assert all(not isinstance(event, RiskDecisionEvent) for event in sink.events)
