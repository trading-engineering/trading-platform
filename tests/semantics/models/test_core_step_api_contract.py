"""Semantics tests for the transitional Core step API skeleton."""

from __future__ import annotations

import copy

import pytest

import tradingchassis_core as tc
import tradingchassis_core.core.domain.processing_step as processing_step_module
from tradingchassis_core.core.domain import run_core_step as domain_run_core_step
from tradingchassis_core.core.domain.event_model import (
    canonical_category_for_type,
    is_canonical_stream_candidate_type,
)
from tradingchassis_core.core.domain.policy_risk_decision import PolicyRiskDecision
from tradingchassis_core.core.domain.processing import process_event_entry
from tradingchassis_core.core.domain.processing_order import EventStreamEntry, ProcessingPosition
from tradingchassis_core.core.domain.processing_step import (
    ControlTimeQueueReevaluationContext,
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
    OrderStateEvent,
    Price,
    Quantity,
)
from tradingchassis_core.core.events.sinks.null_event_bus import NullEventBus
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation
from tradingchassis_core.core.risk.risk_engine import GateDecision, RejectedIntent


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
    assert tuple(it.client_order_id for it in result.candidate_intents) == ("queued-1",)
    assert isinstance(result.core_step_decision, CoreStepDecision)
    assert tuple(
        it.client_order_id for it in result.core_step_decision.dispatchable_intents
    ) == ("accepted-now",)
    assert result.core_step_decision.control_scheduling_obligation is not None
    assert result.core_step_decision.control_scheduling_obligation.due_ts_ns_local == 17
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
        "combine_candidate_intents",
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
        "combine_candidate_intents",
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
