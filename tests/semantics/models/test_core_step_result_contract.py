"""Semantics tests for the CoreStepResult contract model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import tradingchassis_core as tc
from tradingchassis_core.core.domain.event_model import (
    canonical_category_for_type,
    is_canonical_stream_candidate_type,
)
from tradingchassis_core.core.domain.processing import process_canonical_event
from tradingchassis_core.core.domain.state import StrategyState
from tradingchassis_core.core.domain.step_decision import CoreStepDecision
from tradingchassis_core.core.domain.step_result import CoreStepResult
from tradingchassis_core.core.domain.types import (
    CancelOrderIntent,
    NewOrderIntent,
    Price,
    Quantity,
    ReplaceOrderIntent,
)
from tradingchassis_core.core.events.sinks.null_event_bus import NullEventBus
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation
from tradingchassis_core.core.risk.risk_engine import GateDecision


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


def test_default_result_is_empty_and_none_compat() -> None:
    result = CoreStepResult()

    assert result.generated_intents == ()
    assert result.candidate_intents == ()
    assert result.dispatchable_intents == ()
    assert result.control_scheduling_obligation is None
    assert result.core_step_decision is None
    assert result.compat_gate_decision is None


def test_result_is_immutable() -> None:
    result = CoreStepResult()

    with pytest.raises(FrozenInstanceError):
        result.compat_gate_decision = None


def test_dispatchable_intents_normalize_to_tuple() -> None:
    intent_one = _new_intent(client_order_id="new-1")
    intent_two = _new_intent(client_order_id="new-2")

    result = CoreStepResult(dispatchable_intents=[intent_one, intent_two])

    assert isinstance(result.dispatchable_intents, tuple)
    assert result.dispatchable_intents == (intent_one, intent_two)


def test_generated_intents_normalize_to_tuple() -> None:
    intent_one = _new_intent(client_order_id="generated-1")
    intent_two = _new_intent(client_order_id="generated-2")

    result = CoreStepResult(generated_intents=[intent_one, intent_two])

    assert isinstance(result.generated_intents, tuple)
    assert result.generated_intents == (intent_one, intent_two)


def test_generated_intents_are_distinct_from_dispatchable_intents() -> None:
    generated = _new_intent(client_order_id="generated-only")
    candidate = _new_intent(client_order_id="candidate-only")
    dispatchable = _new_intent(client_order_id="dispatchable-only")

    result = CoreStepResult(
        generated_intents=[generated],
        candidate_intents=[candidate],
        dispatchable_intents=[dispatchable],
    )

    assert result.generated_intents == (generated,)
    assert result.candidate_intents == (candidate,)
    assert result.dispatchable_intents == (dispatchable,)


def test_generated_intents_accept_new_replace_cancel_intents() -> None:
    new_intent = _new_intent(client_order_id="new-intent")
    replace_intent = ReplaceOrderIntent(
        ts_ns_local=2,
        instrument="BTC-USDC-PERP",
        client_order_id="replace-intent",
        intents_correlation_id="corr-replace",
        side="buy",
        order_type="limit",
        intended_qty=Quantity(value=2.0, unit="contracts"),
        intended_price=Price(currency="USDC", value=101.0),
    )
    cancel_intent = CancelOrderIntent(
        ts_ns_local=3,
        instrument="BTC-USDC-PERP",
        client_order_id="cancel-intent",
        intents_correlation_id="corr-cancel",
    )

    result = CoreStepResult(
        generated_intents=[new_intent, replace_intent, cancel_intent],
    )

    assert result.generated_intents == (new_intent, replace_intent, cancel_intent)


def test_candidate_intents_normalize_to_tuple() -> None:
    intent_one = _new_intent(client_order_id="candidate-1")
    intent_two = _new_intent(client_order_id="candidate-2")

    result = CoreStepResult(candidate_intents=[intent_one, intent_two])

    assert isinstance(result.candidate_intents, tuple)
    assert result.candidate_intents == (intent_one, intent_two)


def test_candidate_intents_are_not_dispatchable_by_default() -> None:
    candidate = _new_intent(client_order_id="candidate-only")
    result = CoreStepResult(candidate_intents=[candidate])

    assert result.candidate_intents == (candidate,)
    assert result.dispatchable_intents == ()


def test_can_carry_optional_control_scheduling_obligation() -> None:
    obligation = ControlSchedulingObligation(
        due_ts_ns_local=1_000_000_000,
        reason="rate_limit",
        scope_key="instrument:BTC-USDC-PERP",
        source="execution_control_rate_limit",
    )

    result = CoreStepResult(control_scheduling_obligation=obligation)

    assert result.control_scheduling_obligation is obligation


def test_can_carry_optional_compat_gate_decision() -> None:
    accepted_intent = _new_intent(client_order_id="accepted-now")
    compat_decision = GateDecision(
        ts_ns_local=123,
        accepted_now=[accepted_intent],
        queued=[],
        rejected=[],
        replaced_in_queue=[],
        dropped_in_queue=[],
        handled_in_queue=[],
        execution_rejected=[],
        next_send_ts_ns_local=None,
        control_scheduling_obligations=(),
    )

    result = CoreStepResult(compat_gate_decision=compat_decision)

    assert result.compat_gate_decision is compat_decision


def test_can_carry_optional_core_step_decision() -> None:
    dispatchable = _new_intent(client_order_id="dispatchable")
    decision = CoreStepDecision(dispatchable_intents=[dispatchable])

    result = CoreStepResult(core_step_decision=decision)

    assert result.core_step_decision is decision


def test_core_step_result_dispatchable_intents_are_independent_from_core_step_decision() -> None:
    top_level_dispatchable = _new_intent(client_order_id="top-level")
    decision_dispatchable = _new_intent(client_order_id="decision")
    decision = CoreStepDecision(dispatchable_intents=[decision_dispatchable])
    result = CoreStepResult(
        dispatchable_intents=[top_level_dispatchable],
        core_step_decision=decision,
    )

    assert result.dispatchable_intents == (top_level_dispatchable,)
    assert result.core_step_decision is decision
    assert result.core_step_decision.dispatchable_intents == (decision_dispatchable,)


def test_core_step_result_is_non_canonical_and_not_classified() -> None:
    assert is_canonical_stream_candidate_type(CoreStepResult) is False
    assert canonical_category_for_type(CoreStepResult) is None


def test_canonical_processing_boundary_rejects_core_step_result() -> None:
    state = StrategyState(event_bus=NullEventBus())

    with pytest.raises(TypeError, match="Unsupported non-canonical event type"):
        process_canonical_event(state, CoreStepResult())


def test_public_root_export_identity_when_root_exported() -> None:
    assert hasattr(tc, "CoreStepResult")
    assert tc.CoreStepResult is CoreStepResult
