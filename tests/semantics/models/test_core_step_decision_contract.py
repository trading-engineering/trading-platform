"""Semantics tests for the CoreStepDecision scaffold contract model."""

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
from tradingchassis_core.core.domain.types import NewOrderIntent, Price, Quantity
from tradingchassis_core.core.events.sinks.null_event_bus import NullEventBus
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation


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


def test_default_decision_is_empty_and_none_obligation() -> None:
    decision = CoreStepDecision()

    assert decision.policy_rejected_intents == ()
    assert decision.queued_effective_intents == ()
    assert decision.dispatchable_intents == ()
    assert decision.execution_handled_intents == ()
    assert decision.control_scheduling_obligation is None


def test_core_step_decision_tuple_fields_normalize() -> None:
    rejected = _new_intent(client_order_id="rejected")
    queued = _new_intent(client_order_id="queued")
    dispatchable = _new_intent(client_order_id="dispatchable")
    handled = _new_intent(client_order_id="handled")

    decision = CoreStepDecision(
        policy_rejected_intents=[rejected],
        queued_effective_intents=[queued],
        dispatchable_intents=[dispatchable],
        execution_handled_intents=[handled],
    )

    assert decision.policy_rejected_intents == (rejected,)
    assert decision.queued_effective_intents == (queued,)
    assert decision.dispatchable_intents == (dispatchable,)
    assert decision.execution_handled_intents == (handled,)


def test_core_step_decision_is_immutable() -> None:
    decision = CoreStepDecision()

    with pytest.raises(FrozenInstanceError):
        decision.control_scheduling_obligation = None


def test_core_step_decision_is_non_canonical_and_not_classified() -> None:
    assert is_canonical_stream_candidate_type(CoreStepDecision) is False
    assert canonical_category_for_type(CoreStepDecision) is None


def test_canonical_processing_boundary_rejects_core_step_decision() -> None:
    state = StrategyState(event_bus=NullEventBus())

    with pytest.raises(TypeError, match="Unsupported non-canonical event type"):
        process_canonical_event(state, CoreStepDecision())


def test_core_step_decision_can_carry_dispatchable_and_obligation() -> None:
    dispatchable = _new_intent(client_order_id="dispatchable")
    obligation = ControlSchedulingObligation(
        due_ts_ns_local=1_000_000_000,
        reason="rate_limit",
        scope_key="instrument:BTC-USDC-PERP",
        source="execution_control_rate_limit",
    )
    decision = CoreStepDecision(
        dispatchable_intents=[dispatchable],
        control_scheduling_obligation=obligation,
    )

    assert decision.dispatchable_intents == (dispatchable,)
    assert decision.control_scheduling_obligation is obligation


def test_public_root_export_identity_when_root_exported() -> None:
    assert hasattr(tc, "CoreStepDecision")
    assert tc.CoreStepDecision is CoreStepDecision
