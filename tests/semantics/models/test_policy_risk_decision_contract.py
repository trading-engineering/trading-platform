"""Semantics tests for the PolicyRiskDecision scaffold contract model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import tradingchassis_core as tc
from tradingchassis_core.core.domain.event_model import (
    canonical_category_for_type,
    is_canonical_stream_candidate_type,
)
from tradingchassis_core.core.domain.policy_risk_decision import (
    PolicyRiskDecision,
    map_compat_gate_decision_to_policy_risk_decision,
)
from tradingchassis_core.core.domain.processing import process_canonical_event
from tradingchassis_core.core.domain.state import StrategyState
from tradingchassis_core.core.domain.types import NewOrderIntent, Price, Quantity
from tradingchassis_core.core.events.sinks.null_event_bus import NullEventBus
from tradingchassis_core.core.risk.risk_engine import GateDecision, RejectedIntent


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


def test_default_policy_risk_decision_is_empty() -> None:
    decision = PolicyRiskDecision()
    assert decision.accepted_intents == ()
    assert decision.rejected_intents == ()


def test_policy_risk_decision_tuple_fields_normalize() -> None:
    accepted = _new_intent(client_order_id="accepted")
    rejected = _new_intent(client_order_id="rejected")
    decision = PolicyRiskDecision(
        accepted_intents=[accepted],
        rejected_intents=[rejected],
    )
    assert decision.accepted_intents == (accepted,)
    assert decision.rejected_intents == (rejected,)


def test_policy_risk_decision_is_immutable() -> None:
    decision = PolicyRiskDecision()
    with pytest.raises(FrozenInstanceError):
        decision.accepted_intents = ()


def test_policy_risk_decision_is_non_canonical_and_not_classified() -> None:
    assert is_canonical_stream_candidate_type(PolicyRiskDecision) is False
    assert canonical_category_for_type(PolicyRiskDecision) is None


def test_canonical_processing_boundary_rejects_policy_risk_decision() -> None:
    state = StrategyState(event_bus=NullEventBus())
    with pytest.raises(TypeError, match="Unsupported non-canonical event type"):
        process_canonical_event(state, PolicyRiskDecision())


def test_map_compat_gate_decision_to_policy_risk_decision_projection() -> None:
    accepted_now = _new_intent(client_order_id="accepted-now")
    queued = _new_intent(client_order_id="queued")
    rejected = _new_intent(client_order_id="rejected")
    decision = GateDecision(
        ts_ns_local=123,
        accepted_now=[accepted_now],
        queued=[queued],
        rejected=[RejectedIntent(intent=rejected, reason="policy_reject")],
        replaced_in_queue=[],
        dropped_in_queue=[],
        handled_in_queue=[],
        execution_rejected=[],
        next_send_ts_ns_local=None,
        control_scheduling_obligations=(),
    )

    policy = map_compat_gate_decision_to_policy_risk_decision(decision)

    assert policy.accepted_intents == (accepted_now,)
    assert policy.rejected_intents == (rejected,)


def test_public_root_export_identity_when_root_exported() -> None:
    assert hasattr(tc, "PolicyRiskDecision")
    assert tc.PolicyRiskDecision is PolicyRiskDecision
