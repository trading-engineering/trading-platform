"""Semantics tests for the PolicyRiskDecision scaffold contract model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import tradingchassis_core as tc
from tradingchassis_core.core.domain.candidate_intent import (
    CandidateIntentOrigin,
    CandidateIntentRecord,
)
from tradingchassis_core.core.domain.event_model import (
    canonical_category_for_type,
    is_canonical_stream_candidate_type,
)
from tradingchassis_core.core.domain.policy_risk_decision import (
    PolicyAdmissionResult,
    PolicyRejectedCandidate,
    PolicyRiskDecision,
    apply_policy_to_candidate_records,
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


def test_policy_rejected_candidate_is_immutable() -> None:
    record = CandidateIntentRecord(
        intent=_new_intent(client_order_id="generated-rejected"),
        origin=CandidateIntentOrigin.GENERATED,
        logical_key="order:generated-rejected",
        merge_index=0,
        priority=2,
    )
    rejected = PolicyRejectedCandidate(record=record, reason="policy_reject")
    with pytest.raises(FrozenInstanceError):
        rejected.reason = "changed"


def test_policy_admission_result_defaults_are_empty() -> None:
    result = PolicyAdmissionResult()
    assert result.accepted_generated == ()
    assert result.rejected_generated == ()
    assert result.passthrough_queued == ()
    assert result.policy_risk_decision == PolicyRiskDecision()


def test_policy_admission_result_tuple_normalization() -> None:
    generated_record = CandidateIntentRecord(
        intent=_new_intent(client_order_id="generated-accepted"),
        origin=CandidateIntentOrigin.GENERATED,
        logical_key="order:generated-accepted",
        merge_index=1,
        priority=2,
    )
    queued_record = CandidateIntentRecord(
        intent=_new_intent(client_order_id="queued-passthrough"),
        origin=CandidateIntentOrigin.QUEUED,
        logical_key="order:queued-passthrough",
        merge_index=2,
        priority=2,
    )
    rejected = PolicyRejectedCandidate(
        record=generated_record,
        reason="policy_reject",
    )
    result = PolicyAdmissionResult(
        accepted_generated=[generated_record],
        rejected_generated=[rejected],
        passthrough_queued=[queued_record],
    )
    assert result.accepted_generated == (generated_record,)
    assert result.rejected_generated == (rejected,)
    assert result.passthrough_queued == (queued_record,)


def test_policy_admission_result_and_related_models_are_non_canonical() -> None:
    assert is_canonical_stream_candidate_type(PolicyRejectedCandidate) is False
    assert canonical_category_for_type(PolicyRejectedCandidate) is None
    assert is_canonical_stream_candidate_type(PolicyAdmissionResult) is False
    assert canonical_category_for_type(PolicyAdmissionResult) is None


def test_canonical_processing_boundary_rejects_policy_admission_models() -> None:
    state = StrategyState(event_bus=NullEventBus())
    record = CandidateIntentRecord(
        intent=_new_intent(client_order_id="boundary-record"),
        origin=CandidateIntentOrigin.GENERATED,
        logical_key="order:boundary-record",
        merge_index=0,
        priority=2,
    )
    rejected = PolicyRejectedCandidate(record=record, reason="policy_reject")
    with pytest.raises(TypeError, match="Unsupported non-canonical event type"):
        process_canonical_event(state, rejected)
    with pytest.raises(TypeError, match="Unsupported non-canonical event type"):
        process_canonical_event(state, PolicyAdmissionResult())


def test_apply_policy_to_candidate_records_partitions_generated_and_queued() -> None:
    accepted_generated = CandidateIntentRecord(
        intent=_new_intent(client_order_id="accepted-generated"),
        origin=CandidateIntentOrigin.GENERATED,
        logical_key="order:accepted-generated",
        merge_index=0,
        priority=2,
    )
    rejected_generated = CandidateIntentRecord(
        intent=_new_intent(client_order_id="rejected-generated"),
        origin=CandidateIntentOrigin.GENERATED,
        logical_key="order:rejected-generated",
        merge_index=1,
        priority=2,
    )
    passthrough_queued = CandidateIntentRecord(
        intent=_new_intent(client_order_id="queued-record"),
        origin=CandidateIntentOrigin.QUEUED,
        logical_key="order:queued-record",
        merge_index=2,
        priority=2,
    )
    state = StrategyState(event_bus=NullEventBus())
    calls: list[str] = []

    class _Evaluator:
        def evaluate_policy_intent(
            self,
            *,
            intent: NewOrderIntent,
            state: StrategyState,
            now_ts_ns_local: int,
        ) -> tuple[bool, str | None]:
            _ = state
            assert now_ts_ns_local == 123
            calls.append(intent.client_order_id)
            if intent.client_order_id == "rejected-generated":
                return False, "policy_reject"
            return True, None

    result = apply_policy_to_candidate_records(
        (
            accepted_generated,
            rejected_generated,
            passthrough_queued,
        ),
        state=state,
        now_ts_ns_local=123,
        policy_evaluator=_Evaluator(),
    )

    assert calls == ["accepted-generated", "rejected-generated"]
    assert result.accepted_generated == (accepted_generated,)
    assert result.passthrough_queued == (passthrough_queued,)
    assert len(result.rejected_generated) == 1
    assert result.rejected_generated[0].record == rejected_generated
    assert result.rejected_generated[0].reason == "policy_reject"
    assert tuple(it.client_order_id for it in result.policy_risk_decision.accepted_intents) == (
        "accepted-generated",
    )
    assert tuple(it.client_order_id for it in result.policy_risk_decision.rejected_intents) == (
        "rejected-generated",
    )


def test_apply_policy_to_candidate_records_does_not_call_decide_intents() -> None:
    record = CandidateIntentRecord(
        intent=_new_intent(client_order_id="generated-only"),
        origin=CandidateIntentOrigin.GENERATED,
        logical_key="order:generated-only",
        merge_index=0,
        priority=2,
    )

    class _Evaluator:
        def evaluate_policy_intent(
            self,
            *,
            intent: NewOrderIntent,
            state: StrategyState,
            now_ts_ns_local: int,
        ) -> tuple[bool, str | None]:
            _ = (intent, state, now_ts_ns_local)
            return True, None

        def decide_intents(self, **_: object) -> GateDecision:
            raise AssertionError("decide_intents must not be called by policy helper")

    result = apply_policy_to_candidate_records(
        (record,),
        state=StrategyState(event_bus=NullEventBus()),
        now_ts_ns_local=1,
        policy_evaluator=_Evaluator(),  # type: ignore[arg-type]
    )

    assert result.accepted_generated == (record,)
