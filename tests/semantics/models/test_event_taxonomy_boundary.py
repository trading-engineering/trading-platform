"""Semantics tests for the lightweight core event taxonomy boundary."""

from __future__ import annotations

from tradingchassis_core.core.domain.event_model import (
    CANONICAL_EVENT_CATEGORY_NAMES,
    COMPATIBILITY_PROJECTION_TYPES,
    NON_CANONICAL_CONTROL_HELPER_TYPES,
    TELEMETRY_EVENT_TYPES,
    CanonicalEventCategory,
    canonical_category_for_type,
    is_canonical_stream_candidate_type,
)
from tradingchassis_core.core.domain.processing import process_canonical_event
from tradingchassis_core.core.domain.state import StrategyState
from tradingchassis_core.core.domain.types import (
    ControlTimeEvent,
    FillEvent,
    MarketEvent,
    OrderExecutionFeedbackEvent,
    OrderStateEvent,
    OrderSubmittedEvent,
)
from tradingchassis_core.core.events.event_bus import EventBus
from tradingchassis_core.core.events.events import (
    DerivedFillEvent,
    DerivedPnLEvent,
    ExposureDerivedEvent,
    OrderStateTransitionEvent,
    RiskDecisionEvent,
)
from tradingchassis_core.core.events.sinks.null_event_bus import NullEventBus
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation
from tradingchassis_core.core.risk.risk_engine import GateDecision


def test_canonical_event_category_names_are_stable() -> None:
    """Canonical category names remain docs-aligned and stable."""

    assert CANONICAL_EVENT_CATEGORY_NAMES == (
        "market",
        "intent_related",
        "execution",
        "control",
    )


def test_canonical_stream_candidate_classification_current_slice() -> None:
    """Current slice markers keep canonical candidates explicit and minimal."""

    assert is_canonical_stream_candidate_type(MarketEvent) is True
    assert canonical_category_for_type(MarketEvent) == CanonicalEventCategory.MARKET

    assert is_canonical_stream_candidate_type(FillEvent) is True
    assert canonical_category_for_type(FillEvent) == CanonicalEventCategory.EXECUTION

    assert is_canonical_stream_candidate_type(OrderExecutionFeedbackEvent) is True
    assert (
        canonical_category_for_type(OrderExecutionFeedbackEvent)
        == CanonicalEventCategory.EXECUTION
    )

    assert is_canonical_stream_candidate_type(OrderSubmittedEvent) is True
    assert (
        canonical_category_for_type(OrderSubmittedEvent)
        == CanonicalEventCategory.INTENT_RELATED
    )

    assert is_canonical_stream_candidate_type(ControlTimeEvent) is True
    assert canonical_category_for_type(ControlTimeEvent) == CanonicalEventCategory.CONTROL

    # Compatibility execution feedback remains non-canonical in this slice.
    assert is_canonical_stream_candidate_type(OrderStateEvent) is False
    assert OrderStateEvent in COMPATIBILITY_PROJECTION_TYPES


def test_event_bus_is_not_canonical_stream_record() -> None:
    """EventBus remains a transport abstraction, not a canonical event."""

    assert is_canonical_stream_candidate_type(EventBus) is False
    assert canonical_category_for_type(EventBus) is None


def test_gate_decision_is_not_canonical_stream_record() -> None:
    """GateDecision remains a compatibility decision contract, not an event."""

    assert is_canonical_stream_candidate_type(GateDecision) is False
    assert canonical_category_for_type(GateDecision) is None


def test_control_scheduling_obligation_is_not_an_event() -> None:
    """ControlSchedulingObligation is explicitly non-canonical."""

    assert is_canonical_stream_candidate_type(ControlSchedulingObligation) is False
    assert canonical_category_for_type(ControlSchedulingObligation) is None
    assert ControlSchedulingObligation in NON_CANONICAL_CONTROL_HELPER_TYPES


def test_telemetry_records_are_not_canonical_stream_candidates() -> None:
    """Telemetry/observability records remain outside canonical stream markers."""

    telemetry_types = (
        RiskDecisionEvent,
        DerivedPnLEvent,
        ExposureDerivedEvent,
        OrderStateTransitionEvent,
    )

    for record_type in telemetry_types:
        assert record_type in TELEMETRY_EVENT_TYPES
        assert is_canonical_stream_candidate_type(record_type) is False
        assert canonical_category_for_type(record_type) is None

    # Compatibility projection artifact is also non-canonical.
    assert DerivedFillEvent in COMPATIBILITY_PROJECTION_TYPES
    assert is_canonical_stream_candidate_type(DerivedFillEvent) is False
    assert canonical_category_for_type(DerivedFillEvent) is None


def test_process_canonical_event_rejects_order_state_event_guard() -> None:
    """Canonical processing boundary rejects compatibility OrderStateEvent records."""

    state = StrategyState(event_bus=NullEventBus())
    compatibility_record = OrderStateEvent(
        ts_ns_local=1,
        ts_ns_exch=1,
        instrument="BTC-USDC-PERP",
        client_order_id="compat-1",
        order_type="limit",
        state_type="accepted",
        side="buy",
        intended_price={"currency": "USDC", "value": 100.0},
        filled_price=None,
        intended_qty={"unit": "contracts", "value": 1.0},
        cum_filled_qty=None,
        remaining_qty=None,
        time_in_force="GTC",
        reason=None,
        raw={"req": 0, "source": "snapshot"},
    )

    try:
        process_canonical_event(state, compatibility_record)
    except TypeError as exc:
        assert "Unsupported non-canonical event type" in str(exc)
    else:
        raise AssertionError("Expected process_canonical_event to reject OrderStateEvent")


def test_process_canonical_event_rejects_derived_fill_event_guard() -> None:
    """Canonical processing boundary rejects compatibility DerivedFillEvent records."""

    state = StrategyState(event_bus=NullEventBus())
    compatibility_record = DerivedFillEvent(
        ts_ns_local=1,
        instrument="BTC-USDC-PERP",
        client_order_id="compat-derived-1",
        side="buy",
        delta_qty=0.1,
        cum_qty=0.1,
        price=100.5,
    )

    try:
        process_canonical_event(state, compatibility_record)
    except TypeError as exc:
        assert "Unsupported non-canonical event type" in str(exc)
    else:
        raise AssertionError("Expected process_canonical_event to reject DerivedFillEvent")


def test_process_canonical_event_rejects_control_scheduling_obligation_guard() -> None:
    """Canonical processing boundary rejects non-canonical control obligations."""

    state = StrategyState(event_bus=NullEventBus())
    non_canonical_helper = ControlSchedulingObligation(
        due_ts_ns_local=1_000_000_000,
        reason="rate_limit",
        scope_key="instrument:BTC-USDC-PERP",
        source="execution_control_rate_limit",
    )

    try:
        process_canonical_event(state, non_canonical_helper)
    except TypeError as exc:
        assert "Unsupported non-canonical event type" in str(exc)
    else:
        raise AssertionError(
            "Expected process_canonical_event to reject ControlSchedulingObligation"
        )

