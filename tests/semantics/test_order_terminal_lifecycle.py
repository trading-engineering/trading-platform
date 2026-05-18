"""Terminal order lifecycle canonical Event reduction coverage."""

from __future__ import annotations

import pytest

import tradingchassis_core as tc

INSTRUMENT = "BTC-USDC-PERP"
ORDER_ID = "terminal-order-1"


def _submitted_entry(index: int, ts: int) -> tc.EventStreamEntry:
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=tc.OrderSubmittedEvent(
            ts_ns_local_dispatch=ts,
            instrument=INSTRUMENT,
            client_order_id=ORDER_ID,
            side="buy",
            order_type="limit",
            intended_price=tc.Price(currency="USDC", value=100.0),
            intended_qty=tc.Quantity(value=1.0, unit="contracts"),
            time_in_force="GTC",
            intent_correlation_id=None,
            dispatch_attempt_id=None,
            runtime_correlation=None,
        ),
    )


def _control_entry(index: int, ts: int) -> tc.EventStreamEntry:
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=tc.ControlTimeEvent(
            ts_ns_local_control=ts,
            reason="scheduled_control_recheck",
            due_ts_ns_local=ts,
            realized_ts_ns_local=ts,
            obligation_reason="rate_limit",
            obligation_due_ts_ns_local=ts,
            runtime_correlation=None,
        ),
    )


def _terminal_event(kind: str, ts: int) -> object:
    if kind == "canceled":
        return tc.OrderCanceledEvent(
            ts_ns_local_feedback=ts,
            instrument=INSTRUMENT,
            client_order_id=ORDER_ID,
        )
    if kind == "rejected":
        return tc.OrderRejectedEvent(
            ts_ns_local_feedback=ts,
            instrument=INSTRUMENT,
            client_order_id=ORDER_ID,
        )
    if kind == "expired":
        return tc.OrderExpiredEvent(
            ts_ns_local_feedback=ts,
            instrument=INSTRUMENT,
            client_order_id=ORDER_ID,
        )
    raise AssertionError(f"unsupported terminal kind: {kind}")


def _terminal_entry(index: int, kind: str, ts: int) -> tc.EventStreamEntry:
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=_terminal_event(kind, ts),
    )


@pytest.mark.parametrize("symbol", ("OrderCanceledEvent", "OrderRejectedEvent", "OrderExpiredEvent"))
def test_terminal_events_are_public_exports(symbol: str) -> None:
    assert hasattr(tc, symbol)


@pytest.mark.parametrize("terminal_kind", ("canceled", "rejected", "expired"))
def test_submitted_then_terminal_event_removes_working_order_updates_projection_and_clears_inflight(
    terminal_kind: str,
) -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    tc.process_event_entry(state, _submitted_entry(0, 100))
    state.mark_intent_sent(INSTRUMENT, ORDER_ID, "replace")
    assert state.has_inflight(INSTRUMENT, ORDER_ID)
    assert state.has_working_order(INSTRUMENT, ORDER_ID)

    tc.process_event_entry(state, _terminal_entry(1, terminal_kind, 101))

    assert not state.has_working_order(INSTRUMENT, ORDER_ID)
    assert not state.has_inflight(INSTRUMENT, ORDER_ID)
    projection = state.canonical_orders[(INSTRUMENT, ORDER_ID)]
    assert projection.state == terminal_kind
    assert projection.updated_ts_ns_local == 101


@pytest.mark.parametrize("terminal_kind", ("canceled", "rejected", "expired"))
def test_process_canonical_event_routes_terminal_events(terminal_kind: str) -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    tc.process_event_entry(state, _submitted_entry(0, 100))
    tc.process_canonical_event(state, _terminal_event(terminal_kind, 101))

    assert not state.has_working_order(INSTRUMENT, ORDER_ID)
    assert state.canonical_orders[(INSTRUMENT, ORDER_ID)].state == terminal_kind


@pytest.mark.parametrize("terminal_kind", ("canceled", "rejected", "expired"))
def test_run_core_step_reduces_terminal_event(terminal_kind: str) -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    tc.process_event_entry(state, _submitted_entry(0, 100))

    result = tc.run_core_step(
        state,
        _terminal_entry(1, terminal_kind, 101),
    )

    assert result.generated_intents == ()
    assert result.dispatchable_intents == ()
    assert not state.has_working_order(INSTRUMENT, ORDER_ID)
    assert state.canonical_orders[(INSTRUMENT, ORDER_ID)].state == terminal_kind


@pytest.mark.parametrize("terminal_kind", ("canceled", "rejected", "expired"))
def test_run_core_wakeup_step_reduces_terminal_events_in_processing_order(
    terminal_kind: str,
) -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())

    class _AssertTerminalReducedEvaluator:
        def evaluate(self, context: tc.CoreWakeupStrategyContext) -> list[tc.OrderIntent]:
            assert context.last_position is not None
            assert context.last_position.index == 2
            assert not context.state.orders.get(INSTRUMENT, {}).get(ORDER_ID)
            return []

    result = tc.run_core_wakeup_step(
        state,
        (
            _submitted_entry(0, 100),
            _control_entry(1, 101),
            _terminal_entry(2, terminal_kind, 102),
        ),
        wakeup_strategy_evaluator=_AssertTerminalReducedEvaluator(),
    )

    assert result.generated_intents == ()
    assert not state.has_working_order(INSTRUMENT, ORDER_ID)
    assert state.canonical_orders[(INSTRUMENT, ORDER_ID)].state == terminal_kind


@pytest.mark.parametrize("terminal_kind", ("canceled", "rejected", "expired"))
def test_repeated_terminal_event_is_idempotent(terminal_kind: str) -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    tc.process_event_entry(state, _submitted_entry(0, 100))
    first = _terminal_entry(1, terminal_kind, 101)
    second = _terminal_entry(2, terminal_kind, 101)

    tc.process_event_entry(state, first)
    projection_after_first = state.canonical_orders[(INSTRUMENT, ORDER_ID)]
    state_snapshot = (
        projection_after_first.state,
        projection_after_first.submitted_ts_ns_local,
        projection_after_first.updated_ts_ns_local,
    )

    tc.process_event_entry(state, second)
    projection_after_second = state.canonical_orders[(INSTRUMENT, ORDER_ID)]
    assert not state.has_working_order(INSTRUMENT, ORDER_ID)
    assert (
        projection_after_second.state,
        projection_after_second.submitted_ts_ns_local,
        projection_after_second.updated_ts_ns_local,
    ) == state_snapshot


@pytest.mark.parametrize("terminal_kind", ("canceled", "rejected", "expired"))
def test_terminal_event_for_unknown_order_is_deterministic_and_non_crashing(
    terminal_kind: str,
) -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())

    tc.process_event_entry(state, _terminal_entry(0, terminal_kind, 100))

    assert not state.has_working_order(INSTRUMENT, ORDER_ID)
    projection = state.canonical_orders[(INSTRUMENT, ORDER_ID)]
    assert projection.state == terminal_kind
    assert projection.submitted_ts_ns_local == 100
    assert projection.updated_ts_ns_local == 100


def test_order_rejected_event_is_distinct_from_policy_admission_rejection() -> None:
    class _RejectAllPolicy:
        def evaluate_policy_intent(
            self,
            *,
            intent: tc.OrderIntent,
            state: tc.StrategyState,
            now_ts_ns_local: int,
        ) -> tuple[bool, str | None]:
            _ = (intent, state, now_ts_ns_local)
            return False, "blocked_for_test"

    class _OneIntentEvaluator:
        def evaluate(self, context: object) -> list[tc.NewOrderIntent]:
            _ = context
            return [
                tc.NewOrderIntent(
                    intent_type="new",
                    ts_ns_local=100,
                    instrument=INSTRUMENT,
                    client_order_id="policy-reject-order",
                    intents_correlation_id="corr-policy-reject",
                    side="buy",
                    order_type="limit",
                    intended_qty=tc.Quantity(value=1.0, unit="contracts"),
                    intended_price=tc.Price(currency="USDC", value=100.0),
                    time_in_force="GTC",
                )
            ]

    state = tc.StrategyState(event_bus=tc.NullEventBus())
    result = tc.run_core_step(
        state,
        _control_entry(0, 100),
        strategy_evaluator=_OneIntentEvaluator(),
        policy_admission_context=tc.CorePolicyAdmissionContext(
            policy_evaluator=_RejectAllPolicy(),
            now_ts_ns_local=100,
        ),
        execution_control_apply_context=tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=100,
            activate_dispatchable_outputs=True,
        ),
    )

    assert result.core_step_decision is not None
    assert len(result.core_step_decision.policy_rejected_intents) == 1
    assert (INSTRUMENT, "policy-reject-order") not in state.canonical_orders
