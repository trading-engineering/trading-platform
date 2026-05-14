"""Control scheduling obligation semantics: rate-limit vs inflight deferral.

See ``docs/flows/control-time-and-scheduling.md`` for the normative description.
"""

from __future__ import annotations

import tradingchassis_core as tc


class _AllowAllPolicy:
    def evaluate_policy_intent(
        self,
        *,
        intent: tc.OrderIntent,
        state: tc.StrategyState,
        now_ts_ns_local: int,
    ) -> tuple[bool, str | None]:
        _ = (intent, state, now_ts_ns_local)
        return True, None


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


def _order_submitted_entry(
    index: int,
    ts_dispatch: int,
    *,
    client_order_id: str = "order-a",
    price: float = 100.0,
) -> tc.EventStreamEntry:
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=tc.OrderSubmittedEvent(
            ts_ns_local_dispatch=ts_dispatch,
            instrument="BTC-USDC-PERP",
            client_order_id=client_order_id,
            side="buy",
            order_type="limit",
            intended_price=tc.Price(currency="USDC", value=price),
            intended_qty=tc.Quantity(value=1.0, unit="contracts"),
            time_in_force="GTC",
            intent_correlation_id=None,
            dispatch_attempt_id=None,
            runtime_correlation=None,
        ),
    )


class _NewIntentEvaluator:
    def evaluate(self, context: object) -> list[tc.NewOrderIntent]:
        _ = context
        return [
            tc.NewOrderIntent(
                intent_type="new",
                ts_ns_local=10,
                instrument="BTC-USDC-PERP",
                client_order_id="intent-1",
                intents_correlation_id="corr-1",
                side="buy",
                order_type="limit",
                intended_qty=tc.Quantity(value=1.0, unit="contracts"),
                intended_price=tc.Price(currency="USDC", value=100.0),
                time_in_force="GTC",
            )
        ]


class _ReplaceIntentEvaluator:
    """Emits a single replace against ``order-a`` (requires a working order)."""

    def evaluate(self, context: object) -> list[tc.ReplaceOrderIntent]:
        _ = context
        return [
            tc.ReplaceOrderIntent(
                intent_type="replace",
                ts_ns_local=100,
                instrument="BTC-USDC-PERP",
                client_order_id="order-a",
                intents_correlation_id="corr-repl",
                side="buy",
                order_type="limit",
                intended_qty=tc.Quantity(value=1.0, unit="contracts"),
                intended_price=tc.Price(currency="USDC", value=99.0),
            )
        ]


def _policy_and_apply(
    *,
    now_ts: int,
    max_orders_per_sec: float | None = None,
) -> tuple[tc.CorePolicyAdmissionContext, tc.CoreExecutionControlApplyContext]:
    return (
        tc.CorePolicyAdmissionContext(
            policy_evaluator=_AllowAllPolicy(),
            now_ts_ns_local=now_ts,
        ),
        tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=now_ts,
            max_orders_per_sec=max_orders_per_sec,
            activate_dispatchable_outputs=True,
        ),
    )


def test_rate_limit_deferral_emits_control_scheduling_obligation() -> None:
    """Time-dependent rate limiting produces a non-canonical scheduling obligation."""
    now_ts = 100
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    policy_ctx, apply_ctx = _policy_and_apply(now_ts=now_ts, max_orders_per_sec=0.0)

    result = tc.run_core_step(
        state,
        _control_entry(0, now_ts),
        strategy_evaluator=_NewIntentEvaluator(),
        policy_admission_context=policy_ctx,
        execution_control_apply_context=apply_ctx,
    )

    assert result.dispatchable_intents == ()
    obl = result.control_scheduling_obligation
    assert obl is not None
    assert obl.reason == "rate_limit"
    assert obl.source == "execution_control_rate_limit"
    assert obl.due_ts_ns_local >= now_ts
    assert obl.scope_key == "instrument:BTC-USDC-PERP"


def test_inflight_deferral_does_not_emit_control_scheduling_obligation() -> None:
    """Inflight gating is feedback-dependent; Core does not emit a wake obligation."""
    now_ts = 100
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    tc.process_event_entry(state, _order_submitted_entry(0, now_ts))
    assert state.has_working_order("BTC-USDC-PERP", "order-a")

    state.mark_intent_sent("BTC-USDC-PERP", "order-a", "replace")
    assert state.has_inflight("BTC-USDC-PERP", "order-a")

    policy_ctx, apply_ctx = _policy_and_apply(now_ts=now_ts, max_orders_per_sec=None)

    result = tc.run_core_step(
        state,
        _control_entry(1, now_ts),
        strategy_evaluator=_ReplaceIntentEvaluator(),
        policy_admission_context=policy_ctx,
        execution_control_apply_context=apply_ctx,
    )

    assert result.control_scheduling_obligation is None
    assert result.dispatchable_intents == ()
    assert result.core_step_decision is not None
    assert len(result.core_step_decision.queued_effective_intents) >= 1
    queued = state.queued_intents.get("BTC-USDC-PERP")
    assert queued is not None and len(queued) == 1
    assert queued[0].intent.intent_type == "replace"


def test_inflight_queued_replace_reprocessed_after_order_submitted_feedback() -> None:
    """Canonical OrderSubmittedEvent clears inflight so a later step can dispatch queue."""
    now_ts = 100
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    tc.process_event_entry(state, _order_submitted_entry(0, now_ts))
    state.mark_intent_sent("BTC-USDC-PERP", "order-a", "replace")

    policy_ctx, apply_ctx = _policy_and_apply(now_ts=now_ts, max_orders_per_sec=None)
    blocked = tc.run_core_step(
        state,
        _control_entry(1, now_ts),
        strategy_evaluator=_ReplaceIntentEvaluator(),
        policy_admission_context=policy_ctx,
        execution_control_apply_context=apply_ctx,
    )
    assert blocked.dispatchable_intents == ()
    assert blocked.control_scheduling_obligation is None

    policy_ctx2, apply_ctx2 = _policy_and_apply(now_ts=now_ts + 1, max_orders_per_sec=None)
    cleared = tc.run_core_step(
        state,
        _order_submitted_entry(2, now_ts + 1, price=99.0),
        policy_admission_context=policy_ctx2,
        execution_control_apply_context=apply_ctx2,
    )

    assert not state.has_inflight("BTC-USDC-PERP", "order-a")
    assert len(cleared.dispatchable_intents) == 1
    assert cleared.dispatchable_intents[0].intent_type == "replace"
    assert cleared.control_scheduling_obligation is None
