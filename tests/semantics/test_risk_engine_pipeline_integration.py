"""RiskEngine as PolicyIntentEvaluator inside the real Core step pipeline."""

from __future__ import annotations

import tradingchassis_core as tc
from tradingchassis_core.core.domain.types import NotionalLimits

INSTRUMENT = "BTC-USDC-PERP"
NOW_TS = 100


class _OneNewIntentEvaluator:
    def evaluate(self, context: object) -> list[tc.NewOrderIntent]:
        _ = context
        return [
            tc.NewOrderIntent(
                intent_type="new",
                ts_ns_local=NOW_TS,
                instrument=INSTRUMENT,
                client_order_id="risk-pipeline-intent",
                intents_correlation_id="corr-risk-pipeline",
                side="buy",
                order_type="limit",
                intended_qty=tc.Quantity(value=1.0, unit="contracts"),
                intended_price=tc.Price(currency="USDC", value=100.0),
                time_in_force="GTC",
            )
        ]


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


def _risk_config(*, trading_enabled: bool) -> tc.RiskConfig:
    return tc.RiskConfig(
        scope="test",
        trading_enabled=trading_enabled,
        notional_limits=NotionalLimits(
            currency="USDC",
            max_gross_notional=1_000_000.0,
            max_single_order_notional=1_000_000.0,
        ),
        position_limits=None,
        quote_limits=None,
        order_rate_limits=None,
        max_loss=None,
    )


def _prime_market(state: tc.StrategyState) -> None:
    state.update_market(
        instrument=INSTRUMENT,
        best_bid=99.0,
        best_ask=101.0,
        best_bid_qty=1.0,
        best_ask_qty=1.0,
        tick_size=0.1,
        lot_size=0.01,
        contract_size=1.0,
        ts_ns_local=NOW_TS,
        ts_ns_exch=NOW_TS - 1,
    )


def test_risk_engine_accepts_generated_intent_in_run_core_step() -> None:
    """RiskEngine is a valid optional policy_evaluator for the full step Pipeline."""
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    _prime_market(state)
    policy_engine = tc.RiskEngine(_risk_config(trading_enabled=True))

    result = tc.run_core_step(
        state,
        _control_entry(0, NOW_TS),
        strategy_evaluator=_OneNewIntentEvaluator(),
        policy_admission_context=tc.CorePolicyAdmissionContext(
            policy_evaluator=policy_engine,
            now_ts_ns_local=NOW_TS,
        ),
        execution_control_apply_context=tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=NOW_TS,
            activate_dispatchable_outputs=True,
        ),
    )

    assert tuple(i.client_order_id for i in result.generated_intents) == ("risk-pipeline-intent",)
    assert result.core_step_decision is not None
    policy_decision = result.core_step_decision.policy_risk_decision
    assert policy_decision is not None
    assert tuple(i.client_order_id for i in policy_decision.accepted_intents) == (
        "risk-pipeline-intent",
    )
    assert policy_decision.rejected_intents == ()
    assert tuple(i.client_order_id for i in result.dispatchable_intents) == (
        "risk-pipeline-intent",
    )


def test_risk_engine_rejects_generated_intent_when_trading_disabled() -> None:
    """Trading-disabled RiskConfig rejects new intents through policy admission."""
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    _prime_market(state)
    policy_engine = tc.RiskEngine(_risk_config(trading_enabled=False))

    result = tc.run_core_step(
        state,
        _control_entry(0, NOW_TS),
        strategy_evaluator=_OneNewIntentEvaluator(),
        policy_admission_context=tc.CorePolicyAdmissionContext(
            policy_evaluator=policy_engine,
            now_ts_ns_local=NOW_TS,
        ),
        execution_control_apply_context=tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=NOW_TS,
            activate_dispatchable_outputs=True,
        ),
    )

    assert tuple(i.client_order_id for i in result.generated_intents) == ("risk-pipeline-intent",)
    assert len(result.candidate_intent_records) == 1
    assert result.candidate_intent_records[0].origin is tc.CandidateIntentOrigin.GENERATED
    assert result.dispatchable_intents == ()
    assert result.core_step_decision is not None
    assert len(result.core_step_decision.policy_rejected_intents) == 1
    policy_decision = result.core_step_decision.policy_risk_decision
    assert policy_decision is not None
    assert policy_decision.accepted_intents == ()
    assert len(policy_decision.rejected_intents) == 1
