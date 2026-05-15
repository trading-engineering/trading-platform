"""Core-only example: built-in Risk Engine as ``PolicyIntentEvaluator``.

Demonstrates the full policy + Execution Control apply path. The main quickstart
(``examples/core_step_quickstart.py``) intentionally uses a minimal inline
allow-all policy; use this example when you want the provided Risk Engine gates.

Core returns ``dispatchable_intents``; a Runtime performs Execution and Venue
dispatch later. Core does not dispatch externally.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tradingchassis_core as tc
from tradingchassis_core.core.domain.types import NotionalLimits

INSTRUMENT = "BTC-USDC-PERP"


# Strategy Evaluator: emits one generated Intent for this Core step.
class _OneIntentEvaluator:
    def evaluate(self, context: object) -> list[tc.NewOrderIntent]:
        _ = context
        return [
            tc.NewOrderIntent(
                intent_type="new",
                ts_ns_local=1_000,
                instrument=INSTRUMENT,
                client_order_id="risk-example-intent",
                intents_correlation_id="corr-risk-example",
                side="buy",
                order_type="limit",
                intended_qty=tc.Quantity(value=1.0, unit="contracts"),
                intended_price=tc.Price(currency="USDC", value=100.0),
                time_in_force="GTC",
            )
        ]


def _control_entry(index: int, ts_ns_local: int) -> tc.EventStreamEntry:
    # Control-Time Event as a simple canonical driver (scheduling obligations come from apply).
    return tc.EventStreamEntry(
        position=tc.ProcessingPosition(index=index),
        event=tc.ControlTimeEvent(
            ts_ns_local_control=ts_ns_local,
            reason="scheduled_control_recheck",
            due_ts_ns_local=ts_ns_local,
            realized_ts_ns_local=ts_ns_local,
            obligation_reason="rate_limit",
            obligation_due_ts_ns_local=ts_ns_local,
            runtime_correlation=None,
        ),
    )


def _risk_config() -> tc.RiskConfig:
    return tc.RiskConfig(
        scope="example",
        trading_enabled=True,
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


def main() -> None:
    state = tc.StrategyState(event_bus=tc.NullEventBus())
    state.update_market(
        instrument=INSTRUMENT,
        best_bid=99.0,
        best_ask=101.0,
        best_bid_qty=1.0,
        best_ask_qty=1.0,
        tick_size=0.1,
        lot_size=0.01,
        contract_size=1.0,
        ts_ns_local=1_000,
        ts_ns_exch=999,
    )

    # Built-in Risk Engine (`RiskEngine`) satisfies PolicyIntentEvaluator for policy admission.
    policy_engine = tc.RiskEngine(_risk_config())
    result = tc.run_core_step(
        state,
        _control_entry(0, 1_000),
        strategy_evaluator=_OneIntentEvaluator(),
        policy_admission_context=tc.CorePolicyAdmissionContext(
            policy_evaluator=policy_engine,
            now_ts_ns_local=1_000,
        ),
        execution_control_apply_context=tc.CoreExecutionControlApplyContext(
            execution_control=tc.ExecutionControl(),
            now_ts_ns_local=1_000,
            activate_dispatchable_outputs=True,
        ),
    )

    print("CoreStep with Risk Engine (Core-only; Runtime dispatches later)")
    print("generated:", [i.client_order_id for i in result.generated_intents])
    print("dispatchable:", [i.client_order_id for i in result.dispatchable_intents])


if __name__ == "__main__":
    main()
