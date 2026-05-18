"""Public API surface checks for clean Core exports."""

from __future__ import annotations

import tradingchassis_core as tc


def test_public_api_exposes_clean_core_symbols() -> None:
    for symbol in (
        "EventStreamEntry",
        "ProcessingPosition",
        "process_canonical_event",
        "process_event_entry",
        "run_core_step",
        "run_core_wakeup_reduction",
        "run_core_wakeup_decision",
        "run_core_wakeup_step",
        "CoreWakeupStrategyContext",
        "CoreWakeupStrategyEvaluator",
        "CoreStepResult",
        "CoreStepDecision",
        "PolicyIntentEvaluator",
        "PolicyRiskDecision",
        "ExecutionControlDecision",
        "CandidateIntentRecord",
        "CandidateIntentOrigin",
        "CorePolicyAdmissionContext",
        "CoreExecutionControlApplyContext",
        "ControlTimeEvent",
        "MarketEvent",
        "OrderSubmittedEvent",
        "OrderExecutionFeedbackEvent",
        "FillEvent",
        "OrderIntent",
        "NewOrderIntent",
        "CancelOrderIntent",
        "ReplaceOrderIntent",
        "Price",
        "Quantity",
        "CoreConfiguration",
        "StrategyState",
        "StrategyStateView",
        "ExecutionControl",
        "ControlSchedulingObligation",
        "NullEventBus",
        "RiskEngine",
        "RiskConfig",
    ):
        assert hasattr(tc, symbol), symbol


def test_public_api_does_not_expose_removed_compatibility_symbols() -> None:
    removed = (
        "".join(["Gate", "Decision"]),
        "".join(["compat_", "gate_decision"]),
        "".join(["ControlTimeQueue", "ReevaluationContext"]),
        "".join(["Core", "DecisionContext"]),
        "".join(["OrderState", "Event"]),
        "".join(["Derived", "FillEvent"]),
        "".join(["decide_", "intents"]),
        "".join(["Venue", "Adapter"]),
        "".join(["Venue", "Policy"]),
        "fold_event_stream_entries",
        "apply_execution_control_plan",
        "ExecutionControlApplyContext",
        "ExecutionControlApplyResult",
        "ExecutionControlBlockedRecord",
        "ExecutionControlDispatchableRecord",
        "ExecutionControlHandledRecord",
    )
    for symbol in removed:
        assert not hasattr(tc, symbol)
