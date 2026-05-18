"""Public API surface checks for clean Core exports."""

from __future__ import annotations

import re
from pathlib import Path

import tradingchassis_core as tc

EXPECTED_ROOT_EXPORTS = (
    "CoreConfiguration",
    "RiskConfig",
    "RiskEngine",
    "StrategyState",
    "StrategyStateView",
    "MarketEvent",
    "ControlTimeEvent",
    "OrderSubmittedEvent",
    "OrderCanceledEvent",
    "OrderRejectedEvent",
    "OrderExpiredEvent",
    "OrderExecutionFeedbackEvent",
    "FillEvent",
    "RiskConstraints",
    "NotionalLimits",
    "OrderIntent",
    "NewOrderIntent",
    "CancelOrderIntent",
    "ReplaceOrderIntent",
    "Price",
    "Quantity",
    "SlotKey",
    "stable_slot_order_id",
    "CandidateIntentOrigin",
    "CandidateIntentRecord",
    "ProcessingPosition",
    "EventStreamEntry",
    "process_canonical_event",
    "process_event_entry",
    "run_core_step",
    "run_core_wakeup_reduction",
    "run_core_wakeup_decision",
    "run_core_wakeup_step",
    "CoreStepStrategyContext",
    "CoreStepStrategyEvaluator",
    "CoreWakeupStrategyContext",
    "CoreWakeupStrategyEvaluator",
    "CoreExecutionControlApplyContext",
    "CorePolicyAdmissionContext",
    "CoreWakeupReductionResult",
    "ExecutionControlDecision",
    "PolicyIntentEvaluator",
    "PolicyRiskDecision",
    "PolicyRejectedCandidate",
    "PolicyAdmissionResult",
    "CoreStepDecision",
    "CoreStepResult",
    "ExecutionControl",
    "ControlSchedulingObligation",
    "NullEventBus",
    "__version__",
)


def test_public_api_exposes_clean_core_symbols() -> None:
    for symbol in EXPECTED_ROOT_EXPORTS:
        assert hasattr(tc, symbol), symbol


def test_public_api_docs_mention_root_exports() -> None:
    docs_path = Path(__file__).resolve().parents[2] / "docs" / "reference" / "public-api.md"
    docs_content = docs_path.read_text(encoding="utf-8")
    documented_symbols = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", docs_content))

    missing = sorted(symbol for symbol in EXPECTED_ROOT_EXPORTS if symbol not in documented_symbols)
    assert missing == [], f"Root exports missing from public-api docs: {missing}"


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
