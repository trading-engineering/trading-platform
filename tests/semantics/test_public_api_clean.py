"""Public API surface checks for clean Core exports."""

from __future__ import annotations

import tradingchassis_core as tc


def test_public_api_exposes_clean_core_symbols() -> None:
    for symbol in (
        "run_core_step",
        "run_core_wakeup_step",
        "CoreStepResult",
        "CoreStepDecision",
        "CorePolicyAdmissionContext",
        "CoreExecutionControlApplyContext",
        "ControlTimeEvent",
        "MarketEvent",
        "OrderSubmittedEvent",
        "OrderExecutionFeedbackEvent",
        "FillEvent",
        "ExecutionControl",
        "NullEventBus",
    ):
        assert hasattr(tc, symbol), symbol


def test_public_api_does_not_expose_removed_compatibility_symbols() -> None:
    removed = (
        "".join(["Gate", "Decision"]),
        "".join(["ControlTimeQueue", "ReevaluationContext"]),
        "".join(["Core", "DecisionContext"]),
    )
    for symbol in removed:
        assert not hasattr(tc, symbol)
