"""Public API for the tradingchassis_core package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from tradingchassis_core.core.domain.candidate_intent import (
    CandidateIntentOrigin,
    CandidateIntentRecord,
)
from tradingchassis_core.core.domain.configuration import CoreConfiguration
from tradingchassis_core.core.domain.execution_control_apply import (
    ExecutionControlApplyContext,
    ExecutionControlApplyResult,
    ExecutionControlBlockedRecord,
    ExecutionControlDispatchableRecord,
    ExecutionControlHandledRecord,
    apply_execution_control_plan,
)
from tradingchassis_core.core.domain.execution_control_decision import (
    ExecutionControlDecision,
)
from tradingchassis_core.core.domain.policy_risk_decision import (
    PolicyAdmissionResult,
    PolicyRejectedCandidate,
    PolicyRiskDecision,
)
from tradingchassis_core.core.domain.processing import (
    fold_event_stream_entries,
    process_canonical_event,
    process_event_entry,
)
from tradingchassis_core.core.domain.processing_order import (
    EventStreamEntry,
    ProcessingPosition,
)
from tradingchassis_core.core.domain.processing_step import (
    CoreExecutionControlApplyContext,
    CorePolicyAdmissionContext,
    CoreStepStrategyContext,
    CoreStepStrategyEvaluator,
    CoreWakeupReductionResult,
    run_core_step,
    run_core_wakeup_decision,
    run_core_wakeup_reduction,
    run_core_wakeup_step,
)
from tradingchassis_core.core.domain.slots import (
    SlotKey,
    stable_slot_order_id,
)
from tradingchassis_core.core.domain.state import StrategyState
from tradingchassis_core.core.domain.step_decision import CoreStepDecision
from tradingchassis_core.core.domain.step_result import CoreStepResult
from tradingchassis_core.core.domain.types import (
    CancelOrderIntent,
    ControlTimeEvent,
    FillEvent,
    MarketEvent,
    NewOrderIntent,
    NotionalLimits,
    OrderExecutionFeedbackEvent,
    OrderIntent,
    OrderSubmittedEvent,
    Price,
    Quantity,
    ReplaceOrderIntent,
    RiskConstraints,
)
from tradingchassis_core.core.events.sinks.null_event_bus import NullEventBus
from tradingchassis_core.core.execution_control.execution_control import ExecutionControl
from tradingchassis_core.core.execution_control.types import ControlSchedulingObligation
from tradingchassis_core.core.risk.risk_config import RiskConfig
from tradingchassis_core.core.risk.risk_engine import RiskEngine

__all__ = [
    "CoreConfiguration",
    "RiskConfig",
    "RiskEngine",
    "StrategyState",
    "MarketEvent",
    "ControlTimeEvent",
    "OrderSubmittedEvent",
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
    "fold_event_stream_entries",
    "run_core_step",
    "run_core_wakeup_reduction",
    "run_core_wakeup_decision",
    "run_core_wakeup_step",
    "CoreStepStrategyContext",
    "CoreStepStrategyEvaluator",
    "CoreExecutionControlApplyContext",
    "CorePolicyAdmissionContext",
    "CoreWakeupReductionResult",
    "ExecutionControlDecision",
    "ExecutionControlApplyContext",
    "ExecutionControlApplyResult",
    "ExecutionControlBlockedRecord",
    "ExecutionControlDispatchableRecord",
    "ExecutionControlHandledRecord",
    "apply_execution_control_plan",
    "PolicyRiskDecision",
    "PolicyRejectedCandidate",
    "PolicyAdmissionResult",
    "CoreStepDecision",
    "CoreStepResult",
    "ExecutionControl",
    "ControlSchedulingObligation",
    "NullEventBus",
    "__version__",
]

try:
    __version__ = version("tradingchassis-core")
except PackageNotFoundError:
    __version__ = "0.0.0"
