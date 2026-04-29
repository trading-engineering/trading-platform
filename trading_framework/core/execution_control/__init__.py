"""Execution control (internal).

This package intentionally hosts internal components that govern queue admission,
inflight gating, and timing/rate limiting, while keeping RiskEngine focused on
policy decisions.
"""

from trading_framework.core.execution_control.execution_control import ExecutionControl

__all__ = ["ExecutionControl"]

