"""Execution control (internal).

This package hosts internal components that govern Queue admission, inflight
gating, and rate limiting. Policy admission stays in the Risk Engine / domain
layer.

``ControlSchedulingObligation`` (in ``types``) is a non-canonical scheduling hint
for **rate-limit** deferral only in the current Core slice; **inflight** deferral
does not emit that obligation by default. See ``docs/flows/control-time-and-scheduling.md``.
"""

from tradingchassis_core.core.execution_control.execution_control import ExecutionControl

__all__ = ["ExecutionControl"]

