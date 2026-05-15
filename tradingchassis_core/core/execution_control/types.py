"""Execution-control internal semantic types.

The types in this module are non-canonical runtime helpers. They are not
canonical Events and are not part of the Event Stream taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ControlSchedulingObligation:
    """Internal runtime-facing scheduling obligation.

    This is a derived control signal (not an Event) and does not mutate State.
    """

    due_ts_ns_local: int
    reason: str
    scope_key: str
    source: str
    obligation_key: str = ""

    def __post_init__(self) -> None:
        if self.obligation_key:
            return
        object.__setattr__(
            self,
            "obligation_key",
            (
                f"{self.source}|{self.scope_key}|{self.reason}|{self.due_ts_ns_local}"
            ),
        )

    @property
    def ts_ns_local(self) -> int:
        """Compatibility alias for pre-16F callers/tests."""
        return self.due_ts_ns_local

