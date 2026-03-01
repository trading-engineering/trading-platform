from __future__ import annotations

from typing import Protocol


class EngineContext(Protocol):
    """Read-only execution context exposed to strategies.

    This context intentionally hides engine / runtime specifics.
    """

    @property
    def tick_size(self) -> float:
        """Minimum price increment."""
