"""
Event sink interface.

Sinks consume domain events emitted by the system.
"""
from __future__ import annotations

from typing import Any, Protocol


class EventSink(Protocol):
    def on_event(self, event: Any) -> None:
        """Consume a domain event."""
