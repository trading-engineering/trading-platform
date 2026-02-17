"""
Simple synchronous event bus.
"""
from __future__ import annotations

from typing import Any, Iterable

from trading_platform.core.events.event_sink import EventSink


class EventBus:
    """Dispatches events to registered sinks."""

    def __init__(self, sinks: Iterable[EventSink] | None = None) -> None:
        self._sinks: list[EventSink] = list(sinks) if sinks is not None else []
        self._closed = False

    def register(self, sink: EventSink) -> None:
        """Register a new sink."""
        self._sinks.append(sink)

    def emit(self, event: Any) -> None:
        """Emit an event to all sinks."""
        for sink in self._sinks:
            sink.on_event(event)

    def close(self) -> None:
        """
        Finalize all sinks that expose a close() method.
        """
        if self._closed:
            return

        for sink in self._sinks:
            close_fn = getattr(sink, "close", None)
            if callable(close_fn):
                close_fn()

        self._closed = True
