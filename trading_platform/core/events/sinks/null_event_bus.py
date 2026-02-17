from __future__ import annotations

from typing import Any

from trading_platform.core.events.event_bus import EventBus


class _NullSink:
    """Event sink that discards all events."""

    def on_event(self, event: Any) -> None:
        return


class NullEventBus(EventBus):
    """EventBus that discards all events (used for tests)."""

    def __init__(self) -> None:
        super().__init__(sinks=[_NullSink()])
