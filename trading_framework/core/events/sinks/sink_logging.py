"""
Logging event sink.
"""
from __future__ import annotations

import logging
from typing import Any


class LoggingEventSink:
    """Logs domain events using the standard logging module."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def on_event(self, event: Any) -> None:
        self._logger.info("domain_event", extra={"event": event})
