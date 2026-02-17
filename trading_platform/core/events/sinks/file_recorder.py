"""
Append-only file recorder sink.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FileRecorderSink:
    """Writes each event as a JSON line to a file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        self._closed = False

    def on_event(self, event: Any) -> None:
        record = event.__dict__ if hasattr(event, "__dict__") else {"event": str(event)}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._fh.flush()
        self._fh.close()
        self._closed = True
