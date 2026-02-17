"""
File segmentation logic.

This module contains utilities for splitting data files into
byte-size-constrained segments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_platform.backtest.orchestrator.manifest import DataFileMeta


def segment_files(
    files: list[DataFileMeta],
    max_bytes: int,
) -> list[list[DataFileMeta]]:
    """
    Split files into ordered segments such that each segment does not
    exceed the given maximum size in bytes.
    """

    segments: list[list[DataFileMeta]] = []
    current_segment: list[DataFileMeta] = []
    current_bytes = 0

    # Sort files by start timestamp to ensure deterministic segmentation
    for file_meta in sorted(files, key=lambda item: item.start_ts_ns):
        exceeds_limit = current_bytes + file_meta.size_bytes > max_bytes

        if current_segment and exceeds_limit:
            segments.append(current_segment)
            current_segment = []
            current_bytes = 0

        current_segment.append(file_meta)
        current_bytes += file_meta.size_bytes

    if current_segment:
        segments.append(current_segment)

    return segments
