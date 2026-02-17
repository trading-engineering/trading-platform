"""
Dataset manifest definitions.

This module defines metadata structures and protocols used to describe
datasets and their underlying data files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DataFileMeta:
    """
    Immutable metadata describing a single data file.
    """

    file_id: str
    object_key: str
    start_ts_ns: int
    end_ts_ns: int
    size_bytes: int
    symbol: str
    venue: str
    datatype: str


class DatasetManifest(Protocol):
    """
    Protocol describing a dataset manifest interface.
    """

    def iter_files(
        self,
        *,
        start_ts_ns: int,
        end_ts_ns: int,
        symbol: str,
        venue: str,
        datatype: str,
    ) -> list[DataFileMeta]:
        """
        Iterate over data files matching the given constraints.
        """
