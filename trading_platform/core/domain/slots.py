"""Utilities for deterministic slot-based order identifiers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SlotKey:
    """Deterministic slot identifier for market making levels.

    The slot is defined by (instrument, side, level_index).
    """

    instrument: str
    side: str
    level_index: int


def stable_slot_order_id(slot: SlotKey, namespace: str) -> str:
    """Return a stable numeric string for a slot.

    The returned value is a decimal string representing a signed 63-bit integer
    (non-negative). It is suitable as a deterministic client_order_id.

    The namespace makes the mapping explicit and versionable.
    """
    if not namespace:
        raise ValueError("namespace must be non-empty")

    payload = (
        f"{slot.instrument}:{slot.side}:{slot.level_index}:{namespace}"
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    order_id = int.from_bytes(digest, "big") & ((1 << 63) - 1)
    return str(order_id)
