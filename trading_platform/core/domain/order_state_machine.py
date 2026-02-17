"""
Order lifecycle state machine definitions.

This module defines the canonical order states and the allowed transitions
between them. It is intentionally passive and validation-only.

The state machine is snapshot-driven and designed for observability,
debugging, and research instrumentation. It must NOT enforce behavior
or raise exceptions in production paths.
"""

from __future__ import annotations

# Terminal order states: once reached, the order is considered complete.
ORDER_TERMINAL_STATES: frozenset[str] = frozenset(
    {
        "filled",
        "canceled",
        "expired",
        "rejected",
    }
)


# Allowed order state transitions.
#
# Key   : previous state (or None if order was not previously observed)
# Value : set of allowed next states
#
# Notes:
# - The state machine is best-effort and snapshot-driven.
# - Repeated states (e.g. partially_filled -> partially_filled) are allowed.
# - This definition is intentionally conservative and venue-agnostic.
ORDER_ALLOWED_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"pending_new"}),

    "pending_new": frozenset(
        {
            "accepted",
            "rejected",
        }
    ),

    "accepted": frozenset(
        {
            "working",
            "canceled",
            "rejected",
        }
    ),

    "working": frozenset(
        {
            "working",
            "partially_filled",
            "filled",
            "canceled",
            "replaced",
        }
    ),

    "partially_filled": frozenset(
        {
            "partially_filled",
            "filled",
            "canceled",
            "replaced",
        }
    ),
}


def is_terminal_state(state: str) -> bool:
    """Return True if the given state is terminal."""
    return state in ORDER_TERMINAL_STATES


def is_valid_transition(prev_state: str | None, next_state: str) -> bool:
    """Return True if the transition prev_state -> next_state is allowed."""
    allowed = ORDER_ALLOWED_TRANSITIONS.get(prev_state)
    if allowed is None:
        return False
    return next_state in allowed
