"""Execution control (internal extraction from RiskEngine).

Owns:
- token bucket rate limiting state & math
- inflight gating that routes NEW/REPLACE to queue
- queue admission via StrategyState.merge_intents_into_queue(...)
- queue-only local handling for certain CANCEL/REPLACE cases
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Callable

from trading_framework.core.domain.types import NewOrderIntent, OrderIntent

if TYPE_CHECKING:
    from trading_framework.core.domain.state import StrategyState


class ExecutionControl:
    """Internal execution control component (stateful)."""

    def __init__(self) -> None:
        # Persistent token buckets keyed by kind.
        self._rate_state: dict[str, dict[str, float]] = {
            "order": {"tokens": 0.0, "last_ts": 0.0},
            "cancel": {"tokens": 0.0, "last_ts": 0.0},
        }

    @staticmethod
    def _sec(ts_ns: int) -> int:
        return ts_ns // 1_000_000_000

    def consume_rate(self, kind: str, ts_ns_local: int, limit_per_sec: float) -> tuple[bool, int]:
        """Token bucket rate limiting.

        Returns:
            (allowed_now, wake_ts_ns_local)

        If not allowed, wake_ts is the earliest local timestamp when one token becomes available.
        """
        if limit_per_sec <= 0:
            sec = self._sec(ts_ns_local)
            return False, (sec + 1) * 1_000_000_000

        state = self._rate_state.setdefault(kind, {"tokens": 0.0, "last_ts": float(ts_ns_local)})
        now_ts = ts_ns_local
        last_ts = state["last_ts"]

        dt_sec = max(0.0, (now_ts - last_ts) / 1_000_000_000)

        # Capacity allows bursts up to ~1 second worth of requests.
        capacity = limit_per_sec

        tokens = state["tokens"]
        tokens = min(capacity, tokens + dt_sec * limit_per_sec)

        if tokens >= 1.0:
            tokens -= 1.0
            state["tokens"] = tokens
            state["last_ts"] = now_ts
            return True, ts_ns_local

        deficit = 1.0 - tokens
        wait_sec = deficit / limit_per_sec
        wait_ns = int(math.ceil(wait_sec * 1_000_000_000))
        wake_ts = ts_ns_local + max(1, wait_ns)

        state["tokens"] = tokens
        state["last_ts"] = now_ts
        return False, wake_ts

    def maybe_route_new_replace_to_queue_on_inflight(
        self,
        it: OrderIntent,
        state: StrategyState,
        to_queue_by_instr: defaultdict[str, list[OrderIntent]],
    ) -> bool:
        """Inflight gating: if an update is already in flight, enqueue and skip sending now."""
        if it.intent_type in ("new", "replace"):
            if state.has_inflight(it.instrument, it.client_order_id):
                to_queue_by_instr[it.instrument].append(it)
                return True
        return False

    def handle_cancel_against_queued_only_state(
        self,
        it: OrderIntent,
        *,
        state: StrategyState,
        replaced_in_queue: list[tuple[OrderIntent, OrderIntent]],
        handled_in_queue: list[OrderIntent],
    ) -> bool:
        """CANCEL against queued-only state: remove queued intents, do not send cancel."""
        if it.intent_type != "cancel":
            return False

        removed = state.pop_queued_intents_for_order(it.instrument, it.client_order_id)
        for qi in removed:
            replaced_in_queue.append((qi.intent, it))
        handled_in_queue.append(it)
        return True

    def handle_replace_against_queued_new(
        self,
        it: OrderIntent,
        *,
        state: StrategyState,
        queued_new: NewOrderIntent,
        replaced_in_queue: list[tuple[OrderIntent, OrderIntent]],
        dropped_in_queue: list[OrderIntent],
        queued: list[OrderIntent],
        handled_in_queue: list[OrderIntent],
    ) -> None:
        """REPLACE acting on queued NEW: transform into updated NEW in the queue."""
        removed = state.pop_queued_intents_for_order(it.instrument, it.client_order_id)
        for qi in removed:
            replaced_in_queue.append((qi.intent, it))

        updated_new = NewOrderIntent(
            ts_ns_local=it.ts_ns_local,
            instrument=it.instrument,
            client_order_id=it.client_order_id,
            intents_correlation_id=it.intents_correlation_id,
            side=it.side,
            order_type=it.order_type,
            intended_qty=it.intended_qty,
            intended_price=it.intended_price,
            time_in_force=queued_new.time_in_force,
        )

        q_items, replaced, dropped = state.merge_intents_into_queue(
            instrument=it.instrument,
            intents=[updated_new],
        )

        handled_in_queue.append(it)
        replaced_in_queue.extend(replaced)
        dropped_in_queue.extend(dropped)
        queued.extend(q_items)

    @staticmethod
    def is_replace_noop_against_working(
        *,
        replace_intent: OrderIntent,
        working_intended_price: float,
        working_intended_qty: float,
        float_equal: Callable[[float, float], bool],
    ) -> bool:
        replace_px = replace_intent.intended_price.value
        replace_qty = replace_intent.intended_qty.value
        return float_equal(working_intended_price, replace_px) and float_equal(working_intended_qty, replace_qty)

    @staticmethod
    def is_replace_noop_against_queued_new(
        *,
        replace_intent: OrderIntent,
        queued_new: NewOrderIntent,
        float_equal: Callable[[float, float], bool],
    ) -> bool:
        replace_px = replace_intent.intended_price.value
        replace_qty = replace_intent.intended_qty.value
        q_px = queued_new.intended_price.value
        q_qty = queued_new.intended_qty.value
        return float_equal(q_px, replace_px) and float_equal(q_qty, replace_qty)

    def merge_to_queue_per_instrument(
        self,
        *,
        state: StrategyState,
        to_queue_by_instr: defaultdict[str, list[OrderIntent]],
        queued: list[OrderIntent],
        replaced_in_queue: list[tuple[OrderIntent, OrderIntent]],
        dropped_in_queue: list[OrderIntent],
    ) -> None:
        for instr, intents in to_queue_by_instr.items():
            if not intents:
                continue
            q, replaced, dropped = state.merge_intents_into_queue(instrument=instr, intents=intents)
            queued.extend(q)
            replaced_in_queue.extend(replaced)
            dropped_in_queue.extend(dropped)

