"""Risk engine implementing hard risk checks and intent gating."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from trading_platform.core.domain.reject_reasons import RejectReason
from trading_platform.core.domain.types import NewOrderIntent, OrderIntent, RiskConstraints
from trading_platform.core.events.events import RiskDecisionEvent
from trading_platform.core.ports.venue_policy import VenuePolicy

if TYPE_CHECKING:
    from risk.risk_config import RiskConfig

    from trading_platform.core.domain.state import StrategyState
    from trading_platform.core.domain.types import QuoteLimits
    from trading_platform.core.events.event_bus import EventBus


# ---------------------------------------------------------------------------
# Gate decision models (internal, not part of JSON schema)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RejectedIntent:
    intent: OrderIntent
    reason: str


@dataclass(slots=True)
class GateDecision:
    """Result of the hard risk/gate layer.

    - accepted_now: intents that may be sent immediately
    - queued: intents that were enqueued into StrategyState.queue (data-only)
    - rejected: hard rejects with reasons
    - replaced_in_queue: (old, new) pairs when queue replacement happened
    - dropped_in_queue: intents that were dropped during queue merge (e.g. superseded)
    - handled_in_queue: intents that were fully handled locally in the queue layer
      (e.g. cancel/replace acting only on queued state) and must not be sent
    - next_send_ts_ns_local: earliest local timestamp where it makes sense to wake up
      to try flushing the queue (best-effort)
    """

    ts_ns_local: int
    accepted_now: list[OrderIntent]
    queued: list[OrderIntent]
    rejected: list[RejectedIntent]
    replaced_in_queue: list[tuple[OrderIntent, OrderIntent]]
    dropped_in_queue: list[OrderIntent]
    handled_in_queue: list[OrderIntent]
    # Populated by the runner after outbound execution.
    execution_rejected: list[RejectedIntent]
    next_send_ts_ns_local: int | None


class RiskEngine:
    """Hard risk and intent gating engine."""

    # pylint: disable=too-many-instance-attributes
    """Hard risk + gate layer.

    This layer is allowed to:
    - hard reject invalid / risk-breaching intents
    - queue intents that should be sent later (rate limits, budgets)
    - accept intents for immediate sending

    It must NOT submit orders itself.
    """

    def __init__(self, risk_cfg: RiskConfig, event_bus: EventBus) -> None:
        self.risk_cfg = risk_cfg
        self._event_bus = event_bus

        venue_policy_cfg = self._parse_venue_policy_config(risk_cfg)
        self._venue_policy = VenuePolicy(
            min_order_notional=venue_policy_cfg["min_order_notional"],
            post_only_mode=venue_policy_cfg["post_only_mode"],
        )

        # Persistent per-second rate buckets keyed by local timestamp second.
        # Example: {sec: {"order": 3, "cancel": 10}}
        self._rate_state: dict[str, dict[str, float]] = {
            "order": {"tokens": 0.0, "last_ts": 0.0},
            "cancel": {"tokens": 0.0, "last_ts": 0.0},
        }

    @staticmethod
    def _parse_venue_policy_config(risk_cfg: RiskConfig) -> dict[str, object]:
        cfg: dict[str, object] = {
            "min_order_notional": 0.0,
            "post_only_mode": "reject",
        }

        extra = risk_cfg.extra
        if not isinstance(extra, dict):
            return cfg

        # Preferred form: extra["venue_policy"] is a nested dict.
        # RiskConstraints.extra requires flat scalar values, therefore
        # nested config must be normalized before being exposed as constraints.
        vp = extra["venue_policy"] if "venue_policy" in extra else None
        if isinstance(vp, dict):
            if "min_order_notional" in vp:
                try:
                    cfg["min_order_notional"] = float(vp["min_order_notional"])
                except (TypeError, ValueError):
                    pass

            if "post_only_mode" in vp:
                mode = str(vp["post_only_mode"])
                if mode in {"reject", "drop"}:
                    cfg["post_only_mode"] = mode

            return cfg

        # Backwards/alternative form: flattened keys.
        if "venue_policy_min_order_notional" in extra:
            try:
                cfg["min_order_notional"] = float(
                extra["venue_policy_min_order_notional"]
            )
            except (TypeError, ValueError):
                pass

        if "venue_policy_post_only_mode" in extra:
            mode = str(extra["venue_policy_post_only_mode"])
            if mode in {"reject", "drop"}:
                cfg["post_only_mode"] = mode

        return cfg

    @staticmethod
    def _constraints_extra(extra: object) -> dict[str, object]:
        """Normalize RiskConfig.extra for RiskConstraints.

        RiskConstraints.extra is defined as a flat mapping of scalar values
        (str/float/bool/None). Nested dicts are not allowed.

        The normalization keeps the original mapping intact on RiskConfig,
        but produces a flattened mapping for strategy constraints.
        """

        if not isinstance(extra, dict):
            return {}

        normalized: dict[str, object] = {}
        for key, value in extra.items():
            if key == "venue_policy" and isinstance(value, dict):
                # Flatten nested venue policy config.
                if "min_order_notional" in value:
                    normalized["venue_policy_min_order_notional"] = value["min_order_notional"]
                if "post_only_mode" in value:
                    normalized["venue_policy_post_only_mode"] = value["post_only_mode"]
                continue

            # Only keep scalar values to match the RiskConstraints schema.
            if value is None or isinstance(value, (str, float, bool)):
                normalized[key] = value
            elif isinstance(value, int):
                normalized[key] = float(value)

        return normalized

    @staticmethod
    def _float_equal(a: float, b: float) -> bool:
        """Best-effort float equality for normalized values."""
        return abs(a - b) <= 1e-12

    # ---------------------------------------------------------------------
    # Soft constraints for strategy
    # ---------------------------------------------------------------------

    def build_constraints(self, current_timestamp_ns_local: int) -> RiskConstraints:
        """Build RiskConstraints handed to the strategy."""
        extra = self._constraints_extra(self.risk_cfg.extra)
        return RiskConstraints(
            ts_ns_local=current_timestamp_ns_local,
            scope=self.risk_cfg.scope,
            trading_enabled=self.risk_cfg.trading_enabled,
            position_limits=self.risk_cfg.position_limits,
            notional_limits=self.risk_cfg.notional_limits,
            quote_limits=self.risk_cfg.quote_limits,
            order_rate_limits=self.risk_cfg.order_rate_limits,
            max_loss=self.risk_cfg.max_loss,
            extra=extra,
        )

    # ---------------------------------------------------------------------
    # Hard gate decision
    # ---------------------------------------------------------------------

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def decide_intents(
        self,
        raw_intents: list[OrderIntent],
        state: StrategyState,
        now_ts_ns_local: int,
    ) -> GateDecision:
        """Hard gate decision.

        - Hard rejects: never send (risk breach / invalid)
        - Queue: send later (rate limits / local budgets)
        - Accept: send now

        NOTE: This implementation *does* enqueue queued intents into StrategyState
        (data-only queue) by calling state.merge_intents_into_queue().
        """

        accepted_now: list[OrderIntent] = []
        to_queue_by_instr: defaultdict[str, list[OrderIntent]] = defaultdict(list)
        rejected: list[RejectedIntent] = []
        replaced_in_queue: list[tuple[OrderIntent, OrderIntent]] = []
        dropped_in_queue: list[OrderIntent] = []
        handled_in_queue: list[OrderIntent] = []

        # Intents that ended up queued due to rate limits or due to queue-only handling.
        queued: list[OrderIntent] = []
        next_send_ts: int | None = None

        # counters for RiskDecisionEvent
        reject_counts: dict[str, int] = {}
        def _count_reject(reason: str) -> None:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

        # --- Trading enabled gate ---
        if not self.risk_cfg.trading_enabled:
            for it in raw_intents:
                if it.intent_type == "cancel":
                    # Cancels are risk-reducing: allow them through even when disabled
                    accepted_now.append(it)
                else:
                    rejected.append(RejectedIntent(it, RejectReason.TRADING_DISABLED))
                    _count_reject(RejectReason.TRADING_DISABLED)

            decision = GateDecision(
                ts_ns_local=now_ts_ns_local,
                accepted_now=accepted_now,
                queued=[],
                rejected=rejected,
                replaced_in_queue=[],
                dropped_in_queue=[],
                handled_in_queue=[],
                execution_rejected=[],
                next_send_ts_ns_local=None,
            )

            # emit summary
            self._event_bus.emit(
                RiskDecisionEvent(
                    ts_ns_local=now_ts_ns_local,
                    accepted=len(accepted_now),
                    queued=0,
                    rejected=len(rejected),
                    handled=len(handled_in_queue),
                    reject_reasons=reject_counts,
                )
            )

            return decision

        # --- Max loss (portfolio drawdown kill-switch) ---
        max_loss_cfg = self.risk_cfg.max_loss
        if max_loss_cfg is not None:
            pnl = state.get_total_pnl()
            if pnl <= max_loss_cfg.max_drawdown:
                for it in raw_intents:
                    if it.intent_type == "cancel":
                        accepted_now.append(it)
                    else:
                        rejected.append(RejectedIntent(it, RejectReason.MAX_LOSS_DRAWDOWN))
                        _count_reject(RejectReason.MAX_LOSS_DRAWDOWN)

                decision = GateDecision(
                    ts_ns_local=now_ts_ns_local,
                    accepted_now=accepted_now,
                    queued=[],
                    rejected=rejected,
                    replaced_in_queue=[],
                    dropped_in_queue=[],
                    handled_in_queue=[],
                    execution_rejected=[],
                    next_send_ts_ns_local=None,
                )

                self._event_bus.emit(
                    RiskDecisionEvent(
                        ts_ns_local=now_ts_ns_local,
                        accepted=len(accepted_now),
                        queued=0,
                        rejected=len(rejected),
                        handled=len(handled_in_queue),
                        reject_reasons=reject_counts,
                    )
                )

                return decision

            # Rolling loss kill-switch (equity change over a fixed window)
            if max_loss_cfg.rolling_loss is not None and max_loss_cfg.rolling_loss_window is not None:
                window_ns = int(max_loss_cfg.rolling_loss_window * 1_000_000_000)
                rolling = state.get_rolling_loss(
                now_ts_ns_local=now_ts_ns_local,
                window_ns=window_ns,
            )
                if rolling is not None and rolling <= max_loss_cfg.rolling_loss:
                    for it in raw_intents:
                        if it.intent_type == "cancel":
                            accepted_now.append(it)
                        else:
                            rejected.append(RejectedIntent(it, RejectReason.MAX_LOSS_ROLLING))
                            _count_reject(RejectReason.MAX_LOSS_ROLLING)

                    decision = GateDecision(
                        ts_ns_local=now_ts_ns_local,
                        accepted_now=accepted_now,
                        queued=[],
                        rejected=rejected,
                        replaced_in_queue=[],
                        dropped_in_queue=[],
                        handled_in_queue=[],
                        execution_rejected=[],
                        next_send_ts_ns_local=None,
                    )

                    self._event_bus.emit(
                        RiskDecisionEvent(
                            ts_ns_local=now_ts_ns_local,
                            accepted=len(accepted_now),
                            queued=0,
                            rejected=len(rejected),
                            handled=len(handled_in_queue),
                            reject_reasons=reject_counts,
                        )
                    )

                    return decision

        # --- Rate limits (per second, local time) ---
        rate_cfg = self.risk_cfg.order_rate_limits
        max_orders_per_sec = None if rate_cfg is None else rate_cfg.max_orders_per_second
        max_cancels_per_sec = None if rate_cfg is None else rate_cfg.max_cancels_per_second

        # --- Position / notional limits ---
        pos_cfg = self.risk_cfg.position_limits
        max_pos = None if (pos_cfg is None or pos_cfg.max_position is None) else pos_cfg.max_position

        notional_cfg = self.risk_cfg.notional_limits
        max_gross_notional = notional_cfg.max_gross_notional
        max_single_order_notional = notional_cfg.max_single_order_notional

        quote_cfg = self.risk_cfg.quote_limits

        quote_book = None
        if quote_cfg is not None:
            quote_book = self._quote_book_global(state)

        # Base portfolio gross notional (best-effort)
        base_gross_notional = self._portfolio_gross_notional(state)

        # -----------------------------------------------------------------
        # Per-intent decision
        # -----------------------------------------------------------------
        for it in raw_intents:
            norm = self._venue_policy.normalize_intent(it, state)
            if norm.reject_reason is not None:
                rejected.append(RejectedIntent(it, norm.reject_reason))
                _count_reject(norm.reject_reason)
                continue
            if norm.dropped:
                handled_in_queue.append(it)
                continue
            if norm.normalized is None:
                rejected.append(RejectedIntent(it, RejectReason.INVALID_QTY))
                _count_reject(RejectReason.INVALID_QTY)
                continue

            it = norm.normalized
            # 0) Existence / uniqueness guards (C1)
            has_working = state.has_working_order(it.instrument, it.client_order_id)
            has_queued = state.has_queued_intent(it.instrument, it.client_order_id)

            # 0.5) Replace delta gating (after venue normalization)
            # Drop replace intents that do not materially change price or quantity.
            if it.intent_type == "replace":
                replace_px = it.intended_price.value
                replace_qty = it.intended_qty.value

                if has_working:
                    working = state.get_working_order_snapshot(it.instrument, it.client_order_id)
                    if working is not None:
                        if (
                            self._float_equal(working.intended_price, replace_px)
                            and self._float_equal(working.intended_qty, replace_qty)
                        ):
                            handled_in_queue.append(it)
                            continue

                if not has_working and has_queued:
                    queued_new = state.find_queued_new_intent(it.instrument, it.client_order_id)
                    if queued_new is not None:
                        q_px = queued_new.intended_price.value
                        q_qty = queued_new.intended_qty.value
                        if self._float_equal(q_px, replace_px) and self._float_equal(q_qty, replace_qty):
                            handled_in_queue.append(it)
                            continue

            if it.intent_type == "new":
                if has_working or has_queued:
                    rejected.append(RejectedIntent(it, RejectReason.DUPLICATE_ID))
                    _count_reject(RejectReason.DUPLICATE_ID)
                    continue

            if it.intent_type == "cancel":
                if not has_working:
                    if has_queued:
                        # Cancel only queued state: remove queued intents and do not send a cancel.
                        removed = state.pop_queued_intents_for_order(it.instrument, it.client_order_id)
                        for qi in removed:
                            replaced_in_queue.append((qi.intent, it))
                        handled_in_queue.append(it)
                        continue

                    rejected.append(RejectedIntent(it, RejectReason.ORDER_NOT_FOUND))
                    _count_reject(RejectReason.ORDER_NOT_FOUND)
                    continue

            if it.intent_type == "replace":
                if not has_working:
                    # Replace acting on queued NEW: transform to NEW (update planned order).
                    queued_new = state.find_queued_new_intent(it.instrument, it.client_order_id)
                    if queued_new is None:
                        rejected.append(RejectedIntent(it, RejectReason.ORDER_NOT_FOUND))
                        _count_reject(RejectReason.ORDER_NOT_FOUND)
                        continue

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
                    continue

            # 0.6) Inflight gating: if an update is already in flight for this
            # order id, do not send another new/replace immediately. Instead,
            # enqueue the latest desired intent to be flushed once inflight clears.
            # NOTE:
            # Inflight gating is best-effort and snapshot-driven.
            # This enforces *eventual consistency*, not ACK-synchronous behavior.
            # An intent may be queued even though the previous request has already
            # reached the venue but is not yet observable via snapshots.
            if it.intent_type in ("new", "replace"):
                if state.has_inflight(it.instrument, it.client_order_id):
                    to_queue_by_instr[it.instrument].append(it)
                    continue

            # 1) Outbound hygiene validation (hard reject)
            ok, reason = self._validate_intent(it, state)
            if not ok:
                rejected.append(RejectedIntent(it, reason))
                _count_reject(reason)
                continue

            # 2) Hard risk checks (hard reject)
            ok, reason = self._hard_checks(
                it,
                state,
                max_pos=max_pos,
                max_single_order_notional=max_single_order_notional,
                max_gross_notional=max_gross_notional,
                base_gross_notional=base_gross_notional,
                quote_cfg=quote_cfg,
                quote_book=quote_book,
            )
            if not ok:
                rejected.append(RejectedIntent(it, reason))
                _count_reject(reason)
                continue

            # 3) Rate limiting -> queue (soft, not reject)
            if it.intent_type == "cancel":
                if max_cancels_per_sec is not None:
                    allowed, wake_ts = self._consume_rate("cancel", now_ts_ns_local, max_cancels_per_sec)
                    if not allowed:
                        to_queue_by_instr[it.instrument].append(it)
                        next_send_ts = wake_ts if next_send_ts is None else min(next_send_ts, wake_ts)
                        continue
                accepted_now.append(it)
                continue

            # new / replace
            if max_orders_per_sec is not None:
                allowed, wake_ts = self._consume_rate("order", now_ts_ns_local, max_orders_per_sec)
                if not allowed:
                    to_queue_by_instr[it.instrument].append(it)
                    next_send_ts = wake_ts if next_send_ts is None else min(next_send_ts, wake_ts)
                    continue

            accepted_now.append(it)

        # -----------------------------------------------------------------
        # Queue merge per instrument (replacement rules live in StrategyState)
        # -----------------------------------------------------------------
        for instr, intents in to_queue_by_instr.items():
            if not intents:
                continue
            q, replaced, dropped = state.merge_intents_into_queue(instrument=instr, intents=intents)
            queued.extend(q)
            replaced_in_queue.extend(replaced)
            dropped_in_queue.extend(dropped)

        decision = GateDecision(
            ts_ns_local=now_ts_ns_local,
            accepted_now=accepted_now,
            queued=queued,
            rejected=rejected,
            replaced_in_queue=replaced_in_queue,
            dropped_in_queue=dropped_in_queue,
            handled_in_queue=handled_in_queue,
            execution_rejected=[],
            next_send_ts_ns_local=next_send_ts,
        )

        self._event_bus.emit(
            RiskDecisionEvent(
                ts_ns_local=now_ts_ns_local,
                accepted=len(accepted_now),
                queued=len(queued),
                rejected=len(rejected),
                handled=len(handled_in_queue),
                reject_reasons=reject_counts,
            )
        )
        
        return decision

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    @staticmethod
    def _sec(ts_ns: int) -> int:
        return ts_ns // 1_000_000_000

    def _consume_rate(self, kind: str, ts_ns_local: int, limit_per_sec: float) -> tuple[bool, int]:
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

    def _validate_intent(self, it: OrderIntent, state: StrategyState) -> tuple[bool, str]:
        """Outbound intent sanity.

        Even if your schemas allow 0 placeholders, outbound intents should still be sensible.
        """
        if it.ts_ns_local <= 0:
            return False, RejectReason.INVALID_TS
        if not it.instrument:
            return False, RejectReason.INVALID_INSTRUMENT

        if it.intent_type == "cancel":
            return True, "OK"

        # new / replace
        if it.intended_qty is None or it.intended_qty.value <= 0:
            return False, RejectReason.INVALID_QTY

        if it.order_type == "limit":
            if it.intended_price is None or it.intended_price.value <= 0:
                return False, RejectReason.INVALID_LIMIT_PRICE

        if it.order_type == "market":
            # if notional checks need a price proxy, require a mid
            if state.get_mid(it.instrument) <= 0:
                return False, RejectReason.NO_MID_FOR_MARKET

        return True, "OK"

    # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements
    def _hard_checks(
        self,
        it: OrderIntent,
        state: StrategyState,
        *,
        max_pos: float | None,
        max_single_order_notional: float | None,
        max_gross_notional: float | None,
        base_gross_notional: float | None,
        quote_cfg: QuoteLimits | None,
        quote_book: dict[tuple[str, str | None, tuple[float, float]]],
    ) -> tuple[bool, str]:
        """Apply hard risk checks. Returns (ok, reason)."""

        # Cancels are always allowed (risk reducing).
        if it.intent_type == "cancel":
            return True, "OK"

        qty = it.intended_qty.value
        px = self._intent_price(it, state) or 0.0
        contract_size = state.get_contract_size(it.instrument)
        notional = abs(px * qty * contract_size)

        # Position limit (symmetric absolute), based on account position
        if max_pos is not None:
            cur_pos = state.account[it.instrument].position if it.instrument in state.account else 0.0
            delta = qty if it.side == "buy" else -qty
            if cur_pos + delta > max_pos or cur_pos + delta < -max_pos:
                return False, RejectReason.MAX_POSITION

        # Single-order notional
        if max_single_order_notional is not None and notional > max_single_order_notional:
            return False, RejectReason.MAX_SINGLE_ORDER_NOTIONAL

        # Portfolio gross notional
        if max_gross_notional is not None and base_gross_notional is not None:
            if base_gross_notional + notional > max_gross_notional:
                return False, RejectReason.MAX_GROSS_NOTIONAL

        # Quote limits (global, queued included)
        if quote_cfg is not None:
            book = self._quote_book_global(state) if quote_book is None else quote_book
            key = (it.instrument, it.client_order_id)

            existing = book.get(key)
            existing_abs = 0.0 if existing is None else existing[0]
            existing_signed = 0.0 if existing is None else existing[1]

            active = len(book)
            gross_q = sum(v[0] for v in book.values())
            net_q = sum(v[1] for v in book.values())

            # Apply delta for this intent (new or replace).
            new_abs = notional
            new_signed = notional if it.side == "buy" else -notional

            active_after = active if existing is not None else active + 1
            gross_after = gross_q - existing_abs + new_abs
            net_after = net_q - existing_signed + new_signed

            if quote_cfg.max_active_quotes is not None:
                if active_after > quote_cfg.max_active_quotes:
                    return False, RejectReason.MAX_ACTIVE_QUOTES

            if quote_cfg.max_gross_quote_notional is not None:
                if gross_after > quote_cfg.max_gross_quote_notional:
                    return False, RejectReason.MAX_GROSS_QUOTE_NOTIONAL

            if quote_cfg.max_net_quote_notional is not None:
                if abs(net_after) > quote_cfg.max_net_quote_notional:
                    return False, RejectReason.MAX_NET_QUOTE_NOTIONAL

        return True, "OK"

    def _intent_price(self, it: OrderIntent, state: StrategyState) -> float | None:
        if it.order_type == "limit":
            return None if it.intended_price is None else it.intended_price.value
        mid = state.get_mid(it.instrument)
        return None if mid <= 0 else mid

    def _portfolio_gross_notional(self, state: StrategyState) -> float | None:
        total = 0.0
        for instr, acct in state.account.items():
            mid = state.get_mid(instr)
            if mid <= 0:
                return None
            total += abs(acct.position * mid * state.get_contract_size(instr))
        return total

    def _quote_book_global(self, state: StrategyState) -> dict[tuple[str, str], tuple[float, float]]:
        """Build a best-effort global quote book including queued intents.

        Returns:
            Mapping (instrument, client_order_id) -> (abs_notional, signed_notional)

        Notes:
            - Working orders are sourced from StrategyState.orders.
            - Queued intents in StrategyState.queued_intents are applied on top.
            - This is used only for quote-limits enforcement.
        """

        book: dict[tuple[str, str], tuple[float, float]] = {}

        # Working orders
        for instr, bucket in state.orders.items():
            contract_size = state.get_contract_size(instr)
            for oid, o in bucket.items():
                qty = o.remaining_qty if o.remaining_qty > 0 else o.intended_qty
                if qty <= 0:
                    continue
                px = o.intended_price
                notional = abs(px * qty * contract_size)
                signed = notional if o.side == "buy" else -notional
                book[(instr, oid)] = (notional, signed)

        # Queued intents (apply on top of working)
        for instr, q in state.queued_intents.items():
            contract_size = state.get_contract_size(instr)
            for qi in q:
                it = qi.intent
                key = (instr, it.client_order_id)

                if it.intent_type == "cancel":
                    if key in book:
                        book.pop(key)
                    continue

                if it.intent_type not in ("new", "replace"):
                    continue

                qty_val = it.intended_qty.value
                px_val = self._intent_price(it, state)
                if px_val is None:
                    continue
                notional = abs(px_val * qty_val * contract_size)
                signed = notional if it.side == "buy" else -notional
                book[key] = (notional, signed)

        return book
