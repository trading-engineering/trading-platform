"""Pure risk policy logic extracted from RiskEngine.

This module is intentionally internal and behavior-preserving:
- It contains only policy checks (validation, kill-switches, hard limits).
- It does not perform queue admission, rate limiting, or inflight gating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trading_framework.core.domain.reject_reasons import RejectReason
from trading_framework.core.domain.types import OrderIntent
from trading_framework.core.ports.venue_policy import NormalizationOutcome, VenuePolicy

if TYPE_CHECKING:
    from trading_framework.core.domain.state import StrategyState
    from trading_framework.core.domain.types import MaxLoss, QuoteLimits


class RiskPolicy:
    """Pure policy layer used by RiskEngine."""

    def __init__(self, *, venue_policy: VenuePolicy) -> None:
        self._venue_policy = venue_policy

    def trading_enabled_gate(
        self,
        *,
        trading_enabled: bool,
        raw_intents: list[OrderIntent],
    ) -> tuple[bool, list[OrderIntent], list[tuple[OrderIntent, str]]]:
        """Trading enabled gate.

        Returns:
            (triggered, accepted_now, rejected_pairs)
        """
        if trading_enabled:
            return False, [], []

        accepted_now: list[OrderIntent] = []
        rejected: list[tuple[OrderIntent, str]] = []
        for it in raw_intents:
            if it.intent_type == "cancel":
                # Cancels are risk-reducing: allow them through even when disabled
                accepted_now.append(it)
            else:
                rejected.append((it, RejectReason.TRADING_DISABLED))
        return True, accepted_now, rejected

    def max_loss_gate(
        self,
        *,
        max_loss_cfg: MaxLoss | None,
        raw_intents: list[OrderIntent],
        state: StrategyState,
        now_ts_ns_local: int,
    ) -> tuple[bool, list[OrderIntent], list[tuple[OrderIntent, str]]]:
        """Max-loss / kill-switch gate.

        Returns:
            (triggered, accepted_now, rejected_pairs)
        """
        if max_loss_cfg is None:
            return False, [], []

        pnl = state.get_total_pnl()
        if pnl <= max_loss_cfg.max_drawdown:
            return True, self._accept_cancels_reject_others(
                raw_intents,
                RejectReason.MAX_LOSS_DRAWDOWN,
            )

        # Rolling loss kill-switch (equity change over a fixed window)
        if max_loss_cfg.rolling_loss is not None and max_loss_cfg.rolling_loss_window is not None:
            window_ns = int(max_loss_cfg.rolling_loss_window * 1_000_000_000)
            rolling = state.get_rolling_loss(
                now_ts_ns_local=now_ts_ns_local,
                window_ns=window_ns,
            )
            if rolling is not None and rolling <= max_loss_cfg.rolling_loss:
                return True, self._accept_cancels_reject_others(
                    raw_intents,
                    RejectReason.MAX_LOSS_ROLLING,
                )

        return False, [], []

    @staticmethod
    def _accept_cancels_reject_others(
        raw_intents: list[OrderIntent],
        reason: str,
    ) -> tuple[list[OrderIntent], list[tuple[OrderIntent, str]]]:
        accepted_now: list[OrderIntent] = []
        rejected: list[tuple[OrderIntent, str]] = []
        for it in raw_intents:
            if it.intent_type == "cancel":
                accepted_now.append(it)
            else:
                rejected.append((it, reason))
        return accepted_now, rejected

    def normalize_intent(self, it: OrderIntent, state: StrategyState) -> NormalizationOutcome:
        return self._venue_policy.normalize_intent(it, state)

    def validate_intent(self, it: OrderIntent, state: StrategyState) -> tuple[bool, str]:
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
    def hard_checks(
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
        px = self.intent_price(it, state) or 0.0
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
            book = self.quote_book_global(state) if quote_book is None else quote_book
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

    def intent_price(self, it: OrderIntent, state: StrategyState) -> float | None:
        if it.order_type == "limit":
            return None if it.intended_price is None else it.intended_price.value
        mid = state.get_mid(it.instrument)
        return None if mid <= 0 else mid

    def portfolio_gross_notional(self, state: StrategyState) -> float | None:
        total = 0.0
        for instr, acct in state.account.items():
            mid = state.get_mid(instr)
            if mid <= 0:
                return None
            total += abs(acct.position * mid * state.get_contract_size(instr))
        return total

    def quote_book_global(self, state: StrategyState) -> dict[tuple[str, str], tuple[float, float]]:
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
                px_val = self.intent_price(it, state)
                if px_val is None:
                    continue
                notional = abs(px_val * qty_val * contract_size)
                signed = notional if it.side == "buy" else -notional
                book[key] = (notional, signed)

        return book

