"""Venue policy normalization and validation logic.

This module applies minimal, venue-agnostic constraints to order intents,
such as tick/lot rounding, post-only enforcement, and minimum notional checks.
The logic is intentionally explicit and branch-heavy to preserve correctness
and debuggability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from trading_platform.core.domain.reject_reasons import RejectReason
from trading_platform.core.domain.types import NewOrderIntent, OrderIntent

if TYPE_CHECKING:
    from trading_platform.core.domain.state import StrategyState


@dataclass(slots=True)
class NormalizationOutcome:
    """Result of venue policy normalization."""

    normalized: OrderIntent | None
    reject_reason: str | None
    dropped: bool


class VenuePolicy:
    """Minimal venue policy layer.

    Scope (kept intentionally small):
    - tick rounding for limit prices
    - lot rounding for intended quantity

    Venue-specific constraints can be enabled in a minimal form:
    - post-only crossing checks (best-effort using top-of-book from state)
    - min-notional checks

    The policy remains best-effort and intentionally avoids venue-specific
    edge cases (self-trade prevention, reduce-only, advanced price sliding).
    """

    def __init__(
        self,
        *,
        min_order_notional: float = 0.0,
        post_only_mode: str = "reject",
    ) -> None:
        self._min_order_notional = float(min_order_notional)

        mode = str(post_only_mode)
        if mode not in {"reject", "drop"}:
            raise ValueError(f"Invalid post_only_mode: {mode}")
        self._post_only_mode = mode

    # pylint: disable=too-many-return-statements
    def normalize_intent(self, intent: OrderIntent, state: StrategyState) -> NormalizationOutcome:
        """Normalize an intent according to venue constraints.

        Returns:
            NormalizationOutcome with one of:
            - normalized != None: normalized intent
            - reject_reason != None: hard reject
            - dropped == True: no-op intent (e.g. qty rounds to 0)
        """

        if intent.intent_type == "cancel":
            return NormalizationOutcome(normalized=intent, reject_reason=None, dropped=False)

        tick_size = state.get_tick_size(intent.instrument)
        lot_size = state.get_lot_size(intent.instrument)

        qty = 0.0 if intent.intended_qty is None else float(intent.intended_qty.value)
        qty_norm = self._round_qty(qty, lot_size)
        if qty_norm <= 0.0:
            return NormalizationOutcome(normalized=None, reject_reason=None, dropped=True)

        px_norm: float | None = None
        if intent.order_type == "limit":
            if intent.intended_price is None or not intent.intended_price.currency:
                return NormalizationOutcome(
                    normalized=None,
                    reject_reason=RejectReason.INVALID_LIMIT_PRICE,
                    dropped=False,
                )

            px = float(intent.intended_price.value)
            px_norm = self._round_price(px, tick_size, side=intent.side)
            if px_norm is None or px_norm <= 0.0:
                return NormalizationOutcome(
                    normalized=None,
                    reject_reason=RejectReason.INVALID_LIMIT_PRICE,
                    dropped=False,
                )

            post_only_outcome = self._enforce_post_only(intent, state, px_norm)
            if post_only_outcome is not None:
                return post_only_outcome

        min_notional_outcome = self._enforce_min_notional(intent, state, qty_norm, px_norm)
        if min_notional_outcome is not None:
            return min_notional_outcome

        if intent.intent_type == "new":
            return NormalizationOutcome(
                normalized=self._clone_new(intent, qty_norm, px_norm),
                reject_reason=None,
                dropped=False,
            )

        # replace
        return NormalizationOutcome(
            normalized=self._clone_replace(intent, qty_norm, px_norm),
            reject_reason=None,
            dropped=False,
        )

    # pylint: disable=too-many-return-statements
    def _enforce_post_only(
        self,
        intent: OrderIntent,
        state: StrategyState,
        px_norm: float,
    ) -> NormalizationOutcome | None:
        if intent.intent_type != "new":
            return None
        if intent.time_in_force != "POST_ONLY":
            return None

        market = state.market[intent.instrument] if intent.instrument in state.market else None
        if market is None:
            return None

        best_bid = float(market.best_bid)
        best_ask = float(market.best_ask)
        if best_bid <= 0.0 or best_ask <= 0.0:
            return None

        would_cross = False
        if intent.side == "buy":
            would_cross = px_norm >= best_ask
        else:
            would_cross = px_norm <= best_bid

        if not would_cross:
            return None

        if self._post_only_mode == "drop":
            return NormalizationOutcome(normalized=None, reject_reason=None, dropped=True)

        return NormalizationOutcome(
            normalized=None,
            reject_reason=RejectReason.POST_ONLY_WOULD_TRADE,
            dropped=False,
        )

    def _enforce_min_notional(
        self,
        intent: OrderIntent,
        state: StrategyState,
        qty_norm: float,
        px_norm: float | None,
    ) -> NormalizationOutcome | None:
        if self._min_order_notional <= 0.0:
            return None

        price = px_norm
        if intent.order_type == "market":
            mid = float(state.get_mid(intent.instrument))
            if mid <= 0.0:
                return None
            price = mid

        if price is None or price <= 0.0:
            return None

        contract_size = float(state.get_contract_size(intent.instrument))
        notional = float(price) * float(qty_norm) * contract_size

        if notional + 1e-12 >= self._min_order_notional:
            return None

        return NormalizationOutcome(
            normalized=None,
            reject_reason=RejectReason.MIN_NOTIONAL,
            dropped=False,
        )

    @staticmethod
    def _round_qty(qty: float, lot_size: float) -> float:
        if qty <= 0.0:
            return 0.0
        if lot_size <= 0.0:
            return float(qty)
        return math.floor(qty / lot_size) * lot_size

    @staticmethod
    def _round_price(price: float, tick_size: float, *, side: str) -> float | None:
        if price <= 0.0:
            return None
        if tick_size <= 0.0:
            return float(price)

        ticks = price / tick_size
        if side == "buy":
            rounded = math.floor(ticks) * tick_size
        else:
            rounded = math.ceil(ticks) * tick_size
        return float(rounded)

    @staticmethod
    def _clone_new(intent: OrderIntent, qty: float, px: float | None) -> NewOrderIntent:
        qty_unit = "contracts" if intent.intended_qty is None else intent.intended_qty.unit
        price_ccy = "UNKNOWN" if intent.intended_price is None else intent.intended_price.currency

        return NewOrderIntent(
            ts_ns_local=intent.ts_ns_local,
            instrument=intent.instrument,
            client_order_id=intent.client_order_id,
            intents_correlation_id=intent.intents_correlation_id,
            side=intent.side,
            order_type=intent.order_type,
            intended_qty={"unit": qty_unit, "value": qty},
            intended_price=None
            if px is None
            else {"currency": price_ccy, "value": px},
            time_in_force=intent.time_in_force,
        )

    @staticmethod
    def _clone_replace(intent: OrderIntent, qty: float, px: float | None) -> OrderIntent:
        # ReplaceOrderIntent shares the same field names as OrderIntent for the used fields.
        payload = intent.model_dump()
        qty_unit = "contracts" if intent.intended_qty is None else intent.intended_qty.unit
        payload["intended_qty"] = {"unit": qty_unit, "value": qty}

        price_ccy = "UNKNOWN" if intent.intended_price is None else intent.intended_price.currency
        payload["intended_price"] = (
            None
            if px is None
            else {"currency": price_ccy, "value": px}
        )
        return type(intent).model_validate(payload)
