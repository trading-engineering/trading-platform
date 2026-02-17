"""Execution adapter for hftbacktest backtests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from hftbacktest import ROIVectorMarketDepthBacktest

    from trading_platform.core.domain.types import OrderIntent

from trading_platform.core.domain.reject_reasons import RejectReason


class ExecutionAdapter(Protocol):
    """Venue-facing execution boundary.

    Strategy, state, and risk layers must not depend on venue-specific
    APIs. Only this adapter is allowed to call into the venue engine.
    """

    def apply_intents(
        self, intents: list[OrderIntent]
    ) -> list[tuple[OrderIntent, str]]:
        """Send a batch of intents to the venue.

        Returns:
            List of (intent, reason) pairs for venue-side failures.
        """


def _to_i64_order_id(external_id: str) -> int:
    """Convert an external string order ID into a signed 64-bit integer."""
    sanitized = external_id.strip()
    if sanitized.isdigit():
        value = int(sanitized)
    else:
        digest = hashlib.blake2b(
            sanitized.encode("utf-8"), digest_size=8
        ).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
    return value & ((1 << 63) - 1)


@dataclass(frozen=True)
class HftBacktestExecutionAdapter(ExecutionAdapter):
    """Execution adapter for hftbacktest."""

    hbt: ROIVectorMarketDepthBacktest
    asset_no: int

    def apply_intents(
        self, intents: list[OrderIntent]
    ) -> list[tuple[OrderIntent, str]]:
        """Apply a batch of order intents to the backtest venue."""
        # pylint: disable=too-many-locals,too-many-branches

        # hftbacktest enums (kept local to the adapter)
        gtc = 0
        gtx = 1  # post-only
        fok = 2
        ioc = 3

        limit = 0
        market = 1

        tif_map = {
            "GTC": gtc,
            "IOC": ioc,
            "FOK": fok,
            "POST_ONLY": gtx,
        }
        order_type_map = {"limit": limit, "market": market}

        execution_errors: list[tuple[OrderIntent, str]] = []

        for intent in intents:
            if intent.intent_type == "new":
                order_id = _to_i64_order_id(intent.client_order_id)
                tif = tif_map[intent.time_in_force]
                order_type = order_type_map[intent.order_type]
                quantity = intent.intended_qty.value
                price = (
                    intent.intended_price.value
                    if intent.intended_price is not None
                    else 0.0
                )

                try:
                    if intent.side == "buy":
                        result_code = self.hbt.submit_buy_order(
                            self.asset_no,
                            order_id,
                            price,
                            quantity,
                            tif,
                            order_type,
                            False,
                        )
                    else:
                        result_code = self.hbt.submit_sell_order(
                            self.asset_no,
                            order_id,
                            price,
                            quantity,
                            tif,
                            order_type,
                            False,
                        )
                except Exception:  # pylint: disable=broad-exception-caught
                    execution_errors.append(
                        (intent, RejectReason.EXCHANGE_ERROR)
                    )
                    continue

                if result_code != 0:
                    execution_errors.append(
                        (intent, RejectReason.EXCHANGE_REJECT)
                    )

            elif intent.intent_type == "cancel":
                order_id = _to_i64_order_id(intent.client_order_id)
                try:
                    result_code = self.hbt.cancel(
                        self.asset_no, order_id, False
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    execution_errors.append(
                        (intent, RejectReason.EXCHANGE_ERROR)
                    )
                    continue

                if result_code != 0:
                    execution_errors.append(
                        (intent, RejectReason.EXCHANGE_REJECT)
                    )

            elif intent.intent_type == "replace":
                order_id = _to_i64_order_id(intent.client_order_id)
                new_price = intent.intended_price.value
                new_quantity = intent.intended_qty.value

                try:
                    result_code = self.hbt.modify(
                        self.asset_no,
                        order_id,
                        new_price,
                        new_quantity,
                        False,
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    execution_errors.append(
                        (intent, RejectReason.EXCHANGE_ERROR)
                    )
                    continue

                if result_code != 0:
                    execution_errors.append(
                        (intent, RejectReason.EXCHANGE_REJECT)
                    )

        return execution_errors
