from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_platform import (
        EngineContext,
        GateDecision,
        MarketEvent,
        RiskConstraints,
        StrategyState,
    )

from trading_platform import (
    NewOrderIntent,
    OrderIntent,
    Price,
    Quantity,
    ReplaceOrderIntent,
    SlotKey,
    Strategy,
    stable_slot_order_id,
)

_SLOT_NAMESPACE = "debug_strategy_v0"


class DebugStrategyV0(Strategy):
    """Very simple market making example strategy."""

    def __init__(
        self,
        spread: float,
        order_qty: float,
        use_price_tick_levels: int,
        post_only: bool,
    ) -> None:
        self.spread = spread
        self.order_qty = order_qty
        self.use_price_tick_levels = use_price_tick_levels
        self.post_only = post_only

        self.intents_on_event: list[OrderIntent] = []
        self.intents_after_risk: list[OrderIntent] = []

    def round_to_tick(self, price: float, tick: float) -> float:
        if tick <= 0:
            raise ValueError("tick must be positive")
        return round(price / tick) * tick

    def on_feed(
        self,
        state: StrategyState,
        event: MarketEvent,
        engine_cfg: EngineContext,
        constraints: RiskConstraints,
    ) -> list[OrderIntent]:
        """Feed-triggered logic (rc=2). Inputs are read-only for Strategy, otherwise considered a bug."""

        self.intents_on_event = []

        # NOTE: keep existing logic as-is for now; we will align field names/types later.
        # This block is only to satisfy the new interface.
        if not constraints.trading_enabled:
            return self.intents_on_event

        if not event.is_book() or event.book is None:
            return self.intents_on_event

        if not event.book.bids or not event.book.asks:
            return self.intents_on_event

        best_bid = float(event.book.bids[0].price.value)
        best_ask = float(event.book.asks[0].price.value)
        mid = 0.5 * (best_bid + best_ask)

        tick = float(engine_cfg.tick_size)
        tif = "POST_ONLY" if self.post_only else "GTC"

        num_levels = int(self.use_price_tick_levels)
        if num_levels <= 0:
            num_levels = 1

        instrument = str(event.instrument)

        def is_slot_busy(client_order_id: str) -> bool:
            return state.is_order_id_busy(instrument, client_order_id)

        def bid_price_for_level(level_index: int) -> float:
            if level_index < len(event.book.bids):
                px = float(event.book.bids[level_index].price.value)
            else:
                px = mid - (self.spread * 0.5) - (float(level_index) * tick)
            return self.round_to_tick(px, tick)

        def ask_price_for_level(level_index: int) -> float:
            if level_index < len(event.book.asks):
                px = float(event.book.asks[level_index].price.value)
            else:
                px = mid + (self.spread * 0.5) + (float(level_index) * tick)
            return self.round_to_tick(px, tick)

        intents: list[OrderIntent] = []

        for level in range(num_levels):
            bid_slot = SlotKey(instrument=instrument, side="buy", level_index=level)
            ask_slot = SlotKey(instrument=instrument, side="sell", level_index=level)

            bid_id = stable_slot_order_id(bid_slot, namespace=_SLOT_NAMESPACE)
            ask_id = stable_slot_order_id(ask_slot, namespace=_SLOT_NAMESPACE)

            bid_px = bid_price_for_level(level)
            ask_px = ask_price_for_level(level)

            if is_slot_busy(bid_id):
                intents.append(
                    ReplaceOrderIntent(
                        ts_ns_local=event.ts_ns_local,
                        instrument=instrument,
                        client_order_id=bid_id,
                        intent_type="replace",
                        order_type="limit",
                        side="buy",
                        intended_price=Price(currency="UNKNOWN", value=bid_px),
                        intended_qty=Quantity(value=self.order_qty, unit="contracts"),
                    )
                )
            else:
                intents.append(
                    NewOrderIntent(
                        ts_ns_local=event.ts_ns_local,
                        instrument=instrument,
                        client_order_id=bid_id,
                        intent_type="new",
                        order_type="limit",
                        side="buy",
                        intended_price=Price(currency="UNKNOWN", value=bid_px),
                        intended_qty=Quantity(value=self.order_qty, unit="contracts"),
                        time_in_force=tif,
                    )
                )

            if is_slot_busy(ask_id):
                intents.append(
                    ReplaceOrderIntent(
                        ts_ns_local=event.ts_ns_local,
                        instrument=instrument,
                        client_order_id=ask_id,
                        intent_type="replace",
                        order_type="limit",
                        side="sell",
                        intended_price=Price(currency="UNKNOWN", value=ask_px),
                        intended_qty=Quantity(value=self.order_qty, unit="contracts"),
                    )
                )
            else:
                intents.append(
                    NewOrderIntent(
                        ts_ns_local=event.ts_ns_local,
                        instrument=instrument,
                        client_order_id=ask_id,
                        intent_type="new",
                        order_type="limit",
                        side="sell",
                        intended_price=Price(currency="UNKNOWN", value=ask_px),
                        intended_qty=Quantity(value=self.order_qty, unit="contracts"),
                        time_in_force=tif,
                    )
                )

        self.intents_on_event.extend(intents)
        return self.intents_on_event

    def on_order_update(
        self,
        state: StrategyState,
        engine_cfg: EngineContext,
        constraints: RiskConstraints,
    ) -> list[OrderIntent]:
        """Order-update-triggered logic (rc=3). Inputs are read-only for Strategy, otherwise considered a bug."""
        return []

    def on_risk_decision(self, decision: GateDecision) -> None:
        self.intents_after_risk = decision.accepted_now
