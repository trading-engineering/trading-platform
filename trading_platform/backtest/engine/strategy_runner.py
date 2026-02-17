"""Strategy execution loop for HFT backtests."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trading_platform.core.domain.state import StrategyState
from trading_platform.core.domain.types import (
    BookLevel,
    BookPayload,
    MarketEvent,
    OrderIntent,
    Price,
    Quantity,
)
from trading_platform.core.events.event_bus import EventBus
from trading_platform.core.events.sinks.file_recorder import FileRecorderSink
from trading_platform.core.events.sinks.sink_logging import LoggingEventSink
from trading_platform.core.ports.venue_adapter import VenueAdapter
from trading_platform.core.risk.risk_config import RiskConfig
from trading_platform.core.risk.risk_engine import RejectedIntent, RiskEngine

if TYPE_CHECKING:
    from trading_platform.backtest.adapters.execution import HftBacktestExecutionAdapter
    from trading_platform.backtest.engine.hft_engine import HftEngineConfig
    from trading_platform.strategies.base import Strategy


MAX_TIMEOUT_NS = 1 << 62  # Effectively "wait forever" without a heartbeat


class HftStrategyRunner:
    """Strategy runner for HFT backtests.

    Invariant:
    - One wait_next() wakeup corresponds to one fully committed timestamp block.
    - Strategy is evaluated at most once per wakeup on a stable state.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        *,
        engine_cfg: HftEngineConfig,
        strategy: Strategy,
        risk_cfg: RiskConfig,
    ) -> None:
        self.engine_cfg = engine_cfg
        self.strategy = strategy

        event_bus = self._build_event_bus(
            path=Path(engine_cfg.event_bus_path),
        )

        self.strategy_state = StrategyState(
            event_bus=event_bus,
        )

        self.risk = RiskEngine(
            risk_cfg=risk_cfg,
            event_bus=event_bus,
        )

        self._next_send_ts_ns_local: int | None = None

    def _build_event_bus(
        self,
        *,
        path: Path,
    ) -> EventBus:
        logger = logging.getLogger("bus")

        sinks = [
            LoggingEventSink(logger),
            FileRecorderSink(path),
        ]

        return EventBus(sinks=sinks)

    def _close_event_bus(self) -> None:
        self.strategy_state._event_bus.close()
        self.risk._event_bus.close()

    def _compute_timeout_ns(self, now_local_ns: int) -> int:
        """Compute wait timeout in nanoseconds."""
        if self._next_send_ts_ns_local is None:
            return MAX_TIMEOUT_NS
        delta = self._next_send_ts_ns_local - now_local_ns
        return 0 if delta <= 0 else delta

    def _sort_intents_for_gate(self, intents: list[OrderIntent]) -> list[OrderIntent]:
        """Sort intents to ensure cancels are evaluated first."""

        def intent_priority(intent: OrderIntent) -> int:
            if intent.intent_type == "cancel":
                return 0
            if intent.intent_type == "replace":
                return 1
            if intent.intent_type == "new":
                return 2
            return 9

        return sorted(intents, key=lambda it: (intent_priority(it), it.ts_ns_local))

    def run(
        self,
        venue: VenueAdapter,
        execution: HftBacktestExecutionAdapter,
        recorder: Any,
    ) -> None:
        """Run the backtest loop."""
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements

        instrument = self.engine_cfg.instrument
        contract_size = self.engine_cfg.contract_size

        # Initialize hftbacktest engine
        # Fetch very first event block to set local timestamp
        venue.wait_next(timeout_ns=MAX_TIMEOUT_NS, include_order_resp=False)
        observed_local_ns = venue.current_timestamp_ns()
        self.strategy_state.update_timestamp(observed_local_ns)
        sim_now_ns = self.strategy_state.sim_ts_ns_local

        while True:
            timeout_ns = self._compute_timeout_ns(self.strategy_state.sim_ts_ns_local)
            rc = venue.wait_next(timeout_ns=timeout_ns, include_order_resp=True)

            if rc == 1:
                self._close_event_bus()
                break

            observed_local_ns = venue.current_timestamp_ns()
            self.strategy_state.update_timestamp(observed_local_ns)
            sim_now_ns = self.strategy_state.sim_ts_ns_local

            raw_intents: list[OrderIntent] = []

            # -----------------------------------------------------------------
            # Market update
            # -----------------------------------------------------------------
            if rc == 2:
                depth = venue.read_market_snapshot()

                bids: list[BookLevel] = []
                asks: list[BookLevel] = []

                max_levels = max(0, int(self.engine_cfg.max_price_tick_levels))
                if max_levels > 0:
                    roi_lb_tick = depth.roi_lb_tick
                    tick_size = depth.tick_size

                    # -----------------------
                    # ASK side (fixed ticks)
                    # -----------------------
                    for offset in range(max_levels):
                        price_tick = depth.best_ask_tick + offset
                        i = price_tick - roi_lb_tick

                        qty = 0.0
                        if 0 <= i < len(depth.ask_depth):
                            qty = depth.ask_depth[i]

                        asks.append(
                            BookLevel(
                                price=Price(
                                    currency="UNKNOWN",
                                    value=price_tick * tick_size,
                                ),
                                quantity=Quantity(
                                    value=qty,
                                    unit="contracts",
                                ),
                            )
                        )

                    # -----------------------
                    # BID side (fixed ticks)
                    # -----------------------
                    for offset in range(max_levels):
                        price_tick = depth.best_bid_tick - offset
                        i = price_tick - roi_lb_tick

                        qty = 0.0
                        if 0 <= i < len(depth.bid_depth):
                            qty = depth.bid_depth[i]

                        bids.append(
                            BookLevel(
                                price=Price(
                                    currency="UNKNOWN",
                                    value=price_tick * tick_size,
                                ),
                                quantity=Quantity(
                                    value=qty,
                                    unit="contracts",
                                ),
                            )
                        )

                market_event = MarketEvent(
                    ts_ns_exch=sim_now_ns,
                    ts_ns_local=sim_now_ns,
                    instrument=instrument,
                    event_type="book",
                    book=BookPayload(
                        book_type="snapshot",
                        bids=bids,
                        asks=asks,
                        depth=min(len(bids), len(asks)),
                    ),
                )

                self.strategy_state.update_market(
                    instrument=instrument,
                    best_bid=depth.best_bid,
                    best_ask=depth.best_ask,
                    best_bid_qty=depth.best_bid_qty,
                    best_ask_qty=depth.best_ask_qty,
                    tick_size=depth.tick_size,
                    lot_size=depth.lot_size,
                    contract_size=contract_size,
                    ts_ns_local=sim_now_ns,
                    ts_ns_exch=sim_now_ns,
                )

                constraints = self.risk.build_constraints(sim_now_ns)
                raw_intents.extend(
                    self.strategy.on_feed(
                        self.strategy_state,
                        market_event,
                        self.engine_cfg,
                        constraints,
                    )
                )

            # -----------------------------------------------------------------
            # Order / account update
            # -----------------------------------------------------------------
            if rc == 3:
                state_values, orders = venue.read_orders_snapshot()

                self.strategy_state.update_account(
                    instrument=instrument,
                    position=state_values.position,
                    balance=state_values.balance,
                    fee=state_values.fee,
                    trading_volume=state_values.trading_volume,
                    trading_value=state_values.trading_value,
                    num_trades=state_values.num_trades,
                )
                self.strategy_state.ingest_order_snapshots(
                    instrument,
                    orders.values(),
                )

                constraints = self.risk.build_constraints(sim_now_ns)
                raw_intents.extend(
                    self.strategy.on_order_update(
                        self.strategy_state,
                        self.engine_cfg,
                        constraints,
                    )
                )

            # -----------------------------------------------------------------
            # Queue flush
            # -----------------------------------------------------------------
            if (
                self._next_send_ts_ns_local is not None
                and sim_now_ns >= self._next_send_ts_ns_local
            ):
                raw_intents.extend(
                    self.strategy_state.pop_queued_intents(instrument)
                )

            # -----------------------------------------------------------------
            # Gate + execution
            # -----------------------------------------------------------------
            if raw_intents:
                combined = self._sort_intents_for_gate(raw_intents)

                decision = self.risk.decide_intents(
                    raw_intents=combined,
                    state=self.strategy_state,
                    now_ts_ns_local=sim_now_ns,
                )

                execution_errors: list[tuple[OrderIntent, str]] = []
                if decision.accepted_now:
                    execution_errors = execution.apply_intents(
                        decision.accepted_now
                    )

                    failed_keys = {
                        (it.instrument, it.client_order_id)
                        for it, _ in execution_errors
                    }

                    for it in decision.accepted_now:
                        if (it.instrument, it.client_order_id) in failed_keys:
                            continue
                        self.strategy_state.mark_intent_sent(
                            it.instrument,
                            it.client_order_id,
                            it.intent_type,
                        )

                if execution_errors:
                    for it, reason in execution_errors:
                        decision.execution_rejected.append(
                            RejectedIntent(it, reason)
                        )

                self.strategy.on_risk_decision(decision)
                self._next_send_ts_ns_local = decision.next_send_ts_ns_local

                # If there are queued intents but the gate did not provide a next_send_ts_ns_local,
                # wake up at the next second boundary to ensure progress.
                if self._next_send_ts_ns_local is None:
                    queue = self.strategy_state.queued_intents.setdefault(
                        instrument,
                        deque(),
                    )
                    if queue:
                        sec = sim_now_ns // 1_000_000_000
                        self._next_send_ts_ns_local = (sec + 1) * 1_000_000_000

            venue.record(recorder)
