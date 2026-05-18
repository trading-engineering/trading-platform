"""Deterministic Core Strategy state.

This state container keeps canonical reducer-owned data and Execution Control
supporting structures (Queue + inflight tracking). Runtime snapshot parsing and
venue lifecycle adaptation are intentionally out of scope for Core.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable, Mapping

from tradingchassis_core.core.domain.processing_order import ProcessingPosition

if TYPE_CHECKING:
    from tradingchassis_core.core.domain.types import (
        ControlTimeEvent,
        FillEvent,
        NewOrderIntent,
        OrderCanceledEvent,
        OrderExecutionFeedbackEvent,
        OrderExpiredEvent,
        OrderIntent,
        OrderRejectedEvent,
        OrderSubmittedEvent,
    )
    from tradingchassis_core.core.events.event_bus import EventBus


@dataclass(slots=True)
class QueuedIntent:
    """An Intent stored for later sending (data-only Queue)."""

    intent: OrderIntent
    queued_at_ts_ns: int
    logical_key: str
    priority: int  # lower = earlier


@dataclass(slots=True)
class InflightInfo:
    """Best-effort inflight tracking per order id."""

    action: str  # new | cancel | replace
    ts_sent_ns_local: int


@dataclass(slots=True)
class MarketState:
    """Best-effort market snapshot needed for risk checks."""

    last_ts_ns_local: int = 0
    last_ts_ns_exch: int = 0
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid: float = 0.0
    best_bid_qty: float = 0.0
    best_ask_qty: float = 0.0
    tick_size: float = 0.0
    lot_size: float = 0.0
    contract_size: float = 1.0


@dataclass(slots=True)
class AccountState:
    """Best-effort account values from canonical execution feedback."""

    position: float = 0.0
    balance: float = 0.0
    fee: float = 0.0
    trading_volume: float = 0.0
    trading_value: float = 0.0
    num_trades: int = 0
    equity: float = 0.0
    initial_equity: float = 0.0
    realized_pnl: float = 0.0


@dataclass(slots=True)
class WorkingOrder:
    """Canonical in-memory view of an active order."""

    instrument: str
    client_order_id: str
    side: str
    intended_price: float
    intended_qty: float
    cum_filled_qty: float
    remaining_qty: float
    state: str
    submitted_ts_ns_local: int
    updated_ts_ns_local: int


@dataclass(slots=True)
class CanonicalOrderProjection:
    """Internal canonical order lifecycle projection."""

    instrument: str
    client_order_id: str
    state: str
    submitted_ts_ns_local: int
    updated_ts_ns_local: int
    side: str | None = None
    intended_price: float | None = None
    intended_qty: float | None = None


@dataclass(frozen=True, slots=True)
class MarketStateView:
    """Immutable market snapshot exposed to Strategy evaluation."""

    last_ts_ns_local: int
    last_ts_ns_exch: int
    best_bid: float
    best_ask: float
    mid: float
    best_bid_qty: float
    best_ask_qty: float
    tick_size: float
    lot_size: float
    contract_size: float


@dataclass(frozen=True, slots=True)
class AccountStateView:
    """Immutable account snapshot exposed to Strategy evaluation."""

    position: float
    balance: float
    fee: float
    trading_volume: float
    trading_value: float
    num_trades: int
    equity: float
    initial_equity: float
    realized_pnl: float


@dataclass(frozen=True, slots=True)
class WorkingOrderView:
    """Immutable working-order snapshot exposed to Strategy evaluation."""

    instrument: str
    client_order_id: str
    side: str
    intended_price: float
    intended_qty: float
    cum_filled_qty: float
    remaining_qty: float
    state: str
    submitted_ts_ns_local: int
    updated_ts_ns_local: int


class StrategyStateView:
    """Read-only snapshot of Strategy State for Strategy evaluation."""

    __slots__ = (
        "_sim_ts_ns_local",
        "_market",
        "_account",
        "_orders",
        "_fills",
        "_fill_cum_qty",
    )

    def __init__(self, state: StrategyState) -> None:
        self._sim_ts_ns_local = state.sim_ts_ns_local

        market_snapshot = {
            instrument: MarketStateView(
                last_ts_ns_local=market.last_ts_ns_local,
                last_ts_ns_exch=market.last_ts_ns_exch,
                best_bid=market.best_bid,
                best_ask=market.best_ask,
                mid=market.mid,
                best_bid_qty=market.best_bid_qty,
                best_ask_qty=market.best_ask_qty,
                tick_size=market.tick_size,
                lot_size=market.lot_size,
                contract_size=market.contract_size,
            )
            for instrument, market in state.market.items()
        }
        self._market: Mapping[str, MarketStateView] = MappingProxyType(market_snapshot)

        account_snapshot = {
            instrument: AccountStateView(
                position=account.position,
                balance=account.balance,
                fee=account.fee,
                trading_volume=account.trading_volume,
                trading_value=account.trading_value,
                num_trades=account.num_trades,
                equity=account.equity,
                initial_equity=account.initial_equity,
                realized_pnl=account.realized_pnl,
            )
            for instrument, account in state.account.items()
        }
        self._account: Mapping[str, AccountStateView] = MappingProxyType(account_snapshot)

        orders_snapshot: dict[str, Mapping[str, WorkingOrderView]] = {}
        for instrument, by_id in state.orders.items():
            orders_snapshot[instrument] = MappingProxyType(
                {
                    client_order_id: WorkingOrderView(
                        instrument=working.instrument,
                        client_order_id=working.client_order_id,
                        side=working.side,
                        intended_price=working.intended_price,
                        intended_qty=working.intended_qty,
                        cum_filled_qty=working.cum_filled_qty,
                        remaining_qty=working.remaining_qty,
                        state=working.state,
                        submitted_ts_ns_local=working.submitted_ts_ns_local,
                        updated_ts_ns_local=working.updated_ts_ns_local,
                    )
                    for client_order_id, working in by_id.items()
                }
            )
        self._orders: Mapping[str, Mapping[str, WorkingOrderView]] = MappingProxyType(
            orders_snapshot
        )

        fills_snapshot = {
            instrument: tuple(fill.model_copy(deep=True) for fill in fills)
            for instrument, fills in state.fills.items()
        }
        self._fills: Mapping[str, tuple[FillEvent, ...]] = MappingProxyType(fills_snapshot)

        fill_cum_snapshot: dict[str, Mapping[str, float]] = {}
        for instrument, fill_cum_by_id in state.fill_cum_qty.items():
            fill_cum_snapshot[instrument] = MappingProxyType(
                {
                    client_order_id: float(qty)
                    for client_order_id, qty in fill_cum_by_id.items()
                }
            )
        self._fill_cum_qty: Mapping[str, Mapping[str, float]] = MappingProxyType(
            fill_cum_snapshot
        )

    @property
    def sim_ts_ns_local(self) -> int:
        return self._sim_ts_ns_local

    @property
    def market(self) -> Mapping[str, MarketStateView]:
        return self._market

    @property
    def account(self) -> Mapping[str, AccountStateView]:
        return self._account

    @property
    def orders(self) -> Mapping[str, Mapping[str, WorkingOrderView]]:
        return self._orders

    @property
    def fills(self) -> Mapping[str, tuple[FillEvent, ...]]:
        return self._fills

    @property
    def fill_cum_qty(self) -> Mapping[str, Mapping[str, float]]:
        return self._fill_cum_qty

    def get_mid(self, instrument: str) -> float:
        market = self._market.get(instrument)
        return 0.0 if market is None else market.mid

    def get_contract_size(self, instrument: str) -> float:
        market = self._market.get(instrument)
        return 1.0 if market is None else market.contract_size

    def get_tick_size(self, instrument: str) -> float:
        market = self._market.get(instrument)
        return 0.0 if market is None else market.tick_size

    def get_lot_size(self, instrument: str) -> float:
        market = self._market.get(instrument)
        return 0.0 if market is None else market.lot_size

    def get_total_equity(self) -> float:
        return float(sum(account.equity for account in self._account.values()))

    def get_total_pnl(self) -> float:
        return float(sum(account.realized_pnl for account in self._account.values()))


class StrategyState:
    """High-level deterministic Strategy state keyed by instrument."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

        self.market: dict[str, MarketState] = {}
        self.account: dict[str, AccountState] = {}
        self.orders: dict[str, dict[str, WorkingOrder]] = {}
        self.fills: dict[str, deque[FillEvent]] = {}
        self.fill_cum_qty: dict[str, dict[str, float]] = {}
        self.queued_intents: dict[str, deque[QueuedIntent]] = {}
        self.inflight: dict[str, dict[str, InflightInfo]] = {}
        self.canonical_orders: dict[tuple[str, str], CanonicalOrderProjection] = {}
        self.last_sent_intents: dict[str, dict[str, tuple[int, str]]] = {}
        self.rolling_equity: deque[tuple[int, float]] = deque()

        self._last_realized_pnl: dict[str, float] = {}
        self._last_exposure: dict[str, float] = {}
        self.last_ts_ns_local: int = 0
        self._last_processing_position_index: int | None = None

    def update_timestamp(self, ts_ns_local: int) -> None:
        self.last_ts_ns_local = max(self.last_ts_ns_local, ts_ns_local)

    @property
    def sim_ts_ns_local(self) -> int:
        """Canonical monotone simulation time (ns, local axis)."""
        return self.last_ts_ns_local

    def _advance_processing_position(self, position: ProcessingPosition) -> None:
        last = self._last_processing_position_index
        next_index = position.index
        if last is not None and next_index <= last:
            raise ValueError(
                "Non-monotonic ProcessingPosition index: "
                f"received {next_index} after {last}."
            )
        self._last_processing_position_index = next_index

    def mark_intent_sent(self, instrument: str, client_order_id: str, intent_type: str) -> None:
        """Record that an intent was sent to the execution layer."""
        bucket = self.last_sent_intents.get(instrument)
        if bucket is None:
            bucket = {}
            self.last_sent_intents[instrument] = bucket

        ts_now = self.last_ts_ns_local
        bucket[client_order_id] = (ts_now, intent_type)

        inflight_bucket = self.inflight.get(instrument)
        if inflight_bucket is None:
            inflight_bucket = {}
            self.inflight[instrument] = inflight_bucket
        inflight_bucket[client_order_id] = InflightInfo(action=intent_type, ts_sent_ns_local=ts_now)

        if intent_type != "new":
            return

        key = (instrument, client_order_id)
        if key in self.canonical_orders:
            return
        self.canonical_orders[key] = CanonicalOrderProjection(
            instrument=instrument,
            client_order_id=client_order_id,
            state="submitted",
            submitted_ts_ns_local=ts_now,
            updated_ts_ns_local=ts_now,
        )

    def apply_order_submitted_event(self, event: OrderSubmittedEvent) -> None:
        """Reduce canonical submitted-entry into active-order projections."""
        self.update_timestamp(event.ts_ns_local_dispatch)
        key = (event.instrument, event.client_order_id)
        projection = self.canonical_orders.get(key)
        if projection is None:
            projection = CanonicalOrderProjection(
                instrument=event.instrument,
                client_order_id=event.client_order_id,
                state="submitted",
                submitted_ts_ns_local=event.ts_ns_local_dispatch,
                updated_ts_ns_local=event.ts_ns_local_dispatch,
            )
            self.canonical_orders[key] = projection

        projection.state = "submitted"
        projection.updated_ts_ns_local = max(
            projection.updated_ts_ns_local, event.ts_ns_local_dispatch
        )
        projection.side = event.side
        projection.intended_price = event.intended_price.value
        projection.intended_qty = event.intended_qty.value

        order_bucket = self.orders.setdefault(event.instrument, {})
        order_bucket[event.client_order_id] = WorkingOrder(
            instrument=event.instrument,
            client_order_id=event.client_order_id,
            side=event.side,
            intended_price=event.intended_price.value,
            intended_qty=event.intended_qty.value,
            cum_filled_qty=0.0,
            remaining_qty=event.intended_qty.value,
            state="submitted",
            submitted_ts_ns_local=event.ts_ns_local_dispatch,
            updated_ts_ns_local=event.ts_ns_local_dispatch,
        )
        self._clear_inflight(event.instrument, event.client_order_id)

    def _apply_terminal_order_event(
        self,
        *,
        instrument: str,
        client_order_id: str,
        ts_ns_local_feedback: int,
        terminal_state: str,
    ) -> None:
        self.update_timestamp(ts_ns_local_feedback)

        order_bucket = self.orders.get(instrument)
        if order_bucket is not None:
            order_bucket.pop(client_order_id, None)

        key = (instrument, client_order_id)
        projection = self.canonical_orders.get(key)
        if projection is None:
            projection = CanonicalOrderProjection(
                instrument=instrument,
                client_order_id=client_order_id,
                state=terminal_state,
                submitted_ts_ns_local=ts_ns_local_feedback,
                updated_ts_ns_local=ts_ns_local_feedback,
            )
            self.canonical_orders[key] = projection

        projection.state = terminal_state
        projection.updated_ts_ns_local = max(
            projection.updated_ts_ns_local, ts_ns_local_feedback
        )

        self._clear_inflight(instrument, client_order_id)

    def apply_order_canceled_event(self, event: OrderCanceledEvent) -> None:
        """Reduce canonical canceled-order feedback into terminal order projection."""
        self._apply_terminal_order_event(
            instrument=event.instrument,
            client_order_id=event.client_order_id,
            ts_ns_local_feedback=event.ts_ns_local_feedback,
            terminal_state="canceled",
        )

    def apply_order_rejected_event(self, event: OrderRejectedEvent) -> None:
        """Reduce canonical rejected-order feedback into terminal order projection."""
        self._apply_terminal_order_event(
            instrument=event.instrument,
            client_order_id=event.client_order_id,
            ts_ns_local_feedback=event.ts_ns_local_feedback,
            terminal_state="rejected",
        )

    def apply_order_expired_event(self, event: OrderExpiredEvent) -> None:
        """Reduce canonical expired-order feedback into terminal order projection."""
        self._apply_terminal_order_event(
            instrument=event.instrument,
            client_order_id=event.client_order_id,
            ts_ns_local_feedback=event.ts_ns_local_feedback,
            terminal_state="expired",
        )

    def apply_control_time_event(self, event: ControlTimeEvent) -> None:
        """Reduce canonical control-time Event without side effects."""
        self.update_timestamp(event.ts_ns_local_control)

    def _clear_inflight(self, instrument: str, client_order_id: str) -> None:
        inflight_bucket = self.inflight.get(instrument)
        if inflight_bucket is None:
            return
        inflight_bucket.pop(client_order_id, None)

    def has_inflight(self, instrument: str, client_order_id: str) -> bool:
        inflight_bucket = self.inflight.get(instrument)
        if inflight_bucket is None:
            return False
        return client_order_id in inflight_bucket

    def update_market(
        self,
        instrument: str,
        best_bid: float,
        best_ask: float,
        best_bid_qty: float,
        best_ask_qty: float,
        tick_size: float,
        lot_size: float,
        contract_size: float,
        *,
        ts_ns_local: int,
        ts_ns_exch: int,
    ) -> None:
        m = self.market.get(instrument)
        if m is None:
            m = MarketState()
            self.market[instrument] = m
        if ts_ns_local < m.last_ts_ns_local:
            return
        if ts_ns_local == m.last_ts_ns_local and ts_ns_exch <= m.last_ts_ns_exch:
            return
        m.last_ts_ns_local = ts_ns_local
        m.last_ts_ns_exch = ts_ns_exch
        m.best_bid = best_bid
        m.best_ask = best_ask
        m.best_bid_qty = best_bid_qty
        m.best_ask_qty = best_ask_qty
        m.tick_size = tick_size
        m.lot_size = lot_size
        m.contract_size = contract_size
        m.mid = 0.5 * (m.best_bid + m.best_ask) if m.best_bid > 0.0 and m.best_ask > 0.0 else 0.0
        self.update_timestamp(ts_ns_local)

    def _update_market_from_positioned_canonical_event(
        self,
        instrument: str,
        best_bid: float,
        best_ask: float,
        best_bid_qty: float,
        best_ask_qty: float,
        tick_size: float,
        lot_size: float,
        contract_size: float,
        *,
        ts_ns_local: int,
        ts_ns_exch: int,
    ) -> None:
        m = self.market.get(instrument)
        if m is None:
            m = MarketState()
            self.market[instrument] = m
        m.last_ts_ns_local = ts_ns_local
        m.last_ts_ns_exch = ts_ns_exch
        m.best_bid = best_bid
        m.best_ask = best_ask
        m.best_bid_qty = best_bid_qty
        m.best_ask_qty = best_ask_qty
        m.tick_size = tick_size
        m.lot_size = lot_size
        m.contract_size = contract_size
        m.mid = 0.5 * (m.best_bid + m.best_ask) if m.best_bid > 0.0 and m.best_ask > 0.0 else 0.0
        self.update_timestamp(ts_ns_local)

    def get_mid(self, instrument: str) -> float:
        m = self.market.get(instrument)
        return 0.0 if m is None else m.mid

    def get_contract_size(self, instrument: str) -> float:
        m = self.market.get(instrument)
        return 1.0 if m is None else m.contract_size

    def get_tick_size(self, instrument: str) -> float:
        m = self.market.get(instrument)
        return 0.0 if m is None else m.tick_size

    def get_lot_size(self, instrument: str) -> float:
        m = self.market.get(instrument)
        return 0.0 if m is None else m.lot_size

    def update_account(
        self,
        instrument: str,
        position: float,
        balance: float,
        fee: float,
        trading_volume: float,
        trading_value: float,
        num_trades: int,
    ) -> None:
        a = self.account.get(instrument)
        if a is None:
            a = AccountState()
            self.account[instrument] = a
        a.position = position
        a.balance = balance
        a.fee = fee
        a.trading_volume = trading_volume
        a.trading_value = trading_value
        a.num_trades = num_trades

        mid = self.get_mid(instrument)
        a.equity = a.balance + a.position * mid * self.get_contract_size(instrument)
        if a.initial_equity == 0.0 and mid > 0.0:
            a.initial_equity = a.equity
        a.realized_pnl = (a.equity - a.initial_equity) if a.initial_equity != 0.0 else 0.0
        self._update_rolling_equity(ts_ns_local=self.last_ts_ns_local)

    def _update_rolling_equity(self, *, ts_ns_local: int) -> None:
        if ts_ns_local <= 0:
            return
        total_equity = sum(x.equity for x in self.account.values())
        dq = self.rolling_equity
        if dq:
            last_ts, last_eq = dq[-1]
            if ts_ns_local < last_ts:
                return
            if ts_ns_local == last_ts and total_equity == last_eq:
                return
        dq.append((ts_ns_local, total_equity))

    def get_total_equity(self) -> float:
        return float(sum(a.equity for a in self.account.values()))

    def get_rolling_loss(self, *, now_ts_ns_local: int, window_ns: int) -> float | None:
        if window_ns <= 0:
            return None
        dq = self.rolling_equity
        if not dq:
            dq.append((now_ts_ns_local, self.get_total_equity()))
            return 0.0
        cutoff = now_ts_ns_local - window_ns
        while len(dq) > 1 and dq[0][0] < cutoff:
            dq.popleft()
        start_ts, start_eq = dq[0]
        if start_ts > now_ts_ns_local:
            return None
        cur_eq = self.get_total_equity()
        return float(cur_eq - start_eq)

    def get_total_pnl(self) -> float:
        return float(sum(a.realized_pnl for a in self.account.values()))

    def apply_fill_event(self, event: FillEvent, *, max_keep: int = 10_000) -> None:
        """Reduce canonical fill deltas into fill and active-order projections."""
        self.update_timestamp(event.ts_ns_local)
        instrument = event.instrument
        client_order_id = event.client_order_id

        bucket = self.fill_cum_qty.setdefault(instrument, {})
        cum_qty = event.cum_filled_qty.value
        last_cum = bucket.get(client_order_id)
        if last_cum is not None and cum_qty <= last_cum + 1e-12:
            return
        bucket[client_order_id] = cum_qty

        dq = self.fills.setdefault(instrument, deque())
        dq.append(event)
        while len(dq) > max_keep:
            dq.popleft()

        order_bucket = self.orders.get(instrument)
        if order_bucket is not None:
            working = order_bucket.get(client_order_id)
            if working is not None:
                working.cum_filled_qty = cum_qty
                if event.remaining_qty is not None:
                    working.remaining_qty = event.remaining_qty.value
                else:
                    working.remaining_qty = max(0.0, working.intended_qty - cum_qty)
                working.state = "filled" if working.remaining_qty <= 1e-12 else "partially_filled"
                working.updated_ts_ns_local = event.ts_ns_local
                if working.state == "filled":
                    order_bucket.pop(client_order_id, None)
                    self._clear_inflight(instrument, client_order_id)

        projection = self.canonical_orders.get((instrument, client_order_id))
        if projection is not None and event.ts_ns_local >= projection.updated_ts_ns_local:
            if event.remaining_qty is not None and event.remaining_qty.value <= 1e-12:
                projection.state = "filled"
            else:
                projection.state = "partially_filled"
            projection.updated_ts_ns_local = event.ts_ns_local

        self._event_bus.emit(event)

    def apply_order_execution_feedback_event(
        self,
        event: OrderExecutionFeedbackEvent,
    ) -> None:
        """Reduce canonical execution feedback into account state only."""
        self.update_timestamp(event.ts_ns_local_feedback)
        self.update_account(
            instrument=event.instrument,
            position=event.position,
            balance=event.balance,
            fee=event.fee,
            trading_volume=event.trading_volume,
            trading_value=event.trading_value,
            num_trades=event.num_trades,
        )

    def get_working_order_snapshot(self, instrument: str, client_order_id: str) -> WorkingOrder | None:
        bucket = self.orders.get(instrument)
        if bucket is None:
            return None
        return bucket.get(client_order_id)

    def has_working_order(self, instrument: str, client_order_id: str) -> bool:
        bucket = self.orders.get(instrument)
        if bucket is None:
            return False
        return client_order_id in bucket

    def has_queued_intent(self, instrument: str, client_order_id: str) -> bool:
        q = self.queued_intents.get(instrument)
        if q is None:
            return False
        key = f"order:{client_order_id}"
        return any(qi.logical_key == key for qi in q)

    def queued_intents_snapshot(self, instrument: str | None = None) -> tuple[OrderIntent, ...]:
        if instrument is not None:
            q = self.queued_intents.get(instrument)
            if q is None:
                return ()
            return tuple(qi.intent for qi in q)
        snapshots: list[OrderIntent] = []
        for instrument_key in sorted(self.queued_intents):
            snapshots.extend(qi.intent for qi in self.queued_intents[instrument_key])
        return tuple(snapshots)

    def pop_queued_intents_for_order(self, instrument: str, client_order_id: str) -> list[QueuedIntent]:
        q = self.queued_intents.get(instrument)
        if q is None or not q:
            return []
        key = f"order:{client_order_id}"
        removed: list[QueuedIntent] = []
        kept: deque[QueuedIntent] = deque()
        for qi in q:
            if qi.logical_key == key:
                removed.append(qi)
            else:
                kept.append(qi)
        self.queued_intents[instrument] = kept
        return removed

    def find_queued_new_intent(self, instrument: str, client_order_id: str) -> NewOrderIntent | None:
        q = self.queued_intents.get(instrument)
        if q is None:
            return None
        key = f"order:{client_order_id}"
        for qi in q:
            if qi.logical_key == key and qi.intent.intent_type == "new":
                return qi.intent  # type: ignore[return-value]
        return None

    def _intent_priority(self, intent: OrderIntent) -> int:
        if intent.intent_type == "cancel":
            return 0
        if intent.intent_type == "replace":
            return 1
        if intent.intent_type == "new":
            return 2
        return 9

    def _compute_logical_key(self, intent: OrderIntent) -> str:
        return f"order:{intent.client_order_id}"

    def merge_intents_into_queue(
        self,
        instrument: str,
        intents: Iterable[OrderIntent],
    ) -> tuple[list[OrderIntent], list[tuple[OrderIntent, OrderIntent]], list[OrderIntent]]:
        q: deque[QueuedIntent] = self.queued_intents.setdefault(instrument, deque())
        queued: list[OrderIntent] = []
        replaced_in_queue: list[tuple[OrderIntent, OrderIntent]] = []
        dropped: list[OrderIntent] = []

        def _matching_entries(key: str) -> list[QueuedIntent]:
            return [qi for qi in q if qi.logical_key == key]

        for intent in intents:
            key = self._compute_logical_key(intent)
            prio = self._intent_priority(intent)
            matches = _matching_entries(key)
            has_cancel = any(qi.intent.intent_type == "cancel" for qi in matches)
            has_replace = any(qi.intent.intent_type == "replace" for qi in matches)
            has_new = any(qi.intent.intent_type == "new" for qi in matches)

            if intent.intent_type == "cancel":
                for qi in list(matches):
                    q.remove(qi)
                    replaced_in_queue.append((qi.intent, intent))
                q.append(
                    QueuedIntent(
                        intent=intent,
                        queued_at_ts_ns=intent.ts_ns_local,
                        logical_key=key,
                        priority=prio,
                    )
                )
                queued.append(intent)
                continue

            if intent.intent_type == "replace":
                if has_cancel:
                    dropped.append(intent)
                    continue
                for qi in list(matches):
                    if qi.intent.intent_type in ("new", "replace"):
                        q.remove(qi)
                        replaced_in_queue.append((qi.intent, intent))
                q.append(
                    QueuedIntent(
                        intent=intent,
                        queued_at_ts_ns=intent.ts_ns_local,
                        logical_key=key,
                        priority=prio,
                    )
                )
                queued.append(intent)
                continue

            if intent.intent_type == "new":
                if has_cancel or has_replace:
                    dropped.append(intent)
                    continue
                if has_new:
                    for qi in list(matches):
                        if qi.intent.intent_type == "new":
                            q.remove(qi)
                            replaced_in_queue.append((qi.intent, intent))
                q.append(
                    QueuedIntent(
                        intent=intent,
                        queued_at_ts_ns=intent.ts_ns_local,
                        logical_key=key,
                        priority=prio,
                    )
                )
                queued.append(intent)
                continue

            dropped.append(intent)

        return queued, replaced_in_queue, dropped
