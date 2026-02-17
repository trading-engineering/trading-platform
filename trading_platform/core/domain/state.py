"""Runtime strategy state management.

This module maintains best-effort market, account, order, and queue state
derived from venue snapshots and events. It is intentionally stateful and
optimized for correctness and determinism rather than minimal complexity.
"""

# pylint: disable=line-too-long,too-many-instance-attributes,too-many-public-methods
# pylint: disable=missing-function-docstring,too-many-locals,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-return-statements
# pylint: disable=too-many-boolean-expressions
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable

from trading_platform.core.domain.order_state_machine import is_valid_transition
from trading_platform.core.domain.slots import SlotKey, stable_slot_order_id
from trading_platform.core.domain.types import OrderStateEvent
from trading_platform.core.events.events import (
    DerivedFillEvent,
    DerivedPnLEvent,
    ExposureDerivedEvent,
    OrderStateTransitionEvent,
)

if TYPE_CHECKING:
    from trading_platform.core.domain.types import FillEvent, NewOrderIntent, OrderIntent
    from trading_platform.core.events.event_bus import EventBus


# ---------------------------------------------------------------------------
# Internal state models
#
# These models are intentionally NOT part of the JSON-schema "source of truth".
# They exist to hold runtime state derived from hftbacktest snapshots/events.
# ---------------------------------------------------------------------------

TERMINAL_ORDER_STATES: set[str] = {"filled", "canceled", "expired", "rejected"}

_UNKNOWN_CCY: str = "UNKNOWN"
_DEFAULT_QTY_UNIT: str = "contracts"


@dataclass(slots=True)
class OrderSnapshot:
    """Best-effort internal order snapshot."""

    instrument: str
    client_order_id: str

    ts_ns_exch: int
    ts_ns_local: int

    order_type: str
    time_in_force: str
    state_type: str

    side: str

    intended_price: float
    filled_price: float

    intended_qty: float
    cum_filled_qty: float
    remaining_qty: float

    # Best-effort request marker from hftbacktest snapshots.
    # Convention: 0 indicates no in-flight request.
    req: int = 0


@dataclass(slots=True)
class QueuedIntent:
    """An intent stored for later sending (data only, no policy)."""

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

    # Receipt (local) time is the strategy time axis.
    last_ts_ns_local: int = 0
    # Venue time is used as a tie-breaker for replacement-style updates.
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
    """Best-effort account values from hftbacktest state_values()."""

    position: float = 0.0
    balance: float = 0.0
    fee: float = 0.0
    trading_volume: float = 0.0
    trading_value: float = 0.0
    num_trades: int = 0

    equity: float = 0.0
    initial_equity: float = 0.0
    realized_pnl: float = 0.0


class StrategyState:
    """High-level strategy state keyed by instrument."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

        self.market: dict[str, MarketState] = {}
        self.account: dict[str, AccountState] = {}
        self.orders: dict[str, dict[str, OrderSnapshot]] = {}
        # Accumulates OrderStateEvents since the last consumer pop.
        # Used to surface edge-events such as "replaced" to strategies.
        self.order_events: dict[str, deque[OrderStateEvent]] = {}
        self.fills: dict[str, deque[FillEvent]] = {}
        # Best-effort idempotence for fill deltas.
        # Tracks last observed cumulative filled quantity per (instrument, client_order_id).
        self.fill_cum_qty: dict[str, dict[str, float]] = {}
        self.queued_intents: dict[str, deque[QueuedIntent]] = {}

        self.inflight: dict[str, dict[str, InflightInfo]] = {}

        # Best-effort tracking of last sent intent per (instrument, client_order_id).
        # Mapping: instrument -> client_order_id -> (ts_ns_local, intent_type)
        self.last_sent_intents: dict[str, dict[str, tuple[int, str]]] = {}

        # Rolling equity series for rolling-loss checks.
        # Stores (ts_ns_local, total_equity).
        self.rolling_equity: deque[tuple[int, float]] = deque()

        self._last_realized_pnl: dict[str, float] = {}
        self._last_exposure: dict[str, float] = {}

        # Canonical monotone simulation time (local/receipt axis).
        # This is the single time reference used for gating and risk decisions.
        self.last_ts_ns_local: int = 0

    # ---- Timestamp ----
    def update_timestamp(self, ts_ns_local: int) -> None:
        # Monotone simulation time: never regress.
        # Using max() here makes the policy explicit.
        self.last_ts_ns_local = max(self.last_ts_ns_local, ts_ns_local)

    @property
    def sim_ts_ns_local(self) -> int:
        """Canonical monotone simulation time (ns, local axis)."""
        return self.last_ts_ns_local

    def mark_intent_sent(self, instrument: str, client_order_id: str, intent_type: str) -> None:
        """Record that an intent was sent to the execution layer.

        This is used for best-effort inflight handling. hftbacktest provides snapshots
        (status/req) rather than explicit ACK events, so inflight is cleared heuristically
        as soon as subsequent snapshots indicate completion.
        """

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

    def _clear_inflight(self, instrument: str, client_order_id: str) -> None:
        inflight_bucket = self.inflight.get(instrument)
        if inflight_bucket is None:
            return
        inflight_bucket.pop(client_order_id, None)

    def has_inflight(self, instrument: str, client_order_id: str) -> bool:
        """Return True if an order id currently has an inflight marker."""
        inflight_bucket = self.inflight.get(instrument)
        if inflight_bucket is None:
            return False
        return client_order_id in inflight_bucket

    def _maybe_clear_inflight_from_snapshot(self, event: OrderStateEvent) -> None:
        """Heuristically clear inflight markers based on snapshot state.

        This avoids leaking inflight entries when orders progress from pending to
        working/terminal states.
        """

        # NOTE:
        # Inflight clearing is heuristic because the venue provides snapshots,
        # not explicit ACK / completion events.
        # This is sufficient for backtest and research purposes, but does NOT
        # guarantee exact ACK ordering as in a live FIX/WebSocket venue.

        inflight_bucket = self.inflight.get(event.instrument)
        if inflight_bucket is None:
            return

        info = inflight_bucket.get(event.client_order_id)
        if info is None:
            return

        if event.ts_ns_local < info.ts_sent_ns_local:
            return

        req_val = 0
        if isinstance(event.raw, dict):
            req_val = event.raw.get("req", 0)

        # If the snapshot indicates no active request, clear inflight as soon
        # as the snapshot is at or after the send time.
        if req_val == 0:
            if event.state_type in TERMINAL_ORDER_STATES or event.state_type == "rejected":
                self._clear_inflight(event.instrument, event.client_order_id)
                return

            if event.state_type in ("accepted", "working", "partially_filled"):
                self._clear_inflight(event.instrument, event.client_order_id)
                return

        if event.state_type in TERMINAL_ORDER_STATES or event.state_type == "rejected":
            self._clear_inflight(event.instrument, event.client_order_id)
            return

        if info.action == "new" and event.state_type in ("accepted", "working", "partially_filled"):
            self._clear_inflight(event.instrument, event.client_order_id)
            return

        if info.action == "cancel" and event.state_type == "canceled":
            self._clear_inflight(event.instrument, event.client_order_id)
            return

        if info.action == "replace" and event.state_type in ("working", "partially_filled"):
            self._clear_inflight(event.instrument, event.client_order_id)

    # ---- Market ----
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

        # Replacement-style update policy:
        # - primary sort key: receipt time (local)
        # - tie-breaker: venue time (best-effort)
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

        if m.best_bid > 0.0 and m.best_ask > 0.0:
            m.mid = 0.5 * (m.best_bid + m.best_ask)
        else:
            m.mid = 0.0

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

    # ---- Account ----
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

        # ---- Derived Realized PnL detection ----
        last = self._last_realized_pnl.get(instrument)
        cur = a.realized_pnl

        if last is None:
            # First observation: initialize baseline, no event
            self._last_realized_pnl[instrument] = cur
        elif cur != last:
            self._event_bus.emit(
                DerivedPnLEvent(
                    ts_ns_local=self.last_ts_ns_local,
                    instrument=instrument,
                    delta_pnl=cur - last,
                    cum_realized_pnl=cur,
                )
            )
            self._last_realized_pnl[instrument] = cur

        # ---- Derived Exposure detection ----
        mid = self.get_mid(instrument)
        contract_size = self.get_contract_size(instrument)
        exposure = a.position * mid * contract_size

        last_exposure = self._last_exposure.get(instrument)

        if last_exposure is None:
            # First observation establishes baseline
            self._last_exposure[instrument] = exposure
        elif exposure != last_exposure:
            self._event_bus.emit(
                ExposureDerivedEvent(
                    ts_ns_local=self.last_ts_ns_local,
                    instrument=instrument,
                    exposure=exposure,
                    delta_exposure=exposure - last_exposure,
                )
            )
            self._last_exposure[instrument] = exposure

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

    # ---- Orders ----
    @staticmethod
    def _should_drop_transition_update(cur: OrderSnapshot, event: OrderStateEvent) -> bool:
        """Return True if a late transition-style update should be ignored.

        Transition updates (live-style) must not be dropped purely because they
        arrived late. They should only be dropped if they are idempotent and do
        not advance the current snapshot.

        Accepted as progress:
        - terminal state applied when current state is non-terminal
        - increased cumulative filled quantity
        - decreased remaining quantity
        - clearing a request marker (req!=0 -> req==0)
        """

        if event.state_type in TERMINAL_ORDER_STATES and cur.state_type not in TERMINAL_ORDER_STATES:
            return False

        event_cum = 0.0
        if event.cum_filled_qty is not None:
            event_cum = event.cum_filled_qty.value

        if event_cum > cur.cum_filled_qty:
            return False

        event_remaining = 0.0
        if event.remaining_qty is not None:
            event_remaining = event.remaining_qty.value
        else:
            intended_qty = event.intended_qty.value
            event_remaining = max(0.0, intended_qty - event_cum)

        if event_remaining < cur.remaining_qty:
            return False

        current_req = 0
        if isinstance(event.raw, dict):
            try:
                current_req = event.raw["req"]  # type: ignore[arg-type]
            except KeyError:
                current_req = 0

        if cur.req != current_req and current_req == 0:
            return False

        # Idempotent late update: ignore when it is strictly older.
        if event.ts_ns_local < cur.ts_ns_local:
            return True
        if event.ts_ns_local == cur.ts_ns_local and event.ts_ns_exch < cur.ts_ns_exch:
            return True

        # Equal/newer but not progressing: keep latest by timestamp.
        return False

    def apply_order_state_event(self, event: OrderStateEvent) -> None:
        events_bucket = self.order_events.setdefault(event.instrument, deque())
        bucket = self.orders.setdefault(event.instrument, {})
        cur = bucket.get(event.client_order_id)

        raw_dict: dict[str, object | None] = event.raw if isinstance(event.raw, dict) else None

        current_req = 0
        source = "transition"
        if raw_dict is not None:
            try:
                current_req = raw_dict["req"]  # type: ignore[arg-type]
            except KeyError:
                current_req = 0

            try:
                source = raw_dict["source"]  # type: ignore[arg-type]
            except KeyError:
                source = "transition"

        prev_req = cur.req if cur is not None else 0

        inflight_bucket = self.inflight.get(event.instrument)
        inflight_info = None if inflight_bucket is None else inflight_bucket.get(event.client_order_id)

        if (
            inflight_info is not None
            and inflight_info.action == "replace"
            and prev_req != 0
            and current_req == 0
            and event.state_type not in TERMINAL_ORDER_STATES
            and event.state_type != "rejected"
        ):
            replaced_event = event.model_copy(update={"state_type": "replaced"})
            events_bucket.append(replaced_event)

        if cur is not None:
            # Late-update policy
            #
            # - Replacement-style events ("snapshot") may be safely treated as
            #   overwrites. Older snapshots must not overwrite newer snapshots.
            # - Transition-style events ("transition") should not be dropped
            #   purely because they are late. A late terminal update or a late
            #   fill progression must still be applied.
            is_snapshot = source == "snapshot"

            if is_snapshot:
                if event.ts_ns_local < cur.ts_ns_local:
                    return
                if event.ts_ns_local == cur.ts_ns_local and event.ts_ns_exch < cur.ts_ns_exch:
                    return
            else:
                if self._should_drop_transition_update(cur, event):
                    return

        # Treat 'replaced' as an edge event. The order identity remains and the
        # snapshot should continue to exist in state.
        if event.state_type == "replaced":
            effective_state = "working" if cur is None else cur.state_type
            event = event.model_copy(update={"state_type": effective_state})

        # Order lifecycle state transition validation (observability only)
        prev_state: str | None = None if cur is None else cur.state_type
        next_state: str = event.state_type

        if not is_valid_transition(prev_state, next_state):
            self._event_bus.emit(
                OrderStateTransitionEvent(
                    ts_ns_local=event.ts_ns_local,
                    instrument=event.instrument,
                    client_order_id=event.client_order_id,
                    prev_state=prev_state,
                    next_state=next_state,
                )
            )

        # Derived Fill detection (snapshot-based)
        if (
            cur is not None
            and event.cum_filled_qty is not None
        ):
            prev_cum = cur.cum_filled_qty
            new_cum = event.cum_filled_qty.value

            if new_cum > prev_cum:
                self._event_bus.emit(
                    DerivedFillEvent(
                        ts_ns_local=event.ts_ns_local,
                        instrument=event.instrument,
                        client_order_id=event.client_order_id,
                        side=event.side,
                        delta_qty=new_cum - prev_cum,
                        cum_qty=new_cum,
                        price=(
                            event.filled_price.value
                            if event.filled_price is not None
                            else None
                        ),
                    )
                )

        events_bucket.append(event)

        intended_price = event.intended_price.value
        filled_price = event.filled_price.value if event.filled_price is not None else 0.0

        intended_qty = event.intended_qty.value
        cum_filled_qty = event.cum_filled_qty.value if event.cum_filled_qty is not None else 0.0
        remaining_qty = (
            event.remaining_qty.value
            if event.remaining_qty is not None
            else max(0.0, intended_qty - cum_filled_qty)
        )

        snap = OrderSnapshot(
            instrument=event.instrument,
            client_order_id=event.client_order_id,
            ts_ns_exch=event.ts_ns_exch,
            ts_ns_local=event.ts_ns_local,
            order_type=event.order_type,
            time_in_force=event.time_in_force,
            state_type=event.state_type,
            side=event.side,
            intended_price=intended_price,
            filled_price=filled_price,
            intended_qty=intended_qty,
            cum_filled_qty=cum_filled_qty,
            remaining_qty=remaining_qty,
            req=current_req,
        )

        # Clear inflight heuristically before any early return.
        self._maybe_clear_inflight_from_snapshot(event)

        if snap.state_type in TERMINAL_ORDER_STATES:
            bucket.pop(event.client_order_id, None)
            self._clear_inflight(event.instrument, event.client_order_id)
            last_bucket = self.last_sent_intents.get(event.instrument)
            if last_bucket is not None:
                last_bucket.pop(event.client_order_id, None)
            return

        bucket[event.client_order_id] = snap

    # ---- Fills ----

    # NOTE:
    # Currently unused.
    # hftbacktest does not emit explicit FillEvent deltas; fills are inferred
    # indirectly from order state snapshots instead.
    # This method is reserved for event-driven backends or live trading venues
    # that provide fill-level events.
    def apply_fill_event(self, event: FillEvent, *, max_keep: int = 10_000) -> None:
        instrument = event.instrument
        client_order_id = event.client_order_id

        bucket = self.fill_cum_qty[instrument] if instrument in self.fill_cum_qty else None
        if bucket is None:
            bucket = {}
            self.fill_cum_qty[instrument] = bucket

        cum_qty = event.cum_filled_qty.value
        last_cum: float = bucket[client_order_id] if client_order_id in bucket else None
        if last_cum is not None:
            # Fill events are deltas. Duplicates commonly repeat the same cumulative filled.
            # Late/out-of-order fills can arrive with a smaller cumulative filled.
            # Both cases should be idempotent no-ops.
            if cum_qty <= last_cum + 1e-12:
                return

        bucket[client_order_id] = cum_qty

        dq = self.fills[instrument] if instrument in self.fills else None
        if dq is None:
            dq = deque()
            self.fills[instrument] = dq

        dq.append(event)
        while len(dq) > max_keep:
            dq.popleft()

        self._event_bus.emit(event)

    def ingest_order_snapshots(self, instrument: str, orders_snapshot_iter: Iterable[object]) -> None:
        """Ingest hftbacktest order snapshots and reduce them into internal state.

        hftbacktest provides *snapshots* (not deltas). We translate each snapshot into
        an OrderStateEvent (snapshot) and feed it into apply_order_state_event().
        """

        def map_status(status: int, req: int, client_order_id: str) -> str:
            """Best-effort mapping from hftbacktest (status, req) to schema state.

            Design: terminal status wins. If a request marker is present (req!=0),
            treat this as "in-flight". In that case, "pending_new" is used only
            for in-flight NEW actions; all other in-flight actions map to "accepted".
            """

            if status == 3:
                return "filled"
            if status == 4:
                return "canceled"
            if status == 5:
                return "expired"

            if req != 0:
                inflight_bucket = self.inflight.get(instrument)
                inflight_info = None if inflight_bucket is None else inflight_bucket.get(client_order_id)
                if inflight_info is not None and inflight_info.action == "new":
                    return "pending_new"
                return "accepted"

            if status == 0:
                return "accepted"
            if status == 1:
                return "working"
            if status == 2:
                return "partially_filled"

            return "rejected"

        # hftbacktest "values()" often returns a custom iterator with has_next/get
        if hasattr(orders_snapshot_iter, "has_next") and hasattr(orders_snapshot_iter, "get"):
            it = orders_snapshot_iter

            def _next() -> object | None:
                return it.get() if it.has_next() else None

            while True:
                o = _next()
                if o is None:
                    break
                self._ingest_one_hft_order(instrument, o, map_status)
            return

        # Otherwise assume a normal Python iterable
        for o in orders_snapshot_iter:
            self._ingest_one_hft_order(instrument, o, map_status)

    def _ingest_one_hft_order(
        self,
        instrument: str,
        o: object,
        map_status: Callable[[int, int, int], str],
    ) -> None:
        """Translate a single hftbacktest order snapshot object into an OrderStateEvent."""

        # --- Map primitive enums to your schema enums ---
        order_type: str = "limit" if o.order_type == 0 else "market"

        # hftbacktest typically uses BUY=1, SELL=-1
        side: str = "buy" if o.side == 1 else "sell"

        tif: int = o.time_in_force
        if tif == 0:
            time_in_force = "GTC"
        elif tif == 1:
            time_in_force = "IOC"
        elif tif == 2:
            time_in_force = "FOK"
        elif tif == 3:
            time_in_force = "POST_ONLY"
        else:
            time_in_force = "GTC"

        req_val: int = o.req

        state_type: str = map_status(o.status, req_val, o.order_id)

        # --- Prices / quantities (schema requires structured objects) ---
        intended_price: dict[str, str | float] = {"currency": _UNKNOWN_CCY, "value": o.price}

        exec_price: float = o.exec_price
        filled_price: dict[str, str | float] = None if exec_price <= 0.0 else {"currency": _UNKNOWN_CCY, "value": exec_price}

        intended_qty: dict[str, str | float] = {"value": o.qty, "unit": _DEFAULT_QTY_UNIT}

        exec_qty: float = o.exec_qty
        cum_filled_qty: dict[str, str | float] = None if exec_qty <= 0.0 else {"value": exec_qty, "unit": _DEFAULT_QTY_UNIT}

        leaves_qty: float = o.leaves_qty
        remaining_qty: dict[str, str | float] = {"value": leaves_qty, "unit": _DEFAULT_QTY_UNIT}

        event = OrderStateEvent(
            ts_ns_exch=o.exch_timestamp,
            ts_ns_local=o.local_timestamp,
            instrument=instrument,
            client_order_id=str(o.order_id),
            order_type=order_type,
            state_type=state_type,
            side=side,
            intended_price=intended_price,
            filled_price=filled_price,
            intended_qty=intended_qty,
            cum_filled_qty=cum_filled_qty,
            remaining_qty=remaining_qty,
            time_in_force=time_in_force,
            reason=None,
            raw={"status": o.status, "req": req_val, "source": "snapshot"},
        )

        self.apply_order_state_event(event)

    def get_orders(self, instrument: str) -> dict[str, OrderSnapshot]:
        """Return active order snapshots for an instrument (read-only view)."""
        return self.orders.get(instrument, {})

    # NOTE:
    # Currently unused.
    # OrderStateEvents are accumulated for observability (e.g. replaced, invalid
    # transitions), but no strategy currently consumes edge-events explicitly.
    # This hook is reserved for strategies that require order lifecycle events.
    def pop_order_events(self, instrument: str) -> list[OrderStateEvent]:
        """Return and clear accumulated OrderStateEvents for an instrument.

        The engine calls the strategy without passing a per-event stream. Strategies
        that need edge-events (e.g. replaced) can consume them via this method.
        """

        dq = self.order_events.get(instrument)
        if dq is None or not dq:
            return []

        out: list[OrderStateEvent] = list(dq)
        dq.clear()
        return out

    def get_working_order_snapshot(self, instrument: str, client_order_id: str) -> OrderSnapshot | None:
        """Return an active order snapshot for an order id.

        Returns None if no active snapshot exists.
        """

        bucket = self.orders.get(instrument)
        if bucket is None:
            return None
        return bucket.get(client_order_id)

    # ---- Slot helpers (multi-level quoting) ----
    def slot_client_order_id(self, slot: SlotKey, namespace: str) -> str:
        """Return the stable client_order_id for a slot."""
        return stable_slot_order_id(slot, namespace=namespace)

    def is_slot_busy(self, slot: SlotKey, namespace: str) -> bool:
        """Return True if a slot is busy in queued ∪ working."""

        client_order_id = self.slot_client_order_id(slot, namespace=namespace)
        return self.is_order_id_busy(slot.instrument, client_order_id)

    # ---- Queue / existence helpers (C1) ----

    # NOTE:
    # Currently unused.
    # SlotKey helpers are intended for slot-based / multi-level market making
    # strategies with deterministic order identifiers.
    # No slot-based strategy is implemented at this time.
    def slot_key(self, instrument: str, side: str, level_index: int) -> SlotKey:
        """Create a SlotKey for a given instrument/side/level."""

        return SlotKey(instrument=str(instrument), side=str(side), level_index=int(level_index))

    # NOTE:
    # Currently unused.
    # Deterministic slot-based order IDs are reserved for slot-driven quoting
    # strategies. Current strategies generate order IDs explicitly.
    def slot_order_id(self, slot: SlotKey, namespace: str) -> str:
        """Return the deterministic client_order_id for a slot."""

        return stable_slot_order_id(slot, namespace=namespace)

    def is_order_id_busy(self, instrument: str, client_order_id: str) -> bool:
        """Return True if an order id exists in queued ∪ working."""

        return bool(
            self.has_working_order(instrument, client_order_id)
            or self.has_queued_intent(instrument, client_order_id)
        )

    # NOTE:
    # Currently unused.
    # Slot occupancy checks are only relevant for slot-based strategies
    # that enforce one active order per slot.
    def is_slot_key_busy(self, slot: SlotKey, namespace: str) -> bool:
        """Return True if a slot has a working order or queued intent."""

        order_id = stable_slot_order_id(slot, namespace=namespace)
        return self.is_order_id_busy(slot.instrument, order_id)

    def has_working_order(self, instrument: str, client_order_id: str) -> bool:
        """Check whether an active (non-terminal) order exists in working state."""
        bucket = self.orders.get(instrument)
        if bucket is None:
            return False
        return client_order_id in bucket

    def has_queued_intent(self, instrument: str, client_order_id: str) -> bool:
        """Check whether any queued intent exists for the given order id."""
        q = self.queued_intents.get(instrument)
        if q is None:
            return False
        key = f"order:{client_order_id}"
        return any(qi.logical_key == key for qi in q)

    def pop_queued_intents_for_order(self, instrument: str, client_order_id: str) -> list[QueuedIntent]:
        """Remove and return all queued intents for the given order id."""
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
        """Return the queued NEW intent for the given order id, if present."""
        q = self.queued_intents.get(instrument)
        if q is None:
            return None
        key = f"order:{client_order_id}"
        for qi in q:
            if qi.logical_key == key and qi.intent.intent_type == "new":
                return qi.intent  # type: ignore[return-value]
        return None

    def _intent_priority(self, intent: OrderIntent) -> int:
        """Lower number means higher priority for flushing."""
        if intent.intent_type == "cancel":
            return 0
        if intent.intent_type == "replace":
            return 1
        if intent.intent_type == "new":
            return 2
        return 9

    def _compute_logical_key(self, intent: OrderIntent) -> str:
        """Compute a stable key for queue replacement/deduplication.

        Contract:
        - All order lifecycle operations are keyed by client_order_id.
        - flags/target identifiers are intentionally not supported.
        """
        return f"order:{intent.client_order_id}"

    def merge_intents_into_queue(
        self,
        instrument: str,
        intents: Iterable[OrderIntent],
    ) -> tuple[list[OrderIntent], list[tuple[OrderIntent, OrderIntent]], list[OrderIntent]]:
        """Merge intents into the outbox queue with replacement semantics.

        This is OUTBOX DATA management only (no policy). The Risk/Gate decides what goes here.

        Replacement rules per logical_key:
        - CANCEL dominates:
            remove any queued NEW/REPLACE for that key and queue only CANCEL.
            if CANCEL already queued, additional CANCEL replaces the older CANCEL (keep latest).
        - REPLACE replaces queued NEW/REPLACE for that key.
            if CANCEL is queued, the REPLACE is dropped (cancel dominates).
        - NEW replaces queued NEW for that key.
            if REPLACE or CANCEL is queued, the NEW is dropped (new is obsolete).
        """
        q: deque[QueuedIntent] = self.queued_intents.setdefault(instrument, deque())

        queued: list[OrderIntent] = []
        replaced_in_queue: list[tuple[OrderIntent, OrderIntent]] = []
        dropped: list[OrderIntent] = []

        # Helper: find all queued entries matching key
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
                # Remove all existing entries for the key (including older cancel) and keep latest cancel only.
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
                # If a cancel is already queued, replace is obsolete.
                if has_cancel:
                    dropped.append(intent)
                    continue

                # Remove queued NEW/REPLACE for that key, keep only latest replace.
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
                # If cancel or replace is already queued, new is obsolete.
                if has_cancel or has_replace:
                    dropped.append(intent)
                    continue

                # Replace only queued NEW for that key (keep latest new).
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

            # Unknown intent types are dropped to avoid silent weirdness.
            dropped.append(intent)

        return queued, replaced_in_queue, dropped

    def pop_queued_intents(
        self,
        instrument: str,
        *,
        max_items: int | None = None,
    ) -> list[OrderIntent]:
        """Pop flush candidates from the outbox queue.

        Ordering:
        - priority (cancel -> replace -> new)
        - FIFO by queued_at_ts_ns within same priority

        This function removes the selected items from the queue and returns their intents.
        Gate may decide to re-queue them again if still rate-limited.
        """
        q: deque[QueuedIntent] = self.queued_intents.setdefault(instrument, deque())
        if not q:
            return []

        items = list(q)
        items.sort(key=lambda qi: (qi.priority, qi.queued_at_ts_ns))

        # Inflight gating: do not emit intents for order ids that currently have
        # an outbound request in flight. This keeps the queue stable and avoids
        # sending replace storms while ACKs are pending.
        filtered: list[QueuedIntent] = []
        for qi in items:
            if self.has_inflight(instrument, qi.intent.client_order_id):
                continue
            filtered.append(qi)

        if max_items is None:
            selected = filtered
        else:
            if max_items <= 0:
                return []
            selected = filtered[:max_items]

        selected_ids = {id(x) for x in selected}

        out: list[OrderIntent] = []
        new_q: deque[QueuedIntent] = deque()

        for qi in q:
            if id(qi) in selected_ids:
                out.append(qi.intent)
            else:
                new_q.append(qi)

        self.queued_intents[instrument] = new_q
        return out
