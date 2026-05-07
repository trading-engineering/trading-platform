"""Risk engine implementing hard risk checks and intent gating."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tradingchassis_core.core.domain.reject_reasons import RejectReason
from tradingchassis_core.core.domain.types import OrderIntent, RiskConstraints
from tradingchassis_core.core.events.events import RiskDecisionEvent
from tradingchassis_core.core.execution_control import ExecutionControl
from tradingchassis_core.core.execution_control.types import (
    ControlSchedulingObligation,
)
from tradingchassis_core.core.ports.venue_policy import VenuePolicy
from tradingchassis_core.core.risk.risk_policy import RiskPolicy

if TYPE_CHECKING:
    from risk.risk_config import RiskConfig

    from tradingchassis_core.core.domain.state import StrategyState
    from tradingchassis_core.core.events.event_bus import EventBus


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

    Compatibility decision contract consumed by strategy/runtime orchestration.
    This is not an Event and not a canonical Event Stream record.

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
    control_scheduling_obligations: tuple[ControlSchedulingObligation, ...] = ()


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

        self._risk_policy = RiskPolicy(venue_policy=self._venue_policy)

        # Internal execution-control component owns rate state and queue admission logic.
        # RiskEngine must own a single instance to preserve state lifetime semantics.
        self._execution_control = ExecutionControl()

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

    def evaluate_policy_intent(
        self,
        *,
        intent: OrderIntent,
        state: StrategyState,
        now_ts_ns_local: int,
    ) -> tuple[bool, str | None]:
        """Evaluate one intent with policy-only checks and no side effects.

        Side-effect contract:
        - does not call execution-control helpers;
        - does not mutate queue/rate/inflight state;
        - does not emit EventBus events.
        """

        raw_intents = [intent]

        triggered, policy_accepted, policy_rejected = self._risk_policy.trading_enabled_gate(
            trading_enabled=self.risk_cfg.trading_enabled,
            raw_intents=raw_intents,
        )
        if triggered:
            if policy_accepted:
                return True, None
            if policy_rejected:
                return False, policy_rejected[0][1]
            return False, RejectReason.TRADING_DISABLED

        triggered, policy_accepted, policy_rejected = self._risk_policy.max_loss_gate(
            max_loss_cfg=self.risk_cfg.max_loss,
            raw_intents=raw_intents,
            state=state,
            now_ts_ns_local=now_ts_ns_local,
        )
        if triggered:
            if policy_accepted:
                return True, None
            if policy_rejected:
                return False, policy_rejected[0][1]
            return False, RejectReason.MAX_LOSS_DRAWDOWN

        norm = self._risk_policy.normalize_intent(intent, state)
        if norm.reject_reason is not None:
            return False, norm.reject_reason
        if norm.dropped:
            return False, "dropped_by_policy"
        if norm.normalized is None:
            return False, RejectReason.INVALID_QTY

        normalized_intent = norm.normalized

        ok, reason = self._risk_policy.validate_intent(normalized_intent, state)
        if not ok:
            return False, reason

        pos_cfg = self.risk_cfg.position_limits
        max_pos = None if (pos_cfg is None or pos_cfg.max_position is None) else pos_cfg.max_position

        notional_cfg = self.risk_cfg.notional_limits
        max_gross_notional = notional_cfg.max_gross_notional
        max_single_order_notional = notional_cfg.max_single_order_notional

        quote_cfg = self.risk_cfg.quote_limits
        quote_book = None
        if quote_cfg is not None:
            quote_book = self._risk_policy.quote_book_global(state)
        base_gross_notional = self._risk_policy.portfolio_gross_notional(state)

        ok, reason = self._risk_policy.hard_checks(
            normalized_intent,
            state,
            max_pos=max_pos,
            max_single_order_notional=max_single_order_notional,
            max_gross_notional=max_gross_notional,
            base_gross_notional=base_gross_notional,
            quote_cfg=quote_cfg,
            quote_book=quote_book,
        )
        if not ok:
            return False, reason
        return True, None

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
        control_scheduling_obligations: list[ControlSchedulingObligation] = []

        # counters for RiskDecisionEvent
        reject_counts: dict[str, int] = {}
        def _count_reject(reason: str) -> None:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

        # --- Trading enabled gate ---
        triggered, policy_accepted, policy_rejected = self._risk_policy.trading_enabled_gate(
            trading_enabled=self.risk_cfg.trading_enabled,
            raw_intents=raw_intents,
        )
        if triggered:
            accepted_now.extend(policy_accepted)
            for it, reason in policy_rejected:
                rejected.append(RejectedIntent(it, reason))
                _count_reject(reason)

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
                control_scheduling_obligations=(),
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
        triggered, policy_accepted, policy_rejected = self._risk_policy.max_loss_gate(
            max_loss_cfg=self.risk_cfg.max_loss,
            raw_intents=raw_intents,
            state=state,
            now_ts_ns_local=now_ts_ns_local,
        )
        if triggered:
            accepted_now.extend(policy_accepted)
            for it, reason in policy_rejected:
                rejected.append(RejectedIntent(it, reason))
                _count_reject(reason)

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
                control_scheduling_obligations=(),
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
            quote_book = self._risk_policy.quote_book_global(state)

        # Base portfolio gross notional (best-effort)
        base_gross_notional = self._risk_policy.portfolio_gross_notional(state)

        # -----------------------------------------------------------------
        # Per-intent decision
        # -----------------------------------------------------------------
        for it in raw_intents:
            norm = self._risk_policy.normalize_intent(it, state)
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
            # 0) Pre-submission lifecycle / identity / inflight routing compatibility handling.
            continue_to_policy, lifecycle_reject_reason = (
                self._execution_control.route_pre_submission_lifecycle_and_inflight(
                    it,
                    state=state,
                    to_queue_by_instr=to_queue_by_instr,
                    replaced_in_queue=replaced_in_queue,
                    dropped_in_queue=dropped_in_queue,
                    queued=queued,
                    handled_in_queue=handled_in_queue,
                    float_equal=self._float_equal,
                )
            )
            if not continue_to_policy:
                if lifecycle_reject_reason is not None:
                    rejected.append(RejectedIntent(it, lifecycle_reject_reason))
                    _count_reject(lifecycle_reject_reason)
                continue

            # 1) Outbound hygiene validation (hard reject)
            ok, reason = self._risk_policy.validate_intent(it, state)
            if not ok:
                rejected.append(RejectedIntent(it, reason))
                _count_reject(reason)
                continue

            # 2) Hard risk checks (hard reject)
            ok, reason = self._risk_policy.hard_checks(
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
            rate_result = self._execution_control.route_after_policy_rate_limit(
                it,
                now_ts_ns_local=now_ts_ns_local,
                max_orders_per_sec=max_orders_per_sec,
                max_cancels_per_sec=max_cancels_per_sec,
            )
            if rate_result.stage_to_queue:
                to_queue_by_instr[it.instrument].append(it)
                obligation = rate_result.scheduling_obligation
                if obligation is not None:
                    control_scheduling_obligations.append(obligation)
                    next_send_ts = (
                        obligation.due_ts_ns_local
                        if next_send_ts is None
                        else min(next_send_ts, obligation.due_ts_ns_local)
                    )
                continue

            accepted_now.append(it)

        # -----------------------------------------------------------------
        # Queue merge per instrument (replacement rules live in StrategyState)
        # -----------------------------------------------------------------
        self._execution_control.merge_to_queue_per_instrument(
            state=state,
            to_queue_by_instr=to_queue_by_instr,
            queued=queued,
            replaced_in_queue=replaced_in_queue,
            dropped_in_queue=dropped_in_queue,
        )

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
            control_scheduling_obligations=tuple(control_scheduling_obligations),
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
