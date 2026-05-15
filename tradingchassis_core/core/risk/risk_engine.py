"""Policy-risk evaluator used by deterministic Core policy admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingchassis_core.core.domain.reject_reasons import RejectReason
from tradingchassis_core.core.domain.types import OrderIntent, RiskConstraints
from tradingchassis_core.core.risk.execution_constraints_policy import (
    ExecutionConstraintsPolicy,
)
from tradingchassis_core.core.risk.risk_policy import RiskPolicy

if TYPE_CHECKING:
    from tradingchassis_core.core.domain.state import StrategyState
    from tradingchassis_core.core.risk.risk_config import RiskConfig


class RiskEngine:
    """Policy-only evaluator.

    This component is intentionally side-effect-free for the CoreStep policy phase:
    it does not mutate Queue/rate/inflight state and does not perform Execution Control.
    """

    def __init__(self, risk_cfg: RiskConfig) -> None:
        self.risk_cfg = risk_cfg

        min_order_notional, post_only_mode = self._parse_execution_constraints_config(risk_cfg)
        self._constraints_policy = ExecutionConstraintsPolicy(
            min_order_notional=min_order_notional,
            post_only_mode=post_only_mode,
        )
        self._risk_policy = RiskPolicy(constraints_policy=self._constraints_policy)

    @staticmethod
    def _parse_execution_constraints_config(risk_cfg: RiskConfig) -> tuple[float, str]:
        min_order_notional = 0.0
        post_only_mode = "reject"

        extra = risk_cfg.extra
        if not isinstance(extra, dict):
            return min_order_notional, post_only_mode

        constraints = (
            extra["execution_constraints"]
            if "execution_constraints" in extra
            else None
        )
        if isinstance(constraints, dict):
            if "min_order_notional" in constraints:
                try:
                    min_order_notional = float(constraints["min_order_notional"])
                except (TypeError, ValueError):
                    pass
            if "post_only_mode" in constraints:
                mode = str(constraints["post_only_mode"])
                if mode in {"reject", "drop"}:
                    post_only_mode = mode
            return min_order_notional, post_only_mode

        if "execution_constraints_min_order_notional" in extra:
            try:
                min_order_notional = float(
                    extra["execution_constraints_min_order_notional"]
                )
            except (TypeError, ValueError):
                pass

        if "execution_constraints_post_only_mode" in extra:
            mode = str(extra["execution_constraints_post_only_mode"])
            if mode in {"reject", "drop"}:
                post_only_mode = mode

        return min_order_notional, post_only_mode

    @staticmethod
    def _constraints_extra(extra: object) -> dict[str, str | float | bool | None]:
        if not isinstance(extra, dict):
            return {}

        normalized: dict[str, str | float | bool | None] = {}
        for key, value in extra.items():
            if key == "execution_constraints" and isinstance(value, dict):
                if "min_order_notional" in value:
                    normalized["execution_constraints_min_order_notional"] = value[
                        "min_order_notional"
                    ]
                if "post_only_mode" in value:
                    normalized["execution_constraints_post_only_mode"] = value[
                        "post_only_mode"
                    ]
                continue

            if value is None or isinstance(value, (str, float, bool)):
                normalized[key] = value
            elif isinstance(value, int):
                normalized[key] = float(value)

        return normalized

    def build_constraints(self, current_timestamp_ns_local: int) -> RiskConstraints:
        """Build RiskConstraints handed to Strategy evaluation."""
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

    def evaluate_policy_intent(
        self,
        *,
        intent: OrderIntent,
        state: StrategyState,
        now_ts_ns_local: int,
    ) -> tuple[bool, str | None]:
        """Evaluate one Intent with policy-only checks and no side effects."""

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
        max_gross_notional = (
            None if notional_cfg is None else notional_cfg.max_gross_notional
        )
        max_single_order_notional = (
            None if notional_cfg is None else notional_cfg.max_single_order_notional
        )

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
