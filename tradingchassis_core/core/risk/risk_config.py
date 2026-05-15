"""Typed deterministic risk configuration model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradingchassis_core.core.domain.types import (
    MaxLoss,
    NotionalLimits,
    OrderRateLimits,
    PositionLimits,
    QuoteLimits,
)


class RiskConfig(BaseModel):
    """Structured risk configuration used by Core policy evaluation."""

    scope: str = Field(..., min_length=1)
    trading_enabled: bool = True
    position_limits: PositionLimits | None = None
    notional_limits: NotionalLimits | None = None
    quote_limits: QuoteLimits | None = None
    order_rate_limits: OrderRateLimits | None = None
    max_loss: MaxLoss | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_consistency(self) -> RiskConfig:
        if self.notional_limits is None:
            raise ValueError("notional_limits is required")
        return self
