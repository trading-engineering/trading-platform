"""Risk configuration model for backtest and live trading engines."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_platform.core.domain.types import (
    MaxLoss,
    NotionalLimits,
    OrderRateLimits,
    PositionLimits,
    QuoteLimits,
)


class RiskConfig(BaseModel):
    """Structured-only risk configuration."""

    scope: str = Field(..., min_length=1)
    trading_enabled: bool = True

    # Mirrors types.py RiskConstraints fields (types.py is the source of truth)
    position_limits: PositionLimits | None = None
    notional_limits: NotionalLimits | None = None
    quote_limits: QuoteLimits | None = None
    order_rate_limits: OrderRateLimits | None = None
    max_loss: MaxLoss | None = None

    # Optional additional config fields (kept separate from RiskConstraints.extra)
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_json_obj(cls, risk_obj: dict[str, Any]) -> RiskConfig:
        """Create a RiskConfig instance from a JSON-compatible object."""
        return cls.model_validate(risk_obj)

    @model_validator(mode="after")
    def validate_consistency(self) -> RiskConfig:
        """Validate internal consistency of the risk configuration."""
        if self.notional_limits is None:
            raise ValueError("notional_limits is required")
        return self

    @property
    def params(self) -> dict[str, Any]:
        """Return engine-compatible flat risk parameters."""
        return self.to_engine_params()

    def to_engine_params(self) -> dict[str, Any]:
        """Convert the structured configuration into flat engine parameters."""
        params: dict[str, Any] = {}

        if self.position_limits is not None:
            params["position_limits"] = self.position_limits
        if self.notional_limits is not None:
            params["notional_limits"] = self.notional_limits
        if self.quote_limits is not None:
            params["quote_limits"] = self.quote_limits
        if self.order_rate_limits is not None:
            params["order_rate_limits"] = self.order_rate_limits
        if self.max_loss is not None:
            params["max_loss"] = self.max_loss

        if self.extra:
            params.update(self.extra)

        return params
