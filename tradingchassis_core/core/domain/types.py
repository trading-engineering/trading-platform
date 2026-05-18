"""Core shared data models.

Pydantic models in this module are the source of truth for Core contracts.
"""

# pylint: disable=line-too-long,missing-class-docstring,missing-function-docstring
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Money(BaseModel):
    currency: str = Field(..., min_length=1)
    amount: float = Field(...)
    model_config = ConfigDict(extra="forbid")


class Price(BaseModel):
    currency: str = Field(..., min_length=1)
    value: float = Field(..., ge=0)
    model_config = ConfigDict(extra="forbid")


class Quantity(BaseModel):
    value: float = Field(..., ge=0)
    unit: str = Field(..., min_length=1)
    model_config = ConfigDict(extra="forbid")


class BookLevel(BaseModel):
    price: Price
    quantity: Quantity
    model_config = ConfigDict(extra="forbid")


class BookPayload(BaseModel):
    book_type: Literal["snapshot", "delta"]
    bids: list[BookLevel]
    asks: list[BookLevel]
    depth: int | None = Field(default=None, ge=0)
    model_config = ConfigDict(extra="forbid")


class TradePayload(BaseModel):
    side: Literal["buy", "sell"]
    price: Price
    quantity: Quantity
    trade_id: str | None = Field(default=None, min_length=1)
    model_config = ConfigDict(extra="forbid")


class MarketEvent(BaseModel):
    ts_ns_exch: int = Field(..., gt=0)
    ts_ns_local: int = Field(..., gt=0)
    instrument: str = Field(..., min_length=1)
    event_type: Literal["book", "trade"]
    book: BookPayload | None = None
    trade: TradePayload | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_payload_for_event_type(self) -> MarketEvent:
        if self.event_type == "book":
            if self.book is None:
                raise ValueError("book payload is required when event_type is 'book'")
            if self.trade is not None:
                raise ValueError("trade payload must be None when event_type is 'book'")
        elif self.event_type == "trade":
            if self.trade is None:
                raise ValueError("trade payload is required when event_type is 'trade'")
            if self.book is not None:
                raise ValueError("book payload must be None when event_type is 'trade'")
        return self

    def is_book(self) -> bool:
        return self.event_type == "book"

    def is_trade(self) -> bool:
        return self.event_type == "trade"


TimeInForce = Literal["GTC", "IOC", "FOK", "POST_ONLY"]
OrderType = Literal["limit", "market"]
Side = Literal["buy", "sell"]


class OrderIntentBase(BaseModel):
    ts_ns_local: int = Field(..., gt=0)
    instrument: str = Field(..., min_length=1)
    client_order_id: str = Field(..., min_length=1)
    intents_correlation_id: str | None = Field(default=None, min_length=1)
    model_config = ConfigDict(extra="forbid")


class NewOrderIntent(OrderIntentBase):
    intent_type: Literal["new"] = Field("new")
    side: Side = Field(...)
    order_type: OrderType = Field(...)
    intended_qty: Quantity = Field(...)
    intended_price: Price = Field(...)
    time_in_force: TimeInForce = Field(...)


class CancelOrderIntent(OrderIntentBase):
    intent_type: Literal["cancel"] = Field("cancel")


class ReplaceOrderIntent(OrderIntentBase):
    intent_type: Literal["replace"] = Field("replace")
    side: Side = Field(...)
    order_type: Literal["limit"] = Field("limit")
    intended_qty: Quantity = Field(...)
    intended_price: Price = Field(...)


OrderIntent = Annotated[
    NewOrderIntent | CancelOrderIntent | ReplaceOrderIntent,
    Field(discriminator="intent_type"),
]


class PositionLimits(BaseModel):
    currency: str = Field(..., min_length=1)
    max_position: float | None = Field(default=None, ge=0)
    model_config = ConfigDict(extra="forbid")


class NotionalLimits(BaseModel):
    currency: str = Field(..., min_length=1)
    max_gross_notional: float | None = Field(default=None, ge=0)
    max_single_order_notional: float | None = Field(default=None, ge=0)
    model_config = ConfigDict(extra="forbid")


class QuoteLimits(BaseModel):
    currency: str = Field(..., min_length=1)
    max_gross_quote_notional: float | None = Field(default=None, ge=0)
    max_net_quote_notional: float | None = None
    max_active_quotes: int | None = Field(default=None, ge=0)
    model_config = ConfigDict(extra="forbid")


class OrderRateLimits(BaseModel):
    max_orders_per_second: float | None = Field(default=None, ge=0)
    max_cancels_per_second: float | None = Field(default=None, ge=0)
    model_config = ConfigDict(extra="forbid")


class MaxLoss(BaseModel):
    currency: str = Field(..., min_length=1)
    max_drawdown: float = Field(..., lt=0)
    rolling_loss: float | None = Field(default=None, lt=0)
    rolling_loss_window: float | None = Field(default=None, gt=0)
    model_config = ConfigDict(extra="forbid")


class RiskConstraints(BaseModel):
    ts_ns_local: int = Field(..., gt=0)
    scope: str = Field(..., min_length=1)
    trading_enabled: bool
    position_limits: PositionLimits | None = None
    notional_limits: NotionalLimits | None = None
    quote_limits: QuoteLimits | None = None
    order_rate_limits: OrderRateLimits | None = None
    max_loss: MaxLoss | None = None
    extra: dict[str, str | float | bool | None] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class FillEvent(BaseModel):
    ts_ns_exch: int = Field(..., gt=0)
    ts_ns_local: int = Field(..., gt=0)
    instrument: str = Field(..., min_length=1)
    client_order_id: str = Field(..., min_length=1)
    side: Literal["buy", "sell"]
    intended_price: Price | None = None
    filled_price: Price
    intended_qty: Quantity | None = None
    cum_filled_qty: Quantity
    remaining_qty: Quantity | None = None
    time_in_force: Literal["GTC", "IOC", "FOK", "POST_ONLY"]
    liquidity_flag: Literal["maker", "taker", "unknown"]
    fee: Money | None = None
    model_config = ConfigDict(extra="forbid")


class OrderExecutionFeedbackEvent(BaseModel):
    ts_ns_local_feedback: int = Field(..., gt=0)
    instrument: str = Field(..., min_length=1)
    position: float
    balance: float
    fee: float
    trading_volume: float
    trading_value: float
    num_trades: int
    runtime_correlation: dict[str, str | int | float | bool | None] | None = None
    model_config = ConfigDict(extra="forbid")


class OrderSubmittedEvent(BaseModel):
    ts_ns_local_dispatch: int = Field(..., gt=0)
    instrument: str = Field(..., min_length=1)
    client_order_id: str = Field(..., min_length=1)
    side: Literal["buy", "sell"]
    order_type: Literal["limit", "market"]
    intended_price: Price
    intended_qty: Quantity
    time_in_force: Literal["GTC", "IOC", "FOK", "POST_ONLY"]
    intent_correlation_id: str | None = Field(default=None, min_length=1)
    dispatch_attempt_id: str | None = Field(default=None, min_length=1)
    runtime_correlation: dict[str, str | int | float | bool | None] | None = None
    model_config = ConfigDict(extra="forbid")


class OrderCanceledEvent(BaseModel):
    ts_ns_local_feedback: int = Field(..., gt=0)
    instrument: str = Field(..., min_length=1)
    client_order_id: str = Field(..., min_length=1)
    model_config = ConfigDict(extra="forbid")


class OrderRejectedEvent(BaseModel):
    ts_ns_local_feedback: int = Field(..., gt=0)
    instrument: str = Field(..., min_length=1)
    client_order_id: str = Field(..., min_length=1)
    model_config = ConfigDict(extra="forbid")


class OrderExpiredEvent(BaseModel):
    ts_ns_local_feedback: int = Field(..., gt=0)
    instrument: str = Field(..., min_length=1)
    client_order_id: str = Field(..., min_length=1)
    model_config = ConfigDict(extra="forbid")


class ControlTimeEvent(BaseModel):
    ts_ns_local_control: int = Field(..., gt=0)
    reason: str = Field(..., min_length=1)
    due_ts_ns_local: int | None = Field(default=None, gt=0)
    realized_ts_ns_local: int | None = Field(default=None, gt=0)
    obligation_reason: str | None = Field(default=None, min_length=1)
    obligation_due_ts_ns_local: int | None = Field(default=None, gt=0)
    runtime_correlation: dict[str, str | int | float | bool | None] | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_due_or_realized_present(self) -> ControlTimeEvent:
        if self.due_ts_ns_local is None and self.realized_ts_ns_local is None:
            raise ValueError(
                "at least one of due_ts_ns_local or realized_ts_ns_local is required"
            )
        return self
