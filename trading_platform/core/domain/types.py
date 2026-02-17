"""Core shared data models and schemas.

This module defines the canonical Pydantic models used across the system for
market data, order intents, risk constraints, and execution events. These
types are treated as schema definitions and intentionally prioritize
structural clarity over minimal class size.
"""

# pylint: disable=line-too-long,missing-class-docstring,missing-function-docstring
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Common models
# ---------------------------------------------------------------------------


class Money(BaseModel):
    currency: str = Field(..., min_length=1)
    amount: float = Field(...,)

    model_config = ConfigDict(extra="forbid")


class Price(BaseModel):
    currency: str = Field(..., min_length=1)
    value: float = Field(..., ge=0)

    model_config = ConfigDict(extra="forbid")


class Quantity(BaseModel):
    value: float = Field(..., ge=0)
    unit: str = Field(..., min_length=1)  # e.g. "shares", "contracts", "BTC"

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Market data models (MarketEvent + payloads)
# ---------------------------------------------------------------------------


class BookLevel(BaseModel):
    price: Price
    quantity: Quantity

    model_config = ConfigDict(extra="forbid")


class BookPayload(BaseModel):
    book_type: Literal["snapshot", "delta"]
    bids: list[BookLevel]
    asks: list[BookLevel]
    # depth is optional in the JSON schema, but must be >= 0 when present
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
        """
        Enforce the conditional requirements from the JSON schema:
        - If event_type == "book": book must be present, trade must be None
        - If event_type == "trade": trade must be present, book must be None
        """
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


# ---------------------------------------------------------------------------
# Order intent models (discriminated union)
# ---------------------------------------------------------------------------


TimeInForce = Literal["GTC", "IOC", "FOK", "POST_ONLY"]
OrderType = Literal["limit", "market"]
Side = Literal["buy", "sell"]


class OrderIntentBase(BaseModel):
    """
    Base fields shared by all order intents.

    Notes:
    - client_order_id maps to the execution binding's order_id and is used for new/cancel/replace.
    - intents_correlation_id is optional and can be used to link multiple intents together.
    """

    ts_ns_local: int = Field(
        ...,
        gt=0,
        description="Local intent timestamp in nanoseconds since Unix epoch.",
    )
    instrument: str = Field(
        ...,
        min_length=1,
        description="Instrument identifier used for routing/execution binding (e.g., symbol, asset code).",
    )
    client_order_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Order identifier (maps to the execution binding's order_id). "
            "Used for new/cancel/replace. Must be unique while an order with the same ID exists."
        ),
    )
    intents_correlation_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional correlation identifier to link multiple intents "
            "(e.g., decision bundles) across the order lifecycle."
        ),
    )

    model_config = ConfigDict(extra="forbid")


class NewOrderIntent(OrderIntentBase):
    """
    Create a new order.

    Important:
    - intended_price is required for both limit and market orders to match the execution binding signature.
    """

    intent_type: Literal["new"] = Field(
        "new",
        description="Intent type describing the order lifecycle action.",
    )

    side: Side = Field(..., description="Order side.")
    order_type: OrderType = Field(..., description="Order type.")
    intended_qty: Quantity = Field(
        ...,
        description="Intended total order quantity.",
    )
    intended_price: Price = Field(
        ...,
        description=(
            "Intended order price. Required for both limit and market orders "
            "to match the execution binding signature."
        ),
    )
    time_in_force: TimeInForce = Field(
        ...,
        description="Time in force. Required for new intents.",
    )


class CancelOrderIntent(OrderIntentBase):
    """
    Cancel an existing order identified by client_order_id.

    This intent deliberately forbids order-creation fields (side, order_type, qty, price, tif).
    """

    intent_type: Literal["cancel"] = Field(
        "cancel",
        description="Intent type describing the order lifecycle action.",
    )


class ReplaceOrderIntent(OrderIntentBase):
    """
    Modify an existing order (limit-only). The order ID remains the same (client_order_id).

    Notes:
    - order_type is constrained to 'limit'.
    - time_in_force is intentionally not present here because the execution binding does not support modifying it.
    - intended_qty is the new TOTAL quantity (not a delta).
    """

    intent_type: Literal["replace"] = Field(
        "replace",
        description="Intent type describing the order lifecycle action.",
    )

    side: Side = Field(..., description="Order side.")
    order_type: Literal["limit"] = Field(
        "limit",
        description="Order type. For replace intents this must be 'limit'.",
    )
    intended_qty: Quantity = Field(
        ...,
        description="Intended total order quantity (new total quantity, not a delta).",
    )
    intended_price: Price = Field(
        ...,
        description="Intended order price.",
    )


# Discriminated union: Pydantic will select the correct model based on intent_type.
OrderIntent = Annotated[
    NewOrderIntent | CancelOrderIntent | ReplaceOrderIntent,
    Field(discriminator="intent_type"),
]


# ---------------------------------------------------------------------------
# Risk constraints models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FillEvent model (delta event)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# OrderStateEvent model (snapshot event)
# ---------------------------------------------------------------------------


class OrderStateEvent(BaseModel):
    ts_ns_exch: int = Field(..., gt=0)
    ts_ns_local: int = Field(..., gt=0)

    instrument: str = Field(..., min_length=1)
    client_order_id: str = Field(..., min_length=1)

    order_type: Literal["limit", "market"]
    state_type: Literal[
        "pending_new",
        "accepted",
        "working",
        "partially_filled",
        "filled",
        "canceled",
        "expired",
        "rejected",
        "replaced",
    ]

    side: Literal["buy", "sell"]
    intended_price: Price

    filled_price: Price | None = None
    intended_qty: Quantity

    cum_filled_qty: Quantity | None = None
    remaining_qty: Quantity | None = None

    time_in_force: Literal["GTC", "IOC", "FOK", "POST_ONLY"]

    reason: str | None = Field(default=None, min_length=1)
    raw: dict[str, Any | None] = None

    model_config = ConfigDict(extra="forbid")
