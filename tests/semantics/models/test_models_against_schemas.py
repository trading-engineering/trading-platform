"""Schema conformance tests for core Pydantic models.

This test suite validates that core Pydantic models both accept valid inputs
and reject invalid ones in strict alignment with their corresponding JSON
Schemas. The tests are intentionally verbose and repetitive to ensure full
coverage and explicit failure modes.
"""

# pylint: disable=line-too-long,missing-function-docstring
# pylint: disable=redefined-outer-name,global-statement
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import pytest
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as jsonschema_validate
from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from trading_platform.core.domain.types import (
    FillEvent,
    MarketEvent,
    OrderIntent,
    OrderStateEvent,
    RiskConstraints,
)

SCHEMA_REGISTRY = Registry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_schema(name: str) -> dict:
    """
    Load JSON schema from project root.
    """
    global SCHEMA_REGISTRY

    root = Path(__file__).parent.parent.parent.parent  # /workspaces/trading-platform
    name = "trading_platform/core/schemas/" + name
    schema_path = root / name

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    schema_id = schema.get("$id")
    if isinstance(schema_id, str) and schema_id:
        resource = Resource.from_contents(schema, default_specification=DRAFT202012)
        SCHEMA_REGISTRY = SCHEMA_REGISTRY.with_resource(schema_id, resource)

    return schema


def dump_for_jsonschema(model: Any) -> dict:
    """
    Dump a Pydantic model to a JSON-compatible dict for schema validation.
    Excludes None values so optional fields are omitted instead of null.
    """
    # All validated objects in this test suite are BaseModel instances.
    return model.model_dump(mode="json", exclude_none=True)


T = TypeVar("T")


def pydantic_validate(model_type: Any, data: dict[str, Any]) -> Any:
    """
    Validate input using Pydantic for both:
    - BaseModel subclasses (e.g., MarketEvent)
    - Discriminated unions (e.g., OrderIntent is an Annotated Union)
    """
    adapter = TypeAdapter(model_type)
    return adapter.validate_python(data)


def assert_pydantic_then_schema_ok(model_type: Any, data: dict[str, Any], schema: dict[str, Any]) -> dict:
    """
    Validate with Pydantic first, then validate the dumped instance with JSON Schema.
    Returns the dumped instance.
    """
    obj = pydantic_validate(model_type, data)
    instance = dump_for_jsonschema(obj)
    jsonschema_validate(instance=instance, schema=schema, registry=SCHEMA_REGISTRY)
    return instance


def assert_schema_invalid_but_pydantic_rejects(model_type: Any, data: dict[str, Any], schema: dict[str, Any]):
    """
    Ensures Pydantic is at least as strict as the JSON Schema for the given input.
    If schema rejects, Pydantic must reject too (otherwise model is too lax).
    """
    with pytest.raises(JsonSchemaValidationError):
        jsonschema_validate(instance=data, schema=schema, registry=SCHEMA_REGISTRY)

    with pytest.raises(PydanticValidationError):
        pydantic_validate(model_type, data)


def mk_price(value: float, currency: str = "USD") -> dict[str, Any]:
    return {"currency": currency, "value": value}


def mk_qty(value: float, unit: str = "contracts") -> dict[str, Any]:
    return {"value": value, "unit": unit}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _load_common_schema() -> None:
    load_schema("common.schema.json")


@pytest.fixture(scope="module")
def market_event_schema() -> dict:
    return load_schema("market_event.schema.json")


@pytest.fixture(scope="module")
def order_intent_schema() -> dict:
    return load_schema("order_intent.schema.json")


@pytest.fixture(scope="module")
def risk_constraints_schema() -> dict:
    return load_schema("risk_constraints.schema.json")


@pytest.fixture(scope="module")
def fill_event_schema() -> dict:
    return load_schema("fill_event.schema.json")


@pytest.fixture(scope="module")
def order_state_event_schema() -> dict:
    return load_schema("order_state_event.schema.json")


# ---------------------------------------------------------------------------
# MarketEvent
# ---------------------------------------------------------------------------

def make_book_event(**book_overrides) -> dict[str, Any]:
    book = {
        "book_type": "snapshot",
        "bids": [{"price": mk_price(100.0), "quantity": mk_qty(1.0)}],
        "asks": [{"price": mk_price(100.5), "quantity": mk_qty(1.5)}],
    }
    book.update(book_overrides)
    return {
        "ts_ns_local": 123456789,
        "ts_ns_exch": 123456089,
        "instrument": "BTC-USD",
        "event_type": "book",
        "book": book,
    }


def make_trade_event(**trade_overrides) -> dict[str, Any]:
    trade = {
        "side": "buy",
        "price": mk_price(100.25),
        "quantity": mk_qty(0.5),
    }
    trade.update(trade_overrides)
    return {
        "ts_ns_local": 123456790,
        "ts_ns_exch": 123456789,
        "instrument": "BTC-USD",
        "event_type": "trade",
        "trade": trade,
    }


def test_market_event_book_valid_minimal(market_event_schema):
    assert_pydantic_then_schema_ok(MarketEvent, make_book_event(), market_event_schema)


def test_market_event_trade_valid_minimal(market_event_schema):
    assert_pydantic_then_schema_ok(MarketEvent, make_trade_event(trade_id="T123"), market_event_schema)


def test_market_event_enforces_payload_presence():
    with pytest.raises(PydanticValidationError):
        pydantic_validate(
            MarketEvent,
            {"ts_ns_local": 2, "ts_ns_exch": 1, "instrument": "BTC-USD", "event_type": "book"},
        )
    with pytest.raises(PydanticValidationError):
        pydantic_validate(
            MarketEvent,
            {"ts_ns_local": 2, "ts_ns_exch": 1, "instrument": "BTC-USD", "event_type": "trade"},
        )


def test_market_event_rejects_extra_top_level_fields(market_event_schema):
    data = make_book_event()
    data["unexpected"] = 123

    with pytest.raises(PydanticValidationError):
        pydantic_validate(MarketEvent, data)

    with pytest.raises(JsonSchemaValidationError):
        jsonschema_validate(instance=data, schema=market_event_schema, registry=SCHEMA_REGISTRY)


def test_market_event_book_depth_optional_valid_values(market_event_schema):
    assert_pydantic_then_schema_ok(MarketEvent, make_book_event(depth=0), market_event_schema)
    assert_pydantic_then_schema_ok(MarketEvent, make_book_event(depth=1), market_event_schema)


def test_market_event_book_depth_negative_rejected(market_event_schema):
    bad = make_book_event(depth=-1)
    assert_schema_invalid_but_pydantic_rejects(MarketEvent, bad, market_event_schema)


def test_market_event_trade_id_min_length(market_event_schema):
    assert_pydantic_then_schema_ok(MarketEvent, make_trade_event(trade_id="X"), market_event_schema)

    bad = make_trade_event(trade_id="")
    assert_schema_invalid_but_pydantic_rejects(MarketEvent, bad, market_event_schema)


def test_market_event_instrument_min_length(market_event_schema):
    bad = make_book_event()
    bad["instrument"] = ""
    assert_schema_invalid_but_pydantic_rejects(MarketEvent, bad, market_event_schema)


def test_market_event_ts_ns_exclusive_minimum(market_event_schema):
    bad = make_book_event()
    bad["ts_ns_local"] = 0
    assert_schema_invalid_but_pydantic_rejects(MarketEvent, bad, market_event_schema)


def test_market_event_xor_book_trade_enforced(market_event_schema):
    # Schema forbids having both book and trade.
    bad = make_book_event()
    bad["trade"] = make_trade_event()["trade"]

    with pytest.raises(JsonSchemaValidationError):
        jsonschema_validate(instance=bad, schema=market_event_schema, registry=SCHEMA_REGISTRY)

    with pytest.raises(PydanticValidationError):
        pydantic_validate(MarketEvent, bad)


# ---------------------------------------------------------------------------
# OrderIntent
# ---------------------------------------------------------------------------

def make_new_intent(**overrides) -> dict[str, Any]:
    """
    Build a valid minimal 'new' intent (limit by default).
    Note: intended_price is required for both limit and market orders in this system.
    """
    data: dict[str, Any] = {
        "ts_ns_local": 123456790,
        "client_order_id": "C-1",
        "instrument": "BTC-USD",
        "intent_type": "new",
        "order_type": "limit",
        "side": "buy",
        "intended_qty": mk_qty(1.0),
        "intended_price": mk_price(100.0),
        "time_in_force": "GTC",
    }
    data.update(overrides)
    return data


def make_cancel_intent(**overrides) -> dict[str, Any]:
    """
    Build a valid minimal 'cancel' intent.
    """
    data: dict[str, Any] = {
        "ts_ns_local": 123456791,
        "client_order_id": "C-1",
        "instrument": "BTC-USD",
        "intent_type": "cancel",
    }
    data.update(overrides)
    return data


def make_replace_intent(**overrides) -> dict[str, Any]:
    """
    Build a valid minimal 'replace' intent (limit-only).
    time_in_force is not allowed because it is not modifiable by the execution binding.
    """
    data: dict[str, Any] = {
        "ts_ns_local": 123456792,
        "client_order_id": "C-1",
        "instrument": "BTC-USD",
        "intent_type": "replace",
        "side": "buy",
        "order_type": "limit",
        "intended_qty": mk_qty(2.0),
        "intended_price": mk_price(101.0),
    }
    data.update(overrides)
    return data


def test_order_intent_valid_new_limit_with_optional_correlation_id(order_intent_schema):
    data = make_new_intent(intents_correlation_id="CORR-1")
    assert_pydantic_then_schema_ok(OrderIntent, data, order_intent_schema)


def test_order_intent_new_market_requires_intended_price(order_intent_schema):
    data = make_new_intent(order_type="market")
    assert_pydantic_then_schema_ok(OrderIntent, data, order_intent_schema)

    bad = make_new_intent(order_type="market")
    bad.pop("intended_price", None)
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad, order_intent_schema)


def test_order_intent_new_limit_requires_intended_price(order_intent_schema):
    bad = make_new_intent(order_type="limit")
    bad.pop("intended_price", None)
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad, order_intent_schema)


def test_order_intent_cancel_valid_minimal(order_intent_schema):
    assert_pydantic_then_schema_ok(OrderIntent, make_cancel_intent(), order_intent_schema)


def test_order_intent_cancel_forbids_order_fields(order_intent_schema):
    # Cancel must not contain order-creation fields.
    bad = make_cancel_intent(side="buy")
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad, order_intent_schema)

    bad = make_cancel_intent(order_type="limit")
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad, order_intent_schema)

    bad = make_cancel_intent(intended_qty=1.0)
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad, order_intent_schema)

    bad = make_cancel_intent(intended_price=100.0)
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad, order_intent_schema)

    bad = make_cancel_intent(time_in_force="GTC")
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad, order_intent_schema)


def test_order_intent_replace_valid_minimal(order_intent_schema):
    assert_pydantic_then_schema_ok(OrderIntent, make_replace_intent(), order_intent_schema)


def test_order_intent_replace_requires_limit(order_intent_schema):
    bad = make_replace_intent(order_type="market")
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad, order_intent_schema)


def test_order_intent_replace_forbids_time_in_force(order_intent_schema):
    # Replace must not contain time_in_force (not modifiable).
    bad = make_replace_intent(time_in_force="GTC")
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad, order_intent_schema)


def test_order_intent_min_length_optionals(order_intent_schema):
    bad_corr = make_new_intent(intents_correlation_id="")
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad_corr, order_intent_schema)


def test_order_intent_exclusive_minimum_constraints(order_intent_schema):
    bad_ts = make_new_intent(ts_ns_local=0)
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad_ts, order_intent_schema)

    bad_price = make_new_intent(intended_price=mk_price(-1.0))
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad_price, order_intent_schema)

    bad_qty = make_new_intent(intended_qty=mk_qty(-1.0))
    assert_schema_invalid_but_pydantic_rejects(OrderIntent, bad_qty, order_intent_schema)


def test_order_intent_rejects_additional_properties(order_intent_schema):
    data = make_new_intent()
    data["unexpected"] = "x"

    with pytest.raises(PydanticValidationError):
        pydantic_validate(OrderIntent, data)

    with pytest.raises(JsonSchemaValidationError):
        jsonschema_validate(instance=data, schema=order_intent_schema, registry=SCHEMA_REGISTRY)


# ---------------------------------------------------------------------------
# RiskConstraints
# ---------------------------------------------------------------------------

def make_risk_constraints(**overrides) -> dict[str, Any]:
    data = {
        "ts_ns_local": 987654321,
        "scope": "BTC-USD:default",
        "trading_enabled": True,
    }
    data.update(overrides)
    return data


def test_risk_constraints_valid_with_optionals(risk_constraints_schema):
    data = make_risk_constraints(
        notional_limits={
            "currency": "USD",
            "max_gross_notional": 1_000_000.0,
            "max_single_order_notional": 100_000.0,
        },
        position_limits={"currency": "BTC", "max_position": 10.0},
        quote_limits={
            "currency": "USD",
            "max_gross_quote_notional": 500_000.0,
            "max_net_quote_notional": 0.0,
            "max_active_quotes": 100,
        },
        order_rate_limits={"max_orders_per_second": 50.0, "max_cancels_per_second": 100.0},
        max_loss={"currency": "USD", "max_drawdown": -10_000.0, "rolling_loss": -1000.0, "rolling_loss_window": 60},
        extra={"desk": "alpha", "risk_mode": "conservative", "debug": None},
    )
    assert_pydantic_then_schema_ok(RiskConstraints, data, risk_constraints_schema)


def test_risk_constraints_missing_optional_notional_limits_is_ok(risk_constraints_schema):
    data = make_risk_constraints()
    assert_pydantic_then_schema_ok(RiskConstraints, data, risk_constraints_schema)


def test_risk_constraints_minimum_constraints(risk_constraints_schema):
    bad_pos = make_risk_constraints(position_limits={"currency": "BTC", "max_position": -1.0})
    assert_schema_invalid_but_pydantic_rejects(RiskConstraints, bad_pos, risk_constraints_schema)

    bad_notional = make_risk_constraints(notional_limits={"currency": "USD", "max_gross_notional": -1.0})
    assert_schema_invalid_but_pydantic_rejects(RiskConstraints, bad_notional, risk_constraints_schema)

    bad_quotes = make_risk_constraints(quote_limits={"currency": "USD", "max_gross_quote_notional": -1.0})
    assert_schema_invalid_but_pydantic_rejects(RiskConstraints, bad_quotes, risk_constraints_schema)

    bad_active = make_risk_constraints(quote_limits={"currency": "USD", "max_gross_quote_notional": 1.0, "max_active_quotes": -1})
    assert_schema_invalid_but_pydantic_rejects(RiskConstraints, bad_active, risk_constraints_schema)

    bad_rate = make_risk_constraints(order_rate_limits={"max_orders_per_second": -0.1})
    assert_schema_invalid_but_pydantic_rejects(RiskConstraints, bad_rate, risk_constraints_schema)


def test_risk_constraints_rejects_additional_properties(risk_constraints_schema):
    data = make_risk_constraints()
    data["unexpected"] = 1

    with pytest.raises(PydanticValidationError):
        pydantic_validate(RiskConstraints, data)

    with pytest.raises(JsonSchemaValidationError):
        jsonschema_validate(instance=data, schema=risk_constraints_schema, registry=SCHEMA_REGISTRY)


def test_risk_constraints_extra_values_are_schema_compatible(risk_constraints_schema):
    data = make_risk_constraints(extra={"ok": "x", "n": 1, "b": True, "z": None})
    assert_pydantic_then_schema_ok(RiskConstraints, data, risk_constraints_schema)

    bad = make_risk_constraints(extra={"nested": {"a": 1}})
    with pytest.raises(JsonSchemaValidationError):
        jsonschema_validate(instance=bad, schema=risk_constraints_schema, registry=SCHEMA_REGISTRY)
    # Pydantic will reject too because extra is typed as primitive union.


# ---------------------------------------------------------------------------
# FillEvent
# ---------------------------------------------------------------------------

def make_fill(**overrides) -> dict[str, Any]:
    data = {
        "ts_ns_local": 123456789,
        "ts_ns_exch": 123456089,
        "instrument": "BTC-USD",
        "client_order_id": "C-1",
        "side": "buy",
        "filled_price": mk_price(100.5),
        "cum_filled_qty": mk_qty(0.25),
        "time_in_force": "GTC",
        "liquidity_flag": "maker",
    }
    data.update(overrides)
    return data


def test_fill_event_valid_with_all_optionals(fill_event_schema):
    data = make_fill(
        intended_price=mk_price(100.0),
        intended_qty=mk_qty(1.0),
        remaining_qty=mk_qty(0.75),
        fee={"currency": "USD", "amount": -0.1},
    )
    assert_pydantic_then_schema_ok(FillEvent, data, fill_event_schema)


def test_fill_event_exclusive_minimum_constraints(fill_event_schema):
    bad_ts = make_fill(ts_ns_local=0)
    assert_schema_invalid_but_pydantic_rejects(FillEvent, bad_ts, fill_event_schema)


def test_fill_event_min_length_optionals(fill_event_schema):
    bad_order_id = make_fill(client_order_id="")
    assert_schema_invalid_but_pydantic_rejects(FillEvent, bad_order_id, fill_event_schema)


def test_fill_event_rejects_additional_properties(fill_event_schema):
    data = make_fill()
    data["unexpected"] = "x"

    with pytest.raises(PydanticValidationError):
        pydantic_validate(FillEvent, data)

    with pytest.raises(JsonSchemaValidationError):
        jsonschema_validate(instance=data, schema=fill_event_schema, registry=SCHEMA_REGISTRY)


# ---------------------------------------------------------------------------
# OrderStateEvent
# ---------------------------------------------------------------------------

def make_state(**overrides) -> dict[str, Any]:
    data = {
        "ts_ns_local": 123456789,
        "ts_ns_exch": 123456089,
        "instrument": "BTC-USD",
        "client_order_id": "C-1",
        "order_type": "limit",
        "state_type": "accepted",
        "side": "buy",
        "intended_price": mk_price(100.0),
        "intended_qty": mk_qty(1.0),
        "time_in_force": "GTC",
    }
    data.update(overrides)
    return data


def test_order_state_event_valid_with_all_optionals(order_state_event_schema):
    data = make_state(
        filled_price=mk_price(100.1),
        cum_filled_qty=mk_qty(0.5),
        remaining_qty=mk_qty(0.5),
        reason="Partial fill",
        raw={"venue_status": "PARTIAL"},
    )
    assert_pydantic_then_schema_ok(OrderStateEvent, data, order_state_event_schema)


def test_order_state_event_min_length_optionals(order_state_event_schema):
    bad_order_id = make_state(client_order_id="")
    assert_schema_invalid_but_pydantic_rejects(OrderStateEvent, bad_order_id, order_state_event_schema)

    bad_reason = make_state(reason="")
    assert_schema_invalid_but_pydantic_rejects(OrderStateEvent, bad_reason, order_state_event_schema)


def test_order_state_event_exclusive_minimum_ts(order_state_event_schema):
    bad = make_state(ts_ns_local=0)
    assert_schema_invalid_but_pydantic_rejects(OrderStateEvent, bad, order_state_event_schema)


def test_order_state_event_rejects_additional_properties(order_state_event_schema):
    data = make_state()
    data["unexpected"] = 1

    with pytest.raises(PydanticValidationError):
        pydantic_validate(OrderStateEvent, data)

    with pytest.raises(JsonSchemaValidationError):
        jsonschema_validate(instance=data, schema=order_state_event_schema, registry=SCHEMA_REGISTRY)
