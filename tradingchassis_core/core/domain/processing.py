"""Minimal canonical Event processing boundary for core.

This module introduces a narrow, docs-aligned processing boundary for current
canonical Event candidates. For these candidates, ``process_canonical_event``
is the preferred top-level canonical state-advance entrypoint in core.

This module is intentionally small:

- it is not a full Event Stream implementation;
- it enforces only minimal positioned monotonicity at the boundary;
- it does not implement replay semantics;
- compatibility ingestion paths remain separate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from tradingchassis_core.core.domain.configuration import CoreConfiguration
from tradingchassis_core.core.domain.event_model import (
    CanonicalEventCategory,
    canonical_category_for_type,
    is_canonical_stream_candidate_type,
)
from tradingchassis_core.core.domain.processing_order import EventStreamEntry, ProcessingPosition
from tradingchassis_core.core.domain.state import StrategyState
from tradingchassis_core.core.domain.types import (
    ControlTimeEvent,
    FillEvent,
    MarketEvent,
    OrderExecutionFeedbackEvent,
    OrderSubmittedEvent,
)


def _extract_required_positive_number(value: object, *, field_path: str) -> float:
    if value is None:
        raise ValueError(f"Missing required market configuration field: {field_path}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Market configuration field must be numeric: {field_path}")

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Market configuration field must be finite: {field_path}")
    if numeric <= 0.0:
        raise ValueError(f"Market configuration field must be > 0: {field_path}")
    return numeric


def _extract_market_instrument_metadata(
    configuration: CoreConfiguration | None,
    *,
    instrument: str,
) -> tuple[float, float, float]:
    if configuration is None:
        raise ValueError(
            "CoreConfiguration is required for positioned canonical MarketEvent processing."
        )

    payload = configuration.payload

    market = payload.get("market")
    if not isinstance(market, Mapping):
        raise ValueError("Missing required market configuration object: payload.market")

    instruments = market.get("instruments")
    if not isinstance(instruments, Mapping):
        raise ValueError(
            "Missing required market configuration object: payload.market.instruments"
        )

    instrument_cfg = instruments.get(instrument)
    if not isinstance(instrument_cfg, Mapping):
        raise ValueError(
            "Missing required market instrument configuration: "
            f"payload.market.instruments.{instrument}"
        )

    tick_size = _extract_required_positive_number(
        instrument_cfg.get("tick_size"),
        field_path=f"payload.market.instruments.{instrument}.tick_size",
    )
    lot_size = _extract_required_positive_number(
        instrument_cfg.get("lot_size"),
        field_path=f"payload.market.instruments.{instrument}.lot_size",
    )
    contract_size = _extract_required_positive_number(
        instrument_cfg.get("contract_size"),
        field_path=f"payload.market.instruments.{instrument}.contract_size",
    )

    return tick_size, lot_size, contract_size


def process_canonical_event(
    state: StrategyState,
    event: object,
    *,
    position: ProcessingPosition | None = None,
    configuration: CoreConfiguration | None = None,
) -> None:
    """Process a canonical Event candidate via existing state reducers.

    Preferred usage for the current slice:
    - use this function as the top-level canonical ingestion boundary for
      currently supported canonical candidates.
    - keep low-level reducer methods as compatibility primitives.

    Accepted canonical candidates in the current slice:
    - ``MarketEvent`` (category: ``market``)
    - ``OrderSubmittedEvent`` (category: ``intent_related``)
    - ``FillEvent`` (category: ``execution``)
    - ``OrderExecutionFeedbackEvent`` (category: ``execution``)
    - ``ControlTimeEvent`` (category: ``control``)

    ``ProcessingPosition`` is accepted as Processing Order metadata at this
    boundary. When provided, positions must be strictly increasing. This
    function is not a full Event Stream or replay layer.

    Non-canonical records (compatibility projections, telemetry payloads, bus
    transports, and helper artifacts) are rejected at this boundary.
    """
    record_type = type(event)
    if not is_canonical_stream_candidate_type(record_type):
        raise TypeError(f"Unsupported non-canonical Event type: {record_type.__name__}")

    category = canonical_category_for_type(record_type)

    if category == CanonicalEventCategory.MARKET and isinstance(event, MarketEvent):
        if event.is_trade():
            raise ValueError(
                "Unsupported MarketEvent for canonical processing in the current Core "
                "baseline: only book MarketEvent payloads are reduced; trade-shaped "
                "MarketEvent payloads are not supported."
            )
        if event.book is None:
            raise ValueError(
                "Unsupported MarketEvent payload for canonical processing in the current "
                "Core baseline: book payload is required."
            )
        if not event.book.bids or not event.book.asks:
            raise ValueError(
                "Unsupported MarketEvent payload for canonical processing in the current "
                "Core baseline: book payload must include at least one bid and one ask "
                "level."
            )

        best_bid_level = event.book.bids[0]
        best_ask_level = event.book.asks[0]

        if position is not None:
            tick_size, lot_size, contract_size = _extract_market_instrument_metadata(
                configuration,
                instrument=event.instrument,
            )
            state._advance_processing_position(position)
            state._update_market_from_positioned_canonical_event(
                instrument=event.instrument,
                best_bid=best_bid_level.price.value,
                best_ask=best_ask_level.price.value,
                best_bid_qty=best_bid_level.quantity.value,
                best_ask_qty=best_ask_level.quantity.value,
                tick_size=tick_size,
                lot_size=lot_size,
                contract_size=contract_size,
                ts_ns_local=event.ts_ns_local,
                ts_ns_exch=event.ts_ns_exch,
            )
        else:
            state.update_market(
                instrument=event.instrument,
                best_bid=best_bid_level.price.value,
                best_ask=best_ask_level.price.value,
                best_bid_qty=best_bid_level.quantity.value,
                best_ask_qty=best_ask_level.quantity.value,
                tick_size=0.0,
                lot_size=0.0,
                contract_size=1.0,
                ts_ns_local=event.ts_ns_local,
                ts_ns_exch=event.ts_ns_exch,
            )
        return

    if category == CanonicalEventCategory.EXECUTION and isinstance(event, FillEvent):
        if position is not None:
            state._advance_processing_position(position)
        state.apply_fill_event(event)
        return

    if (
        category == CanonicalEventCategory.EXECUTION
        and isinstance(event, OrderExecutionFeedbackEvent)
    ):
        if position is not None:
            state._advance_processing_position(position)
        state.apply_order_execution_feedback_event(event)
        return

    if (
        category == CanonicalEventCategory.INTENT_RELATED
        and isinstance(event, OrderSubmittedEvent)
    ):
        if position is not None:
            state._advance_processing_position(position)
        state.apply_order_submitted_event(event)
        return

    if category == CanonicalEventCategory.CONTROL and isinstance(event, ControlTimeEvent):
        if position is not None:
            state._advance_processing_position(position)
        state.apply_control_time_event(event)
        return

    raise TypeError(
        "Unsupported canonical Event candidate for this processing boundary: "
        f"{record_type.__name__}"
    )


def process_event_entry(
    state: StrategyState,
    entry: EventStreamEntry,
    *,
    configuration: CoreConfiguration | None = None,
) -> None:
    """Process one minimal EventStreamEntry via the canonical boundary.

    This wrapper is intentionally minimal:
    - it is not full Event Stream storage;
    - it is not replay orchestration;
    - it is not runtime integration.

    Configuration is accepted as explicit processing input to reflect the
    docs contract. In this slice, positioned canonical MarketEvent reduction
    consumes explicit instrument metadata from configuration.
    Ordering is enforced through ``entry.position`` using existing
    ``ProcessingPosition`` cursor monotonicity logic in canonical processing.
    """
    if configuration is not None and not isinstance(configuration, CoreConfiguration):
        raise TypeError("configuration must be CoreConfiguration or None")
    process_canonical_event(
        state,
        entry.event,
        position=entry.position,
        configuration=configuration,
    )
