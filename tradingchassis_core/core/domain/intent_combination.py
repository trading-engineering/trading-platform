"""Pure helper for Core-step candidate Intent combination."""

from __future__ import annotations

from collections.abc import Sequence

from tradingchassis_core.core.domain.candidate_intent import (
    CandidateIntentOrigin,
    CandidateIntentRecord,
)
from tradingchassis_core.core.domain.types import OrderIntent


def _logical_key(intent: OrderIntent) -> str:
    return f"order:{intent.client_order_id}"


def _intent_priority(intent: OrderIntent) -> int:
    if intent.intent_type == "cancel":
        return 0
    if intent.intent_type == "replace":
        return 1
    if intent.intent_type == "new":
        return 2
    return 9


def _dominance_rank(intent: OrderIntent) -> int:
    if intent.intent_type == "cancel":
        return 3
    if intent.intent_type == "replace":
        return 2
    if intent.intent_type == "new":
        return 1
    return 0


def combine_candidate_intents(
    *,
    generated_intents: Sequence[OrderIntent],
    queued_intents: Sequence[OrderIntent],
) -> tuple[OrderIntent, ...]:
    """Compatibility helper returning only effective intent values.

    Prefer ``combine_candidate_intent_records`` when origin/provenance is needed.
    """
    records = combine_candidate_intent_records(
        generated_intents=generated_intents,
        queued_intents=queued_intents,
    )
    return tuple(record.intent for record in records)


def combine_candidate_intent_records(
    *,
    generated_intents: Sequence[OrderIntent],
    queued_intents: Sequence[OrderIntent],
) -> tuple[CandidateIntentRecord, ...]:
    """Combine queued + generated intents into a deterministic effective set.

    This helper is pure and does not mutate StrategyState.
    Merge order is deterministic: queued first, then generated.
    """
    merged: list[tuple[OrderIntent, CandidateIntentOrigin]] = [
        *((intent, CandidateIntentOrigin.QUEUED) for intent in queued_intents),
        *((intent, CandidateIntentOrigin.GENERATED) for intent in generated_intents),
    ]
    # key -> winning record
    effective_by_key: dict[str, CandidateIntentRecord] = {}

    for merge_index, (intent, origin) in enumerate(merged):
        key = _logical_key(intent)
        incoming = CandidateIntentRecord(
            intent=intent,
            origin=origin,
            logical_key=key,
            merge_index=merge_index,
            priority=_intent_priority(intent),
        )
        existing = effective_by_key.get(key)
        if existing is None:
            effective_by_key[key] = incoming
            continue

        incoming_rank = _dominance_rank(incoming.intent)
        existing_rank = _dominance_rank(existing.intent)
        if incoming_rank > existing_rank:
            effective_by_key[key] = incoming
            continue
        if incoming_rank < existing_rank:
            continue

        # Same-type conflict: latest in deterministic merge order wins.
        effective_by_key[key] = incoming

    ordered = sorted(
        effective_by_key.values(),
        key=lambda item: (item.priority, item.merge_index, item.logical_key),
    )
    return tuple(ordered)
