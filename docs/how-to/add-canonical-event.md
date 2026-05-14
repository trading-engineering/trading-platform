# How To Add a Canonical Event

1. Add or update the Pydantic event model in
   `tradingchassis_core/core/domain/types.py`.
2. Register the event in `core/domain/event_model.py` canonical category mapping.
3. Add reducer handling in `core/domain/processing.py` within
   `process_canonical_event`.
4. Add/update semantics tests under `tests/semantics/`.
5. Export the model from `tradingchassis_core/__init__.py` if part of public API.
6. Update `docs/reference/events-reference.md` and `docs/reference/public-api.md`.

Rules:

- Keep canonical processing deterministic.
- Do not introduce runtime adapter or dispatch logic in reducers.
- Keep Pydantic contracts as source of truth.
