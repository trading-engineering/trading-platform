# TradingChassis Core Docs

This documentation set describes the clean Core package only.

## Contents

- `reference/public-api.md`: supported root exports and step contracts

## Architectural baseline

The only supported processing architecture is:

1. canonical `EventStreamEntry` ingestion
2. deterministic state reduction
3. strategy evaluation
4. candidate intent combination
5. policy admission
6. execution-control planning/apply
7. `CoreStepResult` outputs for runtime dispatch/scheduling

## Contract source of truth

Core contract models are defined in Pydantic classes under
`tradingchassis_core/core/domain/types.py`.
