# TradingChassis Core Documentation

This directory is the current documentation start point for the `core` repository.

Documentation intent follows a Concept -> Flow -> Code -> API progression. In this phase, docs are
still kept in a flat layout. A later phase will reorganize structure and add dedicated code-map and
reference pages.

## Start Here

1. [CoreStep MVP Baseline](core-step-mvp-baseline.md)
2. [Runtime/Core Responsibility Model](core-runtime-responsibility-model.md)
3. [Event Model](event-model.md)
4. [Risk vs ExecutionControl](risk-vs-execution-control.md)
5. [Control Time and Scheduling](control-time-and-scheduling.md)
6. [OrderSubmittedEvent](order-submitted-event.md)
7. [OrderExecutionFeedbackEvent (rc3 MVP path)](order-execution-feedback-event.md)
8. [GateDecision Compatibility](gate-decision-compatibility.md)
9. [Post-MVP Roadmap](post-mvp-roadmap.md)

## Current Status Notes

- The CoreStep MVP baseline is accepted and frozen.
- Migrated paths remain behind flags that are default `false`.
- `GateDecision` remains compatibility for legacy/default-off behavior.
- This MVP is not the final architecture.
