# TradingChassis Core Documentation

This directory is the current documentation start point for the `core` repository.

Documentation intent follows a Concept -> Flow -> Code -> API progression.

Manual Markdown is used for now. MkDocs, mkdocstrings, and Mermaid are optional future tooling
choices and are not part of this baseline.

## Recommended Reading Order

### MVP and status

1. [CoreStep MVP Baseline](mvp/core-step-mvp-baseline.md)
2. [Compatibility Matrix](mvp/compatibility-matrix.md)

### Concepts

3. [Runtime/Core Responsibility Model](concepts/core-runtime-responsibility-model.md)
4. [Event Model](concepts/event-model.md)
5. [Risk vs ExecutionControl](concepts/risk-vs-execution-control.md)
6. [GateDecision Compatibility](concepts/gate-decision-compatibility.md)

### Flows

7. [Control Time and Scheduling](flows/control-time-and-scheduling.md)
8. [OrderSubmittedEvent](flows/order-submitted-event.md)
9. [OrderExecutionFeedbackEvent (rc3 MVP path)](flows/order-execution-feedback-event.md)

### Code map

10. [Repository Map](code-map/repository-map.md)
11. [Core Pipeline Map](code-map/core-pipeline-map.md)

### Reference

12. [Public API Reference](reference/public-api.md)
13. [Events Reference](reference/events-reference.md)

### Roadmap

14. [Post-MVP Roadmap](roadmap/post-mvp-roadmap.md)

## Current Status Notes

- The CoreStep MVP baseline is accepted and frozen.
- Migrated paths remain behind flags that are default `false`.
- `GateDecision` remains compatibility for legacy/default-off behavior.
- This MVP is not the final architecture.
