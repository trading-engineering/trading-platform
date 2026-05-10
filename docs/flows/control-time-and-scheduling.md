# Control Time and Scheduling

## What `ControlSchedulingObligation` is

`ControlSchedulingObligation` is a non-canonical Core output that requests a
future runtime wakeup boundary. It is not an event-stream input by itself.

## Why it is non-canonical

- It is an internal handoff from Core to Runtime.
- It does not directly reduce Core state as an event.
- Canonical status begins only when Runtime realizes the obligation and injects
  a `ControlTimeEvent`.

## What `ControlTimeEvent` is

`ControlTimeEvent` is the canonical control re-entry event that Runtime injects
when a due obligation is realized.

## Current MVP behavior

- Control-time CoreStep path is behind `enable_core_step_control_time_dispatch`.
- Mixed wakeup collapse behavior is behind `enable_core_step_wakeup_collapse`.
- Runtime owns pending obligations and due-time realization.
- No periodic tick model is implied by this MVP.
