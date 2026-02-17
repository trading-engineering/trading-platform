# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog
and this project adheres to Semantic Versioning.

## [Unreleased]

## [0.1.0] - 2026-02-17

Initial public release of the trading platform backtest core.

### Added

#### Core Domain
- Explicit order state machine
- Structured domain types and reject reasons
- Slot-based order tracking
- Event bus and event sink abstractions
- JSON schema validation for domain events

#### Risk Layer
- Configurable risk engine
- Risk constraint enforcement
- Deterministic risk gating before execution

#### Backtest Layer
- Integration with [hftbacktest](https://github.com/nkaz001/hftbacktest)
- Strategy runner abstraction
- Venue adapter interface
- Deterministic event processing pipeline

#### Orchestration
- Segment-based execution model
- Parameter sweep runtime
- Experiment and segment entrypoints
- Prometheus metrics integration
- MLflow-compatible logging hooks

#### Execution Modes
- Fully local execution example
- Cloud-native runtime entrypoints
- S3-compatible storage adapter

#### Strategy Framework
- Base strategy interface
- Structured strategy configuration

#### Testing
- Semantic invariant test suite
- Order state transition validation
- Queue dominance rules
- Risk constraint validation
- Schema conformance tests

#### Tooling
- Dev container configuration
- Development validation scripts
- Dependency compilation helper
