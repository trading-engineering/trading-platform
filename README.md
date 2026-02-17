# Trading Platform

![CI](https://github.com/trading-engineering/trading-platform/actions/workflows/tests.yaml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Deterministic, event-driven trading framework built on top of
[hftbacktest](https://github.com/nkaz001/hftbacktest), extending it with
risk management, explicit order state machines, queue semantics and
research orchestration.

Designed for realistic backtesting and production-ready architecture.

---

## 🧠 What is this?

This project wraps the open-source `hftbacktest` engine and extends it
into a structured trading framework.

While `hftbacktest` provides a high-performance event-driven simulation
core, this platform adds the missing layers required for realistic
research and strategy development:

- Explicit order state machine
- Risk engine with enforceable constraints
- Queue and rate-limit semantics
- Venue abstraction (backtest + live ready)
- Deterministic execution guarantees
- Experiment orchestration (segments, sweeps)
- Schema-validated domain events

The result is a layered trading architecture.

---

## 🧩 What does it solve?

Backtesting setups tend to:

- Ignore realistic order lifecycle behavior
- Have no explicit risk enforcement
- Mix strategy logic with execution logic
- Lack deterministic event modeling
- Do not scale to research workflows

This framework solves those problems by introducing:

- Clear domain boundaries
- Explicit state transitions
- Risk-first execution gating
- Deterministic event pipelines
- Research-grade orchestration

It enables realistic simulation while remaining extensible toward live
trading.

---

## 🏗 Architecture Overview

The system is structured into clear layers with every layer being
exchangeable:

Strategy\
↓\
Risk Engine\
↓\
Venue Abstraction\
↓\
Backtest or Execution Engine

Internally:

- `hftbacktest` remains timestamp-atomic and event-driven.
- The strategy layer operates state-based per timestamp.
- The runner orchestrates event processing deterministically.

Core modules:

- `core/` -- domain models, state machine, risk engine, events
- `backtest/` -- engine adapters, orchestration, runtime entrypoints
- `strategies/` -- base strategy interfaces
- `live/` -- live adapters (work in progress)
- `examples/` -- minimal runnable setups
- `tests/` -- semantic invariant validation
- `scripts/` -- development helper scripts

---

## 🚀 Quickstart

Minimal local example: `examples/local/backtest.py`\
Runs entirely locally using bundled or synthetic example data.\
No S3, cloud storage or live connectivity required.

### Option 1 --- Recommended: Dev Container

A reproducible development environment is provided via a dev container.

```bash
git clone https://github.com/trading-engineering/trading-platform
cd trading-platform
```

Open in an IDE supporting Dev Containers (e.g. VS Code), reopen in
container, then:

```bash
python examples/local/backtest.py --config examples/local/local.json
```

No manual `pip install` required inside the container.

### Option 2 --- Local Python Environment

Python 3.11.x is required.

```bash
pip install -e .
python examples/local/backtest.py --config examples/local/local.json
```

---

## ▶️ Execution Modes

### Local Mode

- Fully local execution
- Uses bundled or synthetic data
- No cloud dependencies
- Suitable for development and testing

```bash
python examples/local/backtest.py --config examples/local/local.json
```

### Cloud / Entrypoint Mode

The backtest runtime exposes entrypoints designed for cloud-native
execution environments.

These enable:

- Remote segment execution
- Distributed parameter sweeps
- Object storage integration
- Experiment orchestration via external workflow engines

Entrypoints are located in:

```
trading_platform/backtest/runtime/
```

Infrastructure and orchestration configuration are intentionally kept separate from the core trading framework.

Cloud execution currently relies on [Oracle Cloud Infrastructure](https://cloud.oracle.com) (OCI) Object Storage accessed via Instance Principals and OCI IAM configuration.
The storage integration is implemented through an S3-compatible adapter in the I/O layer located in:

```
trading_platform/backtest/io/
```

The runtime entrypoints are designed primarily for [Kubernetes](https://kubernetes.io)-based workloads orchestrated via [Argo Workflows](https://argoproj.github.io/workflows).

While the core architecture is cloud-agnostic, the current infrastructure bindings are OCI-specific.
Other cloud providers and execution environments are not yet implemented.

---

## 📊 Data Requirements

The backtest engine expects structured, event-driven market data
compatible with `hftbacktest`.

Key assumptions:

- Timestamp-based atomic event processing
- Deterministic event ordering
- Preprocessed market events
- No implicit reconstruction during runtime

Example synthetic datasets are provided in:

```
tests/data/parts/
```

Example parts:

```
tests/data/parts/part-000.npz
tests/data/parts/part-001.npz
tests/data/parts/part-002.npz
```

### Result Artifacts

Backtest runs produce deterministic result artifacts stored in:

```
tests/data/results/
```

Generated files may include:

```
tests/data/results/stats.npz
tests/data/results/events.json
```

Helper scripts for generating and inspecting synthetic datasets are located in:

```
tests/data/scripts/
```

---

## ⚙️ Configuration

Execution is driven by explicit configuration files
(e.g. `examples/local/local.json`).

Configurations define:

- Data sources
- Risk constraints
- Strategy parameters
- Execution settings

All configuration is explicit and validated.

---

## 🔒 Deterministic Execution

The framework enforces:

- Explicit state transitions
- No hidden side effects
- Ordered event processing
- Reproducible backtest runs

Semantic invariants are verified via dedicated test suites.

Run tests with:

```bash
pytest
```

---

## 🧪 Research & Orchestration

The backtest layer includes:

- Segment-based execution
- Parameter sweeps
- Experiment finalization entrypoints
- Metrics export hooks compatible with [Prometheus](https://prometheus.io)
- Logging integration compatible with [MLflow](https://mlflow.org)

This enables structured research workflows.

Metrics and experiment tracking are designed to integrate naturally into Kubernetes-based deployments, but the core framework does not require a specific monitoring or tracking backend.

---

## 📦 Scope

This repository focuses on:

- Realistic backtesting
- Uniform core domain logic
- Risk-aware backtesting and execution
- Deterministic research workflows

It does not include:

- Data collection pipelines
- Production-grade OMS cloud infrastructure

Live exchange connectivity and production-grade OMS software infrastructure
are under active development and not yet feature-complete.

---

## 🎯 Design Principles

- Determinism over convenience
- Explicit state modeling
- No hidden side effects
- Risk-first architecture
- Clear domain boundaries

---

## 📌 Project Status

- Backtest stack: maturing
- Live adapters: under development
- Cloud execution: experimental

---

## 👥 Who is this for?

- Quant developers building systematic strategies
- Engineers interested in event-driven architectures
- Researchers requiring deterministic simulations
- Contributors exploring structured trading systems
- Reviewers evaluating system design and architecture depth

---

## 🏷️ Versioning

This project follows the MIT license and semantic versioning.
Initial public release: `v0.1.0`
