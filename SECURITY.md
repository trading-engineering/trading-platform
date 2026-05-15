# Security Policy

## Supported Baseline

The supported baseline is the clean standalone Core package line (`0.1.x` and
forward on the active mainline branch).

Older historical commits may not receive fixes.

## Reporting a Vulnerability

Do not report vulnerabilities in public issues.

Use a private security advisory workflow if available for this repository, or
contact project maintainers through the configured private channel.

Include:

- affected component(s)
- reproduction details and impact
- suggested mitigations (if known)

## Scope

This policy covers the Core package in this repository, including:

- canonical Event and Intent contracts
- deterministic CoreStep/CoreWakeupStep decision Pipeline
- package integrity and dependency usage in `tradingchassis_core`

## Secrets and Credentials Policy

Never commit live secrets or account-sensitive data, including:

- API keys and Venue credentials
- account identifiers tied to real accounts
- private trading data dumps

Tests and documentation examples must use synthetic or non-sensitive data only.

## Runtime and Trading Caveat

- TradingChassis Core is a library and does not guarantee safe live trading by
  itself.
- Runtime orchestration, Venue behavior, and deployment hardening remain outside
  this package scope and require separate validation.

## No Financial Performance Guarantee

This package provides deterministic software behavior, not financial advice or
performance guarantees.

## Dependency Vulnerability Handling

- Keep dependencies minimal and pinned by compatible ranges.
- Review dependency advisories and patch vulnerable versions promptly.
- Prefer removing unused dependencies over adding new tooling.
