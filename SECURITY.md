# Security Policy

## Supported Versions and Status

The active supported line is the current `main` branch and accepted MVP baseline for this Core
repository.

Older commits may not receive security fixes.

## Reporting a Vulnerability

Do not report vulnerabilities in public issues.

Use a private security advisory workflow if available for this repository, or contact project
maintainers through the project's configured private channel.

Include:

- affected component(s)
- reproduction details and impact
- suggested mitigations (if known)

## Scope

This policy covers the Core package in this repository, including:

- semantic event-processing contracts
- state and decision model handling
- package integrity and dependency usage in Core

## Out of Scope and Disclaimers

- No financial or trading performance guarantee is provided
- Safe live trading operation is not guaranteed without runtime/venue-specific validation

## Secrets and Credentials

Never commit live secrets to this repository, including:

- API keys and venue credentials
- account identifiers tied to real accounts
- private trading data dumps

Tests and documentation examples must use synthetic or non-sensitive data only.
