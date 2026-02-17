# Security Policy

## Supported Versions

Only the latest version on the `main` branch is actively maintained.

Older commits and historical states of the repository may not receive security updates or patches.

---

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not open a public GitHub issue**.

Instead, report it responsibly via:

- GitHub Security Advisories
- Direct contact with the repository owner (if necessary)

When submitting a report, please include:

- A clear description of the vulnerability
- Steps to reproduce (if applicable)
- Potential impact and affected components
- Any suggested mitigation or fix

Valid reports will be acknowledged in a timely manner and handled through responsible disclosure.

---

## Security Scope

This repository provides:

- Deterministic backtesting architecture
- Risk-aware execution simulation
- Event-driven domain modeling

It does **not** currently provide production-grade live trading infrastructure.

Live exchange connectivity is under development and not feature-complete.

---

## Dependency Security

- Dependencies are explicitly defined
- Python version is pinned (3.11.x)
- External libraries should be kept up to date

Security-related dependency updates are prioritized.

---

## Responsible Usage

This framework is intended for research and controlled environments.

Users are responsible for:

- Secure handling of API credentials
- Secure deployment of live trading components
- Validation of risk configurations

This repository does not assume liability for financial losses
resulting from misuse or incorrect configuration.

---

## Disclosure Policy

Please allow reasonable time for investigation and remediation before public disclosure of any reported vulnerabilities.
