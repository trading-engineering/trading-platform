# Contributing

Thank you for your interest in contributing!

This repository focuses on deterministic, event-driven trading architecture.
Contributions should preserve clarity, explicitness and reproducibility.

---

## Design Principles

All contributions must respect the core design philosophy:

- Determinism over convenience
- Explicit state modeling
- No hidden side effects
- Risk-first architecture
- Clear domain boundaries

Avoid introducing implicit behavior or non-deterministic execution paths.

---

## Workflow

1. Fork the repository
2. Create a feature branch
3. Commit small, logical changes
4. Open a Pull Request with clear description

---

## Commit Style

Use clear messages:

feat: add monitoring overlay  
fix: correct SecretProviderClass parameters  
docs: update bootstrap instructions  

---

## Development Environment

Recommended:

- Python 3.11.x
- Dev Container (provided in this repository)

Alternatively:

```bash
pip install -e .
```

---

## Testing

Before submitting:

- The `./scripts/check.sh` script must pass
- All backtests must complete successfully and produce result artifacts
