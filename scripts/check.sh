#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Running import-linter..."
lint-imports --verbose

echo "⚡ Running ruff (check only)..."
ruff check trading_platform examples tests

echo "🧠 Running mypy..."
mypy trading_platform examples tests

echo "🧪 Running pytest..."
pytest

echo "✅ All checks passed!"
