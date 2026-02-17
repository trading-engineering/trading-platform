#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Compiling requirements with pip-tools..."

python -m pip install --upgrade \
  "pip>=23.3,<25" \
  "setuptools>=68,<81" \
  "wheel>=0.41,<1" \
  "pip-tools>=7.3,<7.6"

python -m piptools compile pyproject.toml -o requirements.txt
python -m piptools compile pyproject.toml --extra dev -o requirements-dev.txt

echo "✅ requirements.txt and requirements-dev.txt updated"
