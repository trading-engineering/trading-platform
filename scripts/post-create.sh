#!/usr/bin/env bash
set -euo pipefail

echo "[post-create] Using venv at /opt/venv"
# shellcheck disable=SC1091
source /opt/venv/bin/activate

cd /workspace/trading-platform

echo "[post-create] Installing dev requirements..."
python -m pip install -r requirements-dev.txt

# If this repo is a Python package, install it in editable mode
if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then
  echo "[post-create] Installing trading-platform in editable mode..."
  python -m pip install -e .
fi

echo "[post-create] Running import-linter..."
lint-imports || true

echo "[post-create] Dev container setup completed."
