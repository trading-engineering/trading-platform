"""Smoke test for the runnable RiskEngine example script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_CORE_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _CORE_ROOT / "examples" / "core_step_with_risk_engine.py"


def test_runnable_risk_engine_example_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, str(_EXAMPLE)],
        cwd=_CORE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "risk-example-intent" in result.stdout
