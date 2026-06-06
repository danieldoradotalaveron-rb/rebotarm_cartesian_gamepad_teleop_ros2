"""Run Ruff via uv (replaces ament_flake8 for this package)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.linter
def test_ruff_check() -> None:
    package_root = Path(__file__).resolve().parents[1]
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")

    result = subprocess.run(
        ["uv", "run", "ruff", "check", "."],
        capture_output=True,
        text=True,
        cwd=package_root,
    )
    assert result.returncode == 0, f"ruff check failed:\n{result.stdout}\n{result.stderr}"
