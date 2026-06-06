"""Bridge so ``colcon test`` runs the pytest suite under ``test/``."""

from __future__ import annotations

import unittest
from pathlib import Path


class PytestSuite(unittest.TestCase):
    """Run the pytest suite as a single unittest test."""

    def test_pytest_suite_passes(self) -> None:
        import pytest

        test_dir = Path(__file__).resolve().parent / "test"
        exit_code = pytest.main(["-q", str(test_dir)])
        self.assertEqual(
            int(exit_code),
            0,
            f"pytest reported exit code {int(exit_code)}",
        )
