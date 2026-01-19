from __future__ import annotations

import unittest
from pathlib import Path
import sys

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from main import load_csv, run_atlas


class ScenarioTests(unittest.TestCase):
    def _run_case(self, filename: str, expected_posture: str) -> None:
        prices, volumes = load_csv(TEST_DIR / "data" / filename)
        result = run_atlas(prices, volumes)
        self.assertEqual(result.get("posture"), expected_posture)
        self.assertTrue(result.get("explanation"))
        self.assertTrue(result.get("agent_signals"))

    def test_calm(self) -> None:
        self._run_case("calm.csv", "GREENLIGHT")

    def test_transition(self) -> None:
        self._run_case("transition.csv", "WATCH")

    def test_crisis(self) -> None:
        self._run_case("crisis.csv", "STAND_DOWN")


if __name__ == "__main__":
    unittest.main()
