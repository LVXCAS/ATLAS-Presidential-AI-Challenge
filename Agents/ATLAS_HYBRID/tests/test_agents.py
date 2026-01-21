"""
Unit tests for ATLAS Track II core agent logic.

Tests cover:
1. Agent aggregation logic (veto, thresholds, data sufficiency)
2. Baseline risk calculation (ATR, RSI, ADX)
3. Stress detection (volatility thresholds)
4. Weighted aggregation (normalization, score calculation)
"""

import unittest
from typing import Any, Dict

# Import functions to test
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_team_utils import (
    baseline_risk,
    baseline_assessment,
    quant_team_assessment,
    infer_stress_flags,
    atr_like,
)


class TestAgentAggregation(unittest.TestCase):
    """Test the agent aggregation logic from quant_team_assessment()."""

    def test_veto_mechanism_forces_stand_down(self):
        """Test that a veto agent with score >= 0.80 forces STAND_DOWN."""
        agent_assessments = {
            "Agent1": {"score": 0.10, "explanation": "Low risk", "weight": 1.0, "is_veto": False},
            "Agent2": {"score": 0.15, "explanation": "Minimal risk", "weight": 1.0, "is_veto": False},
            "VetoAgent": {"score": 0.85, "explanation": "Critical volatility spike", "weight": 1.0, "is_veto": True},
        }

        label, meta = quant_team_assessment(agent_assessments)

        self.assertEqual(label, "STAND_DOWN", "Veto agent should force STAND_DOWN")
        self.assertEqual(meta["method"], "veto", "Method should be 'veto'")
        self.assertEqual(meta["risk_score"], 1.0, "Risk score should be 1.0 for veto")
        self.assertEqual(meta["risk_posture"], "HIGH", "Risk posture should be HIGH")

    def test_veto_below_threshold_no_force(self):
        """Test that veto agent below 0.80 threshold doesn't force STAND_DOWN."""
        agent_assessments = {
            "Agent1": {"score": 0.10, "explanation": "Low risk", "weight": 1.0, "is_veto": False},
            "VetoAgent": {"score": 0.75, "explanation": "Moderate concern", "weight": 1.0, "is_veto": True},
        }

        label, meta = quant_team_assessment(agent_assessments)

        # With weighted average: (0.10 + 0.75) / 2 = 0.425
        self.assertEqual(label, "STAND_DOWN", "Score >= 0.36 should be STAND_DOWN")
        self.assertEqual(meta["method"], "weighted_average", "Should use weighted average, not veto")

    def test_greenlight_threshold_below_0_25(self):
        """Test that aggregated score < 0.25 produces GREENLIGHT."""
        agent_assessments = {
            "Agent1": {"score": 0.20, "explanation": "Low risk", "weight": 1.0, "is_veto": False},
            "Agent2": {"score": 0.15, "explanation": "Minimal risk", "weight": 1.0, "is_veto": False},
            "Agent3": {"score": 0.25, "explanation": "Acceptable", "weight": 1.0, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        # Average: (0.20 + 0.15 + 0.25) / 3 = 0.20
        self.assertEqual(label, "GREENLIGHT", "Score < 0.25 should produce GREENLIGHT")
        self.assertEqual(meta["market_condition"], "CALM", "Market condition should be CALM")
        self.assertEqual(meta["risk_posture"], "LOW", "Risk posture should be LOW")

    def test_watch_threshold_0_25_to_0_36(self):
        """Test that 0.25 <= score < 0.36 produces WATCH."""
        agent_assessments = {
            "Agent1": {"score": 0.25, "explanation": "Some uncertainty", "weight": 1.0, "is_veto": False},
            "Agent2": {"score": 0.30, "explanation": "Moderate risk", "weight": 1.0, "is_veto": False},
            "Agent3": {"score": 0.35, "explanation": "Caution advised", "weight": 1.0, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        # Average: (0.25 + 0.30 + 0.35) / 3 = 0.30
        self.assertEqual(label, "WATCH", "Score 0.25-0.36 should produce WATCH")
        self.assertEqual(meta["market_condition"], "ELEVATED", "Market condition should be ELEVATED")
        self.assertEqual(meta["risk_posture"], "ELEVATED", "Risk posture should be ELEVATED")

    def test_stand_down_threshold_above_0_36(self):
        """Test that score >= 0.36 produces STAND_DOWN."""
        agent_assessments = {
            "Agent1": {"score": 0.40, "explanation": "High volatility", "weight": 1.0, "is_veto": False},
            "Agent2": {"score": 0.50, "explanation": "Risk detected", "weight": 1.0, "is_veto": False},
            "Agent3": {"score": 0.30, "explanation": "Some concern", "weight": 1.0, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        # Average: (0.40 + 0.50 + 0.30) / 3 = 0.40
        self.assertEqual(label, "STAND_DOWN", "Score >= 0.36 should produce STAND_DOWN")
        self.assertEqual(meta["market_condition"], "STRESS", "Market condition should be STRESS")
        self.assertEqual(meta["risk_posture"], "HIGH", "Risk posture should be HIGH")

    def test_insufficient_data_gets_zero_weight(self):
        """Test that agents with data_sufficiency='insufficient' get weight 0."""
        agent_assessments = {
            "Agent1": {
                "score": 0.90,
                "explanation": "No data available",
                "weight": 1.0,
                "is_veto": False,
                "details": {"data_sufficiency": "insufficient"},
            },
            "Agent2": {"score": 0.10, "explanation": "Low risk", "weight": 1.0, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        # Agent1 should be excluded (weight set to 0), so only Agent2 counts
        # Score should be 0.10 (only Agent2)
        self.assertEqual(label, "GREENLIGHT", "Should ignore insufficient data agent")
        self.assertIn("Agent1", meta["insufficient_agents"], "Agent1 should be marked as insufficient")
        self.assertAlmostEqual(meta["aggregated_score"], 0.10, places=2)

    def test_all_insufficient_data_defaults_to_watch(self):
        """Test that when all agents have insufficient data, defaults to WATCH."""
        agent_assessments = {
            "Agent1": {
                "score": 0.90,
                "explanation": "No data",
                "weight": 1.0,
                "details": {"data_sufficiency": "insufficient"},
            },
            "Agent2": {
                "score": 0.85,
                "explanation": "No data",
                "weight": 1.0,
                "details": {"data_sufficiency": "insufficient"},
            },
        }

        label, meta = quant_team_assessment(agent_assessments)

        self.assertEqual(label, "WATCH", "Should default to WATCH with no usable data")
        self.assertEqual(meta["method"], "insufficient_data", "Method should indicate insufficient data")
        self.assertEqual(len(meta["insufficient_agents"]), 2, "Both agents should be marked insufficient")


class TestBaselineRisk(unittest.TestCase):
    """Test baseline risk calculation logic."""

    def test_high_atr_triggers_stand_down(self):
        """Test that ATR >= 25 pips triggers STAND_DOWN."""
        # ATR in absolute terms, need to calculate pips
        # atr_pips = (atr / price) * 10000
        # For price=1.0, atr=0.0025 gives 25 pips
        indicators = {"atr": 0.0025, "_price": 1.0, "rsi": 50.0, "adx": 30.0}

        label, meta = baseline_risk(indicators)

        self.assertEqual(label, "STAND_DOWN", "ATR >= 25 pips should trigger STAND_DOWN")
        self.assertAlmostEqual(meta["atr_pips"], 25.0, places=1)
        self.assertEqual(meta["reason"], "Volatility spike (ATR high)")

    def test_extreme_rsi_high_triggers_watch(self):
        """Test that RSI >= 72 triggers WATCH."""
        indicators = {"atr": 0.0010, "_price": 1.0, "rsi": 75.0, "adx": 30.0}

        label, meta = baseline_risk(indicators)

        self.assertEqual(label, "WATCH", "RSI >= 72 should trigger WATCH")
        self.assertIn("Momentum extreme (RSI)", meta["reasons"])

    def test_extreme_rsi_low_triggers_watch(self):
        """Test that RSI <= 28 triggers WATCH."""
        indicators = {"atr": 0.0010, "_price": 1.0, "rsi": 25.0, "adx": 30.0}

        label, meta = baseline_risk(indicators)

        self.assertEqual(label, "WATCH", "RSI <= 28 should trigger WATCH")
        self.assertIn("Momentum extreme (RSI)", meta["reasons"])

    def test_low_adx_flags_watch(self):
        """Test that ADX <= 18 flags WATCH."""
        indicators = {"atr": 0.0010, "_price": 1.0, "rsi": 50.0, "adx": 15.0}

        label, meta = baseline_risk(indicators)

        self.assertEqual(label, "WATCH", "ADX <= 18 should trigger WATCH")
        self.assertIn("Choppy/uncertain regime (low ADX)", meta["reasons"])

    def test_combined_conditions_watch(self):
        """Test that multiple moderate conditions trigger WATCH."""
        indicators = {
            "atr": 0.0016,  # 16 pips - elevated but not spike
            "_price": 1.0,
            "rsi": 73.0,  # extreme
            "adx": 17.0,  # low
        }

        label, meta = baseline_risk(indicators)

        self.assertEqual(label, "WATCH", "Multiple conditions should trigger WATCH")
        self.assertGreaterEqual(len(meta["reasons"]), 2, "Should have multiple reasons")
        self.assertIn("Momentum extreme (RSI)", meta["reasons"])
        self.assertIn("Choppy/uncertain regime (low ADX)", meta["reasons"])

    def test_normal_conditions_greenlight(self):
        """Test that normal conditions produce GREENLIGHT."""
        indicators = {"atr": 0.0010, "_price": 1.0, "rsi": 50.0, "adx": 30.0}

        label, meta = baseline_risk(indicators)

        self.assertEqual(label, "GREENLIGHT", "Normal conditions should produce GREENLIGHT")
        self.assertEqual(meta["reason"], "No baseline risk flags")


class TestBaselineAssessment(unittest.TestCase):
    """Test baseline_assessment() which maps labels to scores."""

    def test_greenlight_maps_to_low_score(self):
        """Test that GREENLIGHT maps to score 0.20."""
        indicators = {"atr": 0.0010, "_price": 1.0, "rsi": 50.0, "adx": 30.0}

        result = baseline_assessment(indicators)

        self.assertEqual(result["label"], "GREENLIGHT")
        self.assertEqual(result["score"], 0.20)

    def test_watch_maps_to_medium_score(self):
        """Test that WATCH maps to score 0.55."""
        indicators = {"atr": 0.0010, "_price": 1.0, "rsi": 75.0, "adx": 30.0}

        result = baseline_assessment(indicators)

        self.assertEqual(result["label"], "WATCH")
        self.assertEqual(result["score"], 0.55)

    def test_stand_down_maps_to_high_score(self):
        """Test that STAND_DOWN maps to score 0.90."""
        indicators = {"atr": 0.0025, "_price": 1.0, "rsi": 50.0, "adx": 30.0}

        result = baseline_assessment(indicators)

        self.assertEqual(result["label"], "STAND_DOWN")
        self.assertEqual(result["score"], 0.90)


class TestStressDetection(unittest.TestCase):
    """Test stress detection using volatility thresholds."""

    def test_high_volatility_triggers_stress(self):
        """Test that ATR >= 8 pips correctly identifies stress."""
        # Create prices with high volatility
        # For price around 1.10, we need ATR of at least 0.0088 to get 8 pips
        # atr_pips = (atr / price) * 10000
        # Create prices with large moves
        prices = [1.10, 1.1015, 1.1005, 1.102, 1.099, 1.103, 1.098, 1.104, 1.097, 1.105] * 3

        stress_flags = infer_stress_flags(prices, threshold_pips=8.0)

        # Later points should be marked as stress due to high ATR
        self.assertIsInstance(stress_flags, list)
        self.assertEqual(len(stress_flags), len(prices))
        # Check that some later points are marked as stress
        self.assertTrue(any(stress_flags[15:]), "High volatility should trigger stress flags")

    def test_low_volatility_no_stress(self):
        """Test that low volatility doesn't trigger stress."""
        # Create prices with very low volatility
        prices = [1.10 + (i * 0.00001) for i in range(30)]  # Very small increments

        stress_flags = infer_stress_flags(prices, threshold_pips=8.0)

        # Should not trigger stress flags
        self.assertIsInstance(stress_flags, list)
        self.assertEqual(len(stress_flags), len(prices))
        # Most should be False (allowing for edge cases in first few points)
        false_count = sum(1 for flag in stress_flags if not flag)
        self.assertGreater(false_count, len(prices) * 0.9, "Low volatility should not trigger stress")

    def test_stress_threshold_customization(self):
        """Test that custom threshold works correctly."""
        # Moderate volatility prices
        prices = [1.10, 1.1008, 1.1004, 1.1012, 1.1006, 1.1014, 1.1008, 1.1015] * 2

        # With high threshold, should not trigger
        stress_high = infer_stress_flags(prices, threshold_pips=15.0)
        self.assertFalse(any(stress_high[10:]), "High threshold should not trigger on moderate volatility")

        # With low threshold, more likely to trigger
        stress_low = infer_stress_flags(prices, threshold_pips=3.0)
        # Some should be marked (moderate volatility exceeds low threshold)
        true_count = sum(1 for flag in stress_low[10:] if flag)
        self.assertGreater(true_count, 0, "Low threshold should trigger on moderate volatility")

    def test_initial_points_not_stress(self):
        """Test that first 2 points are never marked as stress (insufficient data)."""
        prices = [1.10, 1.12, 1.14, 1.16]  # Even with large moves

        stress_flags = infer_stress_flags(prices, threshold_pips=1.0)  # Very low threshold

        self.assertFalse(stress_flags[0], "First point should never be stress")
        self.assertFalse(stress_flags[1], "Second point should never be stress")


class TestWeightedAggregation(unittest.TestCase):
    """Test weighted aggregation mechanics."""

    def test_weights_properly_normalized(self):
        """Test that different weights are properly normalized in calculation."""
        agent_assessments = {
            "HighWeight": {"score": 0.80, "explanation": "High risk", "weight": 2.0, "is_veto": False},
            "LowWeight": {"score": 0.20, "explanation": "Low risk", "weight": 0.5, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        # Weighted average: (0.80 * 2.0 + 0.20 * 0.5) / (2.0 + 0.5)
        #                  = (1.6 + 0.1) / 2.5 = 1.7 / 2.5 = 0.68
        expected_score = (0.80 * 2.0 + 0.20 * 0.5) / (2.0 + 0.5)
        self.assertAlmostEqual(meta["aggregated_score"], expected_score, places=2)
        self.assertAlmostEqual(meta["total_weight"], 2.5, places=2)

    def test_score_calculation_formula(self):
        """Test that score calculation follows the weighted average formula."""
        agent_assessments = {
            "Agent1": {"score": 0.30, "explanation": "Low-med risk", "weight": 1.0, "is_veto": False},
            "Agent2": {"score": 0.50, "explanation": "Medium risk", "weight": 1.5, "is_veto": False},
            "Agent3": {"score": 0.20, "explanation": "Low risk", "weight": 0.8, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        # Calculate expected weighted average
        total = (0.30 * 1.0) + (0.50 * 1.5) + (0.20 * 0.8)
        total_weight = 1.0 + 1.5 + 0.8
        expected_score = total / total_weight

        self.assertAlmostEqual(meta["aggregated_score"], expected_score, places=2)
        self.assertAlmostEqual(meta["risk_score"], expected_score, places=2)

    def test_equal_weights_simple_average(self):
        """Test that equal weights produce a simple average."""
        agent_assessments = {
            "Agent1": {"score": 0.20, "explanation": "Low", "weight": 1.0, "is_veto": False},
            "Agent2": {"score": 0.40, "explanation": "Med", "weight": 1.0, "is_veto": False},
            "Agent3": {"score": 0.30, "explanation": "Low-med", "weight": 1.0, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        # Simple average: (0.20 + 0.40 + 0.30) / 3 = 0.30
        expected_score = (0.20 + 0.40 + 0.30) / 3
        self.assertAlmostEqual(meta["aggregated_score"], expected_score, places=2)

    def test_single_agent_returns_its_score(self):
        """Test that a single agent returns its exact score."""
        agent_assessments = {
            "OnlyAgent": {"score": 0.42, "explanation": "Solo assessment", "weight": 1.5, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        # With only one agent, score should equal that agent's score
        self.assertAlmostEqual(meta["aggregated_score"], 0.42, places=2)

    def test_zero_weight_agents_excluded(self):
        """Test that agents with zero weight don't affect the score."""
        agent_assessments = {
            "ActiveAgent": {"score": 0.25, "explanation": "Active", "weight": 1.0, "is_veto": False},
            "ZeroWeightAgent": {"score": 0.90, "explanation": "Ignored", "weight": 0.0, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        # Should only count ActiveAgent
        self.assertAlmostEqual(meta["aggregated_score"], 0.25, places=2)
        self.assertEqual(label, "WATCH")  # 0.25 is exactly at WATCH threshold


class TestATRCalculation(unittest.TestCase):
    """Test ATR-like volatility calculation helper."""

    def test_atr_with_stable_prices(self):
        """Test ATR calculation with stable prices."""
        prices = [1.10] * 20  # Completely stable

        atr = atr_like(prices, period=14)

        self.assertAlmostEqual(atr, 0.0, places=6, msg="Stable prices should have zero ATR")

    def test_atr_with_volatile_prices(self):
        """Test ATR calculation with volatile prices."""
        # Create prices with known moves
        prices = [1.10, 1.11, 1.09, 1.12, 1.08, 1.13, 1.07, 1.14] * 3

        atr = atr_like(prices, period=14)

        # Should be non-zero and relatively high
        self.assertGreater(atr, 0.01, "Volatile prices should have measurable ATR")

    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data returns 0."""
        prices = [1.10]  # Only one price

        atr = atr_like(prices, period=14)

        self.assertEqual(atr, 0.0, "Single price should return zero ATR")


class TestDriversAndExplanations(unittest.TestCase):
    """Test that drivers and explanations are properly included in output."""

    def test_drivers_included_in_output(self):
        """Test that top drivers are included in aggregation output."""
        agent_assessments = {
            "HighScoreAgent": {"score": 0.70, "explanation": "High volatility detected", "weight": 1.0, "is_veto": False},
            "MedScoreAgent": {"score": 0.40, "explanation": "Moderate risk", "weight": 1.0, "is_veto": False},
            "LowScoreAgent": {"score": 0.15, "explanation": "Minimal risk", "weight": 1.0, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        self.assertIn("drivers", meta)
        self.assertIsInstance(meta["drivers"], list)
        self.assertGreater(len(meta["drivers"]), 0, "Should have at least one driver")
        # Drivers should be sorted by weighted score
        self.assertEqual(meta["drivers"][0]["agent"], "HighScoreAgent", "Highest score should be first driver")

    def test_explanation_contains_score(self):
        """Test that explanation includes the risk score."""
        agent_assessments = {
            "Agent1": {"score": 0.45, "explanation": "Some risk", "weight": 1.0, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        explanation = meta["explanation"]
        self.assertIn("0.45", explanation, "Explanation should include the risk score")

    def test_risk_flag_codes_populated(self):
        """Test that risk flag codes are populated for known agents."""
        agent_assessments = {
            "TechnicalAgent": {"score": 0.70, "explanation": "High vol", "weight": 1.0, "is_veto": False},
            "CorrelationAgent": {"score": 0.65, "explanation": "Breakdown", "weight": 1.0, "is_veto": False},
        }

        label, meta = quant_team_assessment(agent_assessments)

        self.assertIn("risk_flags", meta)
        # Should include HIGH_VOLATILITY and CORRELATION_BREAKDOWN
        self.assertIn("HIGH_VOLATILITY", meta["risk_flags"])
        self.assertIn("CORRELATION_BREAKDOWN", meta["risk_flags"])


if __name__ == "__main__":
    unittest.main()
