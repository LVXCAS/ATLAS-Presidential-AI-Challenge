"""
Support/Resistance Agent - Identifies key price levels from historical action.

Improves entry/exit precision by detecting where price historically bounces
or breaks through. Uses swing highs/lows, round numbers, and Fibonacci levels.
"""
from typing import Dict, Tuple, List
from .base_agent import BaseAgent
import numpy as np


class SupportResistanceAgent(BaseAgent):
    """
    Identifies and trades support/resistance levels for optimal entry/exit.

    Strategy:
    1. BUY near strong support (bounce setup)
    2. SELL near strong resistance (rejection setup)
    3. BUY on resistance breakout with confirmation
    4. SELL on support breakdown with confirmation
    5. AVOID trading in no-man's-land (between levels)
    """

    def __init__(self, initial_weight: float = 1.7):
        super().__init__(name="SupportResistanceAgent", initial_weight=initial_weight)

        # Tolerance for "at level" (in percentage)
        self.level_tolerance = 0.0015  # 0.15% = ~15 pips on EUR/USD

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Analyze price position relative to key support/resistance levels.

        Returns:
        - BUY if at support or breaking resistance
        - SELL if at resistance or breaking support
        - NEUTRAL if in no-man's land or unclear
        """
        pair = market_data.get("pair", "")
        price = market_data.get("price", 0)
        candles = market_data.get("candles", [])

        if not candles or len(candles) < 50:
            return ("NEUTRAL", 0.3, {"error": "insufficient_data"})

        # Extract price data
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        closes = np.array([c['close'] for c in candles])

        # Identify key levels
        resistance_levels = self._find_resistance_levels(highs, closes)
        support_levels = self._find_support_levels(lows, closes)

        # Find nearest levels
        nearest_resistance = self._find_nearest_level(price, resistance_levels, direction="above")
        nearest_support = self._find_nearest_level(price, support_levels, direction="below")

        # Calculate distances
        dist_to_resistance = ((nearest_resistance - price) / price) if nearest_resistance else float('inf')
        dist_to_support = ((price - nearest_support) / price) if nearest_support else float('inf')

        # Check for round numbers (psychological levels)
        round_number = self._find_nearest_round_number(price, pair)
        dist_to_round = abs(price - round_number) / price if round_number else float('inf')

        # Decision logic
        reasoning = {
            "price": round(price, 5),
            "nearest_resistance": round(nearest_resistance, 5) if nearest_resistance else None,
            "nearest_support": round(nearest_support, 5) if nearest_support else None,
            "dist_to_resistance_%": round(dist_to_resistance * 100, 2),
            "dist_to_support_%": round(dist_to_support * 100, 2),
            "round_number": round(round_number, 5) if round_number else None
        }

        # AT SUPPORT - Bounce setup (BUY)
        if dist_to_support <= self.level_tolerance:
            strength = self._calculate_level_strength(nearest_support, closes, lows)

            vote = "BUY"
            confidence = min(0.75, 0.50 + strength * 0.25)

            return (vote, confidence, {
                **reasoning,
                "signal": "at_support_bounce",
                "level_strength": round(strength, 2),
                "message": f"Price {dist_to_support*100:.2f}% from support {nearest_support:.5f}"
            })

        # AT RESISTANCE - Rejection setup (SELL)
        if dist_to_resistance <= self.level_tolerance:
            strength = self._calculate_level_strength(nearest_resistance, closes, highs)

            vote = "SELL"
            confidence = min(0.75, 0.50 + strength * 0.25)

            return (vote, confidence, {
                **reasoning,
                "signal": "at_resistance_rejection",
                "level_strength": round(strength, 2),
                "message": f"Price {dist_to_resistance*100:.2f}% from resistance {nearest_resistance:.5f}"
            })

        # AT ROUND NUMBER - Watch for bounce/break
        if dist_to_round <= self.level_tolerance:
            # Round numbers act as both support and resistance
            # Neutral but flag it
            return ("NEUTRAL", 0.6, {
                **reasoning,
                "signal": "at_round_number",
                "message": f"Price at psychological level {round_number:.5f}"
            })

        # RESISTANCE BREAKOUT - Price above resistance (bullish)
        if nearest_resistance and price > nearest_resistance * 1.001:  # 0.1% above
            # Check if recent breakout (within last 5 candles)
            recent_closes = closes[-5:]
            if any(c < nearest_resistance for c in recent_closes[:-1]):
                vote = "BUY"
                confidence = 0.70

                return (vote, confidence, {
                    **reasoning,
                    "signal": "resistance_breakout",
                    "message": f"Broke above resistance {nearest_resistance:.5f}"
                })

        # SUPPORT BREAKDOWN - Price below support (bearish)
        if nearest_support and price < nearest_support * 0.999:  # 0.1% below
            # Check if recent breakdown
            recent_closes = closes[-5:]
            if any(c > nearest_support for c in recent_closes[:-1]):
                vote = "SELL"
                confidence = 0.70

                return (vote, confidence, {
                    **reasoning,
                    "signal": "support_breakdown",
                    "message": f"Broke below support {nearest_support:.5f}"
                })

        # IN NO-MAN'S LAND - Between levels, no clear setup
        if dist_to_support > self.level_tolerance and dist_to_resistance > self.level_tolerance:
            # Penalize trades in middle of range
            return ("NEUTRAL", 0.4, {
                **reasoning,
                "signal": "no_clear_level",
                "message": "Price not at key level (avoid low-probability zone)"
            })

        # Default: Normal conditions
        return ("NEUTRAL", 0.5, {
            **reasoning,
            "signal": "normal"
        })

    def _find_resistance_levels(self, highs: np.ndarray, closes: np.ndarray) -> List[float]:
        """
        Find resistance levels using swing highs.

        A swing high is a high that's higher than N bars before and after it.
        """
        levels = []
        window = 5  # Look 5 bars back and forward

        for i in range(window, len(highs) - window):
            high = highs[i]

            # Check if this is a local maximum
            is_swing_high = True
            for j in range(i - window, i + window + 1):
                if j != i and highs[j] >= high:
                    is_swing_high = False
                    break

            if is_swing_high:
                levels.append(high)

        # Cluster nearby levels (within 0.3%)
        clustered = self._cluster_levels(levels, tolerance=0.003)

        return sorted(clustered, reverse=True)  # Highest first

    def _find_support_levels(self, lows: np.ndarray, closes: np.ndarray) -> List[float]:
        """Find support levels using swing lows."""
        levels = []
        window = 5

        for i in range(window, len(lows) - window):
            low = lows[i]

            # Check if this is a local minimum
            is_swing_low = True
            for j in range(i - window, i + window + 1):
                if j != i and lows[j] <= low:
                    is_swing_low = False
                    break

            if is_swing_low:
                levels.append(low)

        # Cluster nearby levels
        clustered = self._cluster_levels(levels, tolerance=0.003)

        return sorted(clustered)  # Lowest first

    def _cluster_levels(self, levels: List[float], tolerance: float = 0.003) -> List[float]:
        """
        Merge levels that are within tolerance of each other.

        Returns the average of clustered levels.
        """
        if not levels:
            return []

        clustered = []
        levels = sorted(levels)

        current_cluster = [levels[0]]

        for level in levels[1:]:
            # If within tolerance of cluster average, add to cluster
            cluster_avg = np.mean(current_cluster)

            if abs(level - cluster_avg) / cluster_avg <= tolerance:
                current_cluster.append(level)
            else:
                # Finish current cluster, start new one
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]

        # Add final cluster
        clustered.append(np.mean(current_cluster))

        return clustered

    def _find_nearest_level(self, price: float, levels: List[float], direction: str = "above") -> float:
        """Find the nearest level above or below current price."""
        if not levels:
            return None

        if direction == "above":
            # Find smallest level that's above price
            above_levels = [l for l in levels if l > price]
            return min(above_levels) if above_levels else None
        else:  # below
            # Find largest level that's below price
            below_levels = [l for l in levels if l < price]
            return max(below_levels) if below_levels else None

    def _find_nearest_round_number(self, price: float, pair: str) -> float:
        """
        Find nearest round number (psychological level).

        For EUR/USD: 1.1000, 1.1500, etc.
        For USD/JPY: 150.00, 155.00, etc.
        """
        if pair.endswith("JPY"):
            # Round to nearest 5.00
            increment = 5.0
            return round(price / increment) * increment
        else:
            # Round to nearest 0.0500
            increment = 0.0050
            return round(price / increment) * increment

    def _calculate_level_strength(self, level: float, closes: np.ndarray,
                                  extremes: np.ndarray) -> float:
        """
        Calculate how strong a support/resistance level is.

        Strength = number of times price tested the level without breaking.

        Returns:
        - 0.0-1.0 where 1.0 = very strong level
        """
        if level is None:
            return 0.0

        tolerance = level * 0.002  # 0.2% tolerance

        # Count touches
        touches = 0
        for i in range(len(extremes)):
            if abs(extremes[i] - level) <= tolerance:
                touches += 1

        # More touches = stronger level (but cap at 5)
        strength = min(touches / 5.0, 1.0)

        return strength
