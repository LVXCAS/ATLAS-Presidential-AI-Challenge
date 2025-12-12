"""
Mean Reversion Agent

Trades range-bound markets when ADX < 20 (no trend).
Uses Bollinger Bands, RSI, and support/resistance.

Specialization: Low-volatility ranges, sideways markets.
"""

from typing import Dict, Tuple
from .base_agent import BaseAgent
import numpy as np


class MeanReversionAgent(BaseAgent):
    """
    Mean reversion specialist for range-bound markets.

    Activates when:
    - ADX < 20 (weak/no trend)
    - Price at BB extremes
    - RSI oversold/overbought

    Strategy: Buy support, sell resistance
    """

    def __init__(self, initial_weight: float = 1.5):
        super().__init__(name="MeanReversionAgent", initial_weight=initial_weight)

        # Thresholds
        self.max_adx_for_range = 20  # ADX < 20 = range
        self.rsi_oversold = 30
        self.rsi_overbought = 70

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Analyze for mean reversion opportunities.

        Returns:
            (vote, confidence, reasoning)
        """
        indicators = market_data.get("indicators", {})
        price = market_data.get("price", 0)

        # Extract indicators
        rsi = indicators.get("rsi", 50)
        adx = indicators.get("adx", 25)
        bb_upper = indicators.get("bb_upper", price * 1.02)
        bb_lower = indicators.get("bb_lower", price * 0.98)
        bb_middle = indicators.get("bb_middle", price)

        # Only trade in range-bound conditions
        if adx >= self.max_adx_for_range:
            return ("NEUTRAL", 0.0, {
                "reason": f"ADX {adx:.1f} too high (trending market)",
                "strategy": "mean_reversion_inactive"
            })

        # Calculate BB position
        bb_range = bb_upper - bb_lower
        if bb_range == 0:
            return ("NEUTRAL", 0.0, {"reason": "Invalid BB range"})

        price_position = (price - bb_lower) / bb_range  # 0 = lower, 1 = upper

        # Score signals
        score = 0
        signals = []
        confidence_factors = []

        # === OVERSOLD CONDITIONS (BUY) ===

        # Price near lower BB
        if price_position <= 0.2:  # Lower 20% of BB
            score += 2.0
            signals.append(f"Price at lower BB ({price_position*100:.0f}% position)")
            confidence_factors.append(0.20)

        # RSI oversold
        if rsi <= self.rsi_oversold:
            score += 1.5
            signals.append(f"RSI oversold ({rsi:.1f})")
            confidence_factors.append(0.15)
        elif rsi <= 40:
            score += 0.5
            signals.append(f"RSI approaching oversold ({rsi:.1f})")
            confidence_factors.append(0.05)

        # === OVERBOUGHT CONDITIONS (SELL) ===

        # Price near upper BB
        if price_position >= 0.8:  # Upper 20% of BB
            score -= 2.0
            signals.append(f"Price at upper BB ({price_position*100:.0f}% position)")
            confidence_factors.append(0.20)

        # RSI overbought
        if rsi >= self.rsi_overbought:
            score -= 1.5
            signals.append(f"RSI overbought ({rsi:.1f})")
            confidence_factors.append(0.15)
        elif rsi >= 60:
            score -= 0.5
            signals.append(f"RSI approaching overbought ({rsi:.1f})")
            confidence_factors.append(0.05)

        # === RANGE QUALITY BONUS ===

        # Low ADX = clean range
        if adx < 15:
            abs_score_boost = 0.5
            if score > 0:
                score += abs_score_boost
            elif score < 0:
                score -= abs_score_boost
            signals.append(f"Strong range environment (ADX {adx:.1f})")
            confidence_factors.append(0.10)

        # === DECISION ===

        # Normalize confidence
        confidence = min(sum(confidence_factors), 0.85)

        if score >= 2.5:
            vote = "BUY"
            confidence = max(confidence, 0.60)
        elif score <= -2.5:
            vote = "SELL"
            confidence = max(confidence, 0.60)
        else:
            vote = "NEUTRAL"
            confidence = 0.3

        reasoning = {
            "agent": self.name,
            "vote": vote,
            "score": round(score, 2),
            "adx": adx,
            "rsi": rsi,
            "bb_position": round(price_position * 100, 1),
            "signals": signals,
            "strategy": "mean_reversion"
        }

        return (vote, confidence, reasoning)
