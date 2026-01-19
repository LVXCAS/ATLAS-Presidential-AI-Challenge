"""
Support/Resistance Agent (Educational, Simulation-Only)

Support/resistance are areas where price repeatedly bounced or stalled in the past.
Being very close to those levels can increase uncertainty because price may either:
- bounce, or
- break through

This implementation is intentionally simple and uses only `price_history`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base_agent import AgentAssessment, BaseAgent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class SupportResistanceAgent(BaseAgent):
    def __init__(self, initial_weight: float = 0.9):
        super().__init__(name="SupportResistanceAgent", initial_weight=initial_weight)
        self.lookback = 60
        self.level_tolerance_pct = 0.15  # 0.15% ~ 15 pips on EUR/USD

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        history: List[float] = list(market_data.get("price_history", []) or [])
        price = float(market_data.get("price", history[-1] if history else 0.0) or 0.0)

        if price <= 0 or len(history) < 10:
            return AgentAssessment(
                score=0.50,
                explanation="Not enough price history to estimate support/resistance.",
                details={"history_len": len(history), "price": price, "data_sufficiency": "insufficient"},
            )

        window = history[-self.lookback :] if len(history) >= self.lookback else history
        support = min(window)
        resistance = max(window)
        range_pct = ((resistance - support) / price) if price else 0.0

        dist_support_pct = ((price - support) / price) * 100.0
        dist_resistance_pct = ((resistance - price) / price) * 100.0
        nearest_dist_pct = min(dist_support_pct, dist_resistance_pct)
        tolerance = self.level_tolerance_pct

        if range_pct < 0.001:  # <0.1% range
            score = 0.65
            explanation = "Price is in a very tight range; breakouts and fakeouts are more likely."
        elif nearest_dist_pct <= tolerance:
            score = 0.60
            explanation = "Price is very close to a recent support/resistance level; outcomes are less predictable."
        elif nearest_dist_pct <= tolerance * 2:
            score = 0.45
            explanation = "Price is near a recent key level; keep an eye on potential reversals or breakouts."
        else:
            score = 0.30
            explanation = "Price is not near a recent key level; support/resistance risk looks lower."

        # Slight bump when the overall range is wide (more room for swings).
        score = _clamp01(score + _clamp01((range_pct - 0.01) / 0.02) * 0.10)

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={
                "lookback": len(window),
                "support": support,
                "resistance": resistance,
                "range_pct": round(range_pct * 100, 2),
                "dist_support_pct": round(dist_support_pct, 2),
                "dist_resistance_pct": round(dist_resistance_pct, 2),
            },
        )
