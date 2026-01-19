"""
Multi-Timeframe Agent (Educational, Simulation-Only)

This agent checks whether short-, medium-, and long-term trends agree.
When timeframes disagree, the situation is usually more uncertain.

It uses only `market_data["price_history"]` (historical/delayed) and does not
request real-time data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_agent import AgentAssessment, BaseAgent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _ema(values: List[float], period: int) -> Optional[float]:
    if period <= 1 or len(values) < 2:
        return None
    alpha = 2.0 / (period + 1.0)
    current = float(values[0])
    for v in values[1:]:
        current = (alpha * float(v)) + ((1.0 - alpha) * current)
    return current


def _sign(x: float, eps: float = 1e-9) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


class MultiTimeframeAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.0):
        super().__init__(name="MultiTimeframeAgent", initial_weight=initial_weight)

        # "Timeframes" represented as different EMA lookbacks.
        self.short_period = 20
        self.mid_period = 50
        self.long_period = 100

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        history = list(market_data.get("price_history", []) or [])
        price = float(market_data.get("price", history[-1] if history else 0.0) or 0.0)

        if len(history) < self.long_period:
            return AgentAssessment(
                score=0.50,
                explanation="Not enough price history for multi-timeframe alignment.",
                details={"history_len": len(history), "needed": self.long_period, "data_sufficiency": "insufficient"},
            )

        ema_short = _ema(history[-(self.long_period + 1) :], self.short_period)
        ema_mid = _ema(history[-(self.long_period + 1) :], self.mid_period)
        ema_long = _ema(history[-(self.long_period + 1) :], self.long_period)

        if ema_short is None or ema_mid is None or ema_long is None or price <= 0:
            return AgentAssessment(
                score=0.50,
                explanation="Could not compute EMA alignment; treating as uncertain.",
                details={
                    "ema_short": ema_short,
                    "ema_mid": ema_mid,
                    "ema_long": ema_long,
                    "price": price,
                    "data_sufficiency": "insufficient",
                },
            )

        s_short = _sign(price - ema_short)
        s_mid = _sign(price - ema_mid)
        s_long = _sign(price - ema_long)
        signs = [s_short, s_mid, s_long]

        # Alignment strength: how many timeframes agree on direction.
        agree_up = sum(1 for s in signs if s > 0)
        agree_down = sum(1 for s in signs if s < 0)
        agree = max(agree_up, agree_down)

        if agree == 3 and 0 not in signs:
            score = 0.25
            explanation = "Short-, mid-, and long-term trends agree — lower uncertainty."
            alignment = "aligned"
        elif agree == 2:
            score = 0.45
            explanation = "Most timeframes agree, but one disagrees — moderate uncertainty."
            alignment = "mostly_aligned"
        else:
            score = 0.70
            explanation = "Timeframes disagree — mixed trends increase uncertainty."
            alignment = "conflicted"

        # Small extra bump if EMAs are tightly packed (often a transition zone).
        spread = (max(ema_short, ema_mid, ema_long) - min(ema_short, ema_mid, ema_long)) / price
        score = _clamp01(score + _clamp01((0.0007 - spread) / 0.0007) * 0.10)

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={
                "price": price,
                "ema_short": ema_short,
                "ema_mid": ema_mid,
                "ema_long": ema_long,
                "signs": {"short": s_short, "mid": s_mid, "long": s_long},
                "alignment": alignment,
                "ema_spread_pct": round(spread * 100, 3),
            },
        )
