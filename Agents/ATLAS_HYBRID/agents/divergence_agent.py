"""
Divergence Agent (Educational, Simulation-Only)

"Divergence" is when price moves one way but a momentum indicator (like RSI)
moves the other way. That can signal weakening momentum and higher uncertainty.

This agent uses only `price_history` (historical/delayed) and computes RSI locally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_agent import AgentAssessment, BaseAgent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _rsi_series(prices: List[float], period: int = 14) -> List[float]:
    if len(prices) < period + 1:
        return []
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, period + 1):
        delta = prices[i] - prices[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    out: List[float] = []
    for i in range(period + 1, len(prices) + 1):
        if avg_loss == 0:
            out.append(100.0)
        else:
            rs = avg_gain / avg_loss
            out.append(100.0 - (100.0 / (1.0 + rs)))

        if i == len(prices):
            break

        delta = prices[i] - prices[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    return out


class DivergenceAgent(BaseAgent):
    def __init__(self, initial_weight: float = 0.9):
        super().__init__(name="DivergenceAgent", initial_weight=initial_weight)
        self.lookback = 20
        self.rsi_period = 14

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        history: List[float] = list(market_data.get("price_history", []) or [])
        if len(history) < self.rsi_period + self.lookback + 1:
            return AgentAssessment(
                score=0.45,
                explanation="Not enough history to check RSI divergence.",
                details={
                    "history_len": len(history),
                    "needed": self.rsi_period + self.lookback + 1,
                    "data_sufficiency": "insufficient",
                },
            )

        rsi = _rsi_series(history, period=self.rsi_period)
        if len(rsi) < self.lookback + 1:
            return AgentAssessment(
                score=0.45,
                explanation="Could not compute RSI history; divergence check skipped.",
                details={"rsi_len": len(rsi), "data_sufficiency": "insufficient"},
            )

        price_now = float(history[-1])
        price_then = float(history[-(self.lookback + 1)])
        rsi_now = float(rsi[-1])
        rsi_then = float(rsi[-(self.lookback + 1)])

        price_change_pct = ((price_now - price_then) / price_then) * 100.0 if price_then else 0.0
        rsi_change = rsi_now - rsi_then

        divergence: Optional[str] = None
        if price_change_pct > 0.05 and rsi_change < -2.0:
            divergence = "bearish"
        elif price_change_pct < -0.05 and rsi_change > 2.0:
            divergence = "bullish"

        score = 0.35
        if divergence == "bearish":
            score = 0.65
            explanation = "Possible bearish divergence: price rose, but RSI weakened — uncertainty increases."
        elif divergence == "bullish":
            score = 0.65
            explanation = "Possible bullish divergence: price fell, but RSI improved — uncertainty increases."
        else:
            explanation = "No clear RSI divergence detected in the recent window."

        # Add a small bump if RSI is very extreme (often unstable).
        extreme_bump = 0.0
        if rsi_now >= 70:
            extreme_bump = 0.10
        elif rsi_now <= 30:
            extreme_bump = 0.10
        score = _clamp01(score + extreme_bump)

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={
                "lookback": self.lookback,
                "price_change_pct": round(price_change_pct, 2),
                "rsi_now": round(rsi_now, 1),
                "rsi_change": round(rsi_change, 1),
                "divergence": divergence or "none",
            },
        )
