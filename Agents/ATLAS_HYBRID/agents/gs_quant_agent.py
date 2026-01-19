"""
GSQuant Agent (Educational, Simulation-Only)

The original ATLAS project referenced institutional tooling (e.g., GS Quant).
For this K–12 educational repository we keep the *idea* (institutional-style
risk metrics) but implement it locally with simple math and no external APIs.

This agent estimates a basic "Value at Risk" (VaR-like) number using historical
returns / ATR and converts it into a 0..1 uncertainty score.
"""

from __future__ import annotations

from statistics import pstdev
from typing import Any, Dict, List

from .base_agent import AgentAssessment, BaseAgent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _returns(prices: List[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        cur = prices[i]
        if prev:
            out.append((cur - prev) / prev)
    return out


class GSQuantAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.6):
        super().__init__(name="GSQuantAgent", initial_weight=initial_weight)

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        history: List[float] = list(market_data.get("price_history", []) or [])
        price = float(market_data.get("price", history[-1] if history else 0.0) or 0.0)
        if price <= 0 or len(history) < 20:
            return AgentAssessment(
                score=0.50,
                explanation="Not enough data to estimate VaR-style risk.",
                details={"history_len": len(history), "price": price, "data_sufficiency": "insufficient"},
            )

        indicators = market_data.get("indicators", {}) or {}
        atr = float(indicators.get("atr", 0.0) or 0.0)
        atr_pct = (atr / price) if price else 0.0

        rets = _returns(history[-200:])
        vol = pstdev(rets) if len(rets) >= 2 else 0.0

        # Very rough VaR proxy:
        # - combine return volatility and ATR-based volatility
        vol_proxy = max(vol, atr_pct)
        var95_pct = 1.65 * vol_proxy  # 95% ~ 1.65 sigma

        # Map VaR into a normalized risk score.
        # 0.2% => low, 1.0% => high (for this educational FX-ish simulator).
        score = _clamp01((var95_pct - 0.002) / (0.010 - 0.002))

        if var95_pct >= 0.010:
            explanation = "VaR-style risk is high: large moves are plausible given recent volatility."
        elif var95_pct >= 0.006:
            explanation = "VaR-style risk is elevated: volatility could cause meaningful swings."
        else:
            explanation = "VaR-style risk looks moderate/low based on recent volatility."

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={
                "atr_pct": round(atr_pct * 100, 3),
                "return_vol_pct": round(vol * 100, 3),
                "var95_pct": round(var95_pct * 100, 3),
            },
        )
