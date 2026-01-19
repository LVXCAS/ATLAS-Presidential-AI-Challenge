"""
Market Regime Agent (Educational)

Goal: estimate how "predictable" the current market regime is using *only*
historical/delayed inputs (no live feeds).

Output:
- `score` in 0..1 where higher = more risk/uncertainty
- short explanation string for beginners
"""

from typing import Dict

from .base_agent import AgentAssessment, BaseAgent


class MarketRegimeAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.2):
        super().__init__(name="MarketRegimeAgent", initial_weight=initial_weight)

    def analyze(self, market_data: Dict) -> AgentAssessment:
        indicators = market_data.get("indicators", {}) or {}
        price = float(market_data.get("price", 0.0) or 0.0)

        adx = float(indicators.get("adx", 20.0) or 20.0)  # proxy 10..60 in this project
        ema50 = float(indicators.get("ema50", price) or price)
        ema200 = float(indicators.get("ema200", price) or price)

        # Interpret ADX proxy as "trend clarity".
        if adx < 18:
            regime = "choppy"
            score = 0.70
            explanation = f"Choppy regime (ADX {adx:.0f}) — trends are less reliable."
        elif adx < 25:
            regime = "transition"
            score = 0.50
            explanation = f"Transition regime (ADX {adx:.0f}) — uncertainty is elevated."
        else:
            # Trending regime; still track whether price agrees with long-term EMA.
            aligned = (price >= ema200 and ema50 >= ema200) or (price <= ema200 and ema50 <= ema200)
            regime = "trend_aligned" if aligned else "trend_mixed"
            score = 0.30 if aligned else 0.45
            explanation = (
                f"Trend regime (ADX {adx:.0f}) with EMA alignment."
                if aligned
                else f"Trend regime (ADX {adx:.0f}) but EMAs disagree — mixed signals."
            )

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={"adx": round(adx, 2), "ema50": ema50, "ema200": ema200, "regime": regime},
        )
