"""Market Regime Agent - Detects bull/bear/range market regimes."""
from typing import Dict, Tuple
from .base_agent import BaseAgent

class MarketRegimeAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.2):
        super().__init__(name="MarketRegimeAgent", initial_weight=initial_weight)

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        indicators = market_data.get("indicators", {})
        adx = indicators.get("adx", 20)
        ema50 = indicators.get("ema50", 0)
        ema200 = indicators.get("ema200", 0)
        price = market_data.get("price", 0)
        
        # Detect regime
        if adx > 25 and ema50 > ema200 and price > ema200:
            regime = "bull_trend"
            vote = "BUY"
            confidence = min(adx / 50, 0.70)
        elif adx > 25 and ema50 < ema200 and price < ema200:
            regime = "bear_trend"
            vote = "SELL"
            confidence = min(adx / 50, 0.70)
        elif adx < 20:
            regime = "range"
            vote = "NEUTRAL"
            confidence = 0.3
        else:
            regime = "transition"
            vote = "NEUTRAL"
            confidence = 0.2

        return (vote, confidence, {"regime": regime, "adx": adx})
