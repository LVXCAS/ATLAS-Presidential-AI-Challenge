"""
Technical Agent

Analyzes technical indicators (RSI, MACD, EMAs, Bollinger Bands, ADX, ATR)
to vote on trade direction.

Specialization: Classic technical analysis with proven indicators.
"""

from typing import Dict, Tuple
from .base_agent import BaseAgent


class TechnicalAgent(BaseAgent):
    """
    Technical analysis agent using multiple indicators.

    Indicators used:
    - RSI (14): Momentum oscillator
    - MACD (12, 26, 9): Trend following
    - EMA50 / EMA200: Trend direction
    - Bollinger Bands: Volatility and mean reversion
    - ADX (14): Trend strength
    - ATR (14): Volatility measurement
    """

    def __init__(self, initial_weight: float = 1.5):
        super().__init__(name="TechnicalAgent", initial_weight=initial_weight)

        # Thresholds (can be tuned over time)
        self.rsi_oversold = 40
        self.rsi_overbought = 60
        self.adx_trending = 28
        self.macd_threshold = 0.0001

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Analyze technical indicators and vote.

        Returns:
            (vote, confidence, reasoning)
        """
        indicators = market_data.get("indicators", {})
        price = market_data.get("price")
        direction = market_data.get("direction", "long").lower()  # Default to "long" since ATLAS only trades BUY

        # Extract indicators
        rsi = indicators.get("rsi", 50)
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        macd_hist = indicators.get("macd_hist", 0)
        ema50 = indicators.get("ema50", price)
        ema200 = indicators.get("ema200", price)
        bb_upper = indicators.get("bb_upper", price * 1.01)
        bb_lower = indicators.get("bb_lower", price * 0.99)
        bb_middle = indicators.get("bb_middle", price)
        adx = indicators.get("adx", 20)
        atr = indicators.get("atr", 0.001)

        # === RSI EXHAUSTION VETO FILTER (Prevents EUR/USD-style failures) ===
        # Block trades at momentum exhaustion points to prevent entries at extremes
        if direction == "long" and rsi > 70:
            return ("BLOCK", 0.95, {
                "reason": f"RSI_EXHAUSTION_LONG: RSI {rsi:.1f} indicates overbought exhaustion",
                "rsi": round(rsi, 1),
                "message": f"BLOCKED LONG entry - RSI {rsi:.1f} overbought (>70)",
                "recommendation": "Wait for RSI < 65 before entering LONG positions"
            })
        elif direction == "short" and rsi < 30:
            return ("BLOCK", 0.95, {
                "reason": f"RSI_EXHAUSTION_SHORT: RSI {rsi:.1f} indicates oversold exhaustion",
                "rsi": round(rsi, 1),
                "message": f"BLOCKED SHORT entry - RSI {rsi:.1f} oversold (<30)",
                "recommendation": "Wait for RSI > 35 before entering SHORT positions"
            })

        # Score different technical signals
        score = 0
        signals = []
        confidence_factors = []

        # === TREND ANALYSIS (Strong weight) ===

        # 1. Price vs EMAs (trend direction)
        is_bullish_trend = price > ema200
        is_bearish_trend = price < ema200
        
        if is_bullish_trend:
            score += 2.0
            signals.append("Price above EMA200 (bullish trend)")
            confidence_factors.append(0.20)
        elif is_bearish_trend:
            score -= 2.0
            signals.append("Price below EMA200 (bearish trend)")

        # 2. EMA alignment
        ema_bullish = ema50 > ema200
        ema_bearish = ema50 < ema200
        
        if ema_bullish:
            score += 1.0
            signals.append("EMA50 > EMA200 (golden cross territory)")
            confidence_factors.append(0.10)
        elif ema_bearish:
            score -= 1.0
            signals.append("EMA50 < EMA200 (death cross territory)")

        # === TREND ALIGNMENT BLOCK (Prevents counter-trend trades) ===
        # Block LONG entries in strong downtrends and SHORT entries in strong uptrends
        strong_downtrend = is_bearish_trend and ema_bearish and adx > self.adx_trending
        strong_uptrend = is_bullish_trend and ema_bullish and adx > self.adx_trending
        
        if direction == "long" and strong_downtrend:
            return ("BLOCK", 0.90, {
                "reason": f"COUNTER_TREND_LONG: Price below EMA200 ({price:.5f} < {ema200:.5f}), EMA50 < EMA200, ADX {adx:.1f} (strong downtrend)",
                "trend": "bearish",
                "adx": round(adx, 1),
                "message": f"BLOCKED LONG entry - Trading against strong downtrend (ADX {adx:.1f})",
                "recommendation": "Only trade LONG in uptrends or wait for trend reversal confirmation"
            })
        elif direction == "short" and strong_uptrend:
            return ("BLOCK", 0.90, {
                "reason": f"COUNTER_TREND_SHORT: Price above EMA200 ({price:.5f} > {ema200:.5f}), EMA50 > EMA200, ADX {adx:.1f} (strong uptrend)",
                "trend": "bullish",
                "adx": round(adx, 1),
                "message": f"BLOCKED SHORT entry - Trading against strong uptrend (ADX {adx:.1f})",
                "recommendation": "Only trade SHORT in downtrends or wait for trend reversal confirmation"
            })

        # === MOMENTUM ANALYSIS ===

        # 3. RSI (pullback in trend)
        if price > ema200 and self.rsi_oversold <= rsi <= self.rsi_overbought:
            # Bullish trend + RSI pullback = BUY setup
            score += 1.5
            signals.append(f"RSI {rsi:.1f} pullback in uptrend")
            confidence_factors.append(0.15)
        elif price < ema200 and rsi >= (100 - self.rsi_overbought):
            # Bearish trend + RSI rally = SELL setup
            score -= 1.5
            signals.append(f"RSI {rsi:.1f} rally in downtrend")

        # 4. RSI extreme (caution)
        if rsi > 70:
            score -= 0.5
            signals.append(f"RSI {rsi:.1f} overbought (caution)")
        elif rsi < 30:
            score += 0.5  # Can be bullish in strong trends
            signals.append(f"RSI {rsi:.1f} oversold")

        # === MACD ANALYSIS ===

        # 5. MACD crossover
        if macd > macd_signal and macd_hist > 0:
            score += 1.5
            signals.append("MACD bullish crossover")
            confidence_factors.append(0.15)
        elif macd < macd_signal and macd_hist < 0:
            score -= 1.5
            signals.append("MACD bearish crossover")

        # 6. MACD alignment with trend
        if price > ema200 and macd > 0:
            score += 1.0
            signals.append("MACD confirms bullish trend")
            confidence_factors.append(0.10)
        elif price < ema200 and macd < 0:
            score -= 1.0
            signals.append("MACD confirms bearish trend")

        # === BOLLINGER BANDS (Mean Reversion + Volatility) ===

        # 7. Price position in BB
        if price < bb_lower:
            score += 1.0
            signals.append("Price below lower BB (potential bounce)")
            confidence_factors.append(0.10)
        elif price > bb_upper:
            score -= 1.0
            signals.append("Price above upper BB (potential reversal)")

        # 8. BB squeeze (low volatility before breakout)
        bb_width = (bb_upper - bb_lower) / bb_middle
        if bb_width < 0.02:  # Tight bands
            # Don't trade during squeezes (wait for breakout)
            score *= 0.5
            signals.append("BB squeeze detected (low conviction)")

        # === TREND STRENGTH ===

        # 9. ADX (is there a tradeable trend?)
        if adx > self.adx_trending:
            # Strong trend - technical signals more reliable
            score *= 1.2
            signals.append(f"ADX {adx:.1f} (strong trend)")
            confidence_factors.append(0.15)
        else:
            # Weak trend - reduce confidence
            score *= 0.7
            signals.append(f"ADX {adx:.1f} (weak trend, lower conviction)")

        # === FINAL DECISION ===

        # Determine vote
        if score >= 5.5:
            vote = "BUY"
        elif score <= -5.5:
            vote = "SELL"
        else:
            vote = "NEUTRAL"

        # Calculate confidence (0.0 - 1.0)
        # Based on:
        # - Number of confirming signals
        # - ADX (trend strength)
        # - Score magnitude

        base_confidence = min(abs(score) / 10.0, 1.0)  # Normalize score to 0-1
        signal_bonus = min(sum(confidence_factors), 0.30)  # Max +0.30 from signals
        confidence = min(base_confidence + signal_bonus, 1.0)

        # Reasoning for transparency
        reasoning = {
            "score": round(score, 2),
            "signals": signals,
            "indicators": {
                "rsi": round(rsi, 1),
                "macd": round(macd, 5),
                "adx": round(adx, 1),
                "price_vs_ema200": "above" if price > ema200 else "below",
            }
        }

        return (vote, confidence, reasoning)

    def learn_from_pattern(self, pattern: Dict):
        """
        Adjust technical thresholds based on discovered patterns.

        Example pattern:
        {
            "condition": "RSI between 38-42 during London open",
            "win_rate": 0.78,
            "sample_size": 42
        }
        """
        super().learn_from_pattern(pattern)

        # If pattern involves RSI, adjust thresholds
        if "RSI" in pattern.get("condition", ""):
            if pattern.get("win_rate", 0) > 0.70:
                # High-probability RSI pattern discovered
                # Could fine-tune RSI thresholds here
                pass
