"""
Divergence Agent - Detects RSI/MACD divergence from price for early reversal signals.

Divergence occurs when price makes new highs/lows but indicators don't confirm,
signaling momentum exhaustion and potential trend reversal.
"""
from typing import Dict, Tuple
from .base_agent import BaseAgent
import numpy as np


class DivergenceAgent(BaseAgent):
    """
    Catches trend exhaustion through indicator divergence.

    Types of Divergence:
    1. Bullish Regular: Price lower low, RSI higher low → trend reversal up
    2. Bearish Regular: Price higher high, RSI lower high → trend reversal down
    3. Bullish Hidden: Price higher low, RSI lower low → continuation up
    4. Bearish Hidden: Price lower high, RSI higher high → continuation down

    Focus on Regular divergence (reversals) as they're higher probability.
    """

    def __init__(self, initial_weight: float = 1.6):
        super().__init__(name="DivergenceAgent", initial_weight=initial_weight)

        # Lookback for finding swing points
        self.swing_window = 5

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Detect divergence between price and momentum indicators.

        Returns:
        - BUY on bullish divergence (oversold with momentum reversal)
        - SELL on bearish divergence (overbought with momentum exhaustion)
        - NEUTRAL if no divergence detected
        """
        candles = market_data.get("candles", [])
        indicators = market_data.get("indicators", {})

        if not candles or len(candles) < 30:
            return ("NEUTRAL", 0.3, {"error": "insufficient_data"})

        # Extract data
        closes = np.array([c['close'] for c in candles[-50:]])
        highs = np.array([c['high'] for c in candles[-50:]])
        lows = np.array([c['low'] for c in candles[-50:]])

        # Calculate RSI if not provided
        rsi = indicators.get("rsi")
        if rsi is None:
            rsi_values = self._calculate_rsi(closes, period=14)
        else:
            # Build RSI history (approximate from recent candles)
            rsi_values = self._build_rsi_history(candles[-50:], current_rsi=rsi)

        # Calculate MACD if not provided
        macd = indicators.get("macd", 0)
        macd_hist = indicators.get("macd_hist", 0)

        if len(rsi_values) < 30:
            return ("NEUTRAL", 0.3, {"error": "insufficient_rsi_data"})

        # Find swing highs and lows in price
        price_swing_highs = self._find_swing_highs(highs)
        price_swing_lows = self._find_swing_lows(lows)

        # Find swing highs and lows in RSI
        rsi_swing_highs = self._find_swing_highs(rsi_values)
        rsi_swing_lows = self._find_swing_lows(rsi_values)

        # Detect divergence
        bullish_div = self._detect_bullish_divergence(
            closes, rsi_values, price_swing_lows, rsi_swing_lows
        )

        bearish_div = self._detect_bearish_divergence(
            closes, rsi_values, price_swing_highs, rsi_swing_highs
        )

        # Current RSI for context
        current_rsi = rsi_values[-1] if len(rsi_values) > 0 else 50

        reasoning = {
            "current_rsi": round(current_rsi, 1),
            "macd": round(macd, 6),
            "macd_hist": round(macd_hist, 6),
        }

        # BULLISH DIVERGENCE - Price falling, RSI rising (reversal up)
        if bullish_div:
            # Stronger signal if RSI is oversold
            oversold_boost = 0.15 if current_rsi < 35 else 0

            vote = "BUY"
            confidence = min(0.80, 0.60 + oversold_boost + bullish_div["strength"] * 0.10)

            return (vote, confidence, {
                **reasoning,
                "signal": "bullish_divergence",
                "divergence_type": bullish_div["type"],
                "divergence_bars": bullish_div["bars_apart"],
                "message": f"Bullish divergence: price declining but RSI rising ({bullish_div['type']})"
            })

        # BEARISH DIVERGENCE - Price rising, RSI falling (reversal down)
        if bearish_div:
            # Stronger signal if RSI is overbought
            overbought_boost = 0.15 if current_rsi > 65 else 0

            vote = "SELL"
            confidence = min(0.80, 0.60 + overbought_boost + bearish_div["strength"] * 0.10)

            return (vote, confidence, {
                **reasoning,
                "signal": "bearish_divergence",
                "divergence_type": bearish_div["type"],
                "divergence_bars": bearish_div["bars_apart"],
                "message": f"Bearish divergence: price rising but RSI falling ({bearish_div['type']})"
            })

        # MACD DIVERGENCE CHECK (secondary confirmation)
        if macd_hist < 0 and current_rsi < 40:
            # MACD negative with RSI oversold = potential reversal
            return ("NEUTRAL", 0.6, {
                **reasoning,
                "signal": "macd_oversold",
                "message": "MACD bearish but RSI oversold (watch for bullish div)"
            })

        if macd_hist > 0 and current_rsi > 60:
            # MACD positive with RSI overbought = potential exhaustion
            return ("NEUTRAL", 0.6, {
                **reasoning,
                "signal": "macd_overbought",
                "message": "MACD bullish but RSI overbought (watch for bearish div)"
            })

        # No divergence detected
        return ("NEUTRAL", 0.5, {
            **reasoning,
            "signal": "no_divergence"
        })

    def _find_swing_highs(self, data: np.ndarray) -> list:
        """Find swing highs (local peaks) in data."""
        swings = []

        for i in range(self.swing_window, len(data) - self.swing_window):
            is_swing = True

            for j in range(i - self.swing_window, i + self.swing_window + 1):
                if j != i and data[j] >= data[i]:
                    is_swing = False
                    break

            if is_swing:
                swings.append((i, data[i]))

        return swings

    def _find_swing_lows(self, data: np.ndarray) -> list:
        """Find swing lows (local troughs) in data."""
        swings = []

        for i in range(self.swing_window, len(data) - self.swing_window):
            is_swing = True

            for j in range(i - self.swing_window, i + self.swing_window + 1):
                if j != i and data[j] <= data[i]:
                    is_swing = False
                    break

            if is_swing:
                swings.append((i, data[i]))

        return swings

    def _detect_bullish_divergence(self, closes: np.ndarray, rsi: np.ndarray,
                                   price_lows: list, rsi_lows: list) -> dict:
        """
        Detect bullish divergence: price lower low, RSI higher low.

        Returns dict with divergence info or None if no divergence.
        """
        if len(price_lows) < 2 or len(rsi_lows) < 2:
            return None

        # Get last two price lows
        price_low1 = price_lows[-2]  # (index, value)
        price_low2 = price_lows[-1]

        # Find corresponding RSI lows (within ±3 bars)
        rsi_low1 = self._find_nearest_swing(rsi_lows, price_low1[0], tolerance=3)
        rsi_low2 = self._find_nearest_swing(rsi_lows, price_low2[0], tolerance=3)

        if not rsi_low1 or not rsi_low2:
            return None

        # Check for divergence: price lower low, RSI higher low
        if price_low2[1] < price_low1[1] and rsi_low2[1] > rsi_low1[1]:
            bars_apart = price_low2[0] - price_low1[0]
            strength = (rsi_low2[1] - rsi_low1[1]) / rsi_low1[1]  # % increase in RSI

            return {
                "type": "regular_bullish",
                "bars_apart": bars_apart,
                "strength": strength,
                "price_change_%": ((price_low2[1] - price_low1[1]) / price_low1[1] * 100),
                "rsi_change": (rsi_low2[1] - rsi_low1[1])
            }

        return None

    def _detect_bearish_divergence(self, closes: np.ndarray, rsi: np.ndarray,
                                   price_highs: list, rsi_highs: list) -> dict:
        """
        Detect bearish divergence: price higher high, RSI lower high.
        """
        if len(price_highs) < 2 or len(rsi_highs) < 2:
            return None

        # Get last two price highs
        price_high1 = price_highs[-2]
        price_high2 = price_highs[-1]

        # Find corresponding RSI highs
        rsi_high1 = self._find_nearest_swing(rsi_highs, price_high1[0], tolerance=3)
        rsi_high2 = self._find_nearest_swing(rsi_highs, price_high2[0], tolerance=3)

        if not rsi_high1 or not rsi_high2:
            return None

        # Check for divergence: price higher high, RSI lower high
        if price_high2[1] > price_high1[1] and rsi_high2[1] < rsi_high1[1]:
            bars_apart = price_high2[0] - price_high1[0]
            strength = (rsi_high1[1] - rsi_high2[1]) / rsi_high1[1]  # % decrease in RSI

            return {
                "type": "regular_bearish",
                "bars_apart": bars_apart,
                "strength": strength,
                "price_change_%": ((price_high2[1] - price_high1[1]) / price_high1[1] * 100),
                "rsi_change": (rsi_high2[1] - rsi_high1[1])
            }

        return None

    def _find_nearest_swing(self, swings: list, target_index: int, tolerance: int = 3):
        """Find swing point nearest to target index within tolerance."""
        for swing in swings:
            if abs(swing[0] - target_index) <= tolerance:
                return swing
        return None

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        if len(closes) < period + 1:
            return np.array([])

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros(len(closes))
        avg_loss = np.zeros(len(closes))

        # Initial averages
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        # Smoothed averages
        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

        rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))

        return rsi[period:]

    def _build_rsi_history(self, candles: list, current_rsi: float) -> np.ndarray:
        """Build approximate RSI history from candles."""
        closes = np.array([c['close'] for c in candles])

        if len(closes) >= 20:
            return self._calculate_rsi(closes, period=14)
        else:
            # Fallback: return current RSI repeated
            return np.array([current_rsi] * len(closes))
