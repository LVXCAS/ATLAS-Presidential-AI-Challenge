"""
SHARED TECHNICAL ANALYSIS LIBRARY
Used by FOREX, FUTURES, and CRYPTO bots
Provides consistent TA-Lib indicator calculations across all markets
"""
import numpy as np

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARN] TA-Lib not available - using simplified indicators")


class TechnicalAnalysis:
    """
    Unified technical analysis engine for all asset classes
    Provides RSI, MACD, EMA, ADX, ATR, Bollinger Bands, etc.
    """

    def __init__(self):
        self.talib_available = TALIB_AVAILABLE

    # ============================================================================
    # MOMENTUM INDICATORS
    # ============================================================================

    def calculate_rsi(self, closes, period=14):
        """
        Relative Strength Index (RSI)
        Returns: float (0-100)
        Oversold: < 30, Overbought: > 70
        """
        if not self.talib_available:
            return self._simple_rsi(closes, period)

        closes_array = np.array(closes, dtype=float)
        rsi = talib.RSI(closes_array, timeperiod=period)
        return rsi[-1] if len(rsi) > 0 else 50.0

    def calculate_macd(self, closes, fast=12, slow=26, signal=9):
        """
        Moving Average Convergence Divergence (MACD)
        Returns: dict {'macd': float, 'signal': float, 'histogram': float}
        """
        if not self.talib_available:
            return self._simple_macd(closes, fast, slow, signal)

        closes_array = np.array(closes, dtype=float)
        macd, macd_signal, macd_hist = talib.MACD(closes_array,
                                                    fastperiod=fast,
                                                    slowperiod=slow,
                                                    signalperiod=signal)

        return {
            'macd': macd[-1] if len(macd) > 0 else 0.0,
            'signal': macd_signal[-1] if len(macd_signal) > 0 else 0.0,
            'histogram': macd_hist[-1] if len(macd_hist) > 0 else 0.0
        }

    # ============================================================================
    # TREND INDICATORS
    # ============================================================================

    def calculate_ema(self, closes, period=20):
        """
        Exponential Moving Average (EMA)
        Returns: float
        """
        if not self.talib_available:
            return self._simple_ema(closes, period)

        closes_array = np.array(closes, dtype=float)
        ema = talib.EMA(closes_array, timeperiod=period)
        return ema[-1] if len(ema) > 0 else closes[-1]

    def calculate_adx(self, highs, lows, closes, period=14):
        """
        Average Directional Index (ADX)
        Returns: float (0-100)
        Strong trend: > 25, Very strong: > 50
        """
        if not self.talib_available:
            return 25.0  # Default assumption

        highs_array = np.array(highs, dtype=float)
        lows_array = np.array(lows, dtype=float)
        closes_array = np.array(closes, dtype=float)

        adx = talib.ADX(highs_array, lows_array, closes_array, timeperiod=period)
        return adx[-1] if len(adx) > 0 else 25.0

    # ============================================================================
    # VOLATILITY INDICATORS
    # ============================================================================

    def calculate_atr(self, highs, lows, closes, period=14):
        """
        Average True Range (ATR)
        Returns: float (measures volatility)
        """
        if not self.talib_available:
            return self._simple_atr(highs, lows, closes, period)

        highs_array = np.array(highs, dtype=float)
        lows_array = np.array(lows, dtype=float)
        closes_array = np.array(closes, dtype=float)

        atr = talib.ATR(highs_array, lows_array, closes_array, timeperiod=period)
        return atr[-1] if len(atr) > 0 else 0.001

    def calculate_bollinger_bands(self, closes, period=20, std_dev=2):
        """
        Bollinger Bands
        Returns: dict {'upper': float, 'middle': float, 'lower': float}
        """
        if not self.talib_available:
            return self._simple_bollinger(closes, period, std_dev)

        closes_array = np.array(closes, dtype=float)
        upper, middle, lower = talib.BBANDS(closes_array,
                                              timeperiod=period,
                                              nbdevup=std_dev,
                                              nbdevdn=std_dev)

        return {
            'upper': upper[-1] if len(upper) > 0 else closes[-1],
            'middle': middle[-1] if len(middle) > 0 else closes[-1],
            'lower': lower[-1] if len(lower) > 0 else closes[-1]
        }

    # ============================================================================
    # SIMPLIFIED FALLBACK CALCULATIONS (if TA-Lib not available)
    # ============================================================================

    def _simple_rsi(self, closes, period=14):
        """Simple RSI calculation without TA-Lib"""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _simple_ema(self, closes, period=20):
        """Simple EMA calculation without TA-Lib"""
        if len(closes) < period:
            return closes[-1]

        multiplier = 2 / (period + 1)
        ema = closes[0]

        for price in closes[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _simple_atr(self, highs, lows, closes, period=14):
        """Simple ATR calculation without TA-Lib"""
        if len(closes) < 2:
            return 0.001

        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)

        return np.mean(true_ranges[-period:]) if true_ranges else 0.001

    def _simple_macd(self, closes, fast=12, slow=26, signal=9):
        """Simple MACD calculation without TA-Lib"""
        ema_fast = self._simple_ema(closes, fast)
        ema_slow = self._simple_ema(closes, slow)
        macd_line = ema_fast - ema_slow

        # Simplified signal line
        signal_line = macd_line * 0.9
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def _simple_bollinger(self, closes, period=20, std_dev=2):
        """Simple Bollinger Bands calculation without TA-Lib"""
        if len(closes) < period:
            return {'upper': closes[-1], 'middle': closes[-1], 'lower': closes[-1]}

        recent = closes[-period:]
        middle = np.mean(recent)
        std = np.std(recent)

        return {
            'upper': middle + (std * std_dev),
            'middle': middle,
            'lower': middle - (std * std_dev)
        }


# Singleton instance
ta = TechnicalAnalysis()
