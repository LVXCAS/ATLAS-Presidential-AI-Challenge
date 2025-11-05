"""
SHARED MULTI-TIMEFRAME ANALYSIS
Used by FOREX, FUTURES, and CRYPTO bots
Provides higher timeframe trend confirmation
"""
from SHARED.technical_analysis import ta


class MultiTimeframeAnalysis:
    """
    Multi-timeframe trend analysis
    Confirms 1H entry signals with 4H trend direction
    """

    def __init__(self):
        self.ta = ta

    def get_higher_timeframe_trend(self, closes_4h, current_price):
        """
        Determine 4-hour timeframe trend direction

        Args:
            closes_4h: Array of 4H closing prices (100+ candles)
            current_price: Current market price

        Returns:
            str: 'bullish', 'bearish', or 'neutral'
        """
        if len(closes_4h) < 50:
            return 'neutral'

        # Calculate 4H EMAs
        ema_fast = self.ta.calculate_ema(closes_4h, period=10)
        ema_slow = self.ta.calculate_ema(closes_4h, period=21)
        ema_trend = self.ta.calculate_ema(closes_4h, period=50)

        # Bullish: Fast > Slow and Price > Trend EMA
        if ema_fast > ema_slow and current_price > ema_trend:
            return 'bullish'

        # Bearish: Fast < Slow and Price < Trend EMA
        elif ema_fast < ema_slow and current_price < ema_trend:
            return 'bearish'

        else:
            return 'neutral'

    def confirm_entry(self, entry_direction, trend_4h):
        """
        Confirm if 1H entry signal aligns with 4H trend

        Args:
            entry_direction: 'long' or 'short' from 1H analysis
            trend_4h: 'bullish', 'bearish', or 'neutral' from 4H analysis

        Returns:
            bool: True if aligned or neutral, False if counter-trend
        """
        if trend_4h == 'neutral':
            return True  # Neutral 4H = allow trade

        if entry_direction == 'long' and trend_4h == 'bullish':
            return True  # Bullish 4H confirms long entry

        if entry_direction == 'short' and trend_4h == 'bearish':
            return True  # Bearish 4H confirms short entry

        return False  # Counter-trend = reject

    def get_trend_strength(self, closes_4h):
        """
        Measure 4H trend strength using ADX

        Args:
            closes_4h: Array of 4H closing prices

        Returns:
            str: 'strong', 'moderate', 'weak'
        """
        if len(closes_4h) < 50:
            return 'weak'

        # Calculate ADX (requires highs/lows but we approximate with closes)
        highs = [c * 1.001 for c in closes_4h]  # Approximate
        lows = [c * 0.999 for c in closes_4h]

        adx = self.ta.calculate_adx(highs, lows, closes_4h, period=14)

        if adx > 40:
            return 'strong'
        elif adx > 25:
            return 'moderate'
        else:
            return 'weak'


# Singleton instance
mtf = MultiTimeframeAnalysis()
