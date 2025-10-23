#!/usr/bin/env python3
"""
Price Action Pattern Detection
Detects candlestick patterns for trade confirmation
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PricePatternDetector:
    """Detect candlestick patterns"""

    def __init__(self):
        self.patterns = {}

    def detect_hammer(self, data: pd.DataFrame) -> bool:
        """
        Hammer: Bullish reversal
        - Small body at top
        - Long lower shadow (2x body)
        - Little/no upper shadow
        """
        if len(data) < 1:
            return False

        candle = data.iloc[-1]
        o, h, l, c = candle['Open'], candle['High'], candle['Low'], candle['Close']

        body = abs(c - o)
        total_range = h - l

        if total_range == 0:
            return False

        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)

        # Hammer criteria
        is_hammer = (
            lower_shadow > body * 2 and  # Long lower shadow
            upper_shadow < body * 0.3 and  # Small upper shadow
            body / total_range < 0.3  # Small body
        )

        return is_hammer

    def detect_engulfing_bullish(self, data: pd.DataFrame) -> bool:
        """
        Bullish Engulfing: Strong reversal
        - Previous candle bearish (red)
        - Current candle bullish (green) and completely engulfs previous
        """
        if len(data) < 2:
            return False

        prev = data.iloc[-2]
        curr = data.iloc[-1]

        prev_bearish = prev['Close'] < prev['Open']
        curr_bullish = curr['Close'] > curr['Open']

        # Current engulfs previous
        engulfs = (
            curr['Open'] < prev['Close'] and
            curr['Close'] > prev['Open']
        )

        return prev_bearish and curr_bullish and engulfs

    def detect_engulfing_bearish(self, data: pd.DataFrame) -> bool:
        """Bearish Engulfing: Strong reversal down"""
        if len(data) < 2:
            return False

        prev = data.iloc[-2]
        curr = data.iloc[-1]

        prev_bullish = prev['Close'] > prev['Open']
        curr_bearish = curr['Close'] < curr['Open']

        engulfs = (
            curr['Open'] > prev['Close'] and
            curr['Close'] < prev['Open']
        )

        return prev_bullish and curr_bearish and engulfs

    def detect_doji(self, data: pd.DataFrame) -> bool:
        """
        Doji: Indecision
        - Open and close nearly equal
        - Can signal reversal
        """
        if len(data) < 1:
            return False

        candle = data.iloc[-1]
        o, c = candle['Open'], candle['Close']
        h, l = candle['High'], candle['Low']

        body = abs(c - o)
        total_range = h - l

        if total_range == 0:
            return False

        # Body is < 10% of total range
        return body / total_range < 0.1

    def detect_morning_star(self, data: pd.DataFrame) -> bool:
        """
        Morning Star: Bullish reversal (3-candle pattern)
        - Candle 1: Large bearish
        - Candle 2: Small body (star)
        - Candle 3: Large bullish
        """
        if len(data) < 3:
            return False

        c1, c2, c3 = data.iloc[-3], data.iloc[-2], data.iloc[-1]

        # Candle 1: Bearish
        c1_bearish = c1['Close'] < c1['Open']
        c1_body = abs(c1['Close'] - c1['Open'])

        # Candle 2: Small (star)
        c2_body = abs(c2['Close'] - c2['Open'])
        c2_small = c2_body < c1_body * 0.3

        # Candle 3: Bullish
        c3_bullish = c3['Close'] > c3['Open']
        c3_body = abs(c3['Close'] - c3['Open'])

        # Pattern validation
        return c1_bearish and c2_small and c3_bullish and c3_body > c1_body * 0.5

    def detect_evening_star(self, data: pd.DataFrame) -> bool:
        """Evening Star: Bearish reversal"""
        if len(data) < 3:
            return False

        c1, c2, c3 = data.iloc[-3], data.iloc[-2], data.iloc[-1]

        c1_bullish = c1['Close'] > c1['Open']
        c1_body = abs(c1['Close'] - c1['Open'])

        c2_body = abs(c2['Close'] - c2['Open'])
        c2_small = c2_body < c1_body * 0.3

        c3_bearish = c3['Close'] < c3['Open']
        c3_body = abs(c3['Close'] - c3['Open'])

        return c1_bullish and c2_small and c3_bearish and c3_body > c1_body * 0.5

    def detect_three_white_soldiers(self, data: pd.DataFrame) -> bool:
        """
        Three White Soldiers: Strong bullish continuation
        - 3 consecutive bullish candles
        - Each opens within previous body
        - Each closes higher
        """
        if len(data) < 3:
            return False

        c1, c2, c3 = data.iloc[-3], data.iloc[-2], data.iloc[-1]

        all_bullish = (
            c1['Close'] > c1['Open'] and
            c2['Close'] > c2['Open'] and
            c3['Close'] > c3['Open']
        )

        progressive_close = c1['Close'] < c2['Close'] < c3['Close']

        opens_within_body = (
            c1['Open'] < c2['Open'] < c1['Close'] and
            c2['Open'] < c3['Open'] < c2['Close']
        )

        return all_bullish and progressive_close and opens_within_body

    def analyze_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Detect all patterns and provide signal

        Returns:
            {
                'bullish_patterns': List[str],
                'bearish_patterns': List[str],
                'signal': str ('BULLISH', 'BEARISH', 'NEUTRAL'),
                'strength': float (0-1),
                'description': str
            }
        """
        if data.empty or len(data) < 3:
            return {
                'bullish_patterns': [],
                'bearish_patterns': [],
                'signal': 'NEUTRAL',
                'strength': 0,
                'description': 'Insufficient data'
            }

        bullish = []
        bearish = []

        # Detect all patterns
        if self.detect_hammer(data):
            bullish.append('HAMMER')

        if self.detect_engulfing_bullish(data):
            bullish.append('BULLISH_ENGULFING')

        if self.detect_engulfing_bearish(data):
            bearish.append('BEARISH_ENGULFING')

        if self.detect_doji(data):
            # Doji can be reversal - context dependent
            if len(data) >= 2:
                prev_trend = data.iloc[-2]['Close'] - data.iloc[-3]['Close'] if len(data) >= 3 else 0
                if prev_trend > 0:
                    bearish.append('DOJI_REVERSAL')
                elif prev_trend < 0:
                    bullish.append('DOJI_REVERSAL')

        if self.detect_morning_star(data):
            bullish.append('MORNING_STAR')

        if self.detect_evening_star(data):
            bearish.append('EVENING_STAR')

        if self.detect_three_white_soldiers(data):
            bullish.append('THREE_WHITE_SOLDIERS')

        # Determine signal
        bull_score = len(bullish)
        bear_score = len(bearish)

        if bull_score > bear_score:
            signal = 'BULLISH'
            strength = min(bull_score / 3.0, 1.0)  # Cap at 1.0
            description = f"Bullish patterns: {', '.join(bullish)}"
        elif bear_score > bull_score:
            signal = 'BEARISH'
            strength = min(bear_score / 3.0, 1.0)
            description = f"Bearish patterns: {', '.join(bearish)}"
        else:
            signal = 'NEUTRAL'
            strength = 0
            description = 'No clear pattern signal'

        return {
            'bullish_patterns': bullish,
            'bearish_patterns': bearish,
            'signal': signal,
            'strength': float(strength),
            'description': description
        }

    def supports_trade(self, data: pd.DataFrame, trade_direction: str) -> Dict:
        """
        Check if price patterns support the trade

        Args:
            data: Price data (OHLC)
            trade_direction: 'CALL' or 'PUT'

        Returns:
            {
                'supported': bool,
                'confidence_boost': float (0-0.15),
                'patterns': List[str],
                'reason': str
            }
        """
        analysis = self.analyze_patterns(data)

        if trade_direction == 'CALL':
            if analysis['signal'] == 'BULLISH':
                return {
                    'supported': True,
                    'confidence_boost': analysis['strength'] * 0.15,
                    'patterns': analysis['bullish_patterns'],
                    'reason': f"Bullish patterns detected: {', '.join(analysis['bullish_patterns'])}"
                }
            elif analysis['signal'] == 'NEUTRAL':
                return {
                    'supported': True,
                    'confidence_boost': 0,
                    'patterns': [],
                    'reason': 'No conflicting patterns'
                }
            else:
                return {
                    'supported': False,
                    'confidence_boost': 0,
                    'patterns': analysis['bearish_patterns'],
                    'reason': f"Bearish patterns conflict with CALL: {', '.join(analysis['bearish_patterns'])}"
                }

        elif trade_direction == 'PUT':
            if analysis['signal'] == 'BEARISH':
                return {
                    'supported': True,
                    'confidence_boost': analysis['strength'] * 0.15,
                    'patterns': analysis['bearish_patterns'],
                    'reason': f"Bearish patterns detected: {', '.join(analysis['bearish_patterns'])}"
                }
            elif analysis['signal'] == 'NEUTRAL':
                return {
                    'supported': True,
                    'confidence_boost': 0,
                    'patterns': [],
                    'reason': 'No conflicting patterns'
                }
            else:
                return {
                    'supported': False,
                    'confidence_boost': 0,
                    'patterns': analysis['bullish_patterns'],
                    'reason': f"Bullish patterns conflict with PUT: {', '.join(analysis['bullish_patterns'])}"
                }

        return {'supported': False, 'confidence_boost': 0, 'patterns': [], 'reason': 'Unknown direction'}


# Global instance
_pattern_detector = None

def get_pattern_detector() -> PricePatternDetector:
    """Get singleton pattern detector"""
    global _pattern_detector
    if _pattern_detector is None:
        _pattern_detector = PricePatternDetector()
    return _pattern_detector


if __name__ == "__main__":
    # Test
    import yfinance as yf

    detector = PricePatternDetector()

    symbol = 'AAPL'
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='5d', interval='1d')

    print(f"PRICE PATTERN ANALYSIS: {symbol}")
    print("="*60)

    analysis = detector.analyze_patterns(data)

    print(f"Signal: {analysis['signal']}")
    print(f"Strength: {analysis['strength']:.2f}")
    print(f"Bullish Patterns: {', '.join(analysis['bullish_patterns']) if analysis['bullish_patterns'] else 'None'}")
    print(f"Bearish Patterns: {', '.join(analysis['bearish_patterns']) if analysis['bearish_patterns'] else 'None'}")
    print(f"Description: {analysis['description']}")

    # Test CALL support
    call_check = detector.supports_trade(data, 'CALL')
    print(f"\nCALL Trade: {'SUPPORTED' if call_check['supported'] else 'NOT SUPPORTED'}")
    print(f"Confidence Boost: +{call_check['confidence_boost']:.2%}")
    print(f"Reason: {call_check['reason']}")
