#!/usr/bin/env python3
"""
Multi-Timeframe Analysis
Checks 1-minute, 5-minute, 1-hour, and daily trends before trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """Analyze multiple timeframes for trade confirmation"""

    def __init__(self):
        self.timeframes = {
            '1m': {'period': '1d', 'interval': '1m', 'weight': 0.15},
            '5m': {'period': '5d', 'interval': '5m', 'weight': 0.25},
            '1h': {'period': '1mo', 'interval': '1h', 'weight': 0.30},
            '1d': {'period': '6mo', 'interval': '1d', 'weight': 0.30}
        }

    def calculate_trend(self, data: pd.DataFrame) -> Dict:
        """Calculate trend direction and strength"""
        if data.empty or len(data) < 10:
            return {'direction': 0, 'strength': 0, 'aligned': False}

        try:
            # Calculate moving averages
            close = data['Close']
            sma_fast = close.rolling(5).mean()
            sma_slow = close.rolling(20).mean()

            # Current values
            current_price = close.iloc[-1]
            current_fast = sma_fast.iloc[-1] if len(sma_fast) > 0 and not pd.isna(sma_fast.iloc[-1]) else current_price
            current_slow = sma_slow.iloc[-1] if len(sma_slow) > 0 and not pd.isna(sma_slow.iloc[-1]) else current_price

            # Trend direction
            if current_fast > current_slow and current_price > current_fast:
                direction = 1  # Bullish
            elif current_fast < current_slow and current_price < current_fast:
                direction = -1  # Bearish
            else:
                direction = 0  # Neutral

            # Trend strength (0-1)
            ma_diff = abs(current_fast - current_slow) / current_price if current_price > 0 else 0
            strength = min(ma_diff * 100, 1.0)  # Cap at 1.0

            # Check if price is aligned with trend
            if direction == 1:
                aligned = current_price > current_fast > current_slow
            elif direction == -1:
                aligned = current_price < current_fast < current_slow
            else:
                aligned = False

            return {
                'direction': direction,
                'strength': float(strength),
                'aligned': aligned,
                'price': float(current_price),
                'sma_fast': float(current_fast),
                'sma_slow': float(current_slow)
            }

        except Exception as e:
            logger.error(f"Trend calculation error: {e}")
            return {'direction': 0, 'strength': 0, 'aligned': False}

    def analyze_all_timeframes(self, symbol: str) -> Dict:
        """
        Analyze all timeframes and provide consensus

        Returns:
            {
                'consensus': str ('STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'),
                'score': float (-1 to 1),
                'timeframes': {
                    '1m': {...},
                    '5m': {...},
                    '1h': {...},
                    '1d': {...}
                },
                'aligned': bool (all timeframes agree),
                'confidence': float (0-1)
            }
        """
        try:
            ticker = yf.Ticker(symbol)
            results = {}
            weighted_score = 0
            total_weight = 0

            # Analyze each timeframe
            for tf_name, tf_config in self.timeframes.items():
                try:
                    # Get data
                    data = ticker.history(period=tf_config['period'], interval=tf_config['interval'])

                    if data.empty:
                        continue

                    # Calculate trend
                    trend = self.calculate_trend(data)
                    trend['timeframe'] = tf_name
                    results[tf_name] = trend

                    # Add to weighted score
                    weight = tf_config['weight']
                    weighted_score += trend['direction'] * trend['strength'] * weight
                    total_weight += weight

                except Exception as e:
                    logger.debug(f"Error analyzing {tf_name} for {symbol}: {e}")
                    continue

            if total_weight == 0:
                return {
                    'consensus': 'NEUTRAL',
                    'score': 0,
                    'timeframes': results,
                    'aligned': False,
                    'confidence': 0
                }

            # Normalize score
            final_score = weighted_score / total_weight if total_weight > 0 else 0

            # Check alignment (all timeframes pointing same direction)
            directions = [r['direction'] for r in results.values()]
            if len(directions) >= 3:
                aligned = all(d > 0 for d in directions) or all(d < 0 for d in directions)
            else:
                aligned = False

            # Determine consensus
            if final_score > 0.5:
                consensus = 'STRONG_BUY'
            elif final_score > 0.2:
                consensus = 'BUY'
            elif final_score < -0.5:
                consensus = 'STRONG_SELL'
            elif final_score < -0.2:
                consensus = 'SELL'
            else:
                consensus = 'NEUTRAL'

            # Confidence based on alignment and strength
            confidence = abs(final_score)
            if aligned:
                confidence *= 1.2  # Boost if aligned
            confidence = min(confidence, 1.0)

            return {
                'consensus': consensus,
                'score': float(final_score),
                'timeframes': results,
                'aligned': aligned,
                'confidence': float(confidence),
                'num_timeframes': len(results)
            }

        except Exception as e:
            logger.error(f"Multi-timeframe analysis failed for {symbol}: {e}")
            return {
                'consensus': 'NEUTRAL',
                'score': 0,
                'timeframes': {},
                'aligned': False,
                'confidence': 0
            }

    def should_enter_trade(self, symbol: str, trade_direction: str) -> Dict:
        """
        Check if multi-timeframe analysis supports entering a trade

        Args:
            symbol: Stock symbol
            trade_direction: 'CALL' or 'PUT'

        Returns:
            {
                'approved': bool,
                'reason': str,
                'confidence_boost': float (0-0.2)
            }
        """
        analysis = self.analyze_all_timeframes(symbol)

        # For CALL trades, need bullish consensus
        if trade_direction == 'CALL':
            if analysis['consensus'] in ['STRONG_BUY', 'BUY']:
                return {
                    'approved': True,
                    'reason': f"Multi-timeframe {analysis['consensus']} (score: {analysis['score']:.2f})",
                    'confidence_boost': analysis['confidence'] * 0.15,
                    'timeframe_score': analysis['score']
                }
            elif analysis['consensus'] == 'NEUTRAL' and analysis['score'] > 0:
                return {
                    'approved': True,
                    'reason': f"Weak bullish trend across timeframes (score: {analysis['score']:.2f})",
                    'confidence_boost': analysis['confidence'] * 0.05,
                    'timeframe_score': analysis['score']
                }
            else:
                return {
                    'approved': False,
                    'reason': f"Multi-timeframe shows {analysis['consensus']} (score: {analysis['score']:.2f}) - conflicts with CALL",
                    'confidence_boost': 0,
                    'timeframe_score': analysis['score']
                }

        # For PUT trades, need bearish consensus
        elif trade_direction == 'PUT':
            if analysis['consensus'] in ['STRONG_SELL', 'SELL']:
                return {
                    'approved': True,
                    'reason': f"Multi-timeframe {analysis['consensus']} (score: {analysis['score']:.2f})",
                    'confidence_boost': analysis['confidence'] * 0.15,
                    'timeframe_score': analysis['score']
                }
            elif analysis['consensus'] == 'NEUTRAL' and analysis['score'] < 0:
                return {
                    'approved': True,
                    'reason': f"Weak bearish trend across timeframes (score: {analysis['score']:.2f})",
                    'confidence_boost': analysis['confidence'] * 0.05,
                    'timeframe_score': analysis['score']
                }
            else:
                return {
                    'approved': False,
                    'reason': f"Multi-timeframe shows {analysis['consensus']} (score: {analysis['score']:.2f}) - conflicts with PUT",
                    'confidence_boost': 0,
                    'timeframe_score': analysis['score']
                }

        return {'approved': False, 'reason': 'Unknown trade direction', 'confidence_boost': 0}


# Global instance
_mtf_analyzer = None

def get_mtf_analyzer() -> MultiTimeframeAnalyzer:
    """Get singleton multi-timeframe analyzer"""
    global _mtf_analyzer
    if _mtf_analyzer is None:
        _mtf_analyzer = MultiTimeframeAnalyzer()
    return _mtf_analyzer


if __name__ == "__main__":
    # Test
    analyzer = MultiTimeframeAnalyzer()

    test_symbols = ['AAPL', 'TSLA', 'SPY']

    print("MULTI-TIMEFRAME ANALYSIS TEST")
    print("="*70)

    for symbol in test_symbols:
        print(f"\n{symbol}:")
        analysis = analyzer.analyze_all_timeframes(symbol)

        print(f"  Consensus: {analysis['consensus']}")
        print(f"  Score: {analysis['score']:.3f}")
        print(f"  Aligned: {analysis['aligned']}")
        print(f"  Confidence: {analysis['confidence']:.2%}")

        print(f"\n  Timeframes:")
        for tf, data in analysis['timeframes'].items():
            direction_str = "BULL" if data['direction'] > 0 else "BEAR" if data['direction'] < 0 else "NEUTRAL"
            print(f"    {tf}: {direction_str} (strength: {data['strength']:.2f}, aligned: {data['aligned']})")

        # Test CALL approval
        call_check = analyzer.should_enter_trade(symbol, 'CALL')
        print(f"\n  CALL Trade: {'APPROVED' if call_check['approved'] else 'REJECTED'}")
        print(f"  Reason: {call_check['reason']}")
