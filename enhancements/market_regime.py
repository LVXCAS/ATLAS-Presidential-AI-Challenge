#!/usr/bin/env python3
"""
Market Regime Detection
Identifies if market is trending, ranging, or volatile
Adapts strategy accordingly
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """Detect and adapt to market regimes"""

    def __init__(self):
        # ADX thresholds
        self.adx_thresholds = {
            'STRONG_TREND': 40,
            'TREND': 25,
            'WEAK_TREND': 20
        }

        # Volatility thresholds (ATR as % of price)
        self.vol_thresholds = {
            'HIGH': 0.04,  # 4%+ daily moves
            'NORMAL': 0.02,  # 2-4% daily moves
            'LOW': 0.01    # <1% daily moves
        }

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX)
        Measures trend strength (0-100)
        <20 = weak/ranging, 20-40 = trending, >40 = strong trend
        """
        try:
            if len(data) < period + 1:
                return 20.0  # Default neutral

            high = data['High']
            low = data['Low']
            close = data['Close']

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate Directional Movement
            up_move = high - high.shift()
            down_move = low.shift() - low

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            # Smooth with EMA
            atr = tr.ewm(span=period, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(span=period, adjust=False).mean()

            return float(adx.iloc[-1]) if not adx.empty else 20.0

        except Exception as e:
            logger.error(f"ADX calculation error: {e}")
            return 20.0

    def calculate_atr_percent(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate ATR as percentage of price
        Measures volatility
        """
        try:
            if len(data) < period:
                return 0.02

            high = data['High']
            low = data['Low']
            close = data['Close']

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.rolling(period).mean().iloc[-1]
            current_price = close.iloc[-1]

            atr_pct = atr / current_price if current_price > 0 else 0.02

            return float(atr_pct)

        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.02

    def detect_regime(self, symbol: str = 'SPY') -> Dict:
        """
        Detect current market regime

        Returns:
            {
                'regime': str ('STRONG_TREND', 'TREND', 'RANGE', 'VOLATILE'),
                'trend_direction': str ('BULL', 'BEAR', 'NEUTRAL'),
                'adx': float,
                'atr_pct': float,
                'description': str,
                'strategy_recommendation': str
            }
        """
        try:
            # Fetch SPY data (market proxy)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='3mo', interval='1d')

            if data.empty or len(data) < 30:
                logger.warning(f"Insufficient data for {symbol} regime detection")
                return self._default_regime()

            # Calculate indicators
            adx = self.calculate_adx(data)
            atr_pct = self.calculate_atr_percent(data)

            # Determine trend direction
            close = data['Close']
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()

            current_price = close.iloc[-1]
            current_sma50 = sma_50.iloc[-1] if len(sma_50) > 0 else current_price
            current_sma200 = sma_200.iloc[-1] if len(sma_200) > 0 else current_price

            if current_price > current_sma50 > current_sma200:
                trend_direction = 'BULL'
            elif current_price < current_sma50 < current_sma200:
                trend_direction = 'BEAR'
            else:
                trend_direction = 'NEUTRAL'

            # Determine regime
            if atr_pct > self.vol_thresholds['HIGH']:
                regime = 'VOLATILE'
                strategy_rec = 'Reduce size, take quick profits, avoid new entries'

            elif adx >= self.adx_thresholds['STRONG_TREND']:
                regime = 'STRONG_TREND'
                strategy_rec = f'Follow {trend_direction} trend, hold longer, larger positions'

            elif adx >= self.adx_thresholds['TREND']:
                regime = 'TREND'
                strategy_rec = f'Trade {trend_direction} direction, normal holds'

            else:  # ADX < 20
                regime = 'RANGE'
                strategy_rec = 'Mean reversion, quick profits, avoid directional bets'

            description = (
                f"{regime} market - "
                f"{trend_direction} trend (ADX {adx:.1f}, ATR {atr_pct*100:.1f}%)"
            )

            return {
                'regime': regime,
                'trend_direction': trend_direction,
                'adx': float(adx),
                'atr_pct': float(atr_pct),
                'description': description,
                'strategy_recommendation': strategy_rec,
                'symbol': symbol
            }

        except Exception as e:
            logger.error(f"Regime detection error for {symbol}: {e}")
            return self._default_regime()

    def _default_regime(self) -> Dict:
        """Return default regime when detection fails"""
        return {
            'regime': 'RANGE',
            'trend_direction': 'NEUTRAL',
            'adx': 20.0,
            'atr_pct': 0.02,
            'description': 'Unknown - using conservative defaults',
            'strategy_recommendation': 'Be cautious, use smaller positions',
            'symbol': 'UNKNOWN'
        }

    def get_strategy_adjustments(self, regime: Dict) -> Dict:
        """
        Get strategy adjustments based on regime

        Returns:
            {
                'directional_bias': str,
                'position_size_mult': float,
                'hold_time_mult': float,
                'profit_target_mult': float,
                'strategy_type': str
            }
        """
        regime_type = regime['regime']
        trend = regime['trend_direction']

        if regime_type == 'STRONG_TREND':
            return {
                'directional_bias': trend,
                'position_size_mult': 1.2,  # Larger positions
                'hold_time_mult': 1.5,  # Hold longer
                'profit_target_mult': 1.3,  # Higher targets
                'strategy_type': 'MOMENTUM',
                'recommendation': f'Strong {trend} trend - ride momentum'
            }

        elif regime_type == 'TREND':
            return {
                'directional_bias': trend,
                'position_size_mult': 1.0,
                'hold_time_mult': 1.2,
                'profit_target_mult': 1.1,
                'strategy_type': 'TREND_FOLLOWING',
                'recommendation': f'Follow {trend} trend, standard sizing'
            }

        elif regime_type == 'RANGE':
            return {
                'directional_bias': 'NEUTRAL',
                'position_size_mult': 0.8,  # Smaller positions
                'hold_time_mult': 0.7,  # Shorter holds
                'profit_target_mult': 0.8,  # Lower targets (take quick profits)
                'strategy_type': 'MEAN_REVERSION',
                'recommendation': 'Ranging market - quick in/out, mean reversion'
            }

        else:  # VOLATILE
            return {
                'directional_bias': 'NEUTRAL',
                'position_size_mult': 0.5,  # Much smaller
                'hold_time_mult': 0.5,  # Very short holds
                'profit_target_mult': 0.6,  # Take any profit
                'strategy_type': 'DEFENSIVE',
                'recommendation': 'High volatility - defensive, small positions'
            }

    def should_trade_direction(self, regime: Dict, intended_direction: str) -> Dict:
        """
        Check if intended trade direction aligns with market regime

        Args:
            regime: Regime dict from detect_regime()
            intended_direction: 'CALL' or 'PUT'

        Returns:
            {
                'approved': bool,
                'confidence_adjustment': float (-0.2 to +0.2),
                'reason': str
            }
        """
        regime_type = regime['regime']
        trend = regime['trend_direction']

        # Convert intended direction
        is_bullish = intended_direction.upper() == 'CALL'

        # In ranging/volatile markets, no strong bias
        if regime_type in ['RANGE', 'VOLATILE']:
            return {
                'approved': True,
                'confidence_adjustment': 0.0,
                'reason': f'{regime_type} market - no directional bias'
            }

        # In trending markets, check alignment
        if regime_type in ['TREND', 'STRONG_TREND']:
            strength_mult = 1.5 if regime_type == 'STRONG_TREND' else 1.0

            if (is_bullish and trend == 'BULL') or (not is_bullish and trend == 'BEAR'):
                # Aligned with trend
                conf_boost = 0.15 * strength_mult
                return {
                    'approved': True,
                    'confidence_adjustment': conf_boost,
                    'reason': f'Aligned with {trend} {regime_type} (+{conf_boost:.0%} conf)'
                }
            elif trend == 'NEUTRAL':
                return {
                    'approved': True,
                    'confidence_adjustment': 0.0,
                    'reason': 'Neutral trend - no bias'
                }
            else:
                # Against the trend
                conf_penalty = -0.10 * strength_mult
                return {
                    'approved': True,  # Don't block, just reduce confidence
                    'confidence_adjustment': conf_penalty,
                    'reason': f'Against {trend} {regime_type} ({conf_penalty:.0%} conf)'
                }

        return {
            'approved': True,
            'confidence_adjustment': 0.0,
            'reason': 'No regime bias'
        }


# Global instance
_market_regime = None

def get_market_regime_detector() -> MarketRegimeDetector:
    """Get singleton market regime detector"""
    global _market_regime
    if _market_regime is None:
        _market_regime = MarketRegimeDetector()
    return _market_regime


if __name__ == "__main__":
    # Test
    detector = MarketRegimeDetector()

    print("="*70)
    print("MARKET REGIME DETECTION TEST")
    print("="*70)

    # Detect current regime
    regime = detector.detect_regime('SPY')

    print(f"\n[CURRENT MARKET REGIME]")
    print(f"Regime: {regime['regime']}")
    print(f"Trend Direction: {regime['trend_direction']}")
    print(f"ADX: {regime['adx']:.1f}")
    print(f"ATR: {regime['atr_pct']*100:.2f}% of price")
    print(f"Description: {regime['description']}")
    print(f"Strategy: {regime['strategy_recommendation']}")

    # Get strategy adjustments
    print(f"\n[STRATEGY ADJUSTMENTS]")
    adjustments = detector.get_strategy_adjustments(regime)
    print(f"Type: {adjustments['strategy_type']}")
    print(f"Directional Bias: {adjustments['directional_bias']}")
    print(f"Position Size Mult: {adjustments['position_size_mult']:.2f}x")
    print(f"Hold Time Mult: {adjustments['hold_time_mult']:.2f}x")
    print(f"Profit Target Mult: {adjustments['profit_target_mult']:.2f}x")
    print(f"Recommendation: {adjustments['recommendation']}")

    # Test directional approval
    print(f"\n[DIRECTIONAL TRADE APPROVAL]")
    for direction in ['CALL', 'PUT']:
        approval = detector.should_trade_direction(regime, direction)
        print(f"\n{direction}:")
        print(f"  Approved: {approval['approved']}")
        print(f"  Conf Adjustment: {approval['confidence_adjustment']:+.0%}")
        print(f"  Reason: {approval['reason']}")
