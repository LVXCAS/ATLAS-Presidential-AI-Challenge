#!/usr/bin/env python3
"""
FUTURES EMA CROSSOVER STRATEGY
Optimized for Micro E-mini Futures (MES, MNQ)

Strategy:
- Fast EMA (10) crosses Slow EMA (20) = Signal
- Trend EMA (200) for direction filter
- Entry only with trend confirmation
- RSI for momentum validation

Target: 60%+ win rate on MES/MNQ
Risk: 2x ATR stops, 3x ATR targets
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional


class FuturesEMAStrategy:
    """
    EMA Crossover Strategy for Futures Trading

    Optimized for:
    - MES (Micro E-mini S&P 500) - $5 per point
    - MNQ (Micro E-mini Nasdaq-100) - $2 per point

    Features:
    - Triple EMA system (10/20/200)
    - RSI momentum filter
    - ATR-based stops/targets
    - Trend alignment required
    """

    def __init__(self, ema_fast=10, ema_slow=20, ema_trend=200, rsi_period=14):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.rsi_period = rsi_period

        # Optimization parameters
        self.min_ema_separation = 0.5  # Minimum points separation between fast/slow
        self.min_trend_distance = 2.0  # Minimum distance from trend EMA (points)
        self.rsi_long_threshold = 55   # RSI threshold for long entries
        self.rsi_short_threshold = 45  # RSI threshold for short entries
        self.score_threshold = 9.0     # Minimum score for entry

        # Risk parameters
        self.atr_stop_multiplier = 2.0   # Stop loss: 2x ATR
        self.atr_target_multiplier = 3.0  # Take profit: 3x ATR

        print(f"[FUTURES EMA STRATEGY] Initialized")
        print(f"  Fast EMA: {ema_fast}")
        print(f"  Slow EMA: {ema_slow}")
        print(f"  Trend EMA: {ema_trend}")
        print(f"  RSI Period: {rsi_period}")
        print(f"  Risk/Reward: 1:{self.atr_target_multiplier / self.atr_stop_multiplier:.1f}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""

        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=self.ema_trend, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR (for stops/targets)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()

        return df

    def analyze_opportunity(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        Analyze if current setup qualifies for entry

        Returns:
            Opportunity dict or None
        """

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Need at least 200+ candles for trend EMA
        if len(df) < self.ema_trend + 10:
            return None

        # Get current values
        current = df.iloc[-1]
        previous = df.iloc[-2]

        price = current['close']
        ema_fast_curr = current['ema_fast']
        ema_slow_curr = current['ema_slow']
        ema_trend_curr = current['ema_trend']
        rsi = current['rsi']
        atr = current['atr']

        ema_fast_prev = previous['ema_fast']
        ema_slow_prev = previous['ema_slow']

        # Check for crossover signals
        bullish_cross = (ema_fast_curr > ema_slow_curr) and (ema_fast_prev <= ema_slow_prev)
        bearish_cross = (ema_fast_curr < ema_slow_curr) and (ema_fast_prev >= ema_slow_prev)

        # Calculate EMA separation (trend strength)
        ema_separation = abs(ema_fast_curr - ema_slow_curr)

        # Calculate distance from trend EMA
        trend_distance = abs(price - ema_trend_curr)

        # Scoring system
        score = 5.0  # Base score
        direction = None

        # LONG Setup
        if bullish_cross:
            # Must be above trend (required)
            if price > ema_trend_curr:
                score += 2.0  # Above trend

                # Price must be meaningfully above trend
                if trend_distance >= self.min_trend_distance:
                    score += 1.0  # Strong trend separation
            else:
                return None  # No trade below trend

            # RSI check
            if rsi > self.rsi_long_threshold:
                score += 2.0  # Bullish momentum

            # Strong momentum bonus
            if rsi > 65:
                score += 1.0  # Very strong momentum

            # EMAs must be separated
            if ema_separation >= self.min_ema_separation:
                score += 1.0  # Clear EMA separation

            # Check threshold
            if score >= self.score_threshold:
                direction = 'LONG'

        # SHORT Setup
        elif bearish_cross:
            # Must be below trend (required)
            if price < ema_trend_curr:
                score += 2.0  # Below trend

                # Price must be meaningfully below trend
                if trend_distance >= self.min_trend_distance:
                    score += 1.0  # Strong trend separation
            else:
                return None  # No trade above trend

            # RSI check
            if rsi < self.rsi_short_threshold:
                score += 2.0  # Bearish momentum

            # Strong momentum bonus
            if rsi < 35:
                score += 1.0  # Very strong momentum

            # EMAs must be separated
            if ema_separation >= self.min_ema_separation:
                score += 1.0  # Clear EMA separation

            # Check threshold
            if score >= self.score_threshold:
                direction = 'SHORT'

        # No signal
        if direction is None:
            return None

        # Calculate entry, stop, target
        if direction == 'LONG':
            entry = price
            stop_loss = entry - (self.atr_stop_multiplier * atr)
            take_profit = entry + (self.atr_target_multiplier * atr)
        else:  # SHORT
            entry = price
            stop_loss = entry + (self.atr_stop_multiplier * atr)
            take_profit = entry - (self.atr_target_multiplier * atr)

        risk_reward = abs(take_profit - entry) / abs(entry - stop_loss)

        # Calculate contract value (for position sizing)
        if 'MES' in symbol:
            point_value = 5.0  # $5 per point
        elif 'MNQ' in symbol:
            point_value = 2.0  # $2 per point
        else:
            point_value = 1.0  # Default

        # Calculate risk per contract
        risk_points = abs(entry - stop_loss)
        risk_per_contract = risk_points * point_value

        opportunity = {
            'symbol': symbol,
            'strategy': 'FUTURES_EMA_CROSSOVER',
            'asset_type': 'FUTURES',
            'direction': direction,
            'score': score,
            'entry_price': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward,
            'point_value': point_value,
            'risk_points': risk_points,
            'risk_per_contract': risk_per_contract,
            'indicators': {
                'ema_fast': ema_fast_curr,
                'ema_slow': ema_slow_curr,
                'ema_trend': ema_trend_curr,
                'rsi': rsi,
                'atr': atr,
                'ema_separation': ema_separation,
                'trend_distance': trend_distance
            },
            'timestamp': datetime.now().isoformat()
        }

        return opportunity

    def validate_rules(self, opportunity: Dict) -> bool:
        """
        Validate opportunity meets all entry rules
        """

        # Check R/R
        if opportunity['risk_reward'] < 1.5:
            return False

        # Check trend alignment
        indicators = opportunity['indicators']
        price = opportunity['entry_price']
        trend_ema = indicators['ema_trend']

        if opportunity['direction'] == 'LONG':
            if price < trend_ema:
                return False

            # Ensure meaningful separation from trend
            if indicators['trend_distance'] < self.min_trend_distance:
                return False

        else:  # SHORT
            if price > trend_ema:
                return False

            # Ensure meaningful separation from trend
            if indicators['trend_distance'] < self.min_trend_distance:
                return False

        # Ensure EMAs are properly separated
        if indicators['ema_separation'] < self.min_ema_separation:
            return False

        return True


def demo():
    """Demo the futures strategy"""

    print("\n" + "="*70)
    print("FUTURES EMA STRATEGY DEMO")
    print("="*70)

    # Create mock data
    dates = pd.date_range(start='2025-01-01', periods=300, freq='15min')
    np.random.seed(42)

    # Simulated MES price data with trend
    trend = np.linspace(4500, 4600, 300)
    noise = np.random.normal(0, 5, 300)
    close_prices = trend + noise

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices - np.random.uniform(0, 3, 300),
        'high': close_prices + np.random.uniform(0, 5, 300),
        'low': close_prices - np.random.uniform(0, 5, 300),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 300)
    })

    # Test strategy
    engine = FuturesEMAStrategy()
    opportunity = engine.analyze_opportunity(df, 'MES')

    if opportunity:
        print("\n[SIGNAL FOUND]")
        print(f"  Symbol: {opportunity['symbol']}")
        print(f"  Direction: {opportunity['direction']}")
        print(f"  Score: {opportunity['score']:.2f}")
        print(f"  Entry: ${opportunity['entry_price']:.2f}")
        print(f"  Stop Loss: ${opportunity['stop_loss']:.2f}")
        print(f"  Take Profit: ${opportunity['take_profit']:.2f}")
        print(f"  Risk/Reward: {opportunity['risk_reward']:.2f}:1")
        print(f"  Risk per contract: ${opportunity['risk_per_contract']:.2f}")

        valid = engine.validate_rules(opportunity)
        print(f"  Passes All Rules: {'YES' if valid else 'NO'}")
    else:
        print("\n[NO SIGNAL] Waiting for high-quality setup...")

    print("\n" + "="*70)
    print("Futures strategy ready for MES/MNQ trading")
    print("="*70)


if __name__ == "__main__":
    demo()
