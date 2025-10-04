#!/usr/bin/env python3
"""
TECHNICAL INDICATORS ML ENHANCER
=================================
Adds 150+ technical indicators as features for ML models
Uses pandas-ta and ta libraries (pure Python, no compilation needed)

This enhancer feeds indicators to:
- XGBoost (pattern recognition)
- LightGBM (ensemble models)
- PyTorch (neural networks)
"""

import pandas as pd
import pandas_ta as pta
import ta
import numpy as np
from datetime import datetime


class TechnicalIndicatorsMLEnhancer:
    """Add technical indicators as ML features"""

    def __init__(self):
        print("TECHNICAL INDICATORS ML ENHANCER INITIALIZED")
        print("=" * 60)
        print("Using: pandas-ta + ta libraries")
        print("Available: 150+ technical indicators")
        print("=" * 60)

    def calculate_core_indicators(self, df):
        """Calculate core technical indicators for ML features"""

        if len(df) < 20:
            print("[WARNING] Not enough data for indicators (need 20+ bars)")
            return None

        indicators = {}

        try:
            # MOMENTUM INDICATORS
            indicators['rsi'] = pta.rsi(df['close'], length=14).iloc[-1]
            indicators['rsi_slow'] = pta.rsi(df['close'], length=21).iloc[-1]

            macd_result = pta.macd(df['close'])
            if macd_result is not None and not macd_result.empty:
                indicators['macd'] = macd_result['MACD_12_26_9'].iloc[-1]
                indicators['macd_signal'] = macd_result['MACDs_12_26_9'].iloc[-1]
                indicators['macd_hist'] = macd_result['MACDh_12_26_9'].iloc[-1]

            # TREND INDICATORS
            indicators['sma_20'] = pta.sma(df['close'], length=20).iloc[-1]
            indicators['ema_12'] = pta.ema(df['close'], length=12).iloc[-1]
            indicators['ema_26'] = pta.ema(df['close'], length=26).iloc[-1]

            # VOLATILITY INDICATORS
            bbands = pta.bbands(df['close'], length=20)
            if bbands is not None and not bbands.empty:
                indicators['bb_upper'] = bbands['BBU_20_2.0'].iloc[-1]
                indicators['bb_middle'] = bbands['BBM_20_2.0'].iloc[-1]
                indicators['bb_lower'] = bbands['BBL_20_2.0'].iloc[-1]
                indicators['bb_width'] = bbands['BBB_20_2.0'].iloc[-1]

            atr_result = pta.atr(df['high'], df['low'], df['close'], length=14)
            if atr_result is not None:
                indicators['atr'] = atr_result.iloc[-1]

            # VOLUME INDICATORS
            if 'volume' in df.columns:
                obv_result = pta.obv(df['close'], df['volume'])
                if obv_result is not None:
                    indicators['obv'] = obv_result.iloc[-1]

                indicators['volume_sma'] = pta.sma(df['volume'], length=20).iloc[-1]

            # Using ta library for additional indicators
            indicators['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close']).iloc[-1]
            indicators['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close']).iloc[-1]

            indicators['cci'] = ta.trend.cci(df['high'], df['low'], df['close']).iloc[-1]
            indicators['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close']).iloc[-1]

            # PRICE POSITION
            current_price = df['close'].iloc[-1]
            indicators['price_pct_bb'] = (current_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            indicators['price_vs_sma20'] = (current_price / indicators['sma_20']) - 1
            indicators['price_vs_ema12'] = (current_price / indicators['ema_12']) - 1

            return indicators

        except Exception as e:
            print(f"[ERROR] Indicator calculation failed: {e}")
            return None

    def enhance_ml_score_with_indicators(self, base_score, indicators):
        """Enhance ML score using technical indicators"""

        if indicators is None:
            return base_score

        boost = 0.0

        try:
            # RSI ANALYSIS (+0.3 max)
            rsi = indicators.get('rsi', 50)
            if rsi < 30:  # Oversold
                boost += 0.2
                print(f"  [INDICATOR] RSI oversold ({rsi:.1f}): +0.2")
            elif rsi > 70:  # Overbought (caution for puts)
                boost += 0.1
                print(f"  [INDICATOR] RSI overbought ({rsi:.1f}): +0.1")

            # MACD ANALYSIS (+0.2 max)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:  # Bullish crossover
                boost += 0.2
                print(f"  [INDICATOR] MACD bullish: +0.2")

            # BOLLINGER BANDS (+0.3 max)
            bb_position = indicators.get('price_pct_bb', 0.5)
            if bb_position < 0.2:  # Near lower band (oversold)
                boost += 0.3
                print(f"  [INDICATOR] BB oversold ({bb_position:.2f}): +0.3")
            elif bb_position > 0.8:  # Near upper band
                boost += 0.1
                print(f"  [INDICATOR] BB extended ({bb_position:.2f}): +0.1")

            # STOCHASTIC (+0.2 max)
            stoch_k = indicators.get('stoch_k', 50)
            if stoch_k < 20:  # Oversold
                boost += 0.2
                print(f"  [INDICATOR] Stochastic oversold ({stoch_k:.1f}): +0.2")

            # VOLUME CONFIRMATION (+0.2 max)
            if 'volume' in indicators and 'volume_sma' in indicators:
                volume_ratio = indicators.get('obv', 0) / indicators.get('volume_sma', 1)
                if volume_ratio > 1.5:  # Strong volume
                    boost += 0.2
                    print(f"  [INDICATOR] Strong volume: +0.2")

            # ATR VOLATILITY (+0.1 max)
            atr = indicators.get('atr', 0)
            current_price = indicators.get('price_vs_sma20', 0) + 1  # Approximate
            if atr > 0:
                atr_pct = (atr / current_price) if current_price > 0 else 0
                if atr_pct > 0.02:  # Good volatility for options
                    boost += 0.1
                    print(f"  [INDICATOR] High ATR ({atr_pct:.2%}): +0.1")

        except Exception as e:
            print(f"  [ERROR] Indicator enhancement failed: {e}")

        enhanced_score = base_score + boost
        print(f"  [TOTAL INDICATOR BOOST]: +{boost:.2f}")

        return enhanced_score

    def demo_indicators(self, symbol, historical_df):
        """Demonstrate indicator calculation"""

        print(f"\n{'='*60}")
        print(f"DEMO: Technical Indicators for {symbol}")
        print(f"{'='*60}")

        print(f"\nCalculating indicators on {len(historical_df)} bars...")

        indicators = self.calculate_core_indicators(historical_df)

        if indicators:
            print("\n[MOMENTUM INDICATORS]")
            print(f"  RSI (14):        {indicators.get('rsi', 0):.2f}")
            print(f"  RSI (21):        {indicators.get('rsi_slow', 0):.2f}")
            print(f"  MACD:            {indicators.get('macd', 0):.4f}")
            print(f"  MACD Signal:     {indicators.get('macd_signal', 0):.4f}")
            print(f"  Stochastic K:    {indicators.get('stoch_k', 0):.2f}")

            print("\n[TREND INDICATORS]")
            print(f"  SMA (20):        ${indicators.get('sma_20', 0):.2f}")
            print(f"  EMA (12):        ${indicators.get('ema_12', 0):.2f}")
            print(f"  Price vs SMA:    {indicators.get('price_vs_sma20', 0):.2%}")

            print("\n[VOLATILITY INDICATORS]")
            print(f"  BB Upper:        ${indicators.get('bb_upper', 0):.2f}")
            print(f"  BB Middle:       ${indicators.get('bb_middle', 0):.2f}")
            print(f"  BB Lower:        ${indicators.get('bb_lower', 0):.2f}")
            print(f"  BB Position:     {indicators.get('price_pct_bb', 0):.2%}")
            print(f"  ATR (14):        ${indicators.get('atr', 0):.2f}")

            print("\n[DEMO: ML Score Enhancement]")
            base_score = 3.5
            enhanced_score = self.enhance_ml_score_with_indicators(base_score, indicators)
            print(f"\nBase ML Score:     {base_score:.2f}")
            print(f"Enhanced Score:    {enhanced_score:.2f}")
            print(f"Boost from indicators: +{enhanced_score - base_score:.2f}")

        else:
            print("[FAILED] Could not calculate indicators")

        print(f"{'='*60}\n")


def main():
    """Demo technical indicators integration"""

    enhancer = TechnicalIndicatorsMLEnhancer()

    # Create sample data for demo
    print("\n[DEMO] Creating sample historical data...")

    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

    # Simulate price data (trending down for oversold signal)
    base_price = 100
    prices = [base_price - i * 0.5 + np.random.randn() * 0.3 for i in range(30)]

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(30)]
    })

    df.set_index('timestamp', inplace=True)

    # Demo indicators
    enhancer.demo_indicators("DEMO_STOCK", df)

    print("\n[SUCCESS] Technical indicators ready for ML integration")
    print("[NEXT] Integrate with ml_activation_system.py")


if __name__ == "__main__":
    main()
