#!/usr/bin/env python3
"""
TIME SERIES MOMENTUM STRATEGY
==============================
One of the most robust quant strategies ever discovered!

Research:
- Moskowitz, Ooi, Pedersen (2012): "Time series momentum"
- Sharpe ratio: 0.5-1.0 across all asset classes
- Works for 200+ years across markets
- Combines beautifully with options trading

Strategy:
- If 1-month return > 0: Upward momentum → Buy calls
- If 1-month return < 0: Downward momentum → Buy puts
- If near zero: No momentum → Sell premium (iron condors)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
from dotenv import load_dotenv

load_dotenv('.env.paper')


class TimeSeriesMomentumStrategy:
    """Time series momentum for options trading"""

    def __init__(self):
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')

        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

        print("=" * 70)
        print("TIME SERIES MOMENTUM STRATEGY")
        print("=" * 70)
        print("Research: Moskowitz, Ooi, Pedersen (2012)")
        print("Sharpe: 0.5-1.0 | Robust across 200+ years")
        print("=" * 70)

    def calculate_momentum_signal(self, symbol, lookback_days=21):
        """
        Calculate time series momentum signal

        Returns:
        - Positive: Upward momentum (buy calls)
        - Negative: Downward momentum (buy puts)
        - Near zero: No momentum (sell premium)
        """

        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 10)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )

            bars = self.data_client.get_stock_bars(request)
            df = bars.df

            if df.empty or len(df) < lookback_days:
                return None

            # Calculate returns
            df = df.reset_index()
            df = df[df['symbol'] == symbol]

            current_price = df['close'].iloc[-1]
            past_price = df['close'].iloc[-lookback_days]

            # Time series momentum = current price / past price - 1
            momentum = (current_price / past_price) - 1

            # Annualized return
            annual_momentum = momentum * (252 / lookback_days)

            # Volatility (for Sharpe ratio)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            # Momentum Sharpe (signal strength)
            momentum_sharpe = annual_momentum / volatility if volatility > 0 else 0

            return {
                'symbol': symbol,
                'current_price': current_price,
                'momentum': momentum,
                'annual_momentum': annual_momentum,
                'volatility': volatility,
                'momentum_sharpe': momentum_sharpe,
                'signal': self._classify_signal(momentum, momentum_sharpe)
            }

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            return None

    def _classify_signal(self, momentum, momentum_sharpe):
        """
        Classify momentum into trading signals

        Strong signals: |momentum| > 5% AND momentum_sharpe > 0.5
        Weak signals: |momentum| < 2%
        """

        if abs(momentum) < 0.02:  # Less than 2%
            return {
                'direction': 'NEUTRAL',
                'strength': 'WEAK',
                'strategy': 'Iron Condor (sell premium)',
                'confidence': 0.6
            }

        elif momentum > 0.05 and momentum_sharpe > 0.5:
            return {
                'direction': 'BULLISH',
                'strength': 'STRONG',
                'strategy': 'Buy Calls or Bull Call Spread',
                'confidence': 0.85
            }

        elif momentum > 0.02:
            return {
                'direction': 'BULLISH',
                'strength': 'MODERATE',
                'strategy': 'Bull Put Spread (collect premium)',
                'confidence': 0.70
            }

        elif momentum < -0.05 and momentum_sharpe < -0.5:
            return {
                'direction': 'BEARISH',
                'strength': 'STRONG',
                'strategy': 'Buy Puts or Bear Put Spread',
                'confidence': 0.85
            }

        elif momentum < -0.02:
            return {
                'direction': 'BEARISH',
                'strength': 'MODERATE',
                'strategy': 'Bear Call Spread (collect premium)',
                'confidence': 0.70
            }

        else:
            return {
                'direction': 'NEUTRAL',
                'strength': 'WEAK',
                'strategy': 'Iron Condor or Butterfly',
                'confidence': 0.65
            }

    def scan_universe(self, symbols=None):
        """Scan stock universe for momentum signals"""

        if symbols is None:
            # Default universe: Liquid options stocks
            symbols = [
                'SPY', 'QQQ', 'IWM',  # ETFs
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Mega caps
                'NVDA', 'AMD', 'INTC',  # Semis
                'JPM', 'BAC', 'GS',  # Banks
                'XOM', 'CVX'  # Energy
            ]

        print(f"\n[SCANNING] {len(symbols)} symbols for time series momentum...")
        print("-" * 70)

        signals = []

        for symbol in symbols:
            result = self.calculate_momentum_signal(symbol, lookback_days=21)

            if result:
                signal = result['signal']

                print(f"\n{symbol}: ${result['current_price']:.2f}")
                print(f"  Momentum: {result['momentum']:+.2%} (annualized: {result['annual_momentum']:+.1%})")
                print(f"  Direction: {signal['direction']} ({signal['strength']})")
                print(f"  Strategy: {signal['strategy']}")
                print(f"  Confidence: {signal['confidence']:.0%}")

                if signal['confidence'] >= 0.70:
                    signals.append(result)
                    print(f"  [QUALIFIED] Confidence >= 70%")

        return signals

    def generate_trade_recommendations(self, signals):
        """Generate specific trade recommendations from signals"""

        print(f"\n{'='*70}")
        print("TRADE RECOMMENDATIONS (Time Series Momentum)")
        print(f"{'='*70}\n")

        recommendations = []

        for sig in signals:
            symbol = sig['symbol']
            price = sig['current_price']
            signal = sig['signal']
            momentum = sig['momentum']

            rec = {
                'symbol': symbol,
                'current_price': price,
                'momentum': momentum,
                'signal': signal['direction'],
                'strength': signal['strength'],
                'confidence': signal['confidence']
            }

            if signal['direction'] == 'BULLISH' and signal['strength'] == 'STRONG':
                # Strong uptrend: Buy calls
                rec['trade'] = f"Buy {symbol} calls, strike ${price*1.03:.1f}, 30 DTE"
                rec['rationale'] = f"Strong upward momentum ({momentum:+.1%})"
                recommendations.append(rec)

            elif signal['direction'] == 'BEARISH' and signal['strength'] == 'STRONG':
                # Strong downtrend: Buy puts
                rec['trade'] = f"Buy {symbol} puts, strike ${price*0.97:.1f}, 30 DTE"
                rec['rationale'] = f"Strong downward momentum ({momentum:+.1%})"
                recommendations.append(rec)

            elif signal['direction'] == 'NEUTRAL':
                # Range-bound: Sell premium
                rec['trade'] = f"{symbol} iron condor, 30 DTE"
                rec['rationale'] = "Low momentum, range-bound market"
                recommendations.append(rec)

        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [{rec['signal']}] {rec['symbol']}")
            print(f"   Trade: {rec['trade']}")
            print(f"   Rationale: {rec['rationale']}")
            print(f"   Confidence: {rec['confidence']:.0%}")
            print()

        return recommendations

    def demo_momentum_scan(self):
        """Demo: Scan and generate recommendations"""

        print("\n[DEMO] Time Series Momentum Scan")
        print("=" * 70)

        # Scan universe
        signals = self.scan_universe()

        # Generate trade recommendations
        if signals:
            recommendations = self.generate_trade_recommendations(signals)

            print(f"{'='*70}")
            print(f"SUMMARY: Found {len(recommendations)} qualifying opportunities")
            print(f"{'='*70}\n")

            # Stats
            bullish = sum(1 for r in recommendations if r['signal'] == 'BULLISH')
            bearish = sum(1 for r in recommendations if r['signal'] == 'BEARISH')
            neutral = sum(1 for r in recommendations if r['signal'] == 'NEUTRAL')

            print(f"Bullish signals: {bullish}")
            print(f"Bearish signals: {bearish}")
            print(f"Neutral signals: {neutral}")

        else:
            print("[NO SIGNALS] No qualifying opportunities found")


def main():
    """Run time series momentum strategy demo"""

    strategy = TimeSeriesMomentumStrategy()
    strategy.demo_momentum_scan()

    print("\n" + "="*70)
    print("INTEGRATION WITH YOUR SYSTEM:")
    print("="*70)
    print("1. Add momentum signals to your ML scoring")
    print("2. Use momentum to select strategy type:")
    print("   - Strong momentum → Directional (calls/puts)")
    print("   - Weak momentum → Premium selling (condors)")
    print("3. Combine with technical indicators for timing")
    print("4. Backtest with VectorBT for validation")
    print("\n[SUCCESS] Time series momentum strategy ready!")


if __name__ == "__main__":
    main()
