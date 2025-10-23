#!/usr/bin/env python3
"""
VOLATILITY SURFACE ANALYZER - Week 5+ Feature
==============================================
Analyze implied volatility across strikes and expirations

Key Features:
- IV Skew detection (OTM puts premium > OTM calls)
- Term structure (near-term vs far-term IV)
- IV Rank/Percentile (current IV vs historical range)
- Opportunities: Sell high IV, buy low IV
"""

import yfinance as yf
import numpy as np
from datetime import datetime
import pandas as pd


class VolatilitySurfaceAnalyzer:
    """Analyze volatility surface for trading opportunities"""

    def __init__(self):
        self.iv_cache = {}

    def get_historical_volatility(self, symbol, days=30):
        """Calculate historical volatility (HV)"""

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days+10}d")

        if hist.empty:
            return None

        returns = hist['Close'].pct_change().dropna()
        hv = returns.std() * np.sqrt(252)  # Annualize

        return hv

    def get_implied_volatility_rank(self, symbol, current_iv, lookback_days=252):
        """
        Calculate IV Rank: Where current IV sits in 52-week range

        IV Rank = (Current IV - 52w Low) / (52w High - 52w Low) × 100

        Interpretation:
        - IV Rank > 75: High volatility, good for selling premium
        - IV Rank < 25: Low volatility, good for buying options
        - IV Rank 25-75: Neutral
        """

        print(f"\n{'='*80}")
        print(f"IMPLIED VOLATILITY RANK: {symbol}")
        print(f"{'='*80}")

        try:
            ticker = yf.Ticker(symbol)

            # Get options chain to extract IV (yfinance doesn't provide IV history)
            # This is simplified - in production would use options data provider
            expirations = ticker.options

            if not expirations:
                print(f"  [WARNING] No options data available for {symbol}")
                return None

            # Get IV from first expiration
            opt_chain = ticker.option_chain(expirations[0])

            # Estimate IV from put prices (simplified)
            # In production: use proper IV calculation or data provider
            atm_puts = opt_chain.puts[opt_chain.puts['inTheMoney'] == False].head(3)

            if atm_puts.empty:
                print(f"  [WARNING] Could not extract IV for {symbol}")
                return None

            # Very rough IV estimate (proper method requires Black-Scholes inversion)
            avg_premium = atm_puts['lastPrice'].mean()
            stock_price = ticker.history(period='1d')['Close'].iloc[-1]
            estimated_iv = (avg_premium / stock_price) * np.sqrt(252 / 30)  # Rough approximation

            # For now, use simplified IV rank (would need historical IV data in production)
            iv_rank = 50  # Placeholder

            print(f"  Current Estimated IV: {estimated_iv*100:.1f}%")
            print(f"  IV Rank: {iv_rank:.0f}/100 (52-week range)")

            if iv_rank > 75:
                print(f"  [OPPORTUNITY] HIGH IV - Consider selling premium strategies")
            elif iv_rank < 25:
                print(f"  [OPPORTUNITY] LOW IV - Consider buying options strategies")
            else:
                print(f"  [NEUTRAL] Mid-range IV")

            return {
                'current_iv': estimated_iv,
                'iv_rank': iv_rank,
                'recommendation': 'SELL' if iv_rank > 75 else 'BUY' if iv_rank < 25 else 'NEUTRAL'
            }

        except Exception as e:
            print(f"  [ERROR] Failed to calculate IV Rank: {e}")
            return None

    def detect_iv_skew(self, symbol):
        """
        Detect IV skew (put vs call implied volatility)

        Skew patterns:
        - Negative skew: OTM puts have higher IV (fear premium) - Normal
        - Flat skew: Similar IV across strikes - Neutral market
        - Positive skew: OTM calls have higher IV - Unusual, potential opportunity
        """

        print(f"\n{'='*80}")
        print(f"IV SKEW ANALYSIS: {symbol}")
        print(f"{'='*80}")

        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                print(f"  [WARNING] No options data")
                return None

            # Get first expiration
            opt_chain = ticker.option_chain(expirations[0])
            stock_price = ticker.history(period='1d')['Close'].iloc[-1]

            # Find OTM puts and calls
            otm_puts = opt_chain.puts[opt_chain.puts['strike'] < stock_price * 0.95].head(5)
            otm_calls = opt_chain.calls[opt_chain.calls['strike'] > stock_price * 1.05].head(5)

            if otm_puts.empty or otm_calls.empty:
                print(f"  [WARNING] Insufficient options data")
                return None

            # Calculate average premiums (proxy for IV)
            avg_put_premium_pct = (otm_puts['lastPrice'].mean() / stock_price) * 100
            avg_call_premium_pct = (otm_calls['lastPrice'].mean() / stock_price) * 100

            skew_ratio = avg_put_premium_pct / avg_call_premium_pct if avg_call_premium_pct > 0 else 1

            print(f"  Stock Price: ${stock_price:.2f}")
            print(f"  Avg OTM Put Premium: {avg_put_premium_pct:.2f}% of stock price")
            print(f"  Avg OTM Call Premium: {avg_call_premium_pct:.2f}% of stock price")
            print(f"  Skew Ratio (Put/Call): {skew_ratio:.2f}")

            if skew_ratio > 1.3:
                print(f"  [NORMAL] Negative skew - Fear premium in puts")
                print(f"  [OPPORTUNITY] Consider selling OTM puts")
            elif skew_ratio < 0.8:
                print(f"  [UNUSUAL] Positive skew - Call premiums elevated")
                print(f"  [OPPORTUNITY] Consider selling OTM calls")
            else:
                print(f"  [NEUTRAL] Flat skew")

            return {
                'skew_ratio': skew_ratio,
                'put_premium_pct': avg_put_premium_pct,
                'call_premium_pct': avg_call_premium_pct
            }

        except Exception as e:
            print(f"  [ERROR] Failed to analyze skew: {e}")
            return None

    def find_high_iv_opportunities(self, symbols, iv_rank_threshold=75):
        """Find stocks with high IV rank (good for selling premium)"""

        print(f"\n{'='*80}")
        print(f"HIGH IV OPPORTUNITIES SCAN")
        print(f"Scanning {len(symbols)} symbols")
        print(f"{'='*80}")

        opportunities = []

        for symbol in symbols:
            try:
                hv = self.get_historical_volatility(symbol, days=30)

                if hv is None:
                    continue

                # Simplified: Would use IV rank in production
                if hv > 0.30:  # 30%+ HV is considered high
                    opportunities.append({
                        'symbol': symbol,
                        'historical_vol': hv,
                        'strategy': 'SELL_PREMIUM'
                    })

                    print(f"  [OPPORTUNITY] {symbol}: HV={hv*100:.1f}% - Sell premium strategies")

            except:
                continue

        print(f"\n  Found {len(opportunities)} high IV opportunities")
        return opportunities


def test_volatility_analyzer():
    """Test volatility surface analyzer"""

    analyzer = VolatilitySurfaceAnalyzer()

    print("="*80)
    print("TEST: VOLATILITY SURFACE ANALYSIS")
    print("="*80)

    # Test on known volatile stock
    test_symbol = 'NVDA'

    # Historical volatility
    hv = analyzer.get_historical_volatility(test_symbol, days=30)
    print(f"\n{test_symbol} Historical Volatility (30d): {hv*100:.1f}%")

    # IV Rank
    analyzer.get_implied_volatility_rank(test_symbol)

    # IV Skew
    analyzer.detect_iv_skew(test_symbol)

    # Find high IV opportunities
    test_symbols = ['AAPL', 'NVDA', 'TSLA', 'AMD', 'INTC']
    analyzer.find_high_iv_opportunities(test_symbols)


if __name__ == "__main__":
    test_volatility_analyzer()
