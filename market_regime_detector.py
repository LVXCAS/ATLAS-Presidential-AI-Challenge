#!/usr/bin/env python3
"""
MARKET REGIME DETECTOR
Analyzes market conditions to determine if Bull Put Spreads are viable today

Key insight from Day 3: Bull Put Spreads need LOW momentum (<3%)
On very bullish days (all stocks 6-38% momentum), ZERO candidates found!
"""

import yfinance as yf
from datetime import datetime, timedelta
from multi_source_data_fetcher import MultiSourceDataFetcher


class MarketRegimeDetector:
    """Detect market regime to determine optimal trading strategies"""

    def __init__(self):
        self.data_fetcher = MultiSourceDataFetcher()

    def analyze_market_regime(self):
        """
        Analyze S&P 500 market regime

        Returns:
            dict: {
                'regime': str ('VERY_BULLISH' | 'BULLISH' | 'NEUTRAL' | 'BEARISH'),
                'sp500_momentum': float,
                'vix_level': float,
                'suitable_strategies': list,
                'bull_put_spread_viable': bool,
                'confidence_adjustment': float,
                'explanation': str
            }
        """

        print("\n" + "="*70)
        print("MARKET REGIME ANALYSIS")
        print("="*70)

        # Get S&P 500 data
        try:
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='30d', interval='1d')

            if spy_hist.empty:
                raise Exception("Cannot fetch SPY data")

            current_price = float(spy_hist['Close'].iloc[-1])
            price_5d_ago = float(spy_hist['Close'].iloc[-6])
            price_10d_ago = float(spy_hist['Close'].iloc[-11])
            price_20d_ago = float(spy_hist['Close'].iloc[-21])

            momentum_5d = (current_price / price_5d_ago) - 1
            momentum_10d = (current_price / price_10d_ago) - 1
            momentum_20d = (current_price / price_20d_ago) - 1

            avg_momentum = (momentum_5d + momentum_10d + momentum_20d) / 3

            print(f"\n[S&P 500 MOMENTUM]")
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  5-day momentum: {momentum_5d:+.1%}")
            print(f"  10-day momentum: {momentum_10d:+.1%}")
            print(f"  20-day momentum: {momentum_20d:+.1%}")
            print(f"  Average: {avg_momentum:+.1%}")

        except Exception as e:
            print(f"\n[ERROR] Cannot analyze S&P 500: {e}")
            return {
                'regime': 'UNKNOWN',
                'sp500_momentum': 0,
                'vix_level': 0,
                'suitable_strategies': ['STOCK'],
                'bull_put_spread_viable': False,
                'confidence_adjustment': 0,
                'explanation': 'Market data unavailable'
            }

        # Get VIX (volatility) data
        try:
            vix = yf.Ticker('^VIX')
            vix_hist = vix.history(period='5d', interval='1d')
            vix_level = float(vix_hist['Close'].iloc[-1]) if not vix_hist.empty else 20

            print(f"\n[VIX (VOLATILITY)]")
            print(f"  Current VIX: {vix_level:.2f}")

        except:
            vix_level = 20  # Default assumption

        # Determine market regime
        if avg_momentum > 0.05:
            regime = 'VERY_BULLISH'
        elif avg_momentum > 0.02:
            regime = 'BULLISH'
        elif avg_momentum > -0.02:
            regime = 'NEUTRAL'
        else:
            regime = 'BEARISH'

        print(f"\n[MARKET REGIME]")
        print(f"  Regime: {regime}")

        # Determine suitable strategies based on regime
        if regime == 'VERY_BULLISH':
            # High momentum - Bull Put Spreads NOT viable
            suitable_strategies = ['DUAL_OPTIONS', 'LONG_CALLS']
            bull_put_spread_viable = False
            confidence_adjustment = -1.0  # LOWER threshold (easier to find high-momentum trades)
            explanation = (
                "VERY BULLISH MARKET: S&P 500 momentum >5%\n"
                "  - Most stocks have HIGH momentum (6-38%+)\n"
                "  - Bull Put Spreads NOT viable (need <3% momentum)\n"
                "  - Recommendation: Use Dual Options or Long Calls\n"
                "  - OR: Wait for market to cool down for high-probability spreads"
            )

        elif regime == 'BULLISH':
            # Moderate momentum - some Bull Put Spread candidates
            suitable_strategies = ['BULL_PUT_SPREAD', 'DUAL_OPTIONS']
            bull_put_spread_viable = True
            confidence_adjustment = -0.5  # Slightly lower threshold
            explanation = (
                "BULLISH MARKET: S&P 500 momentum 2-5%\n"
                "  - Mix of high and moderate momentum stocks\n"
                "  - Bull Put Spreads VIABLE on low-momentum stocks\n"
                "  - Recommendation: Scan for <3% momentum candidates"
            )

        elif regime == 'NEUTRAL':
            # Low momentum - IDEAL for Bull Put Spreads
            suitable_strategies = ['BULL_PUT_SPREAD', 'IRON_CONDOR', 'BUTTERFLY']
            bull_put_spread_viable = True
            confidence_adjustment = 0  # Normal threshold
            explanation = (
                "NEUTRAL MARKET: S&P 500 momentum -2% to +2%\n"
                "  - IDEAL for Bull Put Spreads (high probability)\n"
                "  - Many stocks with <3% momentum available\n"
                "  - Recommendation: Focus on high-probability premium collection"
            )

        else:  # BEARISH
            suitable_strategies = ['BEAR_CALL_SPREAD', 'LONG_PUTS']
            bull_put_spread_viable = False
            confidence_adjustment = -1.0
            explanation = (
                "BEARISH MARKET: S&P 500 momentum <-2%\n"
                "  - Bull Put Spreads NOT viable (market falling)\n"
                "  - Recommendation: Use Bear Call Spreads or defensive strategies"
            )

        print(f"\n[STRATEGY RECOMMENDATIONS]")
        print(f"  Suitable strategies: {', '.join(suitable_strategies)}")
        print(f"  Bull Put Spreads viable: {'YES' if bull_put_spread_viable else 'NO'}")
        print(f"  Confidence threshold adjustment: {confidence_adjustment:+.1f}")

        print(f"\n[EXPLANATION]")
        for line in explanation.split('\n'):
            print(f"  {line}")

        print("="*70 + "\n")

        return {
            'regime': regime,
            'sp500_momentum': avg_momentum,
            'vix_level': vix_level,
            'suitable_strategies': suitable_strategies,
            'bull_put_spread_viable': bull_put_spread_viable,
            'confidence_adjustment': confidence_adjustment,
            'explanation': explanation
        }


def test_regime_detection():
    """Test market regime detection"""

    detector = MarketRegimeDetector()
    regime = detector.analyze_market_regime()

    print("\n" + "="*70)
    print("REGIME DETECTION SUMMARY")
    print("="*70)
    print(f"Market Regime: {regime['regime']}")
    print(f"S&P 500 Momentum: {regime['sp500_momentum']:+.1%}")
    print(f"VIX Level: {regime['vix_level']:.2f}")
    print(f"Bull Put Spreads Viable: {'YES' if regime['bull_put_spread_viable'] else 'NO'}")
    print(f"\nRecommended Strategies:")
    for strategy in regime['suitable_strategies']:
        print(f"  - {strategy}")
    print("="*70)


if __name__ == "__main__":
    test_regime_detection()
