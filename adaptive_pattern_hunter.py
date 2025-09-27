#!/usr/bin/env python3
"""
ADAPTIVE PATTERN HUNTER
Adapts your successful patterns to changing market regimes
Switches strategies when markets switch up
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AdaptivePatternHunter:
    """Adapts patterns to market regime changes"""

    def __init__(self):
        # Your core successful patterns (regime-agnostic)
        self.core_patterns = {
            'volatility_explosion': 'High vol breakdowns',
            'earnings_catalyst': 'Catalyst-driven moves',
            'momentum_reversal': 'Momentum breaks',
            'sector_volatility': 'Sector-specific volatility'
        }

        # Market regime indicators
        self.regime_indicators = {}

    def detect_market_regime(self):
        """Figure out what kind of market we're in RIGHT NOW"""

        print("MARKET REGIME DETECTION")
        print("=" * 50)

        try:
            # Get key market indicators
            spy = yf.Ticker('SPY')
            vix = yf.Ticker('^VIX')

            # Recent data
            spy_data = spy.history(period='1mo')
            vix_data = vix.history(period='1mo')

            if spy_data.empty or vix_data.empty:
                print("Data unavailable - assuming NEUTRAL regime")
                return 'NEUTRAL'

            # Calculate regime signals
            spy_return_30d = (spy_data['Close'][-1] - spy_data['Close'][0]) / spy_data['Close'][0]
            current_vix = vix_data['Close'][-1]
            avg_vix_30d = vix_data['Close'].mean()

            print(f"SPY 30d return: {spy_return_30d:.1%}")
            print(f"Current VIX: {current_vix:.1f}")
            print(f"30d avg VIX: {avg_vix_30d:.1f}")

            # Determine regime
            if spy_return_30d > 0.05 and current_vix < 20:
                regime = 'BULL_LOW_VOL'
            elif spy_return_30d > 0.05 and current_vix > 25:
                regime = 'BULL_HIGH_VOL'
            elif spy_return_30d < -0.05 and current_vix > 25:
                regime = 'BEAR_HIGH_VOL'  # Your puts crushed here
            elif spy_return_30d < -0.05 and current_vix < 20:
                regime = 'BEAR_LOW_VOL'
            else:
                regime = 'CHOPPY'

            print(f"DETECTED REGIME: {regime}")
            return regime

        except Exception as e:
            print(f"Regime detection error: {e}")
            return 'NEUTRAL'

    def adapt_patterns_to_regime(self, regime):
        """Adapt your winning patterns to current market regime"""

        print(f"\nADAPTING PATTERNS FOR {regime} REGIME")
        print("-" * 50)

        adaptations = {}

        if regime == 'BEAR_HIGH_VOL':
            # This is where your puts dominated
            adaptations = {
                'primary_strategy': 'PUT_BUYING',
                'volatility_explosion': 'Continue put buying on vol spikes',
                'earnings_catalyst': 'Put buying on earnings disappointments',
                'momentum_reversal': 'Puts on momentum failures',
                'position_sizing': 'Aggressive (25-35% positions)',
                'targets': '50-100% ROI like your wins'
            }

        elif regime == 'BULL_LOW_VOL':
            # Market switched - adapt!
            adaptations = {
                'primary_strategy': 'CALL_BUYING',
                'volatility_explosion': 'Call buying on breakouts instead',
                'earnings_catalyst': 'Call buying on earnings beats',
                'momentum_reversal': 'Calls on momentum resumption',
                'position_sizing': 'Moderate (15-25% positions)',
                'targets': '30-60% ROI (lower vol environment)'
            }

        elif regime == 'BULL_HIGH_VOL':
            # High vol but up trending
            adaptations = {
                'primary_strategy': 'STRADDLES',
                'volatility_explosion': 'Both directions - straddles/strangles',
                'earnings_catalyst': 'Long vol into earnings',
                'momentum_reversal': 'Trade whipsaws both ways',
                'position_sizing': 'Conservative (10-20% positions)',
                'targets': '40-80% ROI on vol expansion'
            }

        elif regime == 'CHOPPY':
            # Sideways market
            adaptations = {
                'primary_strategy': 'PREMIUM_SELLING',
                'volatility_explosion': 'Sell puts on oversold, calls on overbought',
                'earnings_catalyst': 'Sell straddles into earnings',
                'momentum_reversal': 'Iron condors on range-bound',
                'position_sizing': 'Small frequent (5-15% positions)',
                'targets': '20-40% ROI through premium decay'
            }

        for strategy, description in adaptations.items():
            print(f"{strategy}: {description}")

        return adaptations

    def find_current_opportunities(self, regime, adaptations):
        """Find opportunities matching current regime"""

        print(f"\nCURRENT OPPORTUNITIES FOR {regime}")
        print("-" * 50)

        # Candidates based on your successful symbols
        base_candidates = ['TSLA', 'AMD', 'NVDA', 'META', 'UBER', 'COIN', 'RIVN', 'SNAP']

        opportunities = []

        for symbol in base_candidates:
            if regime == 'BEAR_HIGH_VOL':
                # Your put pattern
                opportunity = f"{symbol} PUTS - Look for breakdown like your wins"
                opportunities.append(opportunity)

            elif regime == 'BULL_LOW_VOL':
                # Flip to calls
                opportunity = f"{symbol} CALLS - Momentum breakouts"
                opportunities.append(opportunity)

            elif regime == 'BULL_HIGH_VOL':
                # Both directions
                opportunity = f"{symbol} STRADDLES - High vol both ways"
                opportunities.append(opportunity)

            elif regime == 'CHOPPY':
                # Premium selling
                opportunity = f"{symbol} PREMIUM SELL - Range trading"
                opportunities.append(opportunity)

        print("TOP OPPORTUNITIES:")
        for i, opp in enumerate(opportunities[:5], 1):
            print(f"{i}. {opp}")

        return opportunities

    def run_adaptive_analysis(self):
        """Full adaptive pattern analysis"""

        print("ADAPTIVE PATTERN HUNTER")
        print("=" * 60)
        print("Adapting your 68.3% avg ROI patterns to current market")
        print("=" * 60)

        # Detect current regime
        regime = self.detect_market_regime()

        # Adapt patterns
        adaptations = self.adapt_patterns_to_regime(regime)

        # Find opportunities
        opportunities = self.find_current_opportunities(regime, adaptations)

        print(f"\n" + "=" * 60)
        print("ADAPTATION COMPLETE")
        print(f"Your patterns adapted for {regime} regime")
        print("Ready to maintain ROI as markets switch up!")
        print("=" * 60)

        return {
            'regime': regime,
            'adaptations': adaptations,
            'opportunities': opportunities
        }

def main():
    """Run adaptive pattern analysis"""
    hunter = AdaptivePatternHunter()
    results = hunter.run_adaptive_analysis()

if __name__ == "__main__":
    main()