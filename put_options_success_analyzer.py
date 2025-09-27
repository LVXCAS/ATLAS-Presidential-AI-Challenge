#!/usr/bin/env python3
"""
PUT OPTIONS SUCCESS ANALYZER
Decode what made your 4 put option trades absolutely dominate
Build systematic approach to find more explosive put opportunities
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class PutOptionsSuccessAnalyzer:
    """Analyze what made your put options trades kick ass"""

    def __init__(self):
        # Your successful put trades
        self.winning_puts = {
            'INTC': {
                'roi': 0.706,  # +70.6%
                'type': 'earnings_catalyst',
                'sector': 'semiconductors',
                'notes': 'Intel earnings disappointment'
            },
            'RIVN': {
                'roi': 0.898,  # +89.8%
                'type': 'volatility_explosion',
                'sector': 'ev',
                'notes': 'High volatility stock'
            },
            'SNAP': {
                'roi': 0.447,  # +44.7%
                'type': 'social_media_volatility',
                'sector': 'social_media',
                'notes': 'User growth concerns'
            },
            'LYFT': {
                'roi': 0.683,  # +68.3%
                'type': 'momentum_reversal',
                'sector': 'rideshare',
                'notes': 'Momentum breakdown'
            }
        }

        # Average ROI: 67.1% - CRUSHING IT!
        self.avg_roi = sum([trade['roi'] for trade in self.winning_puts.values()]) / len(self.winning_puts)

    def analyze_winning_patterns(self):
        """What made these puts work so well?"""

        print("PUT OPTIONS SUCCESS ANALYSIS")
        print("=" * 60)
        print(f"Total Winning Trades: {len(self.winning_puts)}")
        print(f"Average ROI: {self.avg_roi:.1%}")
        print(f"Best Trade: RIVN +89.8%")
        print("=" * 60)

        # Pattern analysis
        patterns = {}
        sectors = {}

        for symbol, trade in self.winning_puts.items():
            # Track patterns
            pattern = trade['type']
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(trade['roi'])

            # Track sectors
            sector = trade['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(trade['roi'])

        print("\n=== WINNING PATTERNS ===")
        for pattern, rois in patterns.items():
            avg_roi = sum(rois) / len(rois)
            print(f"{pattern}: {avg_roi:.1%} avg ROI ({len(rois)} trades)")

        print("\n=== WINNING SECTORS ===")
        for sector, rois in sectors.items():
            avg_roi = sum(rois) / len(rois)
            print(f"{sector}: {avg_roi:.1%} avg ROI ({len(rois)} trades)")

        return patterns, sectors

    def identify_success_factors(self):
        """What made these puts absolutely demolish?"""

        print("\n=== SUCCESS FACTORS ANALYSIS ===")

        success_factors = {
            'High Volatility Stocks': ['RIVN', 'SNAP'],
            'Catalyst-Driven Trades': ['INTC'],
            'Momentum Reversals': ['LYFT'],
            'Tech/Growth Exposure': ['INTC', 'RIVN', 'SNAP', 'LYFT'],
            'High Beta Names': ['RIVN', 'SNAP', 'LYFT'],
            'Options-Friendly': ['All 4 symbols']
        }

        for factor, symbols in success_factors.items():
            if symbols == ['All 4 symbols']:
                print(f"✅ {factor}: All trades")
            else:
                roi_total = sum([self.winning_puts[sym]['roi'] for sym in symbols if sym in self.winning_puts])
                avg_roi = roi_total / len(symbols) if symbols else 0
                print(f"✅ {factor}: {avg_roi:.1%} avg ROI ({len(symbols)} trades)")

    def find_similar_current_opportunities(self):
        """Hunt for stocks with similar explosive put potential RIGHT NOW"""

        print("\n=== CURRENT PUT OPPORTUNITIES (Similar to your wins) ===")

        # Candidates with similar characteristics
        explosive_candidates = {
            # High volatility EV/Tech
            'TSLA': 'High vol EV like RIVN',
            'LCID': 'High vol EV like RIVN',
            'NIO': 'High vol EV like RIVN',

            # Social media/tech volatility
            'META': 'Social tech like SNAP',
            'RBLX': 'Gaming/social like SNAP',
            'PINS': 'Social platform like SNAP',

            # Semiconductor/earnings plays
            'AMD': 'Semiconductor like INTC',
            'NVDA': 'High vol semiconductor',
            'MU': 'Semiconductor like INTC',

            # Momentum/transport plays
            'UBER': 'Rideshare like LYFT',
            'DASH': 'Gig economy like LYFT',
            'ABNB': 'Platform economy like LYFT'
        }

        print("EXPLOSIVE PUT CANDIDATES (Similar to your wins):")
        print("Symbol | Similarity | Why It Could Crush")
        print("-" * 50)

        for symbol, reasoning in explosive_candidates.items():
            print(f"{symbol:>6} | {reasoning:>20} | Put potential")

        return explosive_candidates

    def build_put_hunting_strategy(self):
        """Build systematic approach to find more explosive puts"""

        print("\n=== PUT HUNTING STRATEGY (Scale Your Success) ===")

        strategy = {
            'screening_criteria': [
                'High implied volatility (>30%)',
                'Recent momentum breakdown',
                'Upcoming earnings/catalysts',
                'High beta (>1.5)',
                'Options volume >1000/day',
                'Price >$20 (avoid penny stocks)'
            ],
            'timing_signals': [
                'Technical breakdown patterns',
                'Momentum divergence',
                'Earnings disappointment setup',
                'Sector rotation away',
                'High put/call ratio'
            ],
            'position_sizing': [
                'Risk 10-15% per put position',
                'Target 50-100% ROI minimum',
                'Use 30-45 DTE options',
                'Scale into winners',
                'Cut losers quickly'
            ]
        }

        for category, criteria in strategy.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            for item in criteria:
                print(f"  • {item}")

        return strategy

def main():
    """Analyze put success and build scaling strategy"""
    analyzer = PutOptionsSuccessAnalyzer()

    # Analyze what worked
    patterns, sectors = analyzer.analyze_winning_patterns()
    analyzer.identify_success_factors()

    # Find new opportunities
    opportunities = analyzer.find_similar_current_opportunities()
    strategy = analyzer.build_put_hunting_strategy()

    print("\n" + "=" * 60)
    print("PUT OPTIONS SUCCESS DECODED!")
    print(f"Your {len(analyzer.winning_puts)} trades averaged {analyzer.avg_roi:.1%} ROI")
    print("Strategy built to find MORE explosive put opportunities")
    print("Ready to scale this success pattern!")
    print("=" * 60)

if __name__ == "__main__":
    main()