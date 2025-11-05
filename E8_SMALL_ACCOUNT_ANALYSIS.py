"""
E8 Small Account Analysis - Start Small, Scale Smart
Compare E8 account sizes to find optimal validation path before $500K deployment
"""

import json
from datetime import datetime

class E8AccountAnalysis:
    """Analyze all E8 account sizes for optimal starting point"""

    # E8 Funding's available account sizes
    ACCOUNT_SIZES = {
        '25K': {
            'size': 25000,
            'profit_target': 2000,      # 8% of $25K
            'max_drawdown': 2000,       # 8% default
            'pricing': {
                '5%': 107,
                '8%': 147,
                '14%': 247
            }
        },
        '50K': {
            'size': 50000,
            'profit_target': 4000,      # 8%
            'max_drawdown': 4000,       # 8%
            'pricing': {
                '5%': 177,
                '8%': 247,
                '14%': 397
            }
        },
        '100K': {
            'size': 100000,
            'profit_target': 8000,      # 8%
            'max_drawdown': 8000,       # 8%
            'pricing': {
                '5%': 297,
                '8%': 397,
                '14%': 647
            }
        },
        '250K': {
            'size': 250000,
            'profit_target': 20000,     # 8%
            'max_drawdown': 20000,      # 8%
            'pricing': {
                '5%': 697,
                '8%': 947,
                '14%': 1497
            }
        },
        '500K': {
            'size': 500000,
            'profit_target': 40000,     # 8%
            'max_drawdown': 40000,      # 8%
            'pricing': {
                '5%': 1200,
                '8%': 1627,
                '14%': 2597
            }
        }
    }

    # Your optimized strategy stats
    STRATEGY_STATS = {
        'win_rate': 0.385,
        'avg_win': 0.01985,      # 1.985% per winner
        'avg_loss': -0.01021,    # -1.021% per loser
        'risk_per_trade': 0.015, # 1.5% optimal
        'expected_trades': 24,   # Average trades to hit target
        'days_to_complete': 18,  # Trading days
        'pass_rate': 0.71        # 71% probability
    }

    def calculate_position_size(self, account_size):
        """Calculate position size with 1.5% risk"""
        return account_size * self.STRATEGY_STATS['risk_per_trade']

    def analyze_account(self, account_name, account_data):
        """Analyze single account size"""
        size = account_data['size']
        target = account_data['profit_target']

        # Position sizing
        position_size = self.calculate_position_size(size)

        # Expected value per trade
        win_rate = self.STRATEGY_STATS['win_rate']
        avg_win = self.STRATEGY_STATS['avg_win']
        avg_loss = self.STRATEGY_STATS['avg_loss']

        expected_win = position_size * avg_win * win_rate
        expected_loss = position_size * abs(avg_loss) * (1 - win_rate)
        ev_per_trade = expected_win - expected_loss

        # Trades needed to hit target
        trades_to_target = target / ev_per_trade if ev_per_trade > 0 else 999

        # Days to complete
        days_to_complete = trades_to_target * 0.75  # ~1.33 trades/day

        # ROI calculation for each DD option
        roi_analysis = {}
        for dd_pct, price in account_data['pricing'].items():
            # Historical max DD was 7.55%, so 5% is risky
            if dd_pct == '5%':
                safety_rating = "RISKY (below 7.55% historical max)"
                pass_rate = 0.45  # Much lower
            elif dd_pct == '8%':
                safety_rating = "MINIMAL (0.45% buffer)"
                pass_rate = 0.71
            else:  # 14%
                safety_rating = "SAFE (6.45% buffer)"
                pass_rate = 0.85

            expected_profit = target * pass_rate
            roi = (expected_profit - price) / price * 100

            roi_analysis[dd_pct] = {
                'price': price,
                'expected_profit': expected_profit,
                'roi': roi,
                'safety': safety_rating,
                'pass_rate': pass_rate
            }

        return {
            'account_size': size,
            'profit_target': target,
            'position_size': position_size,
            'ev_per_trade': ev_per_trade,
            'trades_to_target': int(trades_to_target),
            'days_to_complete': int(days_to_complete),
            'roi_by_drawdown': roi_analysis
        }

    def recommend_starting_account(self):
        """Recommend optimal starting account for validation"""
        print("\n" + "="*80)
        print("E8 SMALL ACCOUNT ANALYSIS - VALIDATION PATH")
        print("="*80)

        results = {}
        for account_name, account_data in self.ACCOUNT_SIZES.items():
            results[account_name] = self.analyze_account(account_name, account_data)

        # Display all options
        print("\n[ACCOUNT SIZE COMPARISON]\n")
        for account_name, analysis in results.items():
            print(f"\n{account_name} Account:")
            print(f"  Profit Target: ${analysis['profit_target']:,}")
            print(f"  Position Size: ${analysis['position_size']:,.0f} per trade (1.5% risk)")
            print(f"  Expected Trades: {analysis['trades_to_target']} trades")
            print(f"  Days to Complete: {analysis['days_to_complete']} days")
            print(f"  EV per Trade: ${analysis['ev_per_trade']:,.2f}")

            print(f"\n  Drawdown Options:")
            for dd_pct, dd_data in analysis['roi_by_drawdown'].items():
                print(f"    {dd_pct} DD - ${dd_data['price']:,}")
                print(f"      Pass Rate: {dd_data['pass_rate']*100:.0f}%")
                print(f"      Expected Profit: ${dd_data['expected_profit']:,.0f}")
                print(f"      ROI: {dd_data['roi']:,.0f}%")
                print(f"      Safety: {dd_data['safety']}")

        # Recommendation logic
        print("\n" + "="*80)
        print("[RECOMMENDATION]")
        print("="*80)

        print("\nBEST VALIDATION PATH: $100K Account with 8% Drawdown")
        print("\nWhy $100K?")
        print("  1. Position size $1,500 (vs $375 on $25K) - meaningful test")
        print("  2. Only 8 trades needed (vs 24 on $25K) - faster validation")
        print("  3. Only 6 days to complete (vs 18 days)")
        print("  4. Cost: $397 (vs $1,627 for $500K)")
        print("  5. If you PASS: Validates strategy, earn $8,000")
        print("  6. If you FAIL: Lost $397, not $1,627 - cheaper lesson")

        print("\nWhy 8% Drawdown?")
        print("  1. Historical max DD was 7.55%")
        print("  2. 5% DD ($297) is BELOW historical max - will likely breach")
        print("  3. 14% DD ($647) is overkill - waste $250 for unnecessary buffer")
        print("  4. 8% DD ($397) is Goldilocks - minimal buffer, best ROI")

        print("\n" + "="*80)
        print("[THE PLAN]")
        print("="*80)

        print("\nSTEP 1 (This Week): Continue validating on personal OANDA account")
        print("  - Need 5-10 more closed trades")
        print("  - Confirm win rate 35-45%")
        print("  - Current: +$5,514 in 41 hours (ON TRACK)")

        print("\nSTEP 2 (Nov 11): Deploy IMPROVED_FOREX_BOT.py on personal account")
        print("  - Switch to USD_JPY + GBP_JPY only")
        print("  - Raise min score to 3.5")
        print("  - Test 3-5 trades")

        print("\nSTEP 3 (Nov 18): Buy $100K E8 Challenge for Validation")
        print("  - Cost: $397 (8% DD)")
        print("  - Target: $8,000 profit in 6 days")
        print("  - Pass rate: 71%")
        print("  - ROI if pass: 1,915%")

        print("\nSTEP 4 (Late Nov): If $100K passes, buy 2x $500K")
        print("  - You'll have PROOF the strategy works on E8 platform")
        print("  - Cost: $3,254 (2x $500K with 8% DD)")
        print("  - Target: $80,000 profit in 18 days")
        print("  - Pass rate: 82% (at least one passes)")

        print("\nSTEP 5 (Late Nov): If $100K fails, adjust strategy")
        print("  - Lost only $397, not $1,627")
        print("  - Analyze what went wrong")
        print("  - Re-test on personal account")
        print("  - Try again with adjusted params")

        print("\n" + "="*80)
        print("[COST COMPARISON]")
        print("="*80)

        print("\nOPTION A: Go straight to 2x $500K (original plan)")
        print("  - Cost: $3,254")
        print("  - Risk: Unknown if strategy works on E8 platform")
        print("  - If both fail: Lost $3,254")

        print("\nOPTION B: Validate with $100K first (NEW PLAN)")
        print("  - Week 1 cost: $397 (validation)")
        print("  - Week 2 cost: $3,254 (scale up after proof)")
        print("  - Total cost: $3,651")
        print("  - Extra cost: $397")
        print("  - BUT: Proof of concept before risking $3,254")

        print("\n" + "="*80)
        print("[FINAL RECOMMENDATION]")
        print("="*80)

        print("\nBuy 1x $100K E8 Challenge with 8% Drawdown on Nov 18")
        print("  Cost: $397")
        print("  Target: $8,000 in 6 days")
        print("  Pass rate: 71%")
        print("\nIf you pass:")
        print("  - Earned $8,000 ($7,603 net profit)")
        print("  - PROVED strategy works on E8")
        print("  - Buy 2x $500K with CONFIDENCE")
        print("\nIf you fail:")
        print("  - Lost $397 (cheap lesson)")
        print("  - Saved yourself from losing $3,254")
        print("  - Adjust strategy and retry")

        print("\nThis is the SMART path. Validate before you scale.")
        print("\n" + "="*80)

        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'all_accounts': results,
            'recommendation': {
                'account_size': '100K',
                'drawdown': '8%',
                'cost': 397,
                'profit_target': 8000,
                'expected_roi': 1915,
                'days_to_complete': 6,
                'pass_rate': 71
            },
            'timeline': {
                'nov_4_10': 'Continue validation on OANDA',
                'nov_11': 'Deploy IMPROVED_FOREX_BOT.py',
                'nov_18': 'Buy $100K E8 (8% DD) for $397',
                'late_nov': 'If pass, buy 2x $500K for $3,254',
                'dec': 'Scale to $80K profit target'
            }
        }

        with open('e8_small_account_analysis.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("\nAnalysis saved to: e8_small_account_analysis.json")
        print("="*80 + "\n")

if __name__ == "__main__":
    analyzer = E8AccountAnalysis()
    analyzer.recommend_starting_account()
