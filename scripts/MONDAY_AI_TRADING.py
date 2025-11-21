#!/usr/bin/env python3
"""
MONDAY MORNING AI-ENHANCED TRADING
Complete AI-powered scanner for Options + Forex
Ready for 6:30 AM PT market open

Integration: Traditional Strategies + AI Enhancement + Meta-Learning + Auto-Execution
"""

from ai_enhanced_forex_scanner import AIEnhancedForexScanner
from ai_enhanced_options_scanner import AIEnhancedOptionsScanner
from scanners.futures_scanner import AIEnhancedFuturesScanner
from execution.auto_execution_engine import AutoExecutionEngine
from datetime import datetime
import json
import sys

class MondayAITrading:
    """
    Master trading system with AI integration

    Combines:
    - Optimized EMA Crossover (forex)
    - Bull Put Spreads (options)
    - AI scoring and enhancement
    - Meta-learning from outcomes
    - AUTONOMOUS EXECUTION
    """

    def __init__(self, auto_execute: bool = True, max_trades: int = 2, enable_futures: bool = False):
        print("\n" + "="*70)
        print("MONDAY MORNING AI-ENHANCED TRADING SYSTEM")
        print("="*70)
        print(f"Date: {datetime.now().strftime('%A, %B %d, %Y')}")
        print(f"Time: {datetime.now().strftime('%I:%M %p')}")
        print(f"Mode: {'AUTONOMOUS' if auto_execute else 'MANUAL'}")
        print("="*70)

        # Initialize scanners
        self.forex_scanner = AIEnhancedForexScanner()
        self.options_scanner = AIEnhancedOptionsScanner()

        # Initialize futures scanner (optional)
        self.enable_futures = enable_futures
        if enable_futures:
            self.futures_scanner = AIEnhancedFuturesScanner(paper_trading=True)

            # CONSERVATIVE FUTURES MODE
            self.futures_max_risk = 100.0  # $100 max risk per trade
            self.futures_max_positions = 2  # Only 2 futures at a time
            self.futures_max_total_risk = 500.0  # $500 total across all positions

            print("[FUTURES] Enabled - CONSERVATIVE MODE")
            print(f"  Max Risk Per Trade: ${self.futures_max_risk:.2f}")
            print(f"  Max Positions: {self.futures_max_positions}")
            print(f"  Max Total Risk: ${self.futures_max_total_risk:.2f}")
        else:
            self.futures_scanner = None

        # Initialize auto-execution engine
        self.auto_execute = auto_execute
        self.max_trades = max_trades

        if auto_execute:
            self.execution_engine = AutoExecutionEngine(
                paper_trading=True,
                max_risk_per_trade=500.0
            )
        else:
            self.execution_engine = None

        print("\n[SYSTEM READY] All AI-enhanced scanners initialized")
        if auto_execute:
            print("[AUTO-EXECUTION] Enabled - System will trade autonomously")

    def run_morning_scan(self):
        """Run complete morning scan for all assets"""

        print("\n" + "="*70)
        print("RUNNING MORNING SCAN - AI-ENHANCED")
        print("="*70)

        # Scan Options
        print("\n" + "-"*70)
        print("1. OPTIONS SCAN")
        print("-"*70)

        options_opportunities = self.options_scanner.scan_options([
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'NVDA', 'META', 'SPY', 'QQQ', 'IWM'
        ])

        self.options_scanner.display_opportunities(options_opportunities, top_n=3)

        # Scan Forex
        print("\n" + "-"*70)
        print("2. FOREX SCAN")
        print("-"*70)

        forex_opportunities = self.forex_scanner.scan_forex_pairs(['EUR_USD'])
        self.forex_scanner.display_opportunities(forex_opportunities, top_n=3)

        # Scan Futures (if enabled)
        futures_opportunities = []
        if self.enable_futures and self.futures_scanner:
            print("\n" + "-"*70)
            print("3. FUTURES SCAN (MES/MNQ)")
            print("-"*70)

            futures_opportunities = self.futures_scanner.scan_all_futures()
            self.futures_scanner.display_opportunities(futures_opportunities, top_n=3)

        # Combined recommendations
        print("\n" + "="*70)
        print("MONDAY TRADING RECOMMENDATIONS")
        print("="*70)

        all_opportunities = []

        # Add top options
        for opp in options_opportunities[:2]:
            all_opportunities.append({
                **opp,
                'recommendation': f"Trade {opp['symbol']} Bull Put Spread"
            })

        # Add top forex
        for opp in forex_opportunities[:2]:
            all_opportunities.append({
                **opp,
                'recommendation': f"Trade {opp['symbol']} {opp['direction']}"
            })

        # Add top futures (if enabled)
        if self.enable_futures:
            for opp in futures_opportunities[:2]:
                all_opportunities.append({
                    **opp,
                    'recommendation': f"Trade {opp['symbol']} {opp['direction']}"
                })

        # Sort by final score
        all_opportunities.sort(key=lambda x: x['final_score'], reverse=True)

        # Display top recommendations
        print(f"\nTOP TRADE RECOMMENDATIONS FOR TODAY:\n")

        for i, opp in enumerate(all_opportunities[:5], 1):
            print(f"{i}. {opp['recommendation']}")
            print(f"   Asset: {opp['asset_type']} | Score: {opp['final_score']:.2f} | Confidence: {opp['confidence']:.0%}")

            if opp['asset_type'] == 'FOREX':
                print(f"   Entry: {opp['entry']:.5f} | Stop: {opp['stop']:.5f} | Target: {opp['target']:.5f}")
            else:
                print(f"   Price: ${opp['price']:.2f} | Momentum: {opp['momentum']:+.1%}")

            print()

        # Trading plan
        print("\n" + "="*70)
        print("TODAY'S TRADING PLAN")
        print("="*70)

        print("\n6:30 AM - Market Open:")
        print("  1. Execute top 1-2 Bull Put Spreads (if score 8.0+)")
        print("  2. Monitor EUR/USD for forex entry (if score 9.0+)")
        print("  3. Paper trade only - track ALL outcomes")

        print("\nRisk Management:")
        print("  - Max $500 per options trade")
        print("  - Max 2-3 forex positions (OANDA practice)")
        print("  - Follow stops strictly")

        print("\nEnd of Day (1:00 PM PT):")
        print("  - Journal all trades")
        print("  - Record outcomes for AI learning")
        print("  - Review what worked")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'timestamp': timestamp,
            'options_opportunities': len(options_opportunities),
            'forex_opportunities': len(forex_opportunities),
            'futures_opportunities': len(futures_opportunities) if self.enable_futures else 0,
            'futures_enabled': self.enable_futures,
            'top_recommendations': [
                {
                    'symbol': opp['symbol'],
                    'asset_type': opp['asset_type'],
                    'score': opp['final_score'],
                    'confidence': opp['confidence']
                }
                for opp in all_opportunities[:5]
            ]
        }

        filename = f'monday_ai_scan_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n[SAVED] Scan results: {filename}")

        # AUTO-EXECUTION
        if self.auto_execute and self.execution_engine and all_opportunities:
            print("\n" + "="*70)
            print("AUTONOMOUS EXECUTION - TRADING NOW")
            print("="*70)

            executed_trades = self.execution_engine.auto_execute_opportunities(
                all_opportunities,
                max_trades=self.max_trades
            )

            if executed_trades:
                print("\n" + "="*70)
                print(f"EXECUTION SUMMARY")
                print("="*70)
                for i, trade in enumerate(executed_trades, 1):
                    print(f"\n{i}. {trade['symbol']} {trade['strategy']}")
                    print(f"   Asset: {trade['asset_type']}")
                    print(f"   Status: {trade['status']}")
                    print(f"   AI Score: {trade['ai_score']:.2f}")
                    print(f"   AI Confidence: {trade['ai_confidence']:.0%}")

                    if trade['asset_type'] == 'OPTIONS':
                        print(f"   Credit Collected: ${trade['credit']:.2f}")
                        print(f"   Max Risk: ${trade['max_risk']:.2f}")
                    elif trade['asset_type'] == 'FOREX':
                        print(f"   Entry: {trade['entry_price']:.5f}")
                        print(f"   Stop: {trade['stop_loss']:.5f}")
                        print(f"   Target: {trade['take_profit']:.5f}")

                print("\n" + "="*70)
                print("TRADES EXECUTED - POSITIONS NOW ACTIVE")
                print("="*70)
            else:
                print("\n[NO EXECUTIONS] No trades met execution criteria")

        return all_opportunities

    def record_trade(self, symbol: str, asset_type: str, success: bool, return_pct: float):
        """
        Record trade outcome for AI learning

        Call this at end of day with your trade results!

        Example:
            system.record_trade('AAPL', 'OPTIONS', True, 0.087)  # 8.7% return
            system.record_trade('EUR_USD', 'FOREX', False, -0.015)  # -1.5% loss
        """

        print(f"\n[RECORDING TRADE] {symbol} ({asset_type})")

        if asset_type == 'OPTIONS':
            self.options_scanner.record_trade_outcome(symbol, success, return_pct)
        elif asset_type == 'FOREX':
            self.forex_scanner.record_trade_outcome(symbol, success, return_pct)

        # Save AI learning data
        self.forex_scanner.ai_enhancer.save_learning_data()
        self.options_scanner.ai_enhancer.save_learning_data()

        print(f"[AI LEARNING] System updated with trade outcome")

    def show_ai_performance(self):
        """Show AI learning performance"""

        print("\n" + "="*70)
        print("AI LEARNING PERFORMANCE")
        print("="*70)

        # Forex AI performance
        forex_perf = self.forex_scanner.ai_enhancer.get_performance_summary()
        print(f"\nFOREX AI:")
        print(f"  Total outcomes: {forex_perf.get('total_outcomes', 0)}")
        if forex_perf.get('win_rate'):
            print(f"  Win rate: {forex_perf['win_rate']:.1%}")
            print(f"  Avg return: {forex_perf['avg_return']:.1%}")

        # Options AI performance
        options_perf = self.options_scanner.ai_enhancer.get_performance_summary()
        print(f"\nOPTIONS AI:")
        print(f"  Total outcomes: {options_perf.get('total_outcomes', 0)}")
        if options_perf.get('win_rate'):
            print(f"  Win rate: {options_perf['win_rate']:.1%}")
            print(f"  Avg return: {options_perf['avg_return']:.1%}")

        print(f"\n{'='*70}")


def main():
    """Monday morning trading workflow - AUTONOMOUS by default"""

    # Check for command-line arguments
    auto_execute = True  # Default: AUTONOMOUS
    max_trades = 2
    enable_futures = '--futures' in sys.argv  # Enable with --futures flag

    if '--manual' in sys.argv:
        auto_execute = False
        print("\n[MANUAL MODE] Auto-execution disabled")

    if '--max-trades' in sys.argv:
        idx = sys.argv.index('--max-trades')
        if idx + 1 < len(sys.argv):
            max_trades = int(sys.argv[idx + 1])

    # Initialize - AUTONOMOUS by default
    system = MondayAITrading(auto_execute=auto_execute, max_trades=max_trades, enable_futures=enable_futures)

    # Run morning scan (will auto-execute if enabled)
    opportunities = system.run_morning_scan()

    # Show AI performance
    system.show_ai_performance()

    # Instructions
    print("\n" + "="*70)
    print("SESSION COMPLETE")
    print("="*70)

    if auto_execute:
        print("\n[AUTONOMOUS MODE] System executed trades automatically")
        print("\nMonitor positions throughout the day:")
        print("  - Options close at 1:00 PM PT (market close)")
        print("  - Forex trades 24/5 (can close anytime)")
        print("\nAt end of day, record outcomes:")
        print("  system.record_trade('META', 'OPTIONS', True, 0.085)")
        print("  system.record_trade('EUR_USD', 'FOREX', True, 0.025)")
        print("\nAI learns from your outcomes and improves!")
    else:
        print("\n[MANUAL MODE] Review recommendations and execute manually")
        print("\n1. Review top recommendations above")
        print("2. Execute 1-2 highest-scoring trades (paper trading)")
        print("3. At end of day, record outcomes for AI learning")

    print("\n" + "="*70)
    print("AI-Enhanced Autonomous Trading System")
    print("Scan > Score > Execute > Learn > Repeat")
    print("="*70 + "\n")

    print("Usage Options:")
    print("  python MONDAY_AI_TRADING.py                    # Default: Options + Forex")
    print("  python MONDAY_AI_TRADING.py --futures          # Enable Futures (MES/MNQ)")
    print("  python MONDAY_AI_TRADING.py --manual           # Manual mode (no auto-execution)")
    print("  python MONDAY_AI_TRADING.py --max-trades 3     # Change max trades per session")
    print("  python MONDAY_AI_TRADING.py --futures --max-trades 4  # Combine flags\n")


if __name__ == "__main__":
    main()
