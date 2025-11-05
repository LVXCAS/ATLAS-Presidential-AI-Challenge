#!/usr/bin/env python3
"""
MULTI-STRATEGY OPTIONS SCANNER
===============================
Enhanced scanner supporting multiple options strategies:
1. Bull Put Spread (Current - 60-65% WR)
2. Adaptive Dual Options (Current - Adaptive entry)
3. Iron Condor (NEW - 70-80% WR, neutral markets)
4. Butterfly Spread (NEW - High R/R, range-bound)

USAGE:
    python multi_strategy_options_scanner.py                    # All strategies
    python multi_strategy_options_scanner.py --iron-condor-only # Iron Condor only
    python multi_strategy_options_scanner.py --butterfly-only   # Butterfly only
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict
import pandas as pd

# Import existing systems
from MONDAY_AI_TRADING import MondayAITrading
from strategies.iron_condor_engine import IronCondorEngine
from strategies.butterfly_spread_engine import ButterflySpreadEngine

class MultiStrategyOptionsScanner:
    """
    Multi-strategy options scanner

    Scans S&P 500 and applies multiple strategies based on market conditions:
    - Bull markets → Bull Put Spreads
    - Neutral/choppy → Iron Condors
    - Range-bound → Butterfly Spreads
    - Adaptive → Dual Options system
    """

    def __init__(
        self,
        enable_bull_put=True,
        enable_dual_options=True,
        enable_iron_condor=False,
        enable_butterfly=False,
        min_score=7.5
    ):
        self.enable_bull_put = enable_bull_put
        self.enable_dual_options = enable_dual_options
        self.enable_iron_condor = enable_iron_condor
        self.enable_butterfly = enable_butterfly
        self.min_score = min_score

        # Initialize engines
        self.monday_ai = MondayAITrading() if (enable_bull_put or enable_dual_options) else None
        self.iron_condor = IronCondorEngine() if enable_iron_condor else None
        self.butterfly = ButterflySpreadEngine() if enable_butterfly else None

        print("\n" + "="*80)
        print("MULTI-STRATEGY OPTIONS SCANNER")
        print("="*80)
        print("\nEnabled Strategies:")
        if enable_bull_put:
            print("  [X] Bull Put Spreads (60-65% WR, bullish bias)")
        if enable_dual_options:
            print("  [X] Adaptive Dual Options (Adaptive, high conviction)")
        if enable_iron_condor:
            print("  [X] Iron Condor (70-80% WR, neutral markets)")
        if enable_butterfly:
            print("  [X] Butterfly Spreads (High R/R, range-bound)")
        print(f"\nMin Score Threshold: {min_score}")
        print("="*80 + "\n")

    def scan_and_execute(self, max_positions=19):
        """
        Scan market and execute across all enabled strategies

        Args:
            max_positions: Maximum total positions across all strategies
        """

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting multi-strategy scan...")

        results = {
            'bull_put': [],
            'dual_options': [],
            'iron_condor': [],
            'butterfly': [],
            'total_executed': 0
        }

        # Get current position count
        try:
            from alpaca_trade_api import REST
            api = REST(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                os.getenv('ALPACA_BASE_URL'),
                api_version='v2'
            )
            current_positions = len(api.list_positions())
            print(f"Current Positions: {current_positions}/{max_positions}")

            if current_positions >= max_positions:
                print("[WARNING] Maximum positions reached. Skipping scan.")
                return results

        except Exception as e:
            print(f"[ERROR] Could not check positions: {e}")
            current_positions = 0

        # Calculate available slots
        available_slots = max_positions - current_positions
        print(f"Available Slots: {available_slots}\n")

        # Strategy 1 & 2: Bull Put + Dual Options (via Monday AI)
        if (self.enable_bull_put or self.enable_dual_options) and available_slots > 0:
            try:
                print("[SCANNING] Bull Put + Dual Options strategies...")

                # Run Monday AI scan
                opportunities = self.monday_ai.scan_sp500_for_opportunities(
                    min_score=self.min_score,
                    max_results=available_slots
                )

                print(f"  Found {len(opportunities)} opportunities")

                # Execute top opportunities
                for opp in opportunities[:available_slots]:
                    try:
                        # Determine strategy based on market conditions
                        # (Monday AI internally decides Bull Put vs Dual Options)

                        print(f"\n  [EXECUTE] {opp['symbol']} - Score: {opp.get('score', 0):.2f}")

                        # Execute via Monday AI (it handles strategy selection)
                        executed = self.monday_ai.execute_trade(opp)

                        if executed:
                            strategy_type = opp.get('strategy', 'bull_put')
                            if strategy_type == 'dual_options':
                                results['dual_options'].append(opp)
                            else:
                                results['bull_put'].append(opp)
                            results['total_executed'] += 1
                            available_slots -= 1

                    except Exception as e:
                        print(f"    [ERROR] Execution failed: {e}")

            except Exception as e:
                print(f"[ERROR] Monday AI scan failed: {e}")

        # Strategy 3: Iron Condor (neutral markets)
        if self.enable_iron_condor and available_slots > 0:
            try:
                print("\n[SCANNING] Iron Condor opportunities...")

                # Scan for neutral/range-bound stocks
                # (In production, would filter by low beta, consolidation patterns)

                # For now, execute on high-quality stocks with low volatility
                iron_condor_candidates = self._find_iron_condor_candidates(limit=available_slots)

                print(f"  Found {len(iron_condor_candidates)} neutral candidates")

                for candidate in iron_condor_candidates:
                    try:
                        print(f"\n  [EXECUTE] {candidate['symbol']} Iron Condor")

                        executed = self.iron_condor.execute_iron_condor(
                            symbol=candidate['symbol'],
                            current_price=candidate['price'],
                            contracts=1,
                            expiration_days=7
                        )

                        if executed:
                            results['iron_condor'].append(candidate)
                            results['total_executed'] += 1
                            available_slots -= 1

                    except Exception as e:
                        print(f"    [ERROR] Iron Condor execution failed: {e}")

            except Exception as e:
                print(f"[ERROR] Iron Condor scan failed: {e}")

        # Strategy 4: Butterfly Spread (range-bound)
        if self.enable_butterfly and available_slots > 0:
            try:
                print("\n[SCANNING] Butterfly Spread opportunities...")

                # Scan for range-bound stocks with upcoming catalysts
                butterfly_candidates = self._find_butterfly_candidates(limit=available_slots)

                print(f"  Found {len(butterfly_candidates)} range-bound candidates")

                for candidate in butterfly_candidates:
                    try:
                        print(f"\n  [EXECUTE] {candidate['symbol']} Butterfly Spread")

                        executed = self.butterfly.execute_butterfly(
                            symbol=candidate['symbol'],
                            current_price=candidate['price'],
                            contracts=1,
                            expiration_days=14
                        )

                        if executed:
                            results['butterfly'].append(candidate)
                            results['total_executed'] += 1
                            available_slots -= 1

                    except Exception as e:
                        print(f"    [ERROR] Butterfly execution failed: {e}")

            except Exception as e:
                print(f"[ERROR] Butterfly scan failed: {e}")

        # Summary
        print("\n" + "="*80)
        print("SCAN COMPLETE")
        print("="*80)
        print(f"Bull Put Spreads Executed: {len(results['bull_put'])}")
        print(f"Dual Options Executed: {len(results['dual_options'])}")
        print(f"Iron Condors Executed: {len(results['iron_condor'])}")
        print(f"Butterfly Spreads Executed: {len(results['butterfly'])}")
        print(f"Total Executed: {results['total_executed']}")
        print("="*80 + "\n")

        return results

    def _find_iron_condor_candidates(self, limit=5) -> List[Dict]:
        """
        Find stocks suitable for Iron Condor

        Ideal candidates:
        - Low volatility (IV < 30%)
        - Range-bound (consolidating)
        - Large cap (liquid options)
        """

        # For now, return top S&P 500 stocks
        # In production, would filter by:
        # - Historical volatility < 20%
        # - Bollinger Band width < 0.10
        # - Beta < 1.0

        candidates = []
        sp500_sample = ['AAPL', 'MSFT', 'GOOGL', 'META', 'JPM'][:limit]

        for symbol in sp500_sample:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                price = ticker.history(period='1d')['Close'].iloc[-1]

                candidates.append({
                    'symbol': symbol,
                    'price': price,
                    'strategy': 'iron_condor'
                })
            except:
                continue

        return candidates

    def _find_butterfly_candidates(self, limit=5) -> List[Dict]:
        """
        Find stocks suitable for Butterfly Spread

        Ideal candidates:
        - Tight trading range
        - Upcoming earnings/catalyst
        - High implied volatility (premium collection)
        """

        # For now, return sample
        # In production, would filter by:
        # - ATR / Price < 2%
        # - Earnings date 7-21 days out
        # - IV Rank > 50

        candidates = []
        sp500_sample = ['NVDA', 'TSLA', 'AMD', 'NFLX'][:limit]

        for symbol in sp500_sample:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                price = ticker.history(period='1d')['Close'].iloc[-1]

                candidates.append({
                    'symbol': symbol,
                    'price': price,
                    'strategy': 'butterfly'
                })
            except:
                continue

        return candidates


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description='Multi-Strategy Options Scanner')
    parser.add_argument('--iron-condor-only', action='store_true',
                       help='Only scan for Iron Condor opportunities')
    parser.add_argument('--butterfly-only', action='store_true',
                       help='Only scan for Butterfly opportunities')
    parser.add_argument('--no-bull-put', action='store_true',
                       help='Disable Bull Put spreads')
    parser.add_argument('--no-dual-options', action='store_true',
                       help='Disable Dual Options')
    parser.add_argument('--min-score', type=float, default=7.5,
                       help='Minimum score threshold (default: 7.5)')
    parser.add_argument('--max-positions', type=int, default=19,
                       help='Maximum total positions (default: 19)')

    args = parser.parse_args()

    # Determine which strategies to enable
    if args.iron_condor_only:
        enable_bull_put = False
        enable_dual_options = False
        enable_iron_condor = True
        enable_butterfly = False
    elif args.butterfly_only:
        enable_bull_put = False
        enable_dual_options = False
        enable_iron_condor = False
        enable_butterfly = True
    else:
        enable_bull_put = not args.no_bull_put
        enable_dual_options = not args.no_dual_options
        enable_iron_condor = True  # Enable by default for multi-strategy
        enable_butterfly = True     # Enable by default for multi-strategy

    # Create scanner
    scanner = MultiStrategyOptionsScanner(
        enable_bull_put=enable_bull_put,
        enable_dual_options=enable_dual_options,
        enable_iron_condor=enable_iron_condor,
        enable_butterfly=enable_butterfly,
        min_score=args.min_score
    )

    # Run scan
    results = scanner.scan_and_execute(max_positions=args.max_positions)

    print(f"\nScan complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total trades executed: {results['total_executed']}")


if __name__ == "__main__":
    main()
