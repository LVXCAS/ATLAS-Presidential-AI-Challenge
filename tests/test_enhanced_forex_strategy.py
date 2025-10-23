#!/usr/bin/env python3
"""
BACKTEST ENHANCED FOREX EMA STRATEGY v3.0
Test on 90 days of recent data for EUR/USD, GBP/USD, USD/JPY

TARGET: 60%+ win rate on all pairs

BEFORE (v2.0 - 4H timeframe):
- EUR/USD: 51.7% WR
- GBP/USD: 48.3% WR
- USD/JPY: 63.3% WR (but only 4H)
- Overall: 54.5% WR

AFTER (v3.0 - Enhanced with all filters):
- Target: 60%+ on all pairs
- Fix: USD/JPY pip calculation
- Add: Volume filter, MTF confirm, stricter RSI, dynamic stops
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.forex_ema_strategy import ForexEMAStrategy
from data.oanda_data_fetcher import OandaDataFetcher


class ForexStrategyBacktester:
    """
    Comprehensive backtester for forex strategies

    Features:
    - Multi-pair testing
    - Correct pip calculation for all pairs
    - Detailed performance metrics
    - Trade-by-trade analysis
    """

    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []

    def backtest_pair(self, strategy: ForexEMAStrategy, df: pd.DataFrame,
                     symbol: str, verbose: bool = False) -> Dict:
        """
        Backtest strategy on a single forex pair

        Args:
            strategy: Trading strategy instance
            df: Historical price data (1-hour bars)
            symbol: Forex pair (e.g., 'EUR_USD')
            verbose: Print detailed trade info

        Returns:
            Performance metrics dict
        """

        if df is None or df.empty:
            return self._empty_results(symbol)

        # Reset index if needed
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        trades = []
        wins = 0
        losses = 0
        total_pips = 0
        total_profit = 0

        # Walk forward through data
        lookback = 220  # Need 200+ bars for 200 EMA

        if verbose:
            print(f"\n[BACKTEST] {symbol}")
            print(f"  Data: {df.index[0]} to {df.index[-1]}")
            print(f"  Bars: {len(df)}")

        for i in range(lookback, len(df)):
            # Get historical data up to current point
            historical_data = df.iloc[:i+1].copy()

            # Check for signal
            opportunity = strategy.analyze_opportunity(historical_data, symbol)

            if opportunity is None:
                continue

            # Validate
            if not strategy.validate_rules(opportunity):
                continue

            # Simulate trade execution
            entry_price = opportunity['entry_price']
            stop_loss = opportunity['stop_loss']
            take_profit = opportunity['take_profit']
            direction = opportunity['direction']
            entry_time = df.index[i]

            # Look forward to find exit
            exit_price = None
            exit_time = None
            exit_reason = None

            for j in range(i+1, min(i+100, len(df))):  # Max 100 bars (100 hours)
                bar = df.iloc[j]

                if direction == 'LONG':
                    # Check stop loss
                    if bar['low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_time = df.index[j]
                        exit_reason = 'STOP_LOSS'
                        break
                    # Check take profit
                    if bar['high'] >= take_profit:
                        exit_price = take_profit
                        exit_time = df.index[j]
                        exit_reason = 'TAKE_PROFIT'
                        break
                else:  # SHORT
                    # Check stop loss
                    if bar['high'] >= stop_loss:
                        exit_price = stop_loss
                        exit_time = df.index[j]
                        exit_reason = 'STOP_LOSS'
                        break
                    # Check take profit
                    if bar['low'] <= take_profit:
                        exit_price = take_profit
                        exit_time = df.index[j]
                        exit_reason = 'TAKE_PROFIT'
                        break

            # If no exit found, skip trade
            if exit_price is None:
                continue

            # Calculate profit/loss
            if direction == 'LONG':
                price_change = exit_price - entry_price
            else:
                price_change = entry_price - exit_price

            # Calculate pips correctly
            pips = strategy.calculate_pips(symbol, price_change)

            # Calculate P&L (assume $10 per pip for standard lot)
            profit = pips * 10  # $10 per pip

            # Update stats
            total_pips += pips
            total_profit += profit

            if pips > 0:
                wins += 1
            else:
                losses += 1

            trade = {
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pips': pips,
                'profit': profit,
                'exit_reason': exit_reason,
                'score': opportunity['score']
            }

            trades.append(trade)

            if verbose and len(trades) <= 5:  # Show first 5 trades
                print(f"\n  Trade #{len(trades)}:")
                print(f"    Entry: {entry_time} @ {entry_price:.5f} ({direction})")
                print(f"    Exit: {exit_time} @ {exit_price:.5f} ({exit_reason})")
                print(f"    P&L: {pips:+.1f} pips (${profit:+,.2f})")

        # Calculate metrics
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # Calculate profit factor
        winning_pips = sum(t['pips'] for t in trades if t['pips'] > 0)
        losing_pips = abs(sum(t['pips'] for t in trades if t['pips'] < 0))
        profit_factor = (winning_pips / losing_pips) if losing_pips > 0 else 0

        # Calculate average win/loss
        avg_win = np.mean([t['pips'] for t in trades if t['pips'] > 0]) if wins > 0 else 0
        avg_loss = np.mean([t['pips'] for t in trades if t['pips'] < 0]) if losses > 0 else 0

        results = {
            'symbol': symbol,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'total_profit': total_profit,
            'profit_factor': profit_factor,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'trades': trades
        }

        if verbose:
            print(f"\n[RESULTS] {symbol}")
            print(f"  Total Trades: {total_trades}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Total P&L: {total_pips:+,.1f} pips (${total_profit:+,.2f})")
            print(f"  Profit Factor: {profit_factor:.2f}x")
            print(f"  Avg Win: {avg_win:.1f} pips | Avg Loss: {avg_loss:.1f} pips")

        return results

    def _empty_results(self, symbol: str) -> Dict:
        """Return empty results dict"""
        return {
            'symbol': symbol,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pips': 0,
            'total_profit': 0,
            'profit_factor': 0,
            'avg_win_pips': 0,
            'avg_loss_pips': 0,
            'trades': []
        }

    def print_summary(self, all_results: List[Dict]):
        """Print comprehensive summary"""

        print("\n" + "="*80)
        print("ENHANCED FOREX EMA STRATEGY v3.0 - BACKTEST RESULTS")
        print("="*80)

        # Individual pair results
        for result in all_results:
            symbol = result['symbol']
            wr = result['win_rate']
            trades = result['total_trades']
            pips = result['total_pips']
            pf = result['profit_factor']

            status = "PASS" if wr >= 60 else "FAIL"

            print(f"\n{symbol}:")
            print(f"  Trades: {trades}")
            print(f"  Win Rate: {wr:.1f}% [{status}]")
            print(f"  Total P&L: {pips:+,.1f} pips")
            print(f"  Profit Factor: {pf:.2f}x")

        # Overall stats
        total_trades = sum(r['total_trades'] for r in all_results)
        total_wins = sum(r['wins'] for r in all_results)
        total_pips = sum(r['total_pips'] for r in all_results)
        total_profit = sum(r['total_profit'] for r in all_results)
        overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

        print(f"\n" + "-"*80)
        print(f"OVERALL PERFORMANCE:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Overall Win Rate: {overall_wr:.1f}%")
        print(f"  Total P&L: {total_pips:+,.1f} pips (${total_profit:+,.2f})")

        # Check if targets met
        all_pass = all(r['win_rate'] >= 60 for r in all_results)
        overall_pass = overall_wr >= 65

        print(f"\n" + "="*80)
        print(f"TARGET ASSESSMENT:")
        print(f"  EUR/USD 60%+: [{'PASS' if all_results[0]['win_rate'] >= 60 else 'FAIL'}]")
        print(f"  GBP/USD 60%+: [{'PASS' if all_results[1]['win_rate'] >= 60 else 'FAIL'}]")
        print(f"  USD/JPY 60%+: [{'PASS' if all_results[2]['win_rate'] >= 60 else 'FAIL'}]")
        print(f"  Overall 65%+: [{'PASS' if overall_pass else 'FAIL'}]")

        print(f"\n" + "="*80)
        if all_pass and overall_pass:
            print("RECOMMENDATION: [READY TO TRADE]")
            print("Strategy meets all performance targets. Proceed with paper trading.")
        else:
            print("RECOMMENDATION: [NEEDS MORE WORK]")
            print("Strategy needs further optimization. Review filters and parameters.")
        print("="*80)


def main():
    """Run comprehensive backtest on 3 major pairs"""

    print("\n" + "="*80)
    print("BACKTESTING ENHANCED FOREX EMA STRATEGY v3.0")
    print("="*80)
    print("\nENHANCEMENTS:")
    print("  1. Volume/Activity Filter")
    print("  2. Multi-Timeframe Confirmation (4H trend)")
    print("  3. Stricter Entry Conditions (RSI bounds)")
    print("  4. Dynamic ATR-based Stops")
    print("  5. FIXED USD/JPY Pip Calculation")
    print("\nTARGET: 60%+ win rate on all pairs")
    print("="*80)

    # Initialize
    data_fetcher = OandaDataFetcher(practice=True)

    if not data_fetcher.api:
        print("\n[ERROR] OANDA API not available")
        print("This is a demo showing the framework.")
        print("\nTo run actual backtest:")
        print("1. Get free OANDA practice account: https://www.oanda.com/us-en/trading/")
        print("2. Add OANDA_API_KEY and OANDA_ACCOUNT_ID to .env file")
        print("3. Run this script again")
        return

    strategy = ForexEMAStrategy()
    strategy.set_data_fetcher(data_fetcher)  # For MTF confirmation

    backtester = ForexStrategyBacktester()

    # Test pairs
    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']

    all_results = []

    for pair in pairs:
        print(f"\n{'='*80}")
        print(f"Fetching data for {pair}...")

        # Fetch 90 days of 1-hour data
        # 90 days * 24 hours = 2160 hours, fetch 2500 to be safe
        df = data_fetcher.get_bars(pair, timeframe='H1', limit=2500)

        if df is None or df.empty:
            print(f"[ERROR] No data for {pair}")
            all_results.append(backtester._empty_results(pair))
            continue

        print(f"Fetched {len(df)} bars for {pair}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")

        # Run backtest
        result = backtester.backtest_pair(strategy, df, pair, verbose=True)
        all_results.append(result)

    # Print comprehensive summary
    backtester.print_summary(all_results)


if __name__ == "__main__":
    main()
