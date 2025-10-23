#!/usr/bin/env python3
"""
FUTURES BACKTESTING ENGINE
Test EMA crossover strategy on historical MES/MNQ data

Tests:
- Last 6 months of data
- MES and MNQ performance
- Win rate, profit factor, total P&L
- Per-contract statistics

Target: 60%+ win rate
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.futures_data_fetcher import FuturesDataFetcher, MICRO_FUTURES
from strategies.futures_ema_strategy import FuturesEMAStrategy
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict


class FuturesBacktest:
    """
    Backtest futures EMA strategy

    Features:
    - Tests on historical data
    - Simulates real trades
    - Calculates comprehensive statistics
    - Per-symbol breakdown
    """

    def __init__(self):
        print("\n[FUTURES BACKTEST] Initializing...")

        self.data_fetcher = FuturesDataFetcher(paper_trading=True)
        self.strategy = FuturesEMAStrategy()

        # Trading parameters
        self.max_risk_per_trade = 500.0  # Max $500 risk per contract
        self.initial_balance = 10000.0   # Starting capital
        self.commission_per_contract = 0.62  # Typical micro futures commission

        print("[FUTURES BACKTEST] Ready\n")

    def run_backtest(self, symbol: str, timeframe: str = '15Min', lookback_days: int = 180) -> Dict:
        """
        Run backtest on single symbol

        Args:
            symbol: Futures symbol (MES, MNQ)
            timeframe: Candle timeframe
            lookback_days: Days of history to test

        Returns:
            Backtest results dict
        """

        print(f"\n{'='*70}")
        print(f"BACKTESTING: {symbol} - {MICRO_FUTURES[symbol]['name']}")
        print(f"{'='*70}")
        print(f"Timeframe: {timeframe}")
        print(f"Lookback: {lookback_days} days")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"{'='*70}\n")

        # Fetch historical data
        print("[1/4] Fetching historical data...")
        df = self.data_fetcher.get_bars(symbol, timeframe, limit=5000)

        if df is None or df.empty:
            print(f"[ERROR] No data available for {symbol}")
            return None

        print(f"  Retrieved {len(df)} candles")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")

        # Filter to lookback period
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        df = df[df.index >= cutoff_date]

        print(f"  Testing on {len(df)} candles ({lookback_days} days)")

        # Run strategy on each candle
        print("\n[2/4] Scanning for signals...")

        trades = []
        balance = self.initial_balance
        equity_curve = [balance]

        # Need at least 200 candles for EMA trend
        for i in range(200, len(df)):
            # Get historical data up to this point
            hist_df = df.iloc[:i+1].copy()

            # Analyze for opportunity
            opportunity = self.strategy.analyze_opportunity(hist_df, symbol)

            if opportunity:
                # Validate rules
                if self.strategy.validate_rules(opportunity):
                    # Calculate position size (max 1 contract for backtest)
                    contracts = 1

                    # Simulate trade
                    entry_price = opportunity['entry_price']
                    stop_loss = opportunity['stop_loss']
                    take_profit = opportunity['take_profit']
                    direction = opportunity['direction']

                    # Track forward to see if hit stop or target
                    trade_result = self._simulate_trade(
                        df.iloc[i+1:],
                        entry_price,
                        stop_loss,
                        take_profit,
                        direction,
                        contracts,
                        MICRO_FUTURES[symbol]['point_value']
                    )

                    if trade_result:
                        trades.append(trade_result)
                        balance += trade_result['profit_loss']
                        equity_curve.append(balance)

        print(f"  Found {len(trades)} trades")

        # Calculate statistics
        print("\n[3/4] Calculating statistics...")
        results = self._calculate_statistics(trades, symbol, equity_curve)

        # Display results
        print("\n[4/4] Results:")
        self._display_results(results)

        return results

    def _simulate_trade(self, future_df: pd.DataFrame, entry: float, stop: float,
                       target: float, direction: str, contracts: int, point_value: float) -> Dict:
        """
        Simulate trade to find outcome

        Returns:
            Trade result dict or None
        """

        if future_df.empty:
            return None

        entry_time = future_df.index[0]

        # Check each subsequent candle
        for timestamp, candle in future_df.iterrows():
            # Check if stop hit
            if direction == 'LONG':
                if candle['low'] <= stop:
                    # Stop hit
                    points = stop - entry
                    profit_loss = points * point_value * contracts
                    profit_loss -= self.commission_per_contract * contracts * 2  # Round-trip commission

                    return {
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'direction': direction,
                        'entry_price': entry,
                        'exit_price': stop,
                        'stop_loss': stop,
                        'take_profit': target,
                        'contracts': contracts,
                        'points': points,
                        'profit_loss': profit_loss,
                        'outcome': 'LOSS',
                        'hit_type': 'STOP'
                    }

                # Check if target hit
                elif candle['high'] >= target:
                    # Target hit
                    points = target - entry
                    profit_loss = points * point_value * contracts
                    profit_loss -= self.commission_per_contract * contracts * 2

                    return {
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'direction': direction,
                        'entry_price': entry,
                        'exit_price': target,
                        'stop_loss': stop,
                        'take_profit': target,
                        'contracts': contracts,
                        'points': points,
                        'profit_loss': profit_loss,
                        'outcome': 'WIN',
                        'hit_type': 'TARGET'
                    }

            else:  # SHORT
                if candle['high'] >= stop:
                    # Stop hit
                    points = entry - stop
                    profit_loss = points * point_value * contracts
                    profit_loss -= self.commission_per_contract * contracts * 2

                    return {
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'direction': direction,
                        'entry_price': entry,
                        'exit_price': stop,
                        'stop_loss': stop,
                        'take_profit': target,
                        'contracts': contracts,
                        'points': points,
                        'profit_loss': profit_loss,
                        'outcome': 'LOSS',
                        'hit_type': 'STOP'
                    }

                # Check if target hit
                elif candle['low'] <= target:
                    # Target hit
                    points = entry - target
                    profit_loss = points * point_value * contracts
                    profit_loss -= self.commission_per_contract * contracts * 2

                    return {
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'direction': direction,
                        'entry_price': entry,
                        'exit_price': target,
                        'stop_loss': stop,
                        'take_profit': target,
                        'contracts': contracts,
                        'points': points,
                        'profit_loss': profit_loss,
                        'outcome': 'WIN',
                        'hit_type': 'TARGET'
                    }

        # If we get here, trade wasn't closed (end of data)
        return None

    def _calculate_statistics(self, trades: List[Dict], symbol: str, equity_curve: List[float]) -> Dict:
        """Calculate comprehensive statistics"""

        if not trades:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0
            }

        # Basic stats
        total_trades = len(trades)
        wins = [t for t in trades if t['outcome'] == 'WIN']
        losses = [t for t in trades if t['outcome'] == 'LOSS']

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        # P&L stats
        total_pnl = sum(t['profit_loss'] for t in trades)
        gross_profit = sum(t['profit_loss'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['profit_loss'] for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average stats
        avg_win = sum(t['profit_loss'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['profit_loss'] for t in losses) / len(losses) if losses else 0

        # Max drawdown
        max_equity = self.initial_balance
        max_drawdown = 0
        for equity in equity_curve:
            if equity > max_equity:
                max_equity = equity
            drawdown = max_equity - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        max_drawdown_pct = (max_drawdown / self.initial_balance) * 100 if self.initial_balance > 0 else 0

        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'wins': win_count,
            'losses': loss_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'final_balance': self.initial_balance + total_pnl,
            'return_pct': (total_pnl / self.initial_balance) * 100,
            'trades': trades
        }

    def _display_results(self, results: Dict):
        """Display backtest results"""

        print(f"\n{'='*70}")
        print(f"BACKTEST RESULTS: {results['symbol']}")
        print(f"{'='*70}")

        print(f"\nTRADE STATISTICS:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Wins: {results['wins']}")
        print(f"  Losses: {results['losses']}")
        print(f"  Win Rate: {results['win_rate']:.1%}")

        print(f"\nP&L STATISTICS:")
        print(f"  Total P&L: ${results['total_pnl']:,.2f}")
        print(f"  Gross Profit: ${results['gross_profit']:,.2f}")
        print(f"  Gross Loss: ${results['gross_loss']:,.2f}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")

        print(f"\nAVERAGE TRADE:")
        print(f"  Average Win: ${results['avg_win']:,.2f}")
        print(f"  Average Loss: ${results['avg_loss']:,.2f}")
        print(f"  Avg Win/Loss Ratio: {abs(results['avg_win'] / results['avg_loss']):.2f}:1" if results['avg_loss'] != 0 else "N/A")

        print(f"\nRISK METRICS:")
        print(f"  Max Drawdown: ${results['max_drawdown']:,.2f} ({results['max_drawdown_pct']:.1f}%)")

        print(f"\nFINAL RESULTS:")
        print(f"  Starting Balance: ${self.initial_balance:,.2f}")
        print(f"  Ending Balance: ${results['final_balance']:,.2f}")
        print(f"  Return: {results['return_pct']:+.1f}%")

        print(f"\n{'='*70}")

        # Verdict
        if results['win_rate'] >= 0.60 and results['profit_factor'] >= 1.5:
            print("VERDICT: EXCELLENT - Strategy meets targets!")
        elif results['win_rate'] >= 0.55 and results['profit_factor'] >= 1.3:
            print("VERDICT: GOOD - Strategy is profitable")
        elif results['win_rate'] >= 0.50 and results['profit_factor'] >= 1.0:
            print("VERDICT: ACCEPTABLE - Strategy is breakeven+")
        else:
            print("VERDICT: NEEDS OPTIMIZATION")

        print(f"{'='*70}\n")

    def run_full_backtest(self) -> Dict:
        """
        Run backtest on all futures contracts

        Returns:
            Combined results
        """

        print("\n" + "="*70)
        print("FULL FUTURES BACKTEST")
        print("="*70)
        print("Testing: MES, MNQ")
        print("Period: Last 6 months")
        print("="*70)

        all_results = {}

        for symbol in ['MES', 'MNQ']:
            results = self.run_backtest(symbol, timeframe='15Min', lookback_days=180)
            if results:
                all_results[symbol] = results

        # Combined summary
        if all_results:
            print("\n" + "="*70)
            print("COMBINED SUMMARY")
            print("="*70)

            total_trades = sum(r['total_trades'] for r in all_results.values())
            total_wins = sum(r['wins'] for r in all_results.values())
            combined_win_rate = total_wins / total_trades if total_trades > 0 else 0

            total_pnl = sum(r['total_pnl'] for r in all_results.values())
            total_gross_profit = sum(r['gross_profit'] for r in all_results.values())
            total_gross_loss = sum(r['gross_loss'] for r in all_results.values())
            combined_profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf')

            print(f"\nOVERALL STATISTICS:")
            print(f"  Total Trades: {total_trades}")
            print(f"  Combined Win Rate: {combined_win_rate:.1%}")
            print(f"  Combined Profit Factor: {combined_profit_factor:.2f}")
            print(f"  Total P&L: ${total_pnl:,.2f}")

            print(f"\nPER-SYMBOL BREAKDOWN:")
            for symbol, results in all_results.items():
                print(f"\n  {symbol}:")
                print(f"    Trades: {results['total_trades']}")
                print(f"    Win Rate: {results['win_rate']:.1%}")
                print(f"    P&L: ${results['total_pnl']:,.2f}")

            print(f"\n{'='*70}")

            # Overall verdict
            print("\nOVERALL VERDICT:")
            if combined_win_rate >= 0.60 and combined_profit_factor >= 1.5:
                print("  STATUS: READY FOR DEPLOYMENT")
                print("  Quality: Meets all targets")
            elif combined_win_rate >= 0.55:
                print("  STATUS: ACCEPTABLE")
                print("  Quality: Profitable but could improve")
            else:
                print("  STATUS: NEEDS WORK")
                print("  Quality: Requires optimization")

            print(f"{'='*70}\n")

        return all_results


def main():
    """Run futures backtest"""

    backtest = FuturesBacktest()
    results = backtest.run_full_backtest()

    if results:
        print("\n[SUCCESS] Backtest complete")
        print("Results saved in output above")
    else:
        print("\n[ERROR] Backtest failed")


if __name__ == "__main__":
    main()
