#!/usr/bin/env python3
"""
QUICK FOREX BACKTEST
Shows what trades the EMA Crossover strategy would have found
Tests on last 3 months of real OANDA data
"""

from data.oanda_data_fetcher import OandaDataFetcher
from strategies.forex.ema_rsi_crossover_optimized import EMACrossoverOptimized
import pandas as pd

def run_quick_backtest(pair='EUR_USD', candles=500, timeframe='H4'):
    """
    Quick backtest of EMA Crossover strategy

    Args:
        pair: Forex pair to test
        candles: Number of candles to backtest
        timeframe: Timeframe (H4=4-hour, D=daily)
    """

    print("\n" + "="*70)
    print(f"BACKTESTING: {pair}")
    print(f"Timeframe: {timeframe}")
    print(f"Data: Last {candles} candles")
    print("="*70)

    # Initialize
    fetcher = OandaDataFetcher()
    strategy = EMACrossoverOptimized()

    # Fetch data
    print(f"\n[1/4] Fetching {candles} {timeframe} candles from OANDA...")
    df = fetcher.get_bars(pair, timeframe, limit=candles)

    if df is None or df.empty:
        print(f"[ERROR] Could not fetch data for {pair}")
        return None

    df = df.reset_index()
    print(f"[OK] Fetched {len(df)} candles")
    print(f"      Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Run backtest
    print(f"\n[2/4] Scanning for EMA Crossover signals...")

    trades = []
    in_position = False
    current_trade = None

    # Start after 200 candles (need data for 200 EMA)
    for i in range(250, len(df)):
        window = df.iloc[i-250:i].copy()

        # Check for signal
        opp = strategy.analyze_opportunity(window, pair)

        if opp and strategy.validate_rules(opp):
            # Found a signal
            if not in_position:
                # Enter trade
                current_trade = {
                    'entry_time': window['timestamp'].iloc[-1],
                    'entry_price': opp['entry_price'],
                    'direction': opp['direction'],
                    'stop_loss': opp['stop_loss'],
                    'take_profit': opp['take_profit'],
                    'score': opp['score'],
                    'rsi': opp['indicators']['rsi']
                }
                in_position = True

        # Check if trade should close
        if in_position and current_trade:
            current_price = df.iloc[i]['close']

            # Check stop loss and take profit
            if current_trade['direction'] == 'LONG':
                if current_price <= current_trade['stop_loss']:
                    # Stop loss hit - FIX: Use correct pip multiplier for pair
                    pip_multiplier = 100 if 'JPY' in pair else 10000
                    current_trade['exit_time'] = df.iloc[i]['timestamp']
                    current_trade['exit_price'] = current_trade['stop_loss']
                    current_trade['outcome'] = 'LOSS'
                    current_trade['profit_pips'] = (current_trade['exit_price'] - current_trade['entry_price']) * pip_multiplier
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None

                elif current_price >= current_trade['take_profit']:
                    # Take profit hit
                    pip_multiplier = 100 if 'JPY' in pair else 10000
                    current_trade['exit_time'] = df.iloc[i]['timestamp']
                    current_trade['exit_price'] = current_trade['take_profit']
                    current_trade['outcome'] = 'WIN'
                    current_trade['profit_pips'] = (current_trade['exit_price'] - current_trade['entry_price']) * pip_multiplier
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None

            else:  # SHORT
                if current_price >= current_trade['stop_loss']:
                    # Stop loss hit
                    pip_multiplier = 100 if 'JPY' in pair else 10000
                    current_trade['exit_time'] = df.iloc[i]['timestamp']
                    current_trade['exit_price'] = current_trade['stop_loss']
                    current_trade['outcome'] = 'LOSS'
                    current_trade['profit_pips'] = (current_trade['entry_price'] - current_trade['exit_price']) * pip_multiplier
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None

                elif current_price <= current_trade['take_profit']:
                    # Take profit hit
                    pip_multiplier = 100 if 'JPY' in pair else 10000
                    current_trade['exit_time'] = df.iloc[i]['timestamp']
                    current_trade['exit_price'] = current_trade['take_profit']
                    current_trade['outcome'] = 'WIN'
                    current_trade['profit_pips'] = (current_trade['entry_price'] - current_trade['exit_price']) * pip_multiplier
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None

    print(f"[OK] Found {len(trades)} completed trades")

    # Calculate statistics
    if len(trades) == 0:
        print("\n[NO TRADES] Strategy found no completed trades in this period")
        return None

    print(f"\n[3/4] Calculating performance metrics...")

    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']

    win_rate = len(wins) / len(trades) * 100
    total_pips = sum([t['profit_pips'] for t in trades])
    avg_win = sum([t['profit_pips'] for t in wins]) / len(wins) if wins else 0
    avg_loss = sum([t['profit_pips'] for t in losses]) / len(losses) if losses else 0

    print(f"[OK] Analysis complete")

    # Display results
    print(f"\n[4/4] BACKTEST RESULTS:")
    print("="*70)
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Wins: {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losses: {len(losses)} ({100-win_rate:.1f}%)")
    print(f"  Total Profit: {total_pips:.1f} pips")
    print(f"  Avg Win: {avg_win:.1f} pips")
    print(f"  Avg Loss: {avg_loss:.1f} pips")

    if avg_loss != 0:
        profit_factor = abs(sum([t['profit_pips'] for t in wins]) / sum([t['profit_pips'] for t in losses]))
        print(f"  Profit Factor: {profit_factor:.2f}x")

    # Show recent trades
    print(f"\nLAST 5 TRADES:")
    print("-"*70)

    for i, trade in enumerate(trades[-5:], 1):
        outcome_symbol = "[WIN]" if trade['outcome'] == 'WIN' else "[LOSS]"
        print(f"\n{i}. {outcome_symbol} {trade['direction']}")
        print(f"   Entry: {trade['entry_time']} @ {trade['entry_price']:.5f}")
        print(f"   Exit:  {trade['exit_time']} @ {trade['exit_price']:.5f}")
        print(f"   Profit: {trade['profit_pips']:.1f} pips")
        print(f"   Score: {trade['score']:.1f}, RSI: {trade['rsi']:.1f}")

    print("\n" + "="*70)

    return {
        'trades': trades,
        'win_rate': win_rate,
        'total_pips': total_pips,
        'wins': len(wins),
        'losses': len(losses)
    }


def main():
    """Run quick backtest on multiple pairs"""

    print("\n" + "="*70)
    print("QUICK FOREX BACKTEST - OPTIMIZED STRATEGY v2.0")
    print("Testing on 4-HOUR timeframe with 8/21/200 RSI55/45")
    print("Previous: 41.8% WR on 1H | Target: 54.5% WR on 4H")
    print("="*70)

    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']

    all_results = {}

    for pair in pairs:
        result = run_quick_backtest(pair, candles=500, timeframe='H4')
        if result:
            all_results[pair] = result
        print("\n")

    # Summary
    if all_results:
        print("="*70)
        print("OVERALL SUMMARY")
        print("="*70)

        total_trades = sum([r['wins'] + r['losses'] for r in all_results.values()])
        total_wins = sum([r['wins'] for r in all_results.values()])
        overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        total_pips = sum([r['total_pips'] for r in all_results.values()])

        print(f"\nAcross {len(all_results)} pairs:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Overall Win Rate: {overall_win_rate:.1f}%")
        print(f"  Total Profit: {total_pips:.1f} pips")
        print(f"  Avg per pair: {total_pips/len(all_results):.1f} pips")

        print("\nPer-Pair Performance:")
        for pair, result in all_results.items():
            print(f"  {pair}: {result['win_rate']:.1f}% win rate, {result['total_pips']:.1f} pips")

        print("\n" + "="*70)

        if overall_win_rate >= 60:
            print("RESULT: EXCELLENT performance! Target exceeded! [SUCCESS]")
        elif overall_win_rate >= 54:
            print("RESULT: GOOD performance! Within expected range [OK]")
        elif overall_win_rate >= 50:
            print("RESULT: ACCEPTABLE performance [OK]")
        else:
            print("RESULT: NEEDS MORE OPTIMIZATION [WARNING]")

        print("="*70)

        print("\nNOTE: This is a backtest on 4-hour data with optimized parameters.")
        print("      Expected: 54.5% win rate (tested on 88 trades)")
        print("      Real trading will vary based on execution, slippage, etc.")
        print("      Paper trade for 30 days before going live.")

    else:
        print("[WARNING] No results generated. Try again later.")


if __name__ == "__main__":
    main()
