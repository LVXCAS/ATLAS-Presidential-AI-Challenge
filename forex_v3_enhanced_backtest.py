#!/usr/bin/env python3
"""
FOREX V3 ENHANCED BACKTEST
Tests the enhanced v3.0 strategy with relaxed parameters
Target: 55%+ win rate on 1-hour timeframe
"""

from data.oanda_data_fetcher import OandaDataFetcher
from strategies.forex_ema_strategy import ForexEMAStrategy
import pandas as pd

def run_enhanced_backtest(pair='EUR_USD', candles=2000, timeframe='H1'):
    """
    Enhanced backtest of Forex EMA Strategy v3.0

    Args:
        pair: Forex pair to test
        candles: Number of candles to backtest
        timeframe: Timeframe (H1=1-hour, H4=4-hour)
    """

    print("\n" + "="*70)
    print(f"BACKTESTING: {pair}")
    print(f"Strategy: Enhanced EMA v3.0 (Relaxed Parameters)")
    print(f"Timeframe: {timeframe}")
    print(f"Data: Last {candles} candles")
    print("="*70)

    # Initialize
    fetcher = OandaDataFetcher()
    strategy = ForexEMAStrategy()
    strategy.set_data_fetcher(fetcher)

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
    print(f"\n[2/4] Scanning for signals with v3.0 enhanced filters...")

    trades = []
    in_position = False
    current_trade = None

    # Start after 250 candles (need data for indicators + MTF)
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
                    'rsi': opp['indicators']['rsi'],
                    'entry_idx': i
                }
                in_position = True

        # Check if trade should close
        if in_position and current_trade:
            current_price = df.iloc[i]['close']

            # Get correct pip multiplier
            pip_multiplier = 100 if 'JPY' in pair else 10000

            # Check stop loss and take profit
            if current_trade['direction'] == 'LONG':
                if current_price <= current_trade['stop_loss']:
                    # Stop loss hit
                    current_trade['exit_time'] = df.iloc[i]['timestamp']
                    current_trade['exit_price'] = current_trade['stop_loss']
                    current_trade['outcome'] = 'LOSS'
                    current_trade['profit_pips'] = (current_trade['exit_price'] - current_trade['entry_price']) * pip_multiplier
                    current_trade['bars_held'] = i - current_trade['entry_idx']
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None

                elif current_price >= current_trade['take_profit']:
                    # Take profit hit
                    current_trade['exit_time'] = df.iloc[i]['timestamp']
                    current_trade['exit_price'] = current_trade['take_profit']
                    current_trade['outcome'] = 'WIN'
                    current_trade['profit_pips'] = (current_trade['exit_price'] - current_trade['entry_price']) * pip_multiplier
                    current_trade['bars_held'] = i - current_trade['entry_idx']
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None

            else:  # SHORT
                if current_price >= current_trade['stop_loss']:
                    # Stop loss hit
                    current_trade['exit_time'] = df.iloc[i]['timestamp']
                    current_trade['exit_price'] = current_trade['stop_loss']
                    current_trade['outcome'] = 'LOSS'
                    current_trade['profit_pips'] = (current_trade['entry_price'] - current_trade['exit_price']) * pip_multiplier
                    current_trade['bars_held'] = i - current_trade['entry_idx']
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None

                elif current_price <= current_trade['take_profit']:
                    # Take profit hit
                    current_trade['exit_time'] = df.iloc[i]['timestamp']
                    current_trade['exit_price'] = current_trade['take_profit']
                    current_trade['outcome'] = 'WIN'
                    current_trade['profit_pips'] = (current_trade['entry_price'] - current_trade['exit_price']) * pip_multiplier
                    current_trade['bars_held'] = i - current_trade['entry_idx']
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
    avg_bars = sum([t['bars_held'] for t in trades]) / len(trades)

    # Profit factor
    gross_profit = sum([t['profit_pips'] for t in wins]) if wins else 0
    gross_loss = abs(sum([t['profit_pips'] for t in losses])) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

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
    print(f"  Profit Factor: {profit_factor:.2f}x")
    print(f"  Avg Hold Time: {avg_bars:.1f} bars ({avg_bars} hours)")

    # Show sample trades
    print(f"\nSAMPLE TRADES (First 5 and Last 5):")
    print("-"*70)

    sample_trades = trades[:5] + trades[-5:] if len(trades) > 10 else trades

    for i, trade in enumerate(sample_trades, 1):
        outcome_symbol = "[WIN]" if trade['outcome'] == 'WIN' else "[LOSS]"
        print(f"\n{i}. {outcome_symbol} {trade['direction']}")
        print(f"   Entry: {trade['entry_time']} @ {trade['entry_price']:.5f}")
        print(f"   Exit:  {trade['exit_time']} @ {trade['exit_price']:.5f}")
        print(f"   Profit: {trade['profit_pips']:.1f} pips (held {trade['bars_held']} bars)")
        print(f"   Score: {trade['score']:.1f}, RSI: {trade['rsi']:.1f}")

    print("\n" + "="*70)

    return {
        'trades': trades,
        'win_rate': win_rate,
        'total_pips': total_pips,
        'wins': len(wins),
        'losses': len(losses),
        'profit_factor': profit_factor,
        'avg_bars': avg_bars
    }


def main():
    """Run enhanced backtest on multiple pairs"""

    print("\n" + "="*70)
    print("FOREX V3 ENHANCED BACKTEST")
    print("Strategy: Enhanced EMA v3.0 with Relaxed Parameters")
    print("Timeframe: 1-HOUR (H1)")
    print("Improvements: Volume filter, MTF confirmation, Fixed pip calc")
    print("="*70)

    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']

    all_results = {}

    for pair in pairs:
        result = run_enhanced_backtest(pair, candles=2000, timeframe='H1')
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
        avg_profit_factor = sum([r['profit_factor'] for r in all_results.values()]) / len(all_results)

        print(f"\nAcross {len(all_results)} pairs:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Overall Win Rate: {overall_win_rate:.1f}%")
        print(f"  Total Profit: {total_pips:.1f} pips")
        print(f"  Avg per pair: {total_pips/len(all_results):.1f} pips")
        print(f"  Avg Profit Factor: {avg_profit_factor:.2f}x")

        print("\nPer-Pair Performance:")
        for pair, result in all_results.items():
            print(f"  {pair}: {result['win_rate']:.1f}% WR, {result['total_pips']:+.1f} pips, PF: {result['profit_factor']:.2f}x ({result['wins']+result['losses']} trades)")

        print("\n" + "="*70)

        if overall_win_rate >= 60:
            print("RESULT: EXCELLENT! Target exceeded! [SUCCESS]")
        elif overall_win_rate >= 55:
            print("RESULT: GOOD! Close to target [OK]")
        elif overall_win_rate >= 50:
            print("RESULT: ACCEPTABLE performance [OK]")
        else:
            print("RESULT: NEEDS MORE OPTIMIZATION [WARNING]")

        print("="*70)

        print("\nKEY IMPROVEMENTS IN V3.0:")
        print("1. FIXED: USD/JPY pip calculation (2 decimals vs 5 decimals)")
        print("2. ADDED: Volume/activity filter (avoids low-volatility periods)")
        print("3. ADDED: Multi-timeframe confirmation (4H trend alignment)")
        print("4. IMPROVED: Relaxed RSI thresholds for more signals")
        print("5. IMPROVED: Lower score threshold for better trade frequency")

        print("\nNEXT STEPS:")
        print("1. If win rate >55%: Ready for paper trading")
        print("2. If win rate <55%: Test different timeframes (4H, daily)")
        print("3. Monitor for 30 days before live trading")

    else:
        print("[WARNING] No results generated. Check OANDA connection.")


if __name__ == "__main__":
    main()
