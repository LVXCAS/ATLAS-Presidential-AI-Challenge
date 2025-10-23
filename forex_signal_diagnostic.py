#!/usr/bin/env python3
"""
FOREX SIGNAL DIAGNOSTIC TOOL
Reveals WHY no signals are being generated

Checks all filter stages and reports what's blocking signals
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.oanda_data_fetcher import OandaDataFetcher
from forex_v4_optimized import ForexV4OptimizedStrategy
from datetime import datetime, time
import pandas as pd


def diagnose_pair(symbol: str):
    """Diagnose why a forex pair isn't generating signals"""

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: {symbol}")
    print(f"{'='*70}\n")

    # Fetch data
    fetcher = OandaDataFetcher(practice=True)
    df = fetcher.get_bars(symbol, timeframe='H1', limit=500)

    if df is None or df.empty:
        print(f"[ERROR] No data for {symbol}")
        return

    print(f"[DATA] {len(df)} candles fetched\n")

    # Initialize strategy
    strategy = ForexV4OptimizedStrategy()
    strategy.set_data_fetcher(fetcher)

    # Calculate indicators
    df = strategy.calculate_indicators(df)

    # Get current values
    current = df.iloc[-1]
    previous = df.iloc[-2]

    price = current['close']
    ema_fast_curr = current['ema_fast']
    ema_slow_curr = current['ema_slow']
    ema_trend_curr = current['ema_trend']
    rsi = current['rsi']
    atr = current['atr']
    adx = current['adx']

    ema_fast_prev = previous['ema_fast']
    ema_slow_prev = previous['ema_slow']

    # Check crossovers
    bullish_cross = (ema_fast_curr > ema_slow_curr) and (ema_fast_prev <= ema_slow_prev)
    bearish_cross = (ema_fast_curr < ema_slow_curr) and (ema_fast_prev >= ema_slow_prev)

    print(f"[CURRENT STATE]")
    print(f"  Price: {price:.5f}")
    print(f"  EMA Fast (10): {ema_fast_curr:.5f}")
    print(f"  EMA Slow (21): {ema_slow_curr:.5f}")
    print(f"  EMA Trend (200): {ema_trend_curr:.5f}")
    print(f"  RSI (14): {rsi:.2f}")
    print(f"  ADX (14): {adx:.2f}")
    print(f"  ATR: {atr:.5f}\n")

    print(f"[CROSSOVER CHECK]")
    print(f"  Bullish Crossover: {'YES' if bullish_cross else 'NO'}")
    print(f"  Bearish Crossover: {'YES' if bearish_cross else 'NO'}\n")

    if not bullish_cross and not bearish_cross:
        print("[BLOCKER] No EMA crossover detected")
        print(f"  Fast vs Slow: {ema_fast_curr:.5f} vs {ema_slow_curr:.5f}")
        print(f"  Previous: {ema_fast_prev:.5f} vs {ema_slow_prev:.5f}")
        print(f"  Direction: {'Fast above Slow' if ema_fast_curr > ema_slow_curr else 'Fast below Slow'}")
        return

    # Determine direction
    direction = 'LONG' if bullish_cross else 'SHORT'
    print(f"[SIGNAL DIRECTION] {direction}\n")

    # Check filters one by one
    print(f"[FILTER CHECKS]")

    # Filter 1: Time of day
    timestamp = current.name if hasattr(current, 'name') else None
    if timestamp:
        trade_time = timestamp.time()
        in_hours = strategy.trading_hours['start'] <= trade_time <= strategy.trading_hours['end']
        print(f"  1. Time-of-Day: {'PASS' if in_hours else 'FAIL'}")
        if not in_hours:
            print(f"     Current: {trade_time}")
            print(f"     Required: {strategy.trading_hours['start']} - {strategy.trading_hours['end']} UTC")
    else:
        print(f"  1. Time-of-Day: PASS (no timestamp)")

    # Filter 2: Volatility regime
    current_atr = df['atr'].iloc[-1]
    recent_atr = df['atr'].iloc[-100:]
    percentile = (recent_atr < current_atr).sum() / len(recent_atr) * 100
    vol_ok = strategy.atr_percentile_min <= percentile <= strategy.atr_percentile_max
    print(f"  2. Volatility Regime: {'PASS' if vol_ok else 'FAIL'}")
    print(f"     Current ATR Percentile: {percentile:.1f}%")
    print(f"     Required: {strategy.atr_percentile_min}% - {strategy.atr_percentile_max}%")

    # Filter 3: ADX trend strength
    adx_ok = pd.notna(adx) and adx >= strategy.adx_threshold
    print(f"  3. ADX Trend Strength: {'PASS' if adx_ok else 'FAIL'}")
    print(f"     Current ADX: {adx:.2f}")
    print(f"     Required: >= {strategy.adx_threshold}")

    # Filter 4: Trend alignment
    if direction == 'LONG':
        trend_ok = price > ema_trend_curr
        print(f"  4. Trend Alignment: {'PASS' if trend_ok else 'FAIL'}")
        print(f"     Price: {price:.5f}")
        print(f"     EMA 200: {ema_trend_curr:.5f}")
        print(f"     Required: Price > EMA 200 for LONG")
    else:
        trend_ok = price < ema_trend_curr
        print(f"  4. Trend Alignment: {'PASS' if trend_ok else 'FAIL'}")
        print(f"     Price: {price:.5f}")
        print(f"     EMA 200: {ema_trend_curr:.5f}")
        print(f"     Required: Price < EMA 200 for SHORT")

    # Filter 5: RSI bounds
    if direction == 'LONG':
        rsi_ok = strategy.rsi_long_lower < rsi < strategy.rsi_long_upper
        print(f"  5. RSI Bounds: {'PASS' if rsi_ok else 'FAIL'}")
        print(f"     Current RSI: {rsi:.2f}")
        print(f"     Required: {strategy.rsi_long_lower} < RSI < {strategy.rsi_long_upper}")
    else:
        rsi_ok = strategy.rsi_short_lower < rsi < strategy.rsi_short_upper
        print(f"  5. RSI Bounds: {'PASS' if rsi_ok else 'FAIL'}")
        print(f"     Current RSI: {rsi:.2f}")
        print(f"     Required: {strategy.rsi_short_lower} < RSI < {strategy.rsi_short_upper}")

    # Filter 6: EMA separation
    ema_separation = abs(ema_fast_curr - ema_slow_curr)
    ema_separation_pct = ema_separation / price
    sep_ok = ema_separation_pct >= strategy.min_ema_separation_pct
    print(f"  6. EMA Separation: {'PASS' if sep_ok else 'FAIL'}")
    print(f"     Current: {ema_separation_pct*100:.4f}%")
    print(f"     Required: >= {strategy.min_ema_separation_pct*100:.4f}%")

    # Filter 7: MTF confirmation
    mtf_ok = strategy.check_higher_timeframe_trend(symbol, direction)
    print(f"  7. Multi-Timeframe (4H): {'PASS' if mtf_ok else 'FAIL'}")

    # Summary
    all_checks = [in_hours if timestamp else True, vol_ok, adx_ok, trend_ok, rsi_ok, sep_ok, mtf_ok]
    passed = sum(all_checks)
    total = len(all_checks)

    print(f"\n[SUMMARY]")
    print(f"  Filters Passed: {passed}/{total}")
    print(f"  Score Threshold: {strategy.score_threshold}+")

    if passed == total:
        print(f"  [OK] All filters passed - signal should generate!")
    else:
        print(f"  [BLOCKED] {total-passed} filter(s) failing")


def main():
    """Run diagnostics on all forex pairs"""

    print("\n" + "="*70)
    print("FOREX SIGNAL DIAGNOSTIC TOOL")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nThis tool reveals why no signals are being generated")
    print("="*70)

    pairs = ['EUR_USD', 'USD_JPY']

    for pair in pairs:
        try:
            diagnose_pair(pair)
        except Exception as e:
            print(f"\n[ERROR] {pair}: {e}")

    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
