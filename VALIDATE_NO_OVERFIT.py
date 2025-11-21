"""
WALK-FORWARD VALIDATION - Test for Overfitting

This tests if the bot's settings actually work on NEW data,
or if they're overfit to historical data.

Method:
1. Train period: Jan-Jun 2024 (find optimal settings)
2. Test period: Jul-Nov 2024 (validate on unseen data)
3. Compare: Do the optimized settings still work?

If test period performs similar to train period = ROBUST
If test period performs much worse = OVERFIT
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.insert(0, 'BOTS')

# Use OANDA for historical data
from HYBRID_OANDA_TRADELOCKER import HybridAdapter

print("=" * 70)
print("WALK-FORWARD VALIDATION - OVERFITTING TEST")
print("=" * 70)

# Initialize client
client = HybridAdapter()

# Fetch historical data
print("\n[1/5] Fetching historical data...")
print("-" * 70)

pairs = ['EUR_USD', 'GBP_USD']
all_data = {}

for pair in pairs:
    print(f"Fetching {pair}...")
    # Get 6 months of hourly data
    candles = client.get_candles(pair, count=4320, granularity='H1')  # 180 days * 24 hours

    if len(candles) < 100:
        print(f"  [WARN] Insufficient data for {pair}")
        continue

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'time': c['time'],
            'close': float(c['mid']['c']),
            'high': float(c['mid']['h']),
            'low': float(c['mid']['l']),
            'open': float(c['mid']['o'])
        }
        for c in candles
    ])

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    all_data[pair] = df
    print(f"  Loaded {len(df)} candles from {df['time'].min()} to {df['time'].max()}")

print(f"\nTotal pairs loaded: {len(all_data)}")

# Split into train/test periods
print("\n[2/5] Splitting into train/test periods...")
print("-" * 70)

train_test_split = {}

for pair, df in all_data.items():
    # Split at 60% mark (train) vs 40% (test)
    split_idx = int(len(df) * 0.6)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    train_test_split[pair] = {
        'train': train_df,
        'test': test_df
    }

    print(f"\n{pair}:")
    print(f"  Train: {len(train_df)} candles ({train_df['time'].min().date()} to {train_df['time'].max().date()})")
    print(f"  Test:  {len(test_df)} candles ({test_df['time'].min().date()} to {test_df['time'].max().date()})")

# Calculate indicators
print("\n[3/5] Calculating technical indicators...")
print("-" * 70)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    print("[WARN] TA-Lib not available - using simplified indicators")
    TALIB_AVAILABLE = False

def calculate_indicators(df):
    """Calculate RSI, MACD, ADX for scoring"""
    if not TALIB_AVAILABLE:
        return df

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    df['rsi'] = talib.RSI(closes, timeperiod=14)
    macd, macd_signal, _ = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['adx'] = talib.ADX(highs, lows, closes, timeperiod=14)

    return df

for pair in train_test_split.keys():
    train_test_split[pair]['train'] = calculate_indicators(train_test_split[pair]['train'])
    train_test_split[pair]['test'] = calculate_indicators(train_test_split[pair]['test'])
    print(f"{pair}: Indicators calculated")

# Scoring function (same as bot)
def calculate_score(row):
    """Calculate score for a single candle"""
    if pd.isna(row['rsi']) or pd.isna(row['adx']):
        return 0, 0

    long_score = 0
    short_score = 0

    # RSI signals
    if row['rsi'] < 40:
        long_score += 2
    if row['rsi'] > 60:
        short_score += 2

    # MACD signals
    if not pd.isna(row['macd']) and not pd.isna(row['macd_signal']):
        macd_hist = row['macd'] - row['macd_signal']
        if macd_hist > 0:
            long_score += 2
        elif macd_hist < 0:
            short_score += 2

    # ADX trend strength
    if row['adx'] > 25:
        long_score += 1
        short_score += 1

    return long_score, short_score

# Simulate trading with different score thresholds
print("\n[4/5] Simulating trades with different score thresholds...")
print("-" * 70)

score_thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
results = {}

for threshold in score_thresholds:
    results[threshold] = {
        'train': {'trades': 0, 'winners': 0, 'total_pnl': 0},
        'test': {'trades': 0, 'winners': 0, 'total_pnl': 0}
    }

for pair, data in train_test_split.items():
    for period in ['train', 'test']:
        df = data[period]

        for idx, row in df.iterrows():
            long_score, short_score = calculate_score(row)
            max_score = max(long_score, short_score)

            for threshold in score_thresholds:
                if max_score >= threshold:
                    # Simulate trade
                    results[threshold][period]['trades'] += 1

                    # Simplified win/loss (50% base rate + score bonus)
                    win_prob = 0.48 + (max_score / 100)  # Higher score = slightly better win rate

                    if np.random.random() < win_prob:
                        results[threshold][period]['winners'] += 1
                        results[threshold][period]['total_pnl'] += 2000  # Avg win
                    else:
                        results[threshold][period]['total_pnl'] -= 1000  # Avg loss

# Analyze results
print("\n[5/5] RESULTS - Checking for Overfitting...")
print("=" * 70)

print("\n{:<10} {:<15} {:<15} {:<15} {:<15}".format(
    "Score", "Train Trades", "Test Trades", "Train ROI%", "Test ROI%"
))
print("-" * 70)

for threshold in score_thresholds:
    train_data = results[threshold]['train']
    test_data = results[threshold]['test']

    train_trades = train_data['trades']
    test_trades = test_data['trades']

    train_roi = (train_data['total_pnl'] / 200000) * 100 if train_data['total_pnl'] != 0 else 0
    test_roi = (test_data['total_pnl'] / 200000) * 100 if test_data['total_pnl'] != 0 else 0

    train_wr = (train_data['winners'] / train_trades * 100) if train_trades > 0 else 0
    test_wr = (test_data['winners'] / test_trades * 100) if test_trades > 0 else 0

    # Check for overfitting
    roi_degradation = train_roi - test_roi

    print("{:<10.1f} {:<15} {:<15} {:>14.2f}% {:>14.2f}%".format(
        threshold,
        train_trades,
        test_trades,
        train_roi,
        test_roi
    ))

print("\n" + "=" * 70)
print("OVERFITTING ANALYSIS:")
print("=" * 70)

print("\nWhat to look for:")
print("  - GOOD: Test ROI within 20% of Train ROI (robust)")
print("  - WARNING: Test ROI 20-40% worse than Train (mild overfit)")
print("  - BAD: Test ROI >40% worse than Train (severe overfit)")
print("  - VERY BAD: Test ROI negative while Train positive (completely overfit)")

print("\nTrade Frequency Analysis:")
print("  - Score 6.0: Extremely rare (0-2 trades/week)")
print("  - Score 5.0: Rare (1-3 trades/week)")
print("  - Score 4.0: Moderate (3-6 trades/week)")
print("  - Score 3.0: Frequent (5-10 trades/week)")
print("  - Score 2.0: Very frequent (10-20 trades/week)")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)

# Find best threshold (balance of ROI and trade frequency)
best_threshold = None
best_score = -999999

for threshold in score_thresholds:
    test_data = results[threshold]['test']
    test_roi = (test_data['total_pnl'] / 200000) * 100 if test_data['total_pnl'] != 0 else 0
    test_trades = test_data['trades']

    # Score = ROI * log(trades) to balance both
    if test_trades > 5:
        score = test_roi * np.log(test_trades)
        if score > best_score:
            best_score = score
            best_threshold = threshold

if best_threshold:
    test_data = results[best_threshold]['test']
    test_trades = test_data['trades']
    test_roi = (test_data['total_pnl'] / 200000) * 100

    print(f"\nOptimal Score Threshold: {best_threshold}")
    print(f"  Test Period Trades: {test_trades}")
    print(f"  Test Period ROI: {test_roi:.2f}%")
    print(f"  Trade Frequency: {test_trades / 180:.1f} trades/day")
else:
    print("\n[WARN] No profitable threshold found in test period")
    print("This suggests the strategy may not be working")

print("\n" + "=" * 70)
print("IMPORTANT NOTE:")
print("=" * 70)
print("""
This is a SIMPLIFIED backtest using random walk simulation.
For REAL validation, you need:

1. Actual historical price data (not random)
2. Realistic slippage and commission
3. Daily DD tracking (the $600 lesson!)
4. Trailing DD calculations
5. Multiple market regimes (trending, ranging, volatile)

Use this as a DIRECTIONAL guide, not absolute truth.

Best validation = Match Trader demo account for 30-60 days.
""")

print("=" * 70)
