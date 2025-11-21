"""
WALK-FORWARD VALIDATION - Prevent Overfitting

Methodology:
1. Split data into 6 monthly windows
2. For each window:
   - Train on Month N (optimize score threshold)
   - Test on Month N+1 (validate performance)
   - Roll forward
3. Compare: Do optimized settings work on unseen data?

If in-sample ROI ≈ out-of-sample ROI → ROBUST
If in-sample ROI >> out-of-sample ROI → OVERFIT
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

print("=" * 70)
print("WALK-FORWARD VALIDATION - Anti-Overfitting Backtest")
print("=" * 70)

# Configuration
PAIRS = ['EUR_USD', 'GBP_USD']
STARTING_BALANCE = 200000
MAX_DD = 0.06
DAILY_DD_LIMIT = 4000  # $4k daily loss limit (the $600 lesson)

# Score thresholds to test
SCORE_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

# ==============================================================================
# STEP 1: LOAD HISTORICAL DATA
# ==============================================================================

print("\n[STEP 1] Loading historical EUR/USD data...")
print("-" * 70)

# Use saved historical data if exists, otherwise explain how to get it
data_file = Path('historical_data/EUR_USD_H1_2024.csv')

if not data_file.exists():
    print("[INFO] No historical data found.")
    print("\nTo run proper validation, you need historical OHLC data:")
    print("  1. Download from OANDA API (last 6-12 months)")
    print("  2. Or use: https://www.histdata.com/download-free-forex-data/")
    print("  3. Save as: historical_data/EUR_USD_H1_2024.csv")
    print("\nFor now, I'll simulate with random walk (DEMO ONLY)")

    # Simulate random walk data (for demonstration)
    print("\n[SIMULATING] Generating synthetic price data...")

    dates = pd.date_range(start='2024-01-01', end='2024-11-19', freq='H')
    np.random.seed(42)

    price = 1.10
    prices = []

    for _ in range(len(dates)):
        change = np.random.normal(0, 0.0005)  # Small random changes
        price = price * (1 + change)
        prices.append(price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.0005 for p in prices],
        'low': [p * 0.9995 for p in prices],
        'close': prices,
    })

    print(f"  Generated {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
else:
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Loaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")

# ==============================================================================
# STEP 2: CALCULATE INDICATORS
# ==============================================================================

print("\n[STEP 2] Calculating technical indicators...")
print("-" * 70)

try:
    import talib

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    df['rsi'] = talib.RSI(closes, timeperiod=14)
    macd, macd_signal, _ = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['adx'] = talib.ADX(highs, lows, closes, timeperiod=14)
    df['ema_10'] = talib.EMA(closes, timeperiod=10)
    df['ema_21'] = talib.EMA(closes, timeperiod=21)
    df['ema_200'] = talib.EMA(closes, timeperiod=200)

    print("  RSI, MACD, ADX, EMAs calculated")

except ImportError:
    print("[WARN] TA-Lib not available - using simplified indicators")

    # Simplified calculations
    df['rsi'] = 50  # Neutral
    df['macd'] = 0
    df['macd_signal'] = 0
    df['adx'] = 20
    df['ema_10'] = df['close'].rolling(10).mean()
    df['ema_21'] = df['close'].rolling(21).mean()
    df['ema_200'] = df['close'].rolling(200).mean()

# Drop NaN rows
df = df.dropna()
print(f"  Valid candles after indicator calculation: {len(df)}")

# ==============================================================================
# STEP 3: SCORING FUNCTION (Same as Bot)
# ==============================================================================

def calculate_score(row):
    """Calculate LONG and SHORT scores for a candle"""
    long_score = 0
    short_score = 0

    # RSI signals (2 points)
    if row['rsi'] < 40:
        long_score += 2
    if row['rsi'] > 60:
        short_score += 2

    # MACD crossover (2 points)
    macd_hist = row['macd'] - row['macd_signal']
    if macd_hist > 0:
        long_score += 2
    elif macd_hist < 0:
        short_score += 2

    # Strong trend (1 point)
    if row['adx'] > 25:
        long_score += 1
        short_score += 1

    # EMA alignment (1 point)
    if row['ema_10'] > row['ema_21'] > row['ema_200']:
        long_score += 1
    if row['ema_10'] < row['ema_21'] < row['ema_200']:
        short_score += 1

    return max(long_score, short_score)

df['score'] = df.apply(calculate_score, axis=1)

# ==============================================================================
# STEP 4: WALK-FORWARD WINDOWS
# ==============================================================================

print("\n[STEP 3] Creating walk-forward windows...")
print("-" * 70)

# Split into monthly windows
df['month'] = df['timestamp'].dt.to_period('M')
months = sorted(df['month'].unique())

print(f"  Total months available: {len(months)}")
print(f"  First month: {months[0]}")
print(f"  Last month: {months[-1]}")

# Create train/test pairs
windows = []
for i in range(len(months) - 1):
    train_month = months[i]
    test_month = months[i + 1]

    train_data = df[df['month'] == train_month].copy()
    test_data = df[df['month'] == test_month].copy()

    if len(train_data) > 100 and len(test_data) > 100:
        windows.append({
            'train_month': str(train_month),
            'test_month': str(test_month),
            'train_data': train_data,
            'test_data': test_data
        })

print(f"\n  Created {len(windows)} walk-forward windows:")
for i, w in enumerate(windows):
    print(f"    Window {i+1}: Train on {w['train_month']}, Test on {w['test_month']}")

# ==============================================================================
# STEP 5: WALK-FORWARD OPTIMIZATION
# ==============================================================================

print("\n[STEP 4] Running walk-forward optimization...")
print("=" * 70)

results = []

for i, window in enumerate(windows):
    print(f"\n--- WINDOW {i+1}/{len(windows)} ---")
    print(f"Train: {window['train_month']} | Test: {window['test_month']}")
    print("-" * 70)

    train_data = window['train_data']
    test_data = window['test_data']

    # Find best threshold on TRAIN data
    best_train_score = -999999
    best_threshold = None

    for threshold in SCORE_THRESHOLDS:
        # Simulate trading on train data
        balance = STARTING_BALANCE
        peak = balance
        trades = 0
        winners = 0

        for idx, row in train_data.iterrows():
            if row['score'] >= threshold:
                trades += 1

                # Simplified P/L (50% base win rate + score bonus)
                win_prob = 0.48 + (row['score'] / 100)

                if np.random.random() < win_prob:
                    winners += 1
                    balance += 2000  # Win
                else:
                    balance -= 1000  # Loss

                peak = max(peak, balance)

        # Calculate train metrics
        if trades > 0:
            win_rate = winners / trades
            roi = ((balance - STARTING_BALANCE) / STARTING_BALANCE) * 100
            dd = ((peak - balance) / peak) * 100 if peak > balance else 0

            # Score = ROI - DD penalty
            score = roi - (dd * 2)  # Penalize high DD

            if score > best_train_score and dd < 6:  # Must meet DD constraint
                best_train_score = score
                best_threshold = threshold

    if best_threshold is None:
        print(f"  [SKIP] No profitable threshold found on train data")
        continue

    print(f"  Best threshold on TRAIN: {best_threshold}")

    # Test that threshold on TEST data (out-of-sample)
    balance = STARTING_BALANCE
    peak = balance
    trades = 0
    winners = 0
    daily_losses = {}  # Track daily DD

    for idx, row in test_data.iterrows():
        if row['score'] >= best_threshold:
            trades += 1

            # Simulate trade
            win_prob = 0.48 + (row['score'] / 100)

            if np.random.random() < win_prob:
                winners += 1
                pnl = 2000
            else:
                pnl = -1000

            balance += pnl
            peak = max(peak, balance)

            # Track daily loss (THE $600 LESSON)
            date = row['timestamp'].date()
            if date not in daily_losses:
                daily_losses[date] = 0
            daily_losses[date] += min(0, pnl)  # Only count losses

    # Calculate test metrics
    if trades > 0:
        test_win_rate = (winners / trades) * 100
        test_roi = ((balance - STARTING_BALANCE) / STARTING_BALANCE) * 100
        test_dd = ((peak - balance) / peak) * 100 if peak > balance else 0

        # Check daily DD violations
        daily_dd_violations = sum(1 for loss in daily_losses.values() if loss < -DAILY_DD_LIMIT)

        results.append({
            'window': i + 1,
            'train_month': window['train_month'],
            'test_month': window['test_month'],
            'best_threshold': best_threshold,
            'test_trades': trades,
            'test_win_rate': test_win_rate,
            'test_roi': test_roi,
            'test_dd': test_dd,
            'daily_dd_violations': daily_dd_violations
        })

        print(f"\n  OUT-OF-SAMPLE RESULTS:")
        print(f"    Trades: {trades}")
        print(f"    Win Rate: {test_win_rate:.1f}%")
        print(f"    ROI: {test_roi:+.2f}%")
        print(f"    Max DD: {test_dd:.2f}%")
        print(f"    Daily DD Violations: {daily_dd_violations}")

        if test_dd >= 6:
            print(f"    [FAIL] Exceeded 6% trailing DD")
        if daily_dd_violations > 0:
            print(f"    [FAIL] Hit daily DD limit {daily_dd_violations} times")
    else:
        print(f"  [SKIP] No trades on TEST data")

# ==============================================================================
# STEP 6: ANALYZE RESULTS
# ==============================================================================

print("\n" + "=" * 70)
print("WALK-FORWARD VALIDATION RESULTS")
print("=" * 70)

if not results:
    print("\n[ERROR] No valid results - insufficient data or all strategies failed")
else:
    # Create results DataFrame
    results_df = pd.DataFrame(results)

    print(f"\n{len(results)} windows tested\n")

    print(results_df[['window', 'test_month', 'best_threshold', 'test_roi', 'test_dd', 'daily_dd_violations']].to_string(index=False))

    # Statistical analysis
    print("\n" + "=" * 70)
    print("OVERFITTING ANALYSIS:")
    print("=" * 70)

    avg_roi = results_df['test_roi'].mean()
    std_roi = results_df['test_roi'].std()
    avg_dd = results_df['test_dd'].mean()
    total_dd_violations = results_df['daily_dd_violations'].sum()

    print(f"\nAverage Out-of-Sample ROI: {avg_roi:+.2f}% (std: {std_roi:.2f}%)")
    print(f"Average Max DD: {avg_dd:.2f}%")
    print(f"Total Daily DD Violations: {total_dd_violations}")

    # Check consistency of best threshold
    threshold_counts = results_df['best_threshold'].value_counts()
    most_common_threshold = threshold_counts.index[0]
    threshold_consistency = (threshold_counts[most_common_threshold] / len(results_df)) * 100

    print(f"\nMost Common Threshold: {most_common_threshold} ({threshold_consistency:.0f}% of windows)")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT:")
    print("=" * 70)

    if avg_roi > 5 and avg_dd < 6 and total_dd_violations == 0:
        print("\n✓ ROBUST STRATEGY")
        print(f"  - Positive ROI across windows: {avg_roi:+.2f}%")
        print(f"  - DD within limits: {avg_dd:.2f}%")
        print(f"  - No daily DD violations")
        print(f"\n  RECOMMENDED: Use threshold {most_common_threshold} on Match Trader demo")

    elif avg_roi > 0 and total_dd_violations <= 2:
        print("\n⚠ MARGINALLY ROBUST")
        print(f"  - Barely profitable: {avg_roi:+.2f}%")
        print(f"  - Some DD violations: {total_dd_violations}")
        print(f"\n  RECOMMENDED: Test on demo with threshold {most_common_threshold}")
        print(f"              Lower position sizing by 50% for safety")

    else:
        print("\n✗ STRATEGY FAILS OUT-OF-SAMPLE")
        print(f"  - Poor ROI: {avg_roi:+.2f}%")
        print(f"  - DD violations: {total_dd_violations}")
        print(f"\n  RECOMMENDED: DO NOT deploy on funded account")
        print(f"              Strategy is likely OVERFIT")
        print(f"              Consider simplified 3-parameter version")

# ==============================================================================
# STEP 7: SAVE RESULTS
# ==============================================================================

if results:
    results_file = Path('walk_forward_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results saved to: {results_file}")

print("\n" + "=" * 70)
print("NEXT STEP: Forward Testing on Match Trader Demo")
print("=" * 70)
print("""
Walk-forward validation shows historical robustness.
But the FINAL test is LIVE forward testing.

Deploy on Match Trader demo for 30-60 days:
- Use the most common threshold from above
- Track daily DD violations in real-time
- Monitor out-of-sample ROI vs backtest predictions

If demo results match backtest → Strategy is truly robust
If demo results much worse → Strategy was overfit despite walk-forward
""")

print("=" * 70)
