# BACKTEST COMPARISON GUIDE

## Safe-Optimized vs Aggressive-Optimized

**Goal:** Use REAL data to determine which approach actually works for E8.

---

## What We're Testing

### Strategy A: Safe-Optimized

```yaml
Position Size: 2.0 lots (fixed)
Trade Frequency: 2-3 per week
Stop Loss: 20-25 pips
Score Threshold: 5.5
Take Profit: Partial (50% at 1R, 50% at 2.5R)
Trading Hours: London/NY opens (priority)
Max Trades/Day: 1

Expected:
  Monthly ROI: 5-9%
  Pass Rate: 60-65%
  Daily DD Violations: <5 over 6 months
```

### Strategy B: Aggressive-Optimized

```yaml
Position Size: 2.5-3.0 lots (dynamic)
Trade Frequency: 5-6 per week
Stop Loss: 15 pips (tighter)
Score Threshold: 4.5 (looser)
Take Profit: Partial (50% at 1R, 50% at 3R)
Trading Hours: Extended (8 AM - 5 PM)
Max Trades/Day: 2

Expected:
  Monthly ROI: 15-25%
  Pass Rate: 10-20%
  Daily DD Violations: 10-30 over 6 months
```

---

## How to Run the Backtest

### Step 1: Install Backtrader (if needed)

```bash
pip install backtrader
pip install pandas numpy
```

### Step 2: Fetch OANDA Data

```bash
cd c:\Users\lucas\PC-HIVE-TRADING
python backtesting/fetch_oanda_data_for_backtest.py
```

**What this does:**
- Downloads last 6 months of H1 data from OANDA
- Pairs: EUR/USD, GBP/USD, USD/JPY
- Saves to `backtesting/data/[PAIR]_H1_6M.csv`
- ~4,320 candles per pair

**Output:**
```
[FETCHING] EUR_USD from 2024-05-20 to 2024-11-20
  Requesting candles from OANDA...
  Retrieved 4320 candles
  [SAVED] backtesting/data/EUR_USD_H1_6M.csv (4320 candles)

[FETCHING] GBP_USD from 2024-05-20 to 2024-11-20
  [SAVED] backtesting/data/GBP_USD_H1_6M.csv (4320 candles)

[FETCHING] USD_JPY from 2024-05-20 to 2024-11-20
  [SAVED] backtesting/data/USD_JPY_H1_6M.csv (4320 candles)

✅ All data ready for backtest
```

### Step 3: Update Backtest Script with Data Loading

Edit `backtesting/BACKTEST_SAFE_VS_AGGRESSIVE.py`:

Find this section:
```python
# TODO: Load actual forex data
# For now, create dummy datafeed structure
```

Replace with:
```python
# Load EUR/USD
data_eurusd = bt.feeds.GenericCSVData(
    dataname='backtesting/data/EUR_USD_H1_6M.csv',
    dtformat='%Y-%m-%d %H:%M:%S',
    datetime=0,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=5,
    openinterest=-1,
    name='EUR_USD'
)
cerebro.adddata(data_eurusd)

# Load GBP/USD
data_gbpusd = bt.feeds.GenericCSVData(
    dataname='backtesting/data/GBP_USD_H1_6M.csv',
    dtformat='%Y-%m-%d %H:%M:%S',
    datetime=0,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=5,
    openinterest=-1,
    name='GBP_USD'
)
cerebro.adddata(data_gbpusd)

# Load USD/JPY
data_usdjpy = bt.feeds.GenericCSVData(
    dataname='backtesting/data/USD_JPY_H1_6M.csv',
    dtformat='%Y-%m-%d %H:%M:%S',
    datetime=0,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=5,
    openinterest=-1,
    name='USD_JPY'
)
cerebro.adddata(data_usdjpy)
```

### Step 4: Run the Backtest

```bash
python backtesting/BACKTEST_SAFE_VS_AGGRESSIVE.py
```

---

## Expected Output

### Part 1: Safe-Optimized Results

```
================================================================================
BACKTESTING: Safe-Optimized (2 lots, 2-3 trades/week, 20-25 pip SL)
Period: 2024-05-20 to 2024-11-20
================================================================================

[INFO] Loading forex data feeds...
[INFO] Running backtest...

2024-05-21 ENTRY: EUR_USD, Score: 5.7, Size: 200000
2024-05-23 TRADE CLOSED: EUR_USD, P/L: $1,850.00
2024-05-28 ENTRY: GBP_USD, Score: 5.9, Size: 200000
2024-05-30 TRADE CLOSED: GBP_USD, P/L: $2,100.00
...

================================================================================
RESULTS: Safe-Optimized
================================================================================

[ACCOUNT PERFORMANCE]
  Starting Balance: $200,000.00
  Final Balance: $214,500.00
  Total Return: +7.25%
  Peak Balance: $215,200.00

[RISK METRICS]
  Max Drawdown: 2.3%
  Daily DD Violations: 2

[TRADING ACTIVITY]
  Total Trades: 56
  Avg Trades/Week: 2.2
  Wins: 36
  Losses: 20
  Win Rate: 64.3%

[E8 CHALLENGE VERDICT]
  ❌ FAIL - Would have failed E8 challenge
    - Did not reach $20k target (only $14,500)
    - Had 2 daily DD violations
```

### Part 2: Aggressive-Optimized Results

```
================================================================================
BACKTESTING: Aggressive-Optimized (3 lots, 5-6 trades/week, 15 pip SL)
Period: 2024-05-20 to 2024-11-20
================================================================================

[ACCOUNT PERFORMANCE]
  Starting Balance: $200,000.00
  Final Balance: $189,200.00
  Total Return: -5.40%
  Peak Balance: $218,500.00

[RISK METRICS]
  Max Drawdown: 13.4%
  Daily DD Violations: 18

[TRADING ACTIVITY]
  Total Trades: 132
  Avg Trades/Week: 5.1
  Wins: 62
  Losses: 70
  Win Rate: 47.0%

[E8 CHALLENGE VERDICT]
  ❌ FAIL - Would have failed E8 challenge
    - Exceeded 6% trailing DD limit (13.4%)
    - Had 18 daily DD violations
```

### Part 3: Side-by-Side Comparison

```
================================================================================
SIDE-BY-SIDE COMPARISON
================================================================================

Metric                    Safe-Optimized            Aggressive-Optimized
--------------------------------------------------------------------------------
Strategy                  Safe-Optimized            Aggressive-Optimized
Final Balance             $214,500.00               $189,200.00
Total Return              7.25%                     -5.40%
Max Drawdown              2.30%                     13.40%
Daily DD Violations       2                         18
Total Trades              56                        132
Win Rate                  64.30%                    47.00%
E8 Verdict                ❌ FAIL                    ❌ FAIL
================================================================================

[CONCLUSION]
  ❌ BOTH strategies FAILED
  - Need to go even MORE conservative
  - Or wait for better market conditions

  Verdict: Don't deploy either - need ultra-conservative (1.5 lots, 1 trade/week)
```

---

## What the Results Tell Us

### Scenario 1: Safe-Optimized PASSES, Aggressive FAILS

```
Safe-Optimized:
  ✅ $220,000+ final balance (>$20k profit)
  ✅ Max DD < 6%
  ✅ Daily DD violations = 0

Aggressive-Optimized:
  ❌ Multiple daily DD violations
  ❌ Account would have been terminated

Verdict: Build Safe-Optimized ATLAS
Reason: Proven to work on real data
```

### Scenario 2: Aggressive PASSES, Safe FAILS

```
Aggressive-Optimized:
  ✅ $225,000+ final balance
  ✅ Max DD < 6%
  ✅ Daily DD violations = 0
  ✅ Higher ROI than Safe

Safe-Optimized:
  ✅ Passes too, but lower ROI

Verdict: Build Aggressive-Optimized ATLAS
Reason: Higher ROI, still safe in practice
Surprise: Tighter stops worked better than expected
```

### Scenario 3: BOTH FAIL

```
Both strategies:
  ❌ Multiple daily DD violations
  OR
  ❌ Didn't reach $20k target in 6 months
  OR
  ❌ Exceeded trailing DD

Verdict: Go ULTRA-conservative
Settings:
  - 1.5 lots max
  - 1-2 trades/week
  - 30 pip stops (wider)
  - Score 6.0+ threshold
```

### Scenario 4: BOTH PASS

```
Both strategies passed:
  Safe: +$22k profit, 0 violations
  Aggressive: +$35k profit, 0 violations

Verdict: Build Aggressive (higher ROI)
Reason: Both are safe, pick the faster one
Timeline: 3-4 months vs 5-6 months
```

---

## Critical Metrics to Watch

### 1. Daily DD Violations (MOST IMPORTANT)

```
This is the account killer.

Safe target: 0 violations in 6 months
Acceptable: 1-2 violations (might survive with luck)
Dangerous: 3-5 violations (low pass probability)
Fatal: 10+ violations (guaranteed failure)

If Aggressive shows >5 violations → DON'T USE IT
```

### 2. Max Drawdown

```
E8 limit: 6% trailing DD

Safe: < 3% max DD (comfortable margin)
Acceptable: 3-4% max DD (some pressure)
Risky: 4.5-5.5% max DD (very close to limit)
Fatal: > 6% max DD (would have been terminated)

If either strategy exceeds 6%, it's eliminated.
```

### 3. Time to $20K Target

```
Extrapolate from 6-month results:

If made $10k in 6 months → Need 12 months total
If made $15k in 6 months → Need 8 months total
If made $20k in 6 months → Already passed!
If made $5k in 6 months → Need 24 months (too slow)

Target: Pass in 3-6 months
Acceptable: 6-12 months
Too slow: > 12 months
```

### 4. Win Rate Impact from Tighter Stops

```
Safe (20-25 pip stops): Expected win rate 60-65%
Aggressive (15 pip stops): Expected win rate 50-55%

If Aggressive backtest shows:
  Win rate > 55% → Tighter stops working well
  Win rate 50-55% → As expected, acceptable
  Win rate < 50% → Tighter stops too tight, losing edge
```

---

## After Backtest: Decision Tree

```
Run backtest → Review results → Make decision

┌─ Both strategies PASS
│  ├─ Aggressive has higher ROI
│  │  └─ Build Aggressive-Optimized ATLAS
│  └─ Safe has higher win rate but lower ROI
│     └─ Build Safe-Optimized ATLAS (more reliable)
│
├─ Only Safe PASSES
│  └─ Build Safe-Optimized ATLAS
│     (Aggressive too risky, as predicted)
│
├─ Only Aggressive PASSES
│  └─ Build Aggressive-Optimized ATLAS
│     (Surprising! Tighter stops worked better)
│
└─ Both strategies FAIL
   ├─ Go ULTRA-conservative
   │  - 1.5 lots max
   │  - 1-2 trades/week
   │  - Score 6.0+ threshold
   │  - 30 pip stops
   └─ Re-run backtest with ultra settings
```

---

## Next Steps After Backtest

### If Safe-Optimized Wins

1. Build ATLAS architecture with Safe-Optimized settings
2. Deploy on Match Trader demo (60 days)
3. If demo passes → Pay $600 for E8 evaluation
4. Expected timeline: 5-6 months to funded account

### If Aggressive-Optimized Wins (Surprising)

1. **Validate the result** - Run backtest on different 6-month period
2. If still passes → Build ATLAS with Aggressive settings
3. Deploy on Match Trader demo (60 days, watch closely)
4. Expected timeline: 3-4 months to funded account

### If Both Fail

1. Don't build ATLAS yet
2. Go back to ultra-conservative settings
3. Run backtest again with 1.5 lots, 1 trade/week
4. If still fails → Market conditions not suitable for E8

---

## Running the Backtest Right Now

**Step-by-step:**

```bash
# 1. Fetch data from OANDA
cd c:\Users\lucas\PC-HIVE-TRADING
python backtesting/fetch_oanda_data_for_backtest.py

# Wait for download to complete (~2-5 minutes)

# 2. Run backtest comparison
python backtesting/BACKTEST_SAFE_VS_AGGRESSIVE.py

# Wait for results (~30-60 seconds)

# 3. Review output and make decision
```

**Time required:** 5-10 minutes total

**What you get:**
- Real data showing which strategy works
- Daily DD violation counts (the critical metric)
- Win rates with different stop sizes
- Actual time to $20k target
- Clear verdict on which to build

---

## The Bottom Line

**Instead of arguing theory, let's see what the DATA says.**

- If Safe wins → Build conservative ATLAS, I was right
- If Aggressive wins → Build aggressive ATLAS, you were right
- If both work → Pick the higher ROI one
- If neither works → Go even more conservative

**The backtest will give us the answer in 10 minutes.**

Ready to run it?

```bash
python backtesting/fetch_oanda_data_for_backtest.py
```

Then:

```bash
python backtesting/BACKTEST_SAFE_VS_AGGRESSIVE.py
```

**The data will tell us which strategy to build.**

No more theory. Just results.
