# LEAN Forex Strategy Backtest - Setup Instructions

## What You Have

**File:** `LEAN_FOREX_STRATEGY.py`

This is a **QuantConnect LEAN-compatible** version of your WORKING_FOREX_OANDA.py strategy.

## What LEAN Will Test

- **Period:** May 1, 2024 - November 1, 2024 (6 months)
- **Starting Capital:** $100,000
- **Pairs:** EURUSD, GBPUSD, USDJPY, GBPJPY
- **Leverage:** 5x (same as your live bot)
- **Stop Loss:** 1% per trade
- **Take Profit:** 2% per trade
- **Max Positions:** 3 concurrent

**Key Differences from Simple Backtest:**
- Realistic bid/ask spreads (1-3 pips on forex)
- Slippage modeling (market orders don't fill at exact price)
- Tick-level precision (catches exact stop/target hits)
- Event-driven architecture (exactly mirrors live trading)

---

## Option 1: QuantConnect Cloud (FASTEST - 5 minutes)

### Step 1: Create Account
1. Go to https://www.quantconnect.com
2. Click "Sign Up" (free tier is fine)
3. Verify email

### Step 2: Upload Strategy
1. Click "Algorithm Lab" → "New Algorithm"
2. Name it: "Forex Momentum Strategy"
3. Delete the default code
4. Copy/paste entire contents of `LEAN_FOREX_STRATEGY.py`
5. Click "Save"

### Step 3: Run Backtest
1. Click "Backtest" button (top right)
2. Wait 2-3 minutes for backtest to complete
3. View results dashboard

### Step 4: Analyze Results
QuantConnect will show you:
- **Equity curve** (visual P/L over time)
- **Win rate** (% profitable trades)
- **Sharpe ratio** (risk-adjusted returns)
- **Max drawdown** (worst peak-to-valley loss)
- **Trade list** (every entry/exit with prices)
- **Monthly returns** (breakdown by month)
- **Portfolio statistics** (alpha, beta, volatility)

### Step 5: Compare to Simple Backtest
**Simple Backtest Results:**
- Win Rate: 39.3%
- Profit Factor: 1.28
- Total Return: +19.34%
- Max Drawdown: -7.48%

**LEAN Expected Results:**
- Win Rate: ~35-37% (more realistic after spreads)
- Profit Factor: ~1.10-1.20 (slippage impact)
- Total Return: ~+12-15% (conservative estimate)
- Max Drawdown: ~-8-10% (more accurate)

If LEAN shows:
- ✅ **Win rate > 38%** = Strategy has edge
- ✅ **Sharpe ratio > 0.5** = Risk-adjusted returns acceptable
- ✅ **Max DD < 15%** = Within E8 prop firm limits
- ✅ **Profit factor > 1.15** = Winners outpace losers

Then continue trading. If any metric fails, redesign strategy.

---

## Option 2: Local LEAN Engine (FULL CONTROL - 30 minutes)

### Step 1: Install LEAN CLI
```bash
pip install lean
```

### Step 2: Initialize LEAN Project
```bash
cd c:\Users\lucas\PC-HIVE-TRADING
lean init
```

### Step 3: Create Algorithm
```bash
lean create-project "ForexMomentum"
```

### Step 4: Copy Strategy
Copy contents of `LEAN_FOREX_STRATEGY.py` into:
`c:\Users\lucas\PC-HIVE-TRADING\ForexMomentum\main.py`

### Step 5: Download Data
LEAN needs historical forex data. You have two options:

**A) Use QuantConnect Data (Recommended)**
```bash
lean cloud pull
```
This downloads institutional-grade data from QuantConnect.

**B) Use Local OANDA Data**
Configure `lean.json` to pull from OANDA API (more complex).

### Step 6: Run Backtest
```bash
lean backtest "ForexMomentum"
```

### Step 7: View Results
LEAN outputs JSON results to:
`c:\Users\lucas\PC-HIVE-TRADING\ForexMomentum\backtests\[timestamp]\`

You'll get:
- `results.json` - Full statistics
- `equity.csv` - Equity curve data
- `trades.csv` - Every trade executed
- `orders.csv` - All order fills

---

## Recommendation

**Use QuantConnect Cloud (Option 1) for now.**

Why?
- ✅ Zero setup time (5 minutes total)
- ✅ Professional data quality (tick-level accuracy)
- ✅ Beautiful web dashboard (easier to analyze)
- ✅ Your current trades are developing - need fast answers

**Use Local LEAN (Option 2) later** if you want:
- Full data privacy (nothing on cloud servers)
- Custom data sources (your own OANDA feeds)
- Automated backtesting pipelines

---

## What to Look For in Results

### Red Flags (Strategy Doesn't Work)
- ❌ Win rate < 35%
- ❌ Sharpe ratio < 0.3
- ❌ Max drawdown > 20%
- ❌ Profit factor < 1.1

### Yellow Flags (Needs Tweaking)
- ⚠️ Win rate 35-40%
- ⚠️ Sharpe ratio 0.3-0.5
- ⚠️ Max drawdown 15-20%
- ⚠️ Profit factor 1.1-1.2

### Green Flags (Strategy Works)
- ✅ Win rate > 40%
- ✅ Sharpe ratio > 0.5
- ✅ Max drawdown < 15%
- ✅ Profit factor > 1.2

---

## Next Steps After LEAN Backtest

**If LEAN confirms edge:**
1. Continue trading with confidence
2. Let current GBP_USD/EUR_USD positions develop
3. Focus on USD_JPY and GBP_JPY going forward (best performers)
4. Purchase E8 $500K challenge after 7-day validation

**If LEAN shows no edge:**
1. Close current positions (-$1,090 loss)
2. Redesign strategy with better parameters
3. Re-backtest before going live again
4. Save $1,627 E8 challenge fee until strategy proven

---

## Files You Need

- `LEAN_FOREX_STRATEGY.py` - Your algorithm (already created)
- `LEAN_BACKTEST_INSTRUCTIONS.md` - This file

**Ready to run!** Just upload to QuantConnect.com and hit "Backtest".

---

## Questions?

- QuantConnect docs: https://www.quantconnect.com/docs
- LEAN GitHub: https://github.com/QuantConnect/Lean
- Tutorials: https://www.quantconnect.com/tutorials

Your strategy is coded and ready. Results in 5 minutes if you use QuantConnect Cloud.
