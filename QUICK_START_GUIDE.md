# Quick Start Guide - OPTIONS_BOT

## The Issue

The file `OPTIONS_BOT.py` **doesn't exist**. The documentation references it, but the actual system has **multiple specialized options bots** instead.

---

## Available Options Trading Bots

### 1️⃣ **Options Hunter Bot** (Recommended)
**File**: `options_hunter_bot.py`

**Features**:
- Monte Carlo optimized strategies
- 100% options trading (no stocks)
- Focus on highest win-rate strategies:
  - Bear Put Spreads: 86.4% win rate
  - Bull Call Spreads: 71.7% win rate
- Advanced risk management

**Run**:
```bash
python options_hunter_bot.py
```

---

### 2️⃣ **Autonomous Options Income Agent**
**File**: `autonomous_options_income_agent.py`

**Features**:
- Fully autonomous (no manual intervention)
- Cash-secured puts strategy
- Weekly options focus
- Income generation

**Run**:
```bash
python autonomous_options_income_agent.py
```

---

### 3️⃣ **Real World Options Bot**
**File**: `real_world_options_bot.py`

**Features**:
- Production-ready
- Full Greeks integration (Delta, Gamma, Theta, Vega)
- Multiple strategy support
- Risk management

**Run**:
```bash
python real_world_options_bot.py
```

---

### 4️⃣ **Tomorrow Ready Options Bot**
**File**: `tomorrow_ready_options_bot.py`

**Features**:
- Next-day preparation
- Overnight analysis
- Morning execution planning
- Trade queue management

**Run**:
```bash
python tomorrow_ready_options_bot.py
```

---

### 5️⃣ **Adaptive Dual Options Engine**
**File**: `adaptive_dual_options_engine.py`

**Features**:
- Dual strategy optimization
- Adaptive selection
- Real-time market adjustment

**Run**:
```bash
python adaptive_dual_options_engine.py
```

---

## Easy Launcher (Recommended)

Instead of remembering file names, use the launcher:

```bash
python START_OPTIONS_BOT.py
```

This shows a menu:
```
OPTIONS_BOT LAUNCHER
====================================================================

Available Options Trading Bots:

1. Options Hunter Bot
   File: options_hunter_bot.py
   Description: Monte Carlo optimized, 100% options trading
   Strategies: Bull Call Spreads (71.7%), Bear Put Spreads (86.4%)

2. Autonomous Options Income Agent
   File: autonomous_options_income_agent.py
   Description: Cash-secured puts + selective call buying
   Strategies: Income-focused weekly options

...

Select bot to run (1-5) or 'q' to quit:
```

Just type the number and hit Enter!

---

## Quick Command Reference

```bash
# Interactive launcher (easiest)
python START_OPTIONS_BOT.py

# Direct bot launches
python options_hunter_bot.py                    # Best win rates
python autonomous_options_income_agent.py       # Income focus
python real_world_options_bot.py                # Production ready
python tomorrow_ready_options_bot.py            # Next-day prep
python adaptive_dual_options_engine.py          # Adaptive

# With TSMOM (NEW)
python strategies/time_series_momentum.py       # Test TSMOM
```

---

## Environment Setup

Make sure your `.env` file has:

```bash
# Required for options trading
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Optional for enhanced data
POLYGON_API_KEY=your_polygon_key
```

Check with:
```bash
cat .env | grep ALPACA
```

---

## Testing Before Live

### 1. Test Bot Connection
```bash
python check_paper_account.py
```

### 2. Test Options Chain
```bash
python enhanced_options_checker.py
```

### 3. Test ML Integration
```bash
python test_ml_ensemble.py
python test_bot_startup.py
```

---

## Recommended Starting Point

**For beginners**:
```bash
python START_OPTIONS_BOT.py
# Select: 2 (Autonomous Options Income Agent)
```

**For experienced traders**:
```bash
python options_hunter_bot.py
```

**For testing**:
```bash
python demo_trading_bot.py
```

---

## Why OPTIONS_BOT.py Doesn't Exist

The system **evolved** from a single monolithic bot into **specialized agents**:

| Original Plan | Current Reality |
|--------------|-----------------|
| `OPTIONS_BOT.py` (single file) | 5+ specialized bots |
| Generic options trading | Strategy-specific bots |
| One-size-fits-all | Optimized for use case |

This is **better** because:
- ✅ Each bot is optimized for its strategy
- ✅ Easier to maintain and update
- ✅ Can run multiple bots simultaneously
- ✅ More modular and flexible

---

## Creating a Unified OPTIONS_BOT.py (If You Want)

If you really want a single `OPTIONS_BOT.py` file, I can create one that:
1. Imports all specialized bots
2. Runs them in parallel
3. Coordinates their decisions
4. Provides a unified interface

Let me know if you want this!

---

## Summary

**Don't run**: `python OPTIONS_BOT.py` ❌ (doesn't exist)

**Instead run**:
```bash
python START_OPTIONS_BOT.py              # Interactive menu
# OR
python options_hunter_bot.py             # Best for most users
```

**Your options bots are ready to trade! 🚀**
