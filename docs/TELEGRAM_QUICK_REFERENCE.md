# Telegram Bot Quick Reference Card

**Bot:** @LVXCAS_bot
**Status:** ✅ PRODUCTION READY (15/15 commands tested)

---

## Quick Command List

### 📊 Status & Monitoring
```
/status      System status (Forex/Futures/Options)
/positions   Open positions across all accounts
/pnl         Real-time P&L (OANDA + Alpaca)
/risk        Risk limits & kill-switch status
/regime      Market conditions (Fear & Greed)
/rebalance   Portfolio allocation (40/30/30)
/pipeline    Strategy deployment status
```

### 🔍 Scanners (NEW)
```
/earnings    Upcoming earnings plays (IV > 50)
/confluence  Multi-timeframe setups (1H/4H/Daily align)
/viral       Trending stocks on Reddit
```

### ⚙️ Regime Auto-Switcher
```
/regime auto    Enable auto-switching
/regime manual  Disable auto-switching
/regime status  Check switcher status
```

### 🚀 Remote Control
```
/start_forex    Start Forex Elite
/start_futures  Start Futures scanner
/start_options  Start Options scanner
/restart_all    Restart all systems
```

### 🚨 Emergency
```
/risk override  Reset kill-switch
/stop           Emergency stop all trading
/kill_all       Nuclear option (kill all)
```

### 📚 Help
```
/help           Show all commands
```

---

## Most Useful Commands

**Daily routine:**
1. `/status` - Check what's running
2. `/pnl` - See how you're doing
3. `/earnings` - Check today's opportunities
4. `/confluence` - Find high-probability setups
5. `/viral` - What's trending?

**Weekly routine:**
1. `/rebalance` - Check allocation drift
2. `/pipeline` - See new strategies

**When market conditions change:**
1. `/regime` - Check current regime
2. `/regime auto` - Let system adapt automatically

---

## How to Start Bot

```bash
python telegram_remote_control.py
```

Then send commands to **@LVXCAS_bot** from your phone!

---

## Example Workflow

**Morning (6:30 AM):**
```
You: /status
Bot: Forex Elite RUNNING, Options STOPPED...

You: /pnl
Bot: Total P&L: +$234.56 (+2.34%)

You: /earnings
Bot: AAPL - 10/28, LONG_STRADDLE, IV 68...

You: /confluence
Bot: AAPL - Score: 85, BULLISH, Entry $227.50...
```

**Market Open (9:30 AM):**
```
You: /viral
Bot: TSLA - 150 mentions, 320% spike, BUY signal...

You: /start_options
Bot: Options Scanner STARTED
```

**Midday Check (12:00 PM):**
```
You: /positions
Bot: FOREX: EUR/USD LONG...

You: /risk
Bot: RISK STATUS: OK, Daily loss limit: 2%...
```

**Market Close (4:00 PM):**
```
You: /pnl
Bot: Total P&L: +$456.78 (+4.56%)

You: /rebalance
Bot: Portfolio is balanced ✓
```

---

**Status:** All systems operational
**Test Date:** 2025-10-18
**Pass Rate:** 100% (15/15 commands)
