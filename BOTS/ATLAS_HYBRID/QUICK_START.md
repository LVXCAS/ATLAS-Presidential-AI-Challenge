# ATLAS - QUICK START

## Problem Solved: "No Trades"

**Cause:** E8ComplianceAgent blocked all trades (8.6% DD on OANDA account)
**Fix:** E8ComplianceAgent DISABLED in config
**Status:** READY TO TRADE ✅

---

## Start Trading (Choose One)

### Conservative (0-2 trades/week)
```bash
cd BOTS/ATLAS_HYBRID
python run_paper_training.py --phase validation --days 60
```

### Aggressive (15-25 trades/week) - RECOMMENDED
```bash
cd BOTS/ATLAS_HYBRID
python diagnostics/adjust_threshold.py --mode exploration
python run_paper_training.py --phase exploration --days 20
```

---

## Commands

### Check if running:
```bash
tasklist | findstr python
```

### View logs:
```bash
cd BOTS/ATLAS_HYBRID
type logs\paper_training_*.log | more
```

### Check balance:
```bash
python -c "from adapters.oanda_adapter import OandaAdapter; print(OandaAdapter().get_account_balance())"
```

### Toggle E8ComplianceAgent:
```bash
python diagnostics/disable_e8_agent.py
```

### Adjust threshold:
```bash
python diagnostics/adjust_threshold.py --mode exploration  # 3.5 (15-25 trades/week)
python diagnostics/adjust_threshold.py --mode refinement   # 4.0 (10-15 trades/week)
python diagnostics/adjust_threshold.py --mode validation   # 4.5 (0-2 trades/week)
```

---

## Current Settings

**E8ComplianceAgent:** DISABLED (for OANDA practice)
**Score Threshold:** 4.5 (ultra-conservative)
**Account Balance:** $182,788.16
**Pairs:** EUR/USD, GBP/USD, USD/JPY

---

## Diagnostics

### See why trades are blocked:
```bash
cd BOTS/ATLAS_HYBRID
python diagnostics/trade_blocking_analyzer.py
```

### Check E8ComplianceAgent:
```bash
python diagnostics/check_e8_blocking.py
```

---

## Re-Enable E8ComplianceAgent

**When:** Deploying to real E8 challenge with fresh $200k account

**How:**
```bash
python diagnostics/disable_e8_agent.py  # Toggles it ON
```

---

**Read [START_TRADING_NOW.md](START_TRADING_NOW.md) for full details.**
