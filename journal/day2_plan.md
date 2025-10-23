# Day 2 Battle Plan - October 8, 2025

## Mission: Execute Cleanly, Recover Ground

**Target:** +2-3% on the day
**Trades:** 5/5 successful executions
**Risk:** Max -20% stop loss per position

---

## Pre-Market Checklist (6:00 AM PDT)

### System Status Check
```bash
# 1. Check account status (FIXED version)
python check_account_status.py

# 2. Verify positions
# Look for:
# - AES -21 contracts bug (still there?)
# - INTC winning positions (+$380)
# - Total P&L vs yesterday

# 3. Check buying power
# Need: $20k minimum reserve
# Options BP: Should be ~$44k
```

### Start Background Systems (6:15 AM)
```bash
# 1. Start stop loss monitor (new today!)
start python stop_loss_monitor.py

# Wait 1 minute, verify it's running:
tasklist | findstr python
```

### Launch Scanner (6:25 AM)
```bash
# 2. Kill any old scanner processes first
taskkill /F /IM python.exe

# 3. Start FIXED scanner (5 min before market)
python week2_sp500_scanner.py

# Watch for:
# - No emoji crashes
# - Correct timezone (PDT)
# - "WEEK 2 S&P 500 MOMENTUM SCANNER" header
# - "503 S&P 500 stocks" universe size
```

---

## Trade Execution Plan

### First Scan (6:30-6:35 AM)
**What to expect:**
- 503 stocks scanned
- ~100+ opportunities with score 4.0+
- Top 3-5 will be selected for execution

**Watch for:**
- Order verification logs: `[VERIFY] PRE-SUBMIT: symbol=..., qty=X, side=...`
- Quantity matches: `[VERIFY] PUT filled correctly: X contracts`
- Any warnings: `[WARNING] QTY MISMATCH!` ← This is the AES bug detector

### Execution Rules
1. **Let the scanner run** - Don't interfere unless error
2. **Watch buying power** - Should stay above $20k
3. **Verify each order** - Check logs for quantity mismatches
4. **Count trades** - Should reach 5/5 today (not 3/5)
5. **Monitor P&L** - Check every hour via `check_account_status.py`

---

## Hourly Monitoring Schedule

### 7:00 AM - First Check
```bash
python check_account_status.py
```
**Look for:**
- How many trades executed? (Target: 2-3 by 7 AM)
- Any quantity mismatches?
- P&L trending positive or negative?

### 9:00 AM - Mid-Morning Check
```bash
python check_account_status.py
```
**Look for:**
- How many trades executed? (Target: 4-5 by 9 AM)
- Buying power remaining? (Target: $15k+)
- Stop loss triggered? (Should be none)

### 11:00 AM - Late Morning Check
```bash
python check_account_status.py
```
**Look for:**
- All 5 trades executed? (Target: 5/5 by 11 AM)
- Overall P&L? (Target: +1% minimum)
- Any positions in danger of stop loss?

### 12:30 PM - Pre-Close Check
```bash
python check_account_status.py
```
**Look for:**
- Final P&L? (Target: +2-3%)
- Any last-minute opportunities?
- Prepare end-of-day report

---

## Success Criteria

### Minimum Success (Must Hit)
- ✅ 5/5 trades executed (all slots used)
- ✅ No emoji crashes
- ✅ No quantity mismatches (AES bug fixed)
- ✅ Scanner runs full day without fatal errors
- ✅ End day flat or better (-0% to +1%)

### Target Success (Aiming For)
- 🎯 +2-3% on the day
- 🎯 All trades fill properly (no fallbacks)
- 🎯 Order verification catches any issues
- 🎯 Stop loss doesn't trigger
- 🎯 Buying power managed well

### Stretch Success (Dream Scenario)
- 🚀 +5% on the day
- 🚀 All 5 trades profitable
- 🎯 AES bug self-corrects
- 🚀 New bugs = 0

---

## Risk Management

### Position Limits
- **Max allocation:** 10% per trade ($10k per position)
- **Max contracts:** 1-2 per options trade
- **Max positions:** 5 active trades at once
- **Stop loss:** -20% automatic close

### Buying Power Rules
- **Reserve:** $20k minimum at all times
- **Options BP:** Don't use more than 50% at once
- **Per trade:** ~$4-5k collateral for cash-secured puts

### Emergency Procedures
**If stop loss triggers:**
1. Let monitor close the position automatically
2. Check logs to understand why
3. Reduce position size on next trade

**If AES bug appears again:**
1. Document order ID
2. Check actual fill via Alpaca dashboard
3. Contact Alpaca support
4. Close position manually if needed

**If scanner crashes:**
1. Check error message
2. Fix if obvious (encoding, API, etc.)
3. Restart scanner
4. Don't panic - market is 6.5 hours long

---

## Watch List - Specific Issues

### AES Position (-21 contracts)
- **Current Status:** -$210 loss, -21 contracts
- **Action:** Monitor for self-correction or manual close
- **Decision Point:** If reaches -$300, close manually

### Buying Power Exhaustion
- **Yesterday's Issue:** AAPL forced stock fallback
- **Today's Fix:** 10% max allocation (was 45%)
- **Monitor:** Should have $20k+ reserve after first 2 trades

### Failed Trade Counting
- **Yesterday's Issue:** Failed trades wasted slots
- **Today's Fix:** Bug #3 fixed - failures don't count
- **Monitor:** Should see 5 successful trades, failures retry

---

## End of Day Requirements

### Must Complete
1. **Generate Day 2 journal** (copy Day 1 template)
2. **Record all trades** (entry, exit, P&L)
3. **Note any bugs found** (even small ones)
4. **Update Week 2 targets** (adjust if needed)
5. **Plan Day 3** (what to improve)

### Data to Capture
- Starting portfolio: (from morning check)
- Ending portfolio: (from EOD check)
- Daily P&L: $XXX (+X.XX%)
- Trades executed: X/5
- Bugs found: X (describe each)
- Lessons learned: (minimum 3)

---

## Tomorrow's Theme

**"Execution over perfection"**

You've built the system. You've fixed the bugs. Now execute:
- 5/5 trades
- No crashes
- Positive returns
- Learn from data

If you hit +2-3% today, you're on track.
If you hit +5%, you're ahead of schedule.
If you hit flat, you learned without losing.
If you hit negative, you analyze and improve.

**The only failure is not executing.**

---

## Quick Reference Commands

```bash
# Check status
python check_account_status.py

# Start stop loss monitor
start python stop_loss_monitor.py

# Start scanner (6:25 AM only)
python week2_sp500_scanner.py

# Kill processes if needed
taskkill /F /IM python.exe

# Check running processes
tasklist | findstr python
```

---

**Set alarm for 5:30 AM. Sleep well. Execute tomorrow.**

*Battle plan created: 10:05 PM PDT, October 7, 2025*
