# PRE-LAUNCH CHECKLIST - Options Scanner Fixed

**Date:** October 17, 2025
**Status:** Emergency fixes complete - ready to deploy
**Urgency:** Scanner resumes in hours

---

## BEFORE YOU START

### Run Verification (5 seconds)
```bash
python verify_strategy_fixes.py
```

**Expected Result:**
```
✅ ALL TESTS PASSED - READY FOR DEPLOYMENT
Tests Passed: 6/6
```

If any test fails, DO NOT START - review failed test first.

---

## START SCANNER

### Command
```bash
python week3_production_scanner.py
```

### Watch For (First 30 seconds)
- [ ] Account verification passes
- [ ] Shows correct account ID (first 10 chars match yours)
- [ ] Market regime detected
- [ ] Confidence threshold shows 6.0+ (not 4.0)
- [ ] No errors loading strategy engines

---

## MONITOR FIRST 3 TRADES

### For Each Trade, Verify:

**1. No Stock Fallback**
- [ ] No message: "STOCK FALLBACK: XXX shares"
- [ ] If options fail, should see: "[SKIP] Options not available"
- [ ] No large stock positions created

**2. Strike Selection**
- [ ] Bull put spreads show "15% OTM" (NOT "10% OTM")
- [ ] Example: Stock at $100 → Sell put at $85 (not $90)
- [ ] Buy put $5 below sell put

**3. Confidence Score**
- [ ] All executed trades score 6.0+ (not 4.0+)
- [ ] Should see some trades rejected (quality control)

**4. Filters Working**
- [ ] Some trades show "[FILTER FAIL]" messages (good!)
- [ ] Filters: High volatility, downtrends, low volatility
- [ ] Filtered trades NOT counted as executed

**5. Position Size**
- [ ] All positions under $50k (5% of account)
- [ ] Bull put spreads: 1 contract each
- [ ] No positions over $100k

---

## RED FLAGS (STOP IF YOU SEE)

### STOP IMMEDIATELY IF:

**1. Stock Fallback Occurs**
```
❌ "STOCK FALLBACK: 5000 shares @ $..."
```
Action: Stop scanner, check lines 487-518 in adaptive_dual_options_engine.py

**2. Wrong Strike Selection**
```
❌ Stock at $100 → Sell put at $90 (should be $85)
```
Action: Stop scanner, check lines 43-50 in bull_put_spread_engine.py

**3. Low Confidence Trades**
```
❌ Score: 4.5 executing (should require 6.0+)
```
Action: Stop scanner, check lines 141-148 in week3_production_scanner.py

**4. No Filters Active**
```
❌ No "[FILTER FAIL]" messages at all
```
Action: Stop scanner, check lines 388-404 in week3_production_scanner.py

**5. Large Positions**
```
❌ Position size > $100k
```
Action: Stop scanner, check lines 457-459, 473 in week3_production_scanner.py

---

## AFTER 10 TRADES

### Check Performance

**Win Rate Check:**
- [ ] Winning trades: _____ (target 7+ out of 10)
- [ ] Losing trades: _____ (target 3 or fewer)
- [ ] Win rate: _____% (target 70%+)

**Position Size Check:**
- [ ] Largest position: $_____ (should be under $50k)
- [ ] Average position: $_____ (should be $20-40k)
- [ ] No stock positions: Confirmed ✓

**Quality Control Check:**
- [ ] Trades filtered out: _____ (expect 20-30% rejection rate)
- [ ] All executed scores 6.0+: Confirmed ✓
- [ ] Filters working: Confirmed ✓

---

## END OF DAY

### Review
- [ ] Total trades executed: _____
- [ ] Win rate: _____%
- [ ] Account change: +/- $_____
- [ ] Largest position: $_____
- [ ] Stock fallback trades: _____ (should be 0)

### Compare to Yesterday
| Metric | Yesterday | Today | Better? |
|--------|-----------|-------|---------|
| Win Rate | 33.3% | ____% | [ ] Yes |
| Max Position | $1.4M | $____ | [ ] Yes |
| Account Loss | -$88k | $____ | [ ] Yes |
| Stock Fallbacks | 2 | ____ | [ ] Yes (0) |

---

## FILES TO REVIEW IF ISSUES

1. **`C:\Users\lucas\PC-HIVE-TRADING\FIX_SUMMARY.txt`**
   - Quick overview of all fixes

2. **`C:\Users\lucas\PC-HIVE-TRADING\OPTIONS_STRATEGY_FIX_REPORT.md`**
   - Detailed analysis of problems and solutions

3. **`C:\Users\lucas\PC-HIVE-TRADING\OPTIONS_FIX_QUICK_START.md`**
   - Usage guide and monitoring instructions

4. **`C:\Users\lucas\PC-HIVE-TRADING\verify_strategy_fixes.py`**
   - Automated verification script

---

## EMERGENCY CONTACTS

### Stop Scanner
```bash
# Press Ctrl+C in terminal
# Or close terminal window
```

### Check Logs
```bash
# Scanner logs appear in console
# Look for ERROR messages
# Look for FILTER FAIL messages (good!)
```

### Rollback
If major issues:
1. Stop scanner (Ctrl+C)
2. Close all positions in Alpaca
3. Review logs for root cause
4. Re-run verify_strategy_fixes.py
5. Check documentation files

---

## QUICK REFERENCE

### What Should Happen
✅ Stock fallback disabled (skip if options unavailable)
✅ Strikes at 15% OTM (not 10%)
✅ Only 6.0+ scores execute
✅ Filters reject bad trades
✅ Max $50k positions

### What Should NOT Happen
❌ No stock positions created
❌ No strikes at 10% OTM
❌ No trades scoring under 6.0
❌ No positions over $100k
❌ No ignoring filters

---

## READY TO LAUNCH

**Pre-launch:**
- [ ] Ran verify_strategy_fixes.py → All tests passed
- [ ] Read FIX_SUMMARY.txt
- [ ] Understand what changed
- [ ] Know what to monitor
- [ ] Ready to stop if issues occur

**Launch command:**
```bash
python week3_production_scanner.py
```

**Expected result:**
70-80% win rate, safe positions, quality trades only

---

**Good luck! Monitor closely for first few trades.**

*Last updated: October 17, 2025*
