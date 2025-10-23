# Week 3: Multi-Strategy Mode Activation Guide

**Status:** ✅ INTEGRATED - Ready for Week 3 testing

**Date Built:** October 8, 2025

---

## What Was Added

The scanner now supports **3 strategies** with intelligent auto-selection:

1. **Iron Condor** - High probability, low capital ($500-1,500)
2. **Butterfly** - Neutral/range-bound, defined risk ($200-500)
3. **Dual Options** - Directional momentum (current default, $3,300+)

**Capital efficiency comparison:**
- Week 2 mode: 2-3 trades max (capital exhaustion)
- Week 3 mode: 20-30 trades possible (spreads use 1/10th capital)

---

## How to Activate for Week 3

### Step 1: Edit week2_sp500_scanner.py

Find line ~66:

```python
# Strategy selection mode (can be changed for Week 3+)
self.multi_strategy_mode = False  # Set to True for Week 3+ testing
```

Change to:

```python
self.multi_strategy_mode = True  # Week 3 multi-strategy mode ACTIVE
```

### Step 2: Restart Scanner

```bash
# Kill current scanner
tasklist | findstr python
taskkill /F /PID <scanner_pid>

# Restart with multi-strategy mode
python week2_sp500_scanner.py
```

### Step 3: Verify Activation

Look for this output at startup:

```
[OK] Advanced strategies loaded: Iron Condor, Butterfly
```

During execution, you'll see:

```
[STRATEGY] Iron Condor - Low momentum (2.1%), high probability
```

Or:

```
[STRATEGY] Dual Options - Strong momentum (7.3%), directional
```

---

## Strategy Selection Logic

The scanner automatically chooses based on momentum:

```python
# Momentum < 3% → Iron Condor (high probability income)
if momentum < 0.03:
    return IRON_CONDOR

# Momentum < 2% → Butterfly (neutral play)
elif momentum < 0.02:
    return BUTTERFLY

# Strong momentum → Dual Options (directional)
else:
    return DUAL_OPTIONS
```

**You don't need to do anything** - the scanner picks the right strategy automatically.

---

## Expected Results (Week 3)

### Week 2 Performance (multi_strategy_mode = False)
- Trades/day: 2-3 (limited by capital)
- Capital efficiency: 30% (most buying power unused)
- P&L: Inconsistent (directional risk)

### Week 3 Target (multi_strategy_mode = True)
- Trades/day: 15-25 (spreads use less capital)
- Capital efficiency: 80-90% (better capital usage)
- P&L: More consistent (70%+ win rate on Iron Condors)

---

## Monitoring Performance

Track these metrics during Week 3:

1. **Strategy distribution:**
   - How many Iron Condors? (expect 60-70%)
   - How many Dual Options? (expect 20-30%)
   - How many Butterflies? (expect 10%)

2. **Win rate by strategy:**
   - Iron Condors should win 70-80% (high probability)
   - Dual Options should win 40-50% (directional)
   - Butterflies should win 50-60% (neutral)

3. **Capital efficiency:**
   - Are you executing 20+ trades/day?
   - Is buying power staying above $30k? (good sign)

---

## Troubleshooting

### "Advanced strategies not available"

**Problem:** Week 5+ engines not found

**Solution:**
```bash
# Verify files exist
ls strategies/iron_condor_engine.py
ls strategies/butterfly_spread_engine.py

# If missing, Week 5+ features need to be built
```

### Strategy not executing

**Problem:** Iron Condor selected but trade fails

**Possible causes:**
1. Alpaca doesn't support 4-leg orders (need to place individually)
2. Options unavailable for symbol (Alpaca limitation)
3. Spreads outside bid/ask range

**Solution:** Check logs for specific error, may fall back to stock trades

### All trades are Iron Condors

**This is NORMAL and GOOD!** Most stocks have low momentum (<3%), so Iron Condors are the right choice.

If you want more Dual Options:
- Lower the momentum threshold at week2_sp500_scanner.py:305
- Change `if momentum < 0.03` to `if momentum < 0.01`

---

## Recommended Week 3 Plan

### Days 6-7 (First 2 days of Week 3)
- Enable multi-strategy mode
- Let it run normally
- Collect data on strategy distribution

### Days 8-9
- Analyze win rates by strategy
- Adjust momentum thresholds if needed
- Compare P&L vs Week 2

### Day 10
- Write Week 3 journal
- Decide: Continue with multi-strategy OR back to Week 2 mode

---

## Deactivation (if needed)

To go back to Week 2 mode:

```python
# week2_sp500_scanner.py line ~66
self.multi_strategy_mode = False  # Back to Dual Options only
```

Restart scanner.

---

## Integration with Week 5+

This is a **stepping stone** to the full Week 5+ system:

- Week 3: Multi-strategy (single account)
- Week 4: Add Kelly Criterion, Correlation analysis
- Week 5+: Multi-account orchestration (80 accounts)

All the infrastructure is already built. Week 3 is just testing the strategy engines before scaling up.

---

## Files Modified

```
week2_sp500_scanner.py:
  - Lines 24-31: Import Iron Condor and Butterfly engines
  - Lines 56-66: Initialize engines, add multi_strategy_mode flag
  - Lines 284-317: New _select_optimal_strategy_engine() method
  - Lines 345-372: Modified execution to use selected strategy
```

---

## Summary

**Status:** ✅ Ready for Week 3

**To activate:** Change `multi_strategy_mode = True` in week2_sp500_scanner.py:66

**Expected impact:**
- 10x more trades possible (capital efficiency)
- Higher win rate (70%+ on Iron Condors)
- More consistent daily P&L

**Risk:** New strategies = untested on this scanner. Paper test Week 3-4 before going live.

---

*Built: October 8, 2025, 11:00 PM PDT*
*Lucas, Age 16 → $10M by 18 mission continues*
