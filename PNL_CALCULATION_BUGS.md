# P&L CALCULATION BUGS FOUND

**Date:** October 12, 2025
**Severity:** CRITICAL
**Impact:** Bot showing completely wrong profit/loss numbers

---

## EVIDENCE OF BUG

### Example from Logs:
```
2025-10-06 13:36:28 ET [OPEN] [INFO] MONITORING: META - P&L: $67665.66 (+2301.6%), Signal: ExitSignal.HOLD
2025-10-06 13:36:28 ET [OPEN] [INFO]   Expected P&L: $0.00
```

**Problem:** Bot claims +2301.6% profit ($67,665.66) on META position, but expected P&L is $0. This is physically impossible.

---

## ROOT CAUSES IDENTIFIED

### Bug #1: Symbol Mismatch in Broker P&L Lookup (Line 1273)

**Current Code:**
```python
for pos in all_positions:
    if pos.symbol == symbol and hasattr(pos, 'unrealized_pl'):  # <-- BUG HERE
        real_pnl = float(pos.unrealized_pl)
        return real_pnl
```

**Problem:**
- `symbol` variable contains stock ticker (e.g., "META")
- `pos.symbol` contains **option contract symbol** (e.g., "META251024C00550000")
- **These never match!**
- Bot ALWAYS falls through to paper mode estimation, even in live trading

**Impact:**
- Broker's real P&L is NEVER used
- Bot relies on paper mode estimates which can be wildly inaccurate
- Exit decisions based on fake P&L numbers

---

### Bug #2: Missing Option Symbol Storage

**Problem:**
- When bot opens a position, it tracks the underlying symbol ("META")
- But option contracts have unique symbols like "META251024C00550000"
- Bot doesn't store the `option_symbol` in `position_data`
- Can't lookup actual broker position later

**Evidence in Code (Line 1343):**
```python
option_symbol = opportunity.get('option_symbol')  # <-- This is likely None/missing
if option_symbol:
    quote = await self.options_broker.get_option_quote(option_symbol)
```

**Result:** Falls back to theoretical pricing instead of real market quotes

---

### Bug #3: Paper Mode P&L Multiplier Error (Line 1311)

**Current Code:**
```python
# Calculate P&L based on contract value difference
# Each contract represents 100 shares, so multiply by 100
price_per_contract_change = current_option_price - entry_price
total_pnl = price_per_contract_change * quantity * 100  # <-- CORRECT
```

**This part is actually CORRECT**, but the problem is `current_option_price` is being estimated incorrectly.

---

### Bug #4: Option Price Estimation Using Wrong Values

The `estimate_current_option_price()` function (lines 1326-1403) tries to:
1. Get live option quote (fails - no option_symbol stored)
2. Use professional Black-Scholes pricing (may work but can be inaccurate)
3. Fall back to simplified estimation (very inaccurate)

**Issue:** Without the actual option contract symbol, it can't get real market data and must estimate.

---

### Bug #5: Capping Gains at 2x Masks Real Losses (Lines 1303-1306)

**Current Code:**
```python
# STRICT validation to prevent fake P&L values
max_reasonable_gain = entry_price * 2  # Cap at 2x gain (100%)
if current_option_price > max_reasonable_gain:
    self.log_trade(f"Estimated price ${current_option_price:.2f} seems too high vs entry ${entry_price:.2f}, capping at 2x", "WARN")
    current_option_price = max_reasonable_gain
```

**Problem:**
- Options can easily go up 300-500% in a day
- Capping at 2x (100% gain) **throws away real profits**
- Meanwhile, there's NO cap on losses
- This asymmetry causes P&L to be systematically understated on winners

**Example:**
- Entry: $2.00
- Real price now: $8.00 (400% gain)
- Bot caps at: $4.00 (100% gain)
- **Bot misses 75% of the profit!**

---

## SPECIFIC EXAMPLES OF WRONG P&L

### META Position (Oct 6):
- **Reported:** $67,665.66 profit (+2301%)
- **Expected:** $0.00
- **Problem:** Clearly impossible, entry price must be wrong OR estimate is inflated

### Entry Prices from Logs:
Recent options filled at:
- AMD: $11.80
- GLD: $12.15, $12.40
- UBER: $3.80
- MRK: $1.06
- PFE: $0.36
- IWM: $1.69

If META showed $67,665 profit:
- **If 1 contract:** Would need price to go from ~$0.03 to $676.65 (impossible)
- **If entry was $25:** Would need to go to $701.65 (also impossible in one day)

**Conclusion:** The P&L calculation is fundamentally broken.

---

## WHY THIS IS CRITICAL

1. **Wrong Exit Decisions**
   - Bot exits based on P&L thresholds
   - With wrong P&L, it exits at wrong times
   - Cuts winners early, lets losers run

2. **Daily Loss Limit Triggered Incorrectly**
   - Bot thinks it hit -4.9% daily loss
   - Stops trading for the day
   - But actual P&L might be different

3. **Cannot Trust Performance Metrics**
   - Win rate calculations are wrong
   - Daily P&L reports are wrong
   - Can't evaluate if bot is actually profitable

4. **Risk Management Failure**
   - Bot doesn't know its real risk exposure
   - Could be down more than it thinks
   - Could blow past risk limits

---

## FIXES REQUIRED

### Fix #1: Use Option Contract Symbol for Broker Lookup

**Change Line 1273:**
```python
# OLD (BROKEN):
if pos.symbol == symbol and hasattr(pos, 'unrealized_pl'):

# NEW (FIXED):
option_symbol = position_data.get('option_symbol') or position_data['opportunity'].get('option_symbol')
if option_symbol and pos.symbol == option_symbol and hasattr(pos, 'unrealized_pl'):
```

### Fix #2: Store Option Symbol When Opening Position

When opening a position, store BOTH:
- `symbol`: Underlying ticker (META)
- `option_symbol`: Contract symbol (META251024C00550000)

Need to find where positions are created and add `option_symbol` to the data structure.

### Fix #3: Remove or Increase 2x Gain Cap

**Change Lines 1303-1306:**
```python
# OLD (TOO RESTRICTIVE):
max_reasonable_gain = entry_price * 2  # Cap at 2x gain (100%)

# NEW (MORE REALISTIC):
max_reasonable_gain = entry_price * 10  # Cap at 10x gain (900%) - realistic for options
```

Or better yet: **Remove the cap entirely** and trust broker data.

### Fix #4: Add Fallback to Last Known Price

If estimation fails, use the last known good price:
```python
if current_option_price <= 0 or current_option_price > entry_price * 10:
    # Use entry price as fallback (assume no change)
    self.log_trade(f"[WARN] Option price estimation failed, using entry price", "WARN")
    current_option_price = entry_price
```

### Fix #5: Better Logging for Debugging

Add detailed logging:
```python
self.log_trade(f"P&L Calc Debug: symbol={symbol}, option_symbol={option_symbol}, "
               f"entry=${entry_price:.2f}, current=${current_option_price:.2f}, "
               f"quantity={quantity}, pnl=${total_pnl:.2f}", "DEBUG")
```

---

## IMMEDIATE ACTION ITEMS

### Priority 1 (Do Today):
1. ✅ Fix symbol matching to use `option_symbol` instead of underlying symbol
2. ✅ Increase or remove 2x gain cap (change to 10x minimum)
3. ✅ Add validation: if P&L > $10,000 or > 500%, log warning and investigate

### Priority 2 (Do This Week):
4. ⬜ Store `option_symbol` when opening positions
5. ⬜ Add better error handling and fallbacks
6. ⬜ Log detailed P&L calculation steps for debugging
7. ⬜ Add unit tests for P&L calculations

### Priority 3 (Nice to Have):
8. ⬜ Implement real-time option quote fetching
9. ⬜ Add P&L reconciliation (compare broker vs. calculated)
10. ⬜ Alert on large P&L discrepancies

---

## TESTING PLAN

After fixes:
1. **Monitor next trade:** Check if broker P&L is actually used
2. **Compare numbers:** Broker unrealized_pl vs. calculated P&L
3. **Validate exits:** Ensure exits happen at correct P&L levels
4. **Daily reconciliation:** End-of-day P&L should match broker account

---

## RELATED ISSUES

This ties back to the earlier analysis:
- **Hardcoded max_profit/max_loss** (Lines 2677-2678) - Also affects risk calculations
- **No stop losses** - With wrong P&L, can't set accurate stops
- **No profit targets** - With wrong P&L, can't hit targets

All of these need the P&L calculation to be fixed first.

---

## SUMMARY

**The P&L calculation is completely broken because:**
1. Symbol mismatch prevents using real broker P&L
2. Paper mode estimates are inaccurate without option contract symbol
3. 2x gain cap throws away real profits
4. No proper fallbacks or validation

**Result:** Bot makes decisions on fake numbers, leading to:
- Premature exits
- False daily loss limits
- Can't evaluate real performance
- Risk management failure

**Fix Priority:** CRITICAL - Must fix before bot can be trusted with real money.

---

## FILES THAT NEED CHANGES
- `OPTIONS_BOT.py` - Lines 1261-1403 (P&L calculation functions)
- Position data structure (need to add `option_symbol` field)
- Position opening logic (need to store `option_symbol`)
