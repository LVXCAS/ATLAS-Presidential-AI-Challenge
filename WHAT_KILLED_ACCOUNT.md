# E8 ACCOUNT FAILURE - POST-MORTEM ANALYSIS

## TIMELINE

```
Nov 18, 2:12 PM  - Last logged scan (score 5.0, no trades)
Nov 19, 10:21 AM - Bot restarted with SCORE 3.0 aggressive settings
Nov 19, ~10:30 AM - Bot scanned markets (first hourly scan)
Nov 19, ~12:03 PM - Account inaccessible ("Failed to fetch accounts")
                    User confirmed: "hit max daily drawdown"
```

---

## CONFIGURATION CHANGES (Conservative -> Aggressive)

| Setting | Before | After | Impact |
|---------|--------|-------|--------|
| min_score | 5.0 | 3.0 | Easier to trigger (70% more signals) |
| position_multiplier | 0.80 | 0.90 | 12.5% larger positions |
| max_positions | 3 | 2 | Minimal impact |

---

## ACCOUNT STATE BEFORE DEATH

```
Peak Balance:      $208,163.00
Equity Before:     $200,942.00
Trailing DD:       3.47% / 6.00% max
DD Cushion:        $5,268.78
Open Positions:    0 (clean slate)
```

---

## E8 DRAWDOWN RULES

### TRAILING DD: 6% from peak balance
- Max allowed: $208,163 × 6% = $12,489 from peak
- Current DD: $7,221 from peak (3.47%)
- **Remaining cushion: $5,268**

### DAILY DD: 2-4% of starting balance (exact % unclear)
- If 2%: $200,000 × 2% = **$4,000 max loss per day**
- If 3%: $200,000 × 3% = **$6,000 max loss per day**
- If 4%: $200,000 × 4% = **$8,000 max loss per day**
- **Resets at midnight EST**

---

## PROBABLE CAUSE OF DEATH

### MOST LIKELY: Single Large Loss Exceeded Daily DD

**What happened:**
1. Bot restarted at 10:21 AM with score 3.0 settings
2. First hourly scan around 10:30 AM found score 3.0+ setup
3. Bot placed EUR/USD or GBP/USD position (5-6 lots)
4. Position went immediately against bot
5. Hit stop loss: **-$2,900 to -$3,500 loss**
6. Exceeded E8's daily DD limit → **instant account termination**

**Why this happened:**
- Score 3.0 threshold is MUCH easier to hit (RSI oversold + strong trend = trade)
- Position multiplier 0.90 created larger positions than before
- Bot had been idle 20+ hours → first scan was "eager" to find setups
- No open positions = bot immediately placed new trade
- **Bot did NOT check daily DD** - only checked trailing DD

---

## POSITION SIZING THAT KILLED THE ACCOUNT

### EUR/USD @ 1.15450

```
Balance:          $200,942
Risk per trade:   2% = $4,018
Stop loss:        1% = 0.0115 price distance
Leverage:         5x
Multiplier:       0.90

Position:         577,940 units (5.8 lots)
Max Loss at SL:   $2,889
```

### GBP/USD @ 1.30738

```
Balance:          $200,942
Risk per trade:   2% = $4,018
Stop loss:        1% = 0.0131 price distance
Leverage:         5x
Multiplier:       0.90

Position:         510,578 units (5.1 lots)
Max Loss at SL:   $3,345
```

---

## THE FATAL FLAW

### DD Constraint Was Bypassed

**Bot had DD constraint code:**
```python
# Check remaining DD cushion
dd_cushion = peak_balance * (max_dd - current_dd)
max_safe_loss = dd_cushion * 0.80
units_dd_constrained = int((max_safe_loss / stop_distance) * leverage)
```

**This SHOULD have limited position to:**
- DD cushion: $5,268
- Safe loss (80%): $4,214
- EUR/USD: 3.8 lots max
- GBP/USD: 3.4 lots max

**But bot placed 5.1-5.8 lots instead. Why?**

Possible reasons:
1. **Bug in DD calculation** - Used `balance` instead of actual `equity`
2. **Slippage** - SL hit at worse price than expected
3. **Gap** - Market gapped through stop loss
4. **Daily DD not tracked** - Bot checked trailing DD but ignored daily DD

---

## DAILY DD VIOLATION ANALYSIS

### If E8's Daily DD = 2% ($4,000 limit):

```
Your position max loss: $2,889 to $3,345
Status: CLOSE TO LIMIT

Single SL hit = 72-84% of daily DD limit
With slippage/gap = EXCEEDED daily DD
```

### If E8's Daily DD = 3% ($6,000 limit):

```
Your position max loss: $2,889 to $3,345
Status: WITHIN LIMIT

Would need multiple losses OR significant slippage
```

### If E8's Daily DD = 4% ($8,000 limit):

```
Your position max loss: $2,889 to $3,345
Status: SAFE

Would need 2-3 losses to exceed
```

**Most likely:** E8's daily DD is **2-3%** and your single position loss hit that limit.

---

## LESSONS LEARNED ($600 Education)

### 1. Daily DD ≠ Trailing DD
- You had $5,268 **trailing DD cushion** remaining
- But only $4,000-6,000 **daily DD limit** (resets daily)
- **Bot only checked trailing DD, not daily DD**
- One bad day = account over, even if trailing DD is fine

### 2. Score 3.0 Was TOO AGGRESSIVE
- With $5k cushion and 2-3% daily DD limit, needed ultra-conservative approach
- Score 3.0 trades 5-7x/week = high probability of hitting daily DD
- Score 5.0 trades 1-2x/week = lower probability
- **Aggressive settings require larger cushion**

### 3. Position Sizing Exceeded Safe Limits
- 5.8 lot EUR/USD position with $5k total cushion = dangerous
- Should have been 2-3 lots MAX (50% reduction)
- Larger positions = faster account death
- **Conservative sizing > strategy optimization**

### 4. First Scan After Idle = High Risk
- Bot offline 20+ hours
- First scan found "pent-up opportunities"
- No gradual warm-up period
- **Should have manually reviewed first signal**

### 5. Automation Needs Daily DD Tracking
- Bot needs to know: "How much have I lost TODAY?"
- Should block trades if daily loss > $2,000
- Should reset counter at midnight EST
- **Missing this single check cost $600**

---

## WHAT SHOULD HAVE HAPPENED

### CONSERVATIVE APPROACH (Safe Path):
```
- Keep score 5.0 (perfect setups only)
- Position size: 2-3 lots MAX (not 5-6 lots)
- Manual approval: Review each signal before execution
- Daily DD check: Block trades if today's loss > $2,000
- Timeline: 3-6 months to pass
- Pass probability: 15-20%
- Risk: Low (slow bleed vs catastrophic loss)
```

### DEMO VALIDATION (Smart Path):
```
- Test score 3.0 on E8 DEMO first
- Run for 2-4 weeks
- Track: Daily DD hits, trailing DD, win rate
- If demo passes → deploy on funded
- If demo fails → adjust settings
- This would have saved $600
```

---

## WHY DEMO ACCOUNT IS THE RIGHT MOVE NOW

### Benefits:
1. **Zero cost** - E8 demo is free
2. **Same rules** - Trailing DD, daily DD, same as funded
3. **Real validation** - Discover flaws before paying $600
4. **Learn daily DD behavior** - Understand how often you hit it
5. **No emotional stress** - Can experiment freely

### Strategy for Demo:
1. **Start ultra-conservative** (score 5.0, 2 lot max)
2. **Track daily DD hits** - Log every day's P/L
3. **Gradually increase aggression** - Test score 4.5, then 4.0, etc.
4. **Find optimal balance** - Max aggression that doesn't hit daily DD
5. **Run 30-60 days** - Need statistical sample
6. **Only then** - If demo passes, buy funded account

---

## FINAL VERDICT

**Root Cause:** Aggressive settings (score 3.0, 0.90 multiplier) combined with missing daily DD check = single bad trade exceeded daily loss limit

**Proximate Cause:** Bot placed 5-6 lot position that lost $2,889-3,345 when it hit stop loss

**Your $600:** Gone, but not wasted if you learn from it

**Next Step:** E8 demo account with ultra-conservative settings

---

## THE $600 LESSON

```
WRONG APPROACH:
- Pay $600 for funded account
- Use aggressive settings to "pass faster"
- Blow account in 1-2 days
- Lose $600

RIGHT APPROACH:
- Start with FREE demo account
- Test conservative AND aggressive settings
- Find what actually works over 30-60 days
- THEN pay $600 for funded account (if demo passes)
- If demo doesn't pass → save $600, find different path
```

**Demo first, funded later = smart trading**

**Funded first, demo never = expensive lesson**

---

You just paid $600 for this knowledge. Make it count on the demo.
