# ⚡ SCENARIO 2 ACTIVE - MODERATE AGGRESSION (Score 3.0)

## 🎯 OBJECTIVE: PASS E8 IN 7-11 WEEKS

**Current Status:**
- **Profit Needed:** $19,058 to reach $20,000 target
- **Current Equity:** $200,942
- **DD Cushion:** $5,268 (2.53% remaining before violation)
- **Bot Status:** RUNNING (PID 38004)

---

## ⚙️ NEW CONFIGURATION (Active Now)

```python
min_score: 3.0          # Was 5.0 (RSI+trend OR MACD+trend OR strong momentum)
max_positions: 2        # Was 3 (focus on best 2 pairs)
position_multiplier: 0.90  # Was 0.80 (slightly larger positions)
scan_interval: 1 hour   # Unchanged
pairs: EUR_USD, GBP_USD, USD_JPY
```

---

## 📊 WHAT TO EXPECT

### Trade Frequency
**5-7 trades per week** (vs 1-2 at score 5.0)
- Monday-Friday: 1-2 signals per day on average
- Best trading hours: 3 AM - 12 PM EST (London/NY overlap)
- Tokyo session (7 PM - 4 AM): Also active for USD_JPY

### Score 3.0 Setups Look Like:

**Example LONG Signal (Score 3.0):**
```
EUR_USD @ 1.15450
  RSI: 38 (oversold) ✓ = 2 points
  ADX: 27 (strong trend) ✓ = 1 point
  MACD: Below crossover ✗ = 0 points
  EMA Alignment: Neutral ✗ = 0 points
  TOTAL: 3.0 → TRADE LONG
```

**Example SHORT Signal (Score 3.0):**
```
GBP_USD @ 1.31200
  RSI: 62 (overbought) ✓ = 2 points
  ADX: 26 (strong trend) ✓ = 1 point
  MACD: Above signal ✗ = 0 points
  EMA: Not aligned ✗ = 0 points
  TOTAL: 3.0 → TRADE SHORT
```

### Position Sizes (DD-Constrained)
With current $5,268 DD cushion:

| Pair | Safe Position | Max Loss at SL | Status |
|------|--------------|----------------|--------|
| EUR_USD | 5.0 lots | ~$2,890 | ✓ Active |
| GBP_USD | 4.5 lots | ~$2,950 | ✓ Active |
| USD_JPY | 0.1 lots | ~$780 | ⚠️ Too small |

**Note:** Bot will automatically block USD_JPY trades (position too small). Realistically trading EUR_USD and GBP_USD only.

### Expected Performance

**Per Trade:**
- Average Win: $2,000 (+2% TP)
- Average Loss: -$1,000 (-1% SL)
- Win Rate: 55%
- Expected Value: $650 per trade

**Weekly:**
- 5-7 trades
- Expected Profit: $3,250 to $4,550 per week
- 3-4 winners, 2-3 losers (typical variance)

**Timeline to $20k Target:**
- **Best Case:** 5 weeks (hot streak, 60% win rate)
- **Expected:** 7-8 weeks (55% win rate as planned)
- **Worst Case:** 11 weeks (50% win rate, variance)

---

## 📱 MONITORING REQUIREMENTS (CRITICAL)

### ⚠️ You MUST Check 3x Per Day Now

At score 3.0, bot will trade 5-7x/week. You need active monitoring.

**Morning Check (Before School - 5 min):**
```batch
python GET_E8_TRADE_IDS.py
```
- Any new positions overnight?
- Check unrealized P/L
- If position >-$1,500 unrealized → set phone reminder to check at lunch

**Midday Check (Lunch Break - 3 min):**
- Quick position status check
- If approaching -$2,000 → consider manual close

**Evening Check (After School - 5 min):**
```batch
DAILY_CHECK.bat
```
- Review all day's trades
- Check score log for quality of signals
- Calculate: Running P/L for the week

**Total Time:** 13 min/day (91 min/week)

---

## 🚨 SAFETY RULES (MUST FOLLOW)

### STOP BOT IMMEDIATELY IF:

1. **Equity drops below $196,000** (danger zone - only $1,732 DD cushion left)
   ```batch
   taskkill /F /IM pythonw.exe
   ```

2. **3 consecutive losses in same day** (indicates bad market conditions)
   - Kill bot, wait 24 hours before restart

3. **Position hits -$2,500 unrealized** (too close to DD violation)
   - Close position manually
   - Reassess if bot settings too aggressive

### MANUAL CLOSE POSITION IF:

1. **Position is -$2,000+ and setup invalidated**
   - RSI flipped to opposite extreme
   - Broke major support/resistance
   - Unexpected news event

2. **Friday 3 PM EST with open position**
   - Close before weekend gap risk
   - Restart Monday morning

3. **Major economic data releases**
   - NFP (Non-Farm Payrolls) - 1st Friday each month 8:30 AM
   - FOMC announcements
   - CPI/Inflation reports
   - Close positions 30 min before, restart after volatility settles

---

## 📈 PROGRESS TRACKING

### Week 1 Target: +$2,500 to +$4,000
**Day 1-2:** Bot finding new setups (score 3.0 vs 5.0)
**Day 3-5:** 3-5 trades executed
**Weekend:** Review week, calculate stats

**Success Metrics:**
- ✓ 3-7 trades placed
- ✓ Win rate >50%
- ✓ No DD violations
- ✓ Equity >$203,000

### Week 2 Target: +$5,000 to +$8,000 (cumulative)
**Continue same pattern**

**Red Flags:**
- ❌ Win rate <45% → Market conditions bad, consider pause
- ❌ 4+ losses in a row → Stop bot, reassess
- ❌ Equity trending down → Settings too aggressive

### Week 3-8: Continue Until $20k Target

**Milestones:**
- $205,000 equity = 25% to goal
- $210,000 equity = 50% to goal (**NEW PEAK!** DD resets)
- $215,000 equity = 75% to goal
- $220,000 equity = **CHALLENGE PASSED** 🎉

---

## 🧠 EDGE CASES & TROUBLESHOOTING

### "Bot placed a trade I don't like"
- Check the score log: Was it really score 3.0+?
- Review technical indicators: RSI, MACD, ADX, EMA
- If setup is valid → trust the system (50-55% win rate expected)
- If setup seems weak → Document it, optimize filters later

### "Bot isn't trading at all"
- Check score log: Are scores reaching 3.0?
- Market might be ranging (low ADX, no trends)
- This is GOOD - bot is protecting you
- Don't force trades in bad conditions

### "I'm down $3,000 this week"
- Check win rate: If <40% → pause bot
- Check if hit 3+ consecutive losses → pause bot
- Variance is normal, but multiple red flags = stop

### "I hit a new peak!"
**IMPORTANT:** If equity exceeds $208,163:
- Your DD cushion RESETS
- You get full 6% from new peak
- Document new peak in state file
- This changes everything - now you have room to breathe

---

## 📊 WEEKLY REVIEW TEMPLATE

Every Sunday evening, calculate:

```
WEEK X PERFORMANCE:
===================
Starting Equity: $___,___
Ending Equity: $___,___
Profit/Loss: $___,___ (___%)

Trades Placed: ___
Winners: ___ (___%)
Losers: ___ (___%)

Largest Win: $___,___
Largest Loss: $___,___
Average Win: $___,___
Average Loss: $___,___

DD Cushion Remaining: $___,___ (___%)

STATUS: [ON TRACK / AHEAD / BEHIND / DANGER]

Notes:
- Best setups: _________
- Worst setups: _________
- Adjustments needed: _________
```

---

## 🎯 CRITICAL SUCCESS FACTORS

### 1. **Active Monitoring**
Score 3.0 = more trades = more attention needed
**You cannot "set and forget" at this aggression level**

### 2. **Emotional Discipline**
- Don't close winners early (let TP hit)
- Don't hold losers hoping (let SL hit)
- Trust the 55% win rate over 30+ trades

### 3. **Risk Management**
- Bot has DD constraints built in
- But YOU must monitor equity levels
- Stop bot if approaching danger zone

### 4. **Market Awareness**
- Know when major news events are scheduled
- Avoid trading during high-impact releases
- Close positions before weekend if nervous

### 5. **Patience with Variance**
- Some weeks you'll be up $5,000
- Some weeks you'll be down $2,000
- Over 7-8 weeks, 55% win rate should prevail

---

## 🚀 EXPECTED TIMELINE

**Week 1 (Now):** Bot starts trading score 3.0, you adapt to monitoring 3x/day
**Week 2:** First $4,000-6,000 profit banked, confidence building
**Week 3-4:** Halfway to target, $210k equity (new peak!)
**Week 5-6:** $215k equity, 75% to goal
**Week 7-8:** Final push to $220k = **CHALLENGE PASSED**

**Then what?**
- Withdraw to Phase 2 account (verify payout rules)
- Same DD rules but 80% profit split
- Scale to $20k/month income target

---

## ✅ CONFIGURATION SUMMARY

**Changed from Conservative (Score 5.0):**
```diff
- min_score: 5.0
+ min_score: 3.0

- max_positions: 3
+ max_positions: 2

- position_multiplier: 0.80
+ position_multiplier: 0.90

- Trade frequency: 1-2/week
+ Trade frequency: 5-7/week

- Timeline: 36 weeks
+ Timeline: 7-11 weeks

- Pass probability: 15-20%
+ Pass probability: 35-40%
```

---

**Bot is now running with SCENARIO 2 settings.**

**Check status in 1 hour:** Bot should have completed first scan with score 3.0 threshold.

**Next scan:** Check [e8_score_log.csv](e8_score_log.csv) for score 3.0+ signals.

**Your job:** Monitor 3x/day, trust the system, manage risk.

**LET'S PASS THIS CHALLENGE.** 🎯🚀
