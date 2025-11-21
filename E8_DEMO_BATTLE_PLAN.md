# E8 DEMO ACCOUNT - BATTLE PLAN

## OBJECTIVE: Validate bot on FREE demo before risking another $600

---

## PHASE 1: ULTRA-CONSERVATIVE BASELINE (Week 1-2)

### Configuration:
```python
min_score: 6.0              # Perfect setups ONLY
max_positions: 1            # One position at a time
position_multiplier: 0.60   # 60% of calculated size (very conservative)
risk_per_trade: 0.015       # 1.5% risk (down from 2%)
scan_interval: 3600         # 1 hour
pairs: ['EUR_USD', 'GBP_USD']  # Top 2 only (skip USD_JPY)
```

### Expected Behavior:
- Trade frequency: 0-1 per week (score 6.0 is rare)
- Position size: 1-2 lots max
- Max loss per trade: $1,000-1,500
- Timeline to pass: 6-12 months (if this even works)

### Goals:
- **Don't hit daily DD** (primary objective)
- Track: How many days have >$2k loss
- Track: Win rate at score 6.0
- Baseline: Can the bot survive at all?

### Success Criteria (2 weeks):
- Zero daily DD violations
- At least 1-2 trades placed
- Win rate >40%
- Equity trending up (even if slowly)

---

## PHASE 2: MODERATE CONSERVATIVE (Week 3-4)

*Only proceed if Phase 1 succeeds*

### Configuration:
```python
min_score: 5.0              # Very strong setups
max_positions: 1            # Still one at a time
position_multiplier: 0.70   # 70% sizing
risk_per_trade: 0.018       # 1.8% risk
```

### Expected Behavior:
- Trade frequency: 1-2 per week
- Position size: 2-3 lots
- Max loss per trade: $1,500-2,000

### Goals:
- Still avoid daily DD violations
- Faster profit accumulation than Phase 1
- Find the edge of "safe but effective"

### Success Criteria (2 weeks):
- Max 1 daily DD warning (loss >$3k in one day)
- Zero actual daily DD violations
- Win rate >50%
- Positive equity growth

---

## PHASE 3: BALANCED AGGRESSION (Week 5-6)

*Only proceed if Phase 2 succeeds*

### Configuration:
```python
min_score: 4.0              # Strong setups
max_positions: 2            # Two positions allowed
position_multiplier: 0.75   # 75% sizing
risk_per_trade: 0.02        # Back to 2% risk
```

### Expected Behavior:
- Trade frequency: 2-4 per week
- Position size: 3-4 lots
- Max loss per trade: $2,000-2,500

### Goals:
- Test limits of daily DD tolerance
- Faster path to $20k target
- Identify dangerous market conditions

### Success Criteria (2 weeks):
- Max 2 daily DD warnings
- Zero actual daily DD violations
- Win rate >50%
- On track to pass in 8-12 weeks

---

## PHASE 4: OPTIMAL AGGRESSION (Week 7-8+)

*Only proceed if Phase 3 succeeds*

### Configuration:
```python
min_score: 3.5              # Good setups
max_positions: 2
position_multiplier: 0.80
risk_per_trade: 0.02
```

### Expected Behavior:
- Trade frequency: 3-6 per week
- Position size: 4-5 lots
- Fast path to target (6-10 weeks if works)

### Goals:
- Maximum safe aggression
- Identify the breaking point
- Determine if this aggression level is viable

### Decision Point:
- **If succeeds:** This is your funded account config
- **If fails:** Roll back to Phase 3 settings

---

## DAILY DD TRACKING (CRITICAL)

### Add This to Bot Code:

```python
# Track daily P/L
self.daily_pnl_file = Path('daily_pnl_tracker.json')
self.daily_start_equity = None
self.daily_max_loss = 4000  # E8's likely daily DD limit

def check_daily_dd(self):
    """Check if we've hit daily DD limit"""
    today = datetime.now().strftime('%Y-%m-%d')

    # Load daily tracker
    if self.daily_pnl_file.exists():
        with open(self.daily_pnl_file, 'r') as f:
            tracker = json.load(f)
    else:
        tracker = {}

    # Initialize today's starting equity if needed
    if today not in tracker:
        tracker[today] = {
            'start_equity': self.current_equity,
            'current_loss': 0
        }

    # Calculate today's loss
    start = tracker[today]['start_equity']
    current_loss = start - self.current_equity
    tracker[today]['current_loss'] = current_loss

    # Save tracker
    with open(self.daily_pnl_file, 'w') as f:
        json.dump(tracker, f, indent=2)

    # Check if exceeded
    if current_loss >= self.daily_max_loss:
        print(f"[DAILY DD] Hit daily loss limit: ${current_loss:,.2f}")
        return True  # Block trading for rest of day

    if current_loss >= self.daily_max_loss * 0.75:
        print(f"[WARNING] Approaching daily DD: ${current_loss:,.2f} / ${self.daily_max_loss:,.2f}")

    return False
```

### Usage in Scan:
```python
def scan_forex(self):
    # Check daily DD FIRST
    if self.check_daily_dd():
        print("[SKIP] Daily DD limit reached - no more trades today")
        return

    # Rest of scanning logic...
```

---

## DEMO ACCOUNT SETUP CHECKLIST

### 1. Create E8 Demo Account
- [ ] Go to E8 website
- [ ] Sign up for free demo account
- [ ] Get TradeLocker credentials (email, password, server)
- [ ] Verify account accessible via TradeLocker web

### 2. Update Bot Credentials
- [ ] Edit `.env` file with demo credentials
- [ ] Change server from "E8-Live" to demo server name
- [ ] Test connection with `check_account_emergency.py`

### 3. Install Daily DD Tracking
- [ ] Add `check_daily_dd()` function to bot
- [ ] Add `daily_pnl_tracker.json` initialization
- [ ] Test: Manually set daily loss to $3,500, verify bot blocks trades

### 4. Set Phase 1 Configuration
- [ ] `min_score = 6.0`
- [ ] `max_positions = 1`
- [ ] `position_multiplier = 0.60`
- [ ] `risk_per_trade = 0.015`

### 5. Deploy & Monitor
- [ ] Start bot: `start pythonw BOTS\E8_FOREX_BOT.py`
- [ ] Create tracking spreadsheet:
   - Date | Trades | Winners | Losers | Daily P/L | Daily DD Hit?
- [ ] Check 2x/day (morning, evening)

---

## DEMO SUCCESS = FUNDED ACCOUNT

### After 4-8 Weeks of Demo, Ask:

**1. Did you pass the demo challenge?**
   - Reached $20k profit target
   - Never exceeded daily DD
   - Never exceeded trailing DD
   - **YES:** Buy funded account, use exact same settings
   - **NO:** Keep iterating on demo (costs you $0)

**2. What was your optimal config?**
   - Which phase worked best?
   - Score threshold that balanced opportunity vs safety
   - Position sizing that avoided daily DD

**3. How often did you hit daily DD warnings?**
   - Never: You can be more aggressive
   - 1-2x per month: Perfect balance
   - Weekly: Too aggressive, dial back

**4. What's your realistic timeline?**
   - If demo shows 4-6 month path → that's your funded timeline
   - If demo shows 12+ month path → consider other strategies
   - If demo never passes → saved you $600!

---

## RULES FOR DEMO ACCOUNT

### DO:
- Treat it like real money (no YOLO trades)
- Track every metric (daily DD, win rate, P/L)
- Run for minimum 30 days (preferably 60)
- Test different score thresholds (6.0 → 5.0 → 4.0)
- Document what works and what doesn't

### DON'T:
- Rush to funded account after 1 good week
- Ignore daily DD violations ("it's just demo")
- Change settings every 3 days (need statistical sample)
- Skip the tracking spreadsheet
- Buy funded account until demo PASSES challenge

---

## THE $600 LESSON APPLIED

**What you did before:**
```
1. Buy $600 funded account
2. Start conservative (score 5.0)
3. Get impatient after 1 day
4. Switch to aggressive (score 3.0)
5. Blow account in 2 hours
6. Lose $600
```

**What you'll do now:**
```
1. Start FREE demo account
2. Test ultra-conservative (score 6.0) for 2 weeks
3. Gradually increase aggression ONLY if no daily DD hits
4. Run for 4-8 weeks total
5. If demo passes → buy funded with proven config
6. If demo fails → saved $600, try different approach
```

**Demo = Free education**
**Funded = Expensive test**

**Always learn on demo first.**

---

## TIMELINE

```
Week 1-2:   Phase 1 (Score 6.0, ultra-conservative)
Week 3-4:   Phase 2 (Score 5.0, moderate conservative)
Week 5-6:   Phase 3 (Score 4.0, balanced aggression)
Week 7-8:   Phase 4 (Score 3.5, optimal aggression)

Month 2+:   Continue with optimal config from Phase 3 or 4
            Goal: Pass demo challenge ($20k profit, no DD violations)

Month 3-4:  If demo passed → Buy funded account
            If demo failed → Iterate on demo OR pivot to options
```

---

## NEXT STEPS (Right Now)

**Today:**
1. Read [WHAT_KILLED_ACCOUNT.md](WHAT_KILLED_ACCOUNT.md) - understand what happened
2. Sign up for E8 demo account
3. Get demo credentials

**Tomorrow:**
1. Update `.env` with demo credentials
2. Add daily DD tracking code to bot
3. Set Phase 1 config (score 6.0, 0.60 multiplier)
4. Start bot on demo

**This Week:**
1. Monitor daily (2x/day checks)
2. Track: Trades placed, daily P/L, DD warnings
3. Goal: Zero daily DD hits, learn bot behavior

**Week 2:**
1. Review Week 1 data
2. If safe → proceed to Phase 2
3. If daily DD hit → stay at Phase 1 longer

---

**Demo account is your second chance. Use it wisely.**

**No money at risk. All the learning. That's the smart play.**
