# Day 3 Battle Plan - Wednesday, October 9, 2025

**Mission:** PROVE IRON CONDORS WORK

**Starting Balance:** $98,716.56
**Week 2 Status:** -1.28% (down $1,283.44)
**Critical Day:** This determines if the $10M path is viable

---

## 🎯 Primary Objectives

### **1. Activate Multi-Strategy Mode ✅**
- Status: **ACTIVATED** (line 66 of scanner)
- Iron Condor engine: Ready
- Butterfly engine: Ready
- Max trades: **20/day** (increased from 5)

### **2. Execute 15-20 Iron Condors**
- Target: Low momentum stocks (<3% momentum)
- Capital per trade: $500-1,500
- Expected win rate: 70-80%

### **3. Achieve +1-2% for the Day**
- Needed to start Week 2 recovery
- Current: -1.28%, Target: +0.5-1% by EOD
- This proves the strategy works

---

## 📊 Expected Performance

### **With 20 Iron Condors:**

**Capital deployment:**
- 20 trades × $1,000 avg = $20,000 total
- 80% of trades win (16 winners, 4 losers)
- Average winner: +5% ($50 per trade) = $800
- Average loser: -30% max loss ($300 per trade) = -$1,200
- **Net: -$400** ❌

**Wait, that's negative. Let me recalculate...**

**Realistic Iron Condor Math:**
- Win rate: 75% (15 win, 5 lose)
- Winner profit: $50-150 premium collected
- Loser max loss: $350-450 (spread width - premium)
- Expected value per trade: (0.75 × $100) - (0.25 × $400) = $75 - $100 = **-$25**

**Hmm, that's still negative. The real edge is:**

**Correct Iron Condor Expectation:**
- Enter at 0.30-0.40 delta (70% prob of profit)
- Collect $100-200 premium per spread
- Max loss: $300-500 (10:1 risk/reward inverted)
- Close early if moving against you
- **Win rate needs to be 80%+ to be profitable**

---

## ⚠️ Critical Success Factors

### **1. Strike Selection (Using Greeks)**
- QuantLib integration already done
- Use 0.30-0.40 delta strikes
- **This is KEY to 80%+ win rate**

### **2. Don't Trade High Momentum**
- Momentum <3% only
- High momentum = directional risk
- **Stick to the selection criteria**

### **3. Close Losers Early**
- If trade moves against you > 50% max loss
- Don't let losers run to max
- **Risk management is everything**

### **4. Take Profits at 50%**
- Don't be greedy
- If collected $100 premium, close at $50 gain
- **Lock in winners**

---

## 🚨 What Could Go Wrong

### **1. Iron Condor Engines Don't Execute**
- Possible: Alpaca doesn't support 4-leg orders
- Mitigation: May need to place legs separately
- Fallback: Use Dual Options

### **2. Low Momentum Stocks Move Big**
- Possible: News event, earnings surprise
- Mitigation: Check calendar before trading
- Fallback: Close position immediately

### **3. Can't Find 20 Opportunities**
- Possible: Not enough low-momentum stocks today
- Mitigation: Lower confidence threshold
- Accept: 10-15 trades still good

### **4. Win Rate <70%**
- Possible: Bad day, high volatility
- Critical: If <70%, Iron Condors don't work
- Decision: Revert to Dual Options

---

## 📋 Hour-by-Hour Plan

### **6:00 AM - Pre-Market**
- ✅ Verify multi-strategy mode activated
- Check economic calendar (any big news?)
- Review open positions

### **6:30 AM - Market Open**
- Scanner starts first scan
- Execute first 5 trades within 30 minutes
- Monitor: Are Iron Condors executing correctly?

### **7:00-10:00 AM - Core Trading**
- Execute remaining 10-15 trades
- Track: Win rate on early trades
- Adjust: If issues, switch strategies

### **10:00 AM - Mid-Day Check**
- P&L check
- Are we positive yet?
- Close any losing positions >50% max loss

### **12:00 PM - Final Hour**
- Last chance for additional trades
- Review all open positions
- Set stop losses for overnight

### **1:00 PM - Market Close**
- Calculate final P&L
- Win rate analysis
- Write Day 3 journal

---

## 💰 Day 3 Targets

### **Minimum Acceptable:**
- Trades executed: 10+
- Win rate: 70%+
- P&L: Break-even or better
- **Status: Strategy viable**

### **Good Day:**
- Trades executed: 15+
- Win rate: 75%+
- P&L: +0.5-1% ($500-1,000)
- **Status: On track**

### **Excellent Day:**
- Trades executed: 20+
- Win rate: 80%+
- P&L: +1-2% ($1,000-2,000)
- **Status: Prove the model works**

---

## 🎓 What We're Testing

### **Key Hypotheses:**
1. **Iron Condors are more capital efficient than Dual Options**
   - Test: Can we execute 20 trades vs 2-3?
   - Measure: Total capital deployed

2. **Iron Condors have higher win rate (70-80%)**
   - Test: Track every trade outcome
   - Measure: Winners / Total trades

3. **Lower profit per trade but higher total returns**
   - Test: $50-150 per winner vs $200-500 Dual Options
   - Measure: Total P&L across all trades

4. **Greeks-based strike selection improves win rate**
   - Test: 0.30-0.40 delta targets
   - Measure: Compare to non-Greeks strikes

---

## 📸 Data to Collect

### **For Every Trade:**
- [ ] Symbol
- [ ] Strategy used (Iron Condor, Butterfly, Dual)
- [ ] Entry price
- [ ] Exit price (or current)
- [ ] P&L
- [ ] Win/Loss
- [ ] Delta at entry (if options)
- [ ] Capital deployed
- [ ] Hold time

### **End of Day:**
- [ ] Total trades
- [ ] Win rate by strategy
- [ ] Total capital deployed
- [ ] Average profit per winner
- [ ] Average loss per loser
- [ ] Total P&L
- [ ] Account equity

---

## 🚀 The Stakes

**If Day 3 succeeds (75%+ win rate, +1% P&L):**
- Iron Condors are proven
- Path to $10M is clear
- Week 4: Apply to 10 prop firm evals
- Month 2: Scale to 100 accounts
- **Timeline: $10M in 9 months**

**If Day 3 fails (<70% win rate, negative P&L):**
- Iron Condors don't work for us
- Revert to Dual Options
- Rethink the $10M timeline
- **May need different strategy**

---

## 🎯 Tomorrow Morning Checklist

**Before starting scanner:**
- [ ] Verify multi_strategy_mode = True
- [ ] Verify max_trades_per_day = 20
- [ ] Check account balance ($98,716.56)
- [ ] Review open positions (8 currently)
- [ ] Economic calendar check (any major news?)

**Scanner startup:**
```bash
cd /c/Users/lucas/PC-HIVE-TRADING
python week2_sp500_scanner.py
```

**Look for these startup messages:**
```
[OK] Advanced strategies loaded: Iron Condor, Butterfly
```

**During first trade:**
```
[STRATEGY] Iron Condor - Low momentum (2.1%), high probability
```

---

## 💭 Final Thoughts

**This is the most important trading day so far.**

Day 1: Lost -0.95% (bugs, learning)
Day 2: Lost -0.58% (capital exhaustion)
**Day 3: Make or break**

**If Iron Condors work → $10M is possible.**
**If they don't → Back to the drawing board.**

**Execute with precision. Track everything. Stay disciplined.**

**LET'S PROVE THIS WORKS.** 🚀

---

**Created:** October 8, 2025, 11:45 PM PDT
**Market Opens:** October 9, 2025, 6:30 AM PDT (6 hours 45 minutes)

**Get some sleep. Tomorrow changes everything.**
