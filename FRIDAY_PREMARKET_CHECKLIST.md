# FRIDAY DAY 3 - PRE-MARKET CHECKLIST

**Date**: Friday, October 3, 2025
**Market Open**: 6:30 AM PDT
**Week 1 Day 3/5** - Conservative execution phase

---

## ✅ PRE-MARKET CHECKLIST (Do before 6:30 AM)

### 1. SYSTEM STATUS CHECK
- [ ] Run `python check_positions_now.py` to verify current P&L
- [ ] Confirm all 7 ML/DL/RL systems show [ACTIVE]
- [ ] Verify Time Series Momentum integration working
- [ ] Check GPU acceleration active (GTX 1660 SUPER)

### 2. PORTFOLIO REVIEW
**Current Status (Thursday close)**:
- Portfolio Value: $100,158.38
- Cash Available: $96,197.38
- Daily P&L: +$158.38 (+0.16%)
- Open Positions: 6 (4 winning, 2 losing)

**Best Performer**:
- INTC calls: +$232 (+61.4%) ✓ Validated by +51.6% momentum signal!

**Risk Check**:
- [ ] Confirm daily stop loss NOT hit (-5%)
- [ ] Verify position sizes within Week 1 limits (1.5% max per trade)
- [ ] Check total exposure < 3% of account

### 3. MOMENTUM SIGNALS CHECK
Run `python time_series_momentum_strategy.py` to see current signals:
- [ ] Check which stocks have strong bullish momentum (>+5%)
- [ ] Identify range-bound stocks (<2% momentum) for straddles
- [ ] Note any momentum reversals from yesterday

**Expected High-Momentum Stocks** (from Thursday):
- INTC: +51.6% (Strong BULLISH) ← Monitor for continuation
- TSLA: +28.8% (Strong BULLISH)
- NVDA: +10.0% (Moderate BULLISH)
- AAPL: +7.2% (Moderate BULLISH)

### 4. SCANNER SETTINGS VERIFY
- [ ] Confidence threshold: 4.0+ (Week 1 conservative)
- [ ] Max trades per day: 2 trades
- [ ] Risk per trade: 1.5% max
- [ ] Paper trading mode: ENABLED

### 5. LAUNCH SEQUENCE
```batch
# Option A: Quick launch
FRIDAY_LAUNCH.bat

# Option B: Manual launch
python continuous_week1_scanner.py
```

---

## 📊 WEEK 1 PROGRESS TRACKER

### Days Completed:
- ✅ **Day 1** (Wednesday): System setup, first scans
- ✅ **Day 2** (Thursday): +$158.38 (+0.16%), 6 positions opened
- ⏳ **Day 3** (Friday): Momentum-enhanced scanning
- ⏳ **Day 4** (Saturday): Weekend analysis
- ⏳ **Day 5** (Sunday): Week 1 review

### Week 1 Target:
- **Target**: 5-8% weekly ROI
- **Current**: +0.16% (Day 2 only)
- **Remaining**: 4.84% - 7.84% needed

### Trade Discipline:
- ✅ Only 4.0+ confidence trades
- ✅ Max 2 trades per day
- ✅ 1.5% risk per trade
- ✅ Conservative Week 1 sizing

---

## 🎯 FRIDAY TRADING PLAN

### Morning (6:30 AM - 10:00 AM PDT):
1. **6:15 AM**: Launch scanner with `FRIDAY_LAUNCH.bat`
2. **6:30 AM**: Market opens - monitor for momentum continuation
3. **6:30-7:00 AM**: Prime opportunity window (high volatility)
4. **7:00-10:00 AM**: Continuous 5-minute scanning

### Opportunity Types to Watch:

**Intel-Style Dual Strategies** (4.0+ score):
- Look for stocks with strong bullish momentum (+5%+)
- Scanner will auto-boost scores by +0.5 for strong momentum
- Expected: 0-2 opportunities per day

**Earnings Straddles** (3.5+ score):
- Look for range-bound stocks (low momentum <2%)
- Scanner will boost scores by +0.3 for ideal straddle conditions
- Avoid if strong directional momentum present

### Afternoon (10:00 AM - 1:00 PM PDT):
- Monitor existing positions
- Watch for stop loss triggers
- Prepare end-of-day report

---

## 🚨 RISK MANAGEMENT (Critical!)

### Stop Loss Rules:
- **Daily**: -5% ($5,000 loss) → STOP TRADING
- **Per Trade**: -50% → Close position
- **Portfolio**: -10% from peak → Reduce size

### Position Monitoring:
**Winning Positions**:
- INTC calls: +61.4% → Consider taking profits if >+100%
- AAPL calls (Oct 10): +14.8% → Let run
- AAPL calls (Oct 17): +7.8% → Monitor

**Losing Positions**:
- AAPL put (Oct 10): -38.3% → Watch for -50% stop
- AAPL put (Oct 17): -24.7% → Monitor

### Week 1 Limits (DO NOT EXCEED):
- Max 2 trades per day
- Max 1.5% risk per trade
- Max 3% total portfolio risk
- Only 4.0+ confidence scores

---

## 💡 MOMENTUM-ENHANCED STRATEGY

### How Scanner Works Now:

**Step 1**: Base opportunity detection (existing logic)
**Step 2**: ML enhancement (XGBoost, LightGBM, PyTorch)
**Step 3**: Technical indicators boost (RSI, MACD, Bollinger Bands)
**Step 4**: **NEW** Time series momentum boost

### Example Scoring:
```
Stock: INTC
Base score: 3.0
ML enhancement: +1.2 → 4.2
Momentum check: +51.6% bullish momentum
Momentum boost: +0.5
Final score: 4.7 ✓ QUALIFIED
```

### What To Expect:
- **Better filtering**: Counter-trend trades automatically penalized
- **Higher confidence**: Momentum-aligned trades get boosted
- **Smarter strategy selection**: Directional vs premium based on trend

---

## 📈 SUCCESS METRICS (Track Friday)

### Opportunities:
- [ ] Number of 4.0+ scores found: ____
- [ ] Momentum-boosted opportunities: ____
- [ ] Executed trades: ____ (max 2)

### Performance:
- [ ] Friday P&L: $____
- [ ] Win rate: ____%
- [ ] Best trade: ____
- [ ] Worst trade: ____

### System Performance:
- [ ] Momentum signals accurate? (Y/N)
- [ ] ML systems functioning? (Y/N)
- [ ] No errors or crashes? (Y/N)

---

## 🔧 TROUBLESHOOTING

### If Scanner Crashes:
```batch
# Check for errors
python continuous_week1_scanner.py 2> errors.log

# Restart with fresh data
taskkill /IM python.exe /F
FRIDAY_LAUNCH.bat
```

### If No Opportunities Found:
- **Normal**: Week 1 is conservative (4.0+ threshold)
- **Expected**: 0-2 qualified opportunities per day
- **Don't**: Lower threshold to force trades
- **Do**: Exercise discipline, wait for quality setups

### If API Errors:
```python
# Check Alpaca connection
python check_positions_now.py

# Verify API keys loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv('.env.paper'); print('API Key:', os.getenv('ALPACA_API_KEY')[:10] + '...')"
```

---

## 📝 END OF DAY TASKS

### After Market Close (1:00 PM PDT):
1. [ ] Review all trades executed
2. [ ] Calculate Friday P&L
3. [ ] Update Week 1 progress tracker
4. [ ] Review momentum signals for accuracy
5. [ ] Prepare for Saturday analysis

### Performance Analysis:
- [ ] Did momentum signals predict correctly?
- [ ] Were ML enhancements accurate?
- [ ] Did we maintain discipline (no overtrading)?
- [ ] Are we on track for 5-8% weekly target?

### Weekend Preparation:
- [ ] Review winning strategies (what worked?)
- [ ] Analyze losing trades (what to avoid?)
- [ ] Plan for Week 2 enhancements (if Week 1 successful)
- [ ] Document everything for prop firm application

---

## 🎯 WEEK 1 GOALS (Reminder)

### Primary Objectives:
1. ✅ Prove system discipline (quality over quantity)
2. ⏳ Achieve 5-8% weekly ROI
3. ⏳ Maintain 70%+ win rate
4. ⏳ Zero stop loss violations
5. ⏳ Document professional execution

### Why Week 1 is Conservative:
- **Building confidence** in system
- **Proving discipline** before scaling
- **Establishing track record** for prop firms
- **Testing all integrations** (momentum, ML, indicators)

### Week 2 Preview (If Week 1 succeeds):
- Increase to 3-4 trades per day
- Add more strategy types (iron condors, butterflies)
- Integrate full agentic R&D system
- Target 10-15% weekly ROI

---

## ✅ QUICK START COMMANDS

### Launch Friday Scanner:
```batch
FRIDAY_LAUNCH.bat
```

### Check Current Positions:
```batch
python check_positions_now.py
```

### View Momentum Signals:
```batch
python time_series_momentum_strategy.py
```

### Mission Control Dashboard:
- Launches automatically with scanner
- Shows all 7 ML/DL/RL systems live
- Real-time P&L tracking

---

## 🚀 FRIDAY AFFIRMATIONS

**You are ready because**:
1. ✅ All 7 ML/DL/RL systems active and tested
2. ✅ Time series momentum integrated (200+ years of research)
3. ✅ Current position (INTC) up +61% (momentum validated!)
4. ✅ Professional risk management in place
5. ✅ Conservative Week 1 thresholds enforced

**Your edge**:
- Research-backed momentum system (Moskowitz 2012)
- Institutional-grade ML stack (XGBoost, LightGBM, PyTorch)
- 150+ technical indicators (pandas-ta)
- GPU-accelerated analysis (GTX 1660 SUPER)
- Professional portfolio optimization (FinQuant, PyPortfolioOpt)

**Remember**:
- Quality > Quantity (Week 1 is about discipline)
- 0 trades is better than 1 bad trade
- 5-8% weekly is excellent (not greedy)
- You're building a track record, not gambling

---

**READY FOR FRIDAY! 🎯**

*Checklist created: Thursday October 2, 2025 @ 9:15 PM PDT*
*Market opens: Friday 6:30 AM PDT*
*System: Momentum-Enhanced Week 1 Scanner (7/7 systems active)*
