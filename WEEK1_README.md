# WEEK 1 TRADING SYSTEM - QUICK START GUIDE

**Perfect Paper Month - Week 1: Sep 30 - Oct 4, 2025**
**Target: 5-8% Weekly ROI | Strategy: Hybrid R&D + Conservative Execution**

---

## QUICK START (3 STEPS)

### 1. Launch System (Market Open: 6:30 AM PDT)
```bash
START_WEEK1.bat
```

**That's it!** The system will automatically:
- Scan for opportunities every 5 minutes
- Execute trades when 4.0+ confidence found
- Discover new strategies every 6 hours
- Log everything for prop firm documentation

---

## WHAT'S RUNNING (Week 1 Systems)

### Primary System: Continuous Week 1 Scanner
- **File**: `continuous_week1_scanner.py`
- **Frequency**: Every 5 minutes during market hours
- **Threshold**: 4.0+ confidence (Intel-style), 3.5+ (Earnings)
- **Max Trades**: 2 per day
- **Position Risk**: 1.5% max per trade
- **Execution**: Real options orders via Alpaca

### Secondary System: Continuous R&D Discovery
- **File**: `continuous_rd_discovery.py`
- **Frequency**: Every 6 hours (24/7)
- **Purpose**: Discover new validated strategies
- **Output**: `integrated_research_hybrid_[timestamp].json`
- **Uses**: Historical data + live validation

---

## WEEK 1 STRATEGY (What We're Trading)

### Intel-Style Dual Strategy (Primary)
**Symbols**: INTC, AMD, NVDA, QCOM, MU
**Entry**: 4.0+ confidence score
**Structure**:
- Cash-secured puts (4% OTM, 21-day expiry)
- Long calls (4% OTM, 21-day expiry)

**Target ROI**: 8-15% per trade

### Earnings Straddle Strategy (Secondary)
**Symbols**: AAPL, MSFT, GOOGL (major earnings only)
**Entry**: 3.5+ confidence score
**Structure**:
- ATM call + ATM put (14-day expiry)

**Target ROI**: 12-25% per trade

---

## DAILY TARGETS (Week 1)

| Metric | Target |
|--------|--------|
| **Daily ROI** | 1.0-1.5% |
| **Weekly ROI** | 5-8% |
| **Trades/Day** | 1-2 trades |
| **Win Rate** | 70%+ |
| **Max Drawdown** | 2% weekly |
| **Max Position Risk** | 1.5% per trade |

---

## MONITORING YOUR SYSTEM

### Check Daily Performance
```bash
python todays_performance.py
```

### Check What's Running
```bash
tasklist | findstr python
```

### View Scanner Logs
Check the console window titled "Week1-Scanner"

### View R&D Discovery Results
```bash
# Latest discovery file
ls -lt integrated_research_hybrid_*.json | head -1
```

---

## WEEK 1 SCHEDULE

### Monday-Friday (Sep 30 - Oct 4)
**6:30 AM PDT**: Market opens
- Launch: `START_WEEK1.bat`
- Scanner starts hunting for 4.0+ confidence setups
- Target: 1-2 trades executed

**1:00 PM PDT**: Market closes
- Scanner automatically stops
- Review daily performance
- Check P&L

**Evening**:
- R&D continues discovering strategies
- System prepares for next day

---

## WHAT WE'RE NOT USING (Week 1)

Week 1 = **Conservative validation only**

**Reserved for Week 2+:**
- ❌ ML pattern recognition (XGBoost, neural nets)
- ❌ GPU genetic evolution
- ❌ Reinforcement learning agents
- ❌ Meta-learning optimizers

**Why?** Validate the basics first, then add power.

---

## TROUBLESHOOTING

### Scanner Not Finding Trades
- **Normal**: 4.0+ threshold is conservative
- **Expected**: 0-2 trades/day is perfect for Week 1
- **Check**: View console for opportunity scores

### Scanner Shows Errors
- **API Keys**: Check `.env.paper` has correct Alpaca credentials
- **Connection**: Verify internet connection
- **Market Hours**: Scanner only works during 6:30 AM - 1:00 PM PDT

### Want to Stop System
- Close the console windows
- Or run: `taskkill /F /IM python.exe`

---

## FILES YOU NEED (Already Configured)

### Core System Files
✅ `continuous_week1_scanner.py` - Main scanner (4.0 threshold)
✅ `continuous_rd_discovery.py` - R&D discovery system
✅ `week1_execution_system.py` - Week 1 execution logic
✅ `unified_validated_strategy_system.py` - Base strategy system
✅ `hybrid_rd_system.py` - Hybrid R&D orchestrator
✅ `rd_scanner_integration.py` - R&D scanner integration
✅ `options_executor.py` - Real options execution

### Configuration Files
✅ `.env.paper` - Alpaca API credentials
✅ `config/settings.py` - System settings

### Launcher & Documentation
✅ `START_WEEK1.bat` - One-click launcher
✅ `WEEK1_README.md` - This file

---

## EXPECTED RESULTS (Week 1)

### Conservative Scenario (Most Likely)
- **Daily ROI**: 1.0-1.5%
- **Weekly ROI**: 5-8%
- **Trades**: 5-10 total for the week
- **Win Rate**: 70%+
- **Drawdown**: <2%

### Success Criteria
✅ Consistent positive daily returns
✅ Clean execution with no errors
✅ Prop firm quality documentation
✅ 4/5 positive days
✅ <2% max drawdown

---

## WEEK 1 COMPLETION (Friday Oct 4)

After Week 1 completes successfully:
1. Review full week performance
2. Analyze all trade logs
3. Prepare Week 2 activation (ML systems)
4. Target: 10-15% Week 2 with ML boost

---

## SUPPORT

**System Issues**: Check console output for errors
**API Issues**: Verify `.env.paper` credentials
**Strategy Questions**: Review this README

---

**YOU'RE READY FOR WEEK 1!**

Just run `START_WEEK1.bat` at 6:30 AM PDT and let the system work.

Target: 5-8% weekly ROI with perfect execution.
