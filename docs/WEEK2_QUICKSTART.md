# WEEK 2 - QUICK START GUIDE

## 🚀 Launch Week 2 in 3 Steps

### Step 1: Check Account Status
```bash
python check_positions_now.py
```
- Verify buying power available
- Check current positions
- Confirm account ready

---

### Step 2: Launch Week 2 Scanner
```batch
WEEK2_LAUNCH.bat
```

**OR** (Python direct):
```bash
python week2_sp500_scanner.py
```

---

### Step 3: Monitor Performance
- Watch console output for opportunities
- Review end-of-day report: `week2_sp500_report_YYYYMMDD.json`
- Check actual ROI vs 10-15% weekly target

---

## ✅ Week 2 System Status

**Universe**: 503 S&P 500 stocks ✅
**ML Systems**: 6/6 active ✅
**Target**: 10-15% weekly ROI ✅
**Max Trades**: 5-10 per day ✅
**Risk/Trade**: 2% ✅

---

## 📊 What to Expect

### During Scanning (Every 5 Minutes)
```
SCAN #1 - S&P 500 MOMENTUM SCAN
======================================================================
Time: 06:30:15 AM
Scanning 503 tickers...
  Progress: 25/503 tickers scanned...
  Progress: 50/503 tickers scanned...
  Progress: 100/503 tickers scanned...
  ...
  Progress: 503/503 tickers scanned...

SCAN COMPLETE - Found 23 qualified opportunities
======================================================================

TOP 10 OPPORTUNITIES:
1. NVDA: $125.50
   Score: 5.2 | Momentum: +7.3% (BULLISH)
   Strategy: Bull Call Spread or Long Calls

2. AMD: $145.20
   Score: 4.9 | Momentum: +5.1% (BULLISH)
   Strategy: Bull Call Spread or Long Calls
...
```

### Trade Execution
```
======================================================================
EXECUTING TOP 5 OPPORTUNITIES
======================================================================

[EXECUTE] NVDA: Bull Call Spread or Long Calls
  Score: 5.2
  Momentum: +7.3% (BULLISH)

[EXECUTE] AMD: Bull Call Spread or Long Calls
  Score: 4.9
  Momentum: +5.1% (BULLISH)
...

[SUMMARY] Executed 5 trades
[REMAINING] 5 trades available today
```

---

## 🎯 Daily Targets

### Minimum Success
- **Scans**: 80+ full scans (every 5 min)
- **Opportunities**: 10+ per scan
- **Trades**: 5-10 executed
- **Quality**: All trades 4.0+ confidence

### Weekly Goal
- **ROI**: 10-15%
- **Trades**: 25-50 total
- **Win Rate**: 65-75%
- **Drawdown**: <5% per day

---

## ⚠️ Important Notes

### Market Hours (PDT)
- **Open**: 6:30 AM
- **Close**: 1:00 PM
- **Scanning**: Every 5 minutes during market hours

### Position Management
- Max 2% risk per trade
- Max 10 trades per day
- Max 10% total daily risk
- Stop scanner: Press Ctrl+C

### Quality Control
- Minimum 4.0 confidence score (same as Week 1)
- All trades validated by 6 ML/DL/RL systems
- Momentum-enhanced strategy selection

---

## 📁 Files Reference

### Launch Files
- `WEEK2_LAUNCH.bat` - Main launcher
- `week2_sp500_scanner.py` - Scanner code

### Documentation
- `WEEK2_README.md` - Full documentation
- `WEEK2_UPGRADE_SUMMARY.md` - Week 1 vs Week 2
- `WEEK2_QUICKSTART.md` - This file

### Data Files
- `sp500_complete.json` - All 503 tickers
- `sp500_options_filtered.json` - Scanner input
- `week2_sp500_report_*.json` - Daily reports

---

## 🔧 Troubleshooting

### Scanner Won't Start
```bash
# Check if Week 1 is still running
# If so, stop it first
Ctrl+C in Week 1 terminal
```

### No Opportunities Found
- Check market hours (6:30 AM - 1:00 PM PDT)
- Verify internet connection for data
- Confidence threshold = 4.0+ (high quality bar)

### Too Many/Few Trades
```python
# Edit week2_sp500_scanner.py line 51
self.max_trades_per_day = 10  # Adjust as needed

# Edit line 50 for selectivity
self.confidence_threshold = 4.0  # Increase for fewer trades
```

---

## 📈 Performance Tracking

### Check During Day
```bash
# View current positions
python check_positions_now.py

# View system status
python system_status.py
```

### End of Day Review
- Open: `week2_sp500_report_YYYYMMDD.json`
- Calculate actual ROI
- Compare vs 10-15% weekly target
- Plan for next day

---

## 🎉 Ready to Launch!

### Final Checklist
- [x] 503 S&P 500 tickers loaded
- [x] All ML/DL/RL systems active
- [x] Week 2 scanner tested
- [x] Launch files ready
- [ ] Account checked (do this now)
- [ ] Launch Week 2! (WEEK2_LAUNCH.bat)

---

**Launch Command**: `WEEK2_LAUNCH.bat`

**Target**: 10-15% weekly ROI | 503 stocks | 5-10 trades/day

Good luck! 🚀📈
