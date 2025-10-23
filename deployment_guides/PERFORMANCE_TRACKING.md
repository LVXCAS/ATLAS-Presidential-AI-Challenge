# Performance Tracking Guide

## Track Every System's ROI

### Quick Commands

```bash
# Calculate current P&L
python calculate_pnl.py

# Monitor all systems
python MONITOR_ALL_SYSTEMS.py

# Calculate ROI per system
python calculate_roi.py

# Weekend analysis
python weekend_risk_analysis.py
```

---

## Daily Tracking

### What to Track Every Day

#### 1. Account Snapshot (Before Market Open)
```bash
python check_account_status.py
```

Record:
- Starting equity: $______
- Buying power: $______
- Open positions: ______
- Today's goal: _____%

#### 2. End of Day Performance
```bash
python calculate_pnl.py
```

Record:
- Ending equity: $______
- Daily P&L: $______ (____%)
- Trades taken: ______
- Win rate: _____%

### Daily Tracking Spreadsheet

| Date | Starting Equity | Ending Equity | Daily P&L | Daily % | Trades | Wins | Losses | Win Rate |
|------|----------------|---------------|-----------|---------|--------|------|--------|----------|
| 10/16 | $25,000 | $25,420 | +$420 | +1.68% | 3 | 2 | 1 | 66.7% |
| 10/17 | $25,420 | $26,105 | +$685 | +2.69% | 4 | 3 | 1 | 75.0% |
| 10/18 | $26,105 | $25,890 | -$215 | -0.82% | 2 | 1 | 1 | 50.0% |

Download template: `python generate_tracking_sheet.py`

---

## Per-System Performance

### Track Each System Separately

#### Forex EMA Balanced
```bash
python MONITOR_FOREX_EMA_BALANCED.py
```

Track:
- Total trades: ______
- Winning trades: ______
- Losing trades: ______
- Win rate: _____%
- Average winner: $______
- Average loser: $______
- Profit factor: ______
- Total P&L: $______ (____%)

#### Adaptive Dual Options
```bash
python MONITOR_OPTIONS_SYSTEMS.py --system adaptive_dual
```

Track:
- Positions opened: ______
- Positions closed: ______
- Still open: ______
- Realized P&L: $______
- Unrealized P&L: $______
- ROI: _____%

#### Week3 Scanner
```bash
python MONITOR_WEEK3_SCANNER.py
```

Track:
- Scans completed: ______
- Opportunities found: ______
- Trades executed: ______
- Success rate: _____%
- Total P&L: $______

### System Performance Table

| System | Trades | Win Rate | Profit Factor | Total P&L | ROI % |
|--------|--------|----------|---------------|-----------|-------|
| Forex EMA Balanced | 24 | 75% | 2.8 | +$2,450 | +9.8% |
| Adaptive Dual Options | 12 | 70% | 3.2 | +$3,120 | +12.5% |
| Week3 Scanner | 18 | 68% | 2.5 | +$1,890 | +7.6% |
| **TOTAL** | **54** | **71%** | **2.8** | **+$7,460** | **+29.8%** |

---

## Calculating Important Metrics

### Win Rate
```python
win_rate = (winning_trades / total_trades) * 100

Example:
- Total trades: 54
- Winning trades: 38
- Win rate: (38 / 54) * 100 = 70.4%
```

### Profit Factor
```python
profit_factor = gross_profit / gross_loss

Example:
- Gross profit: $12,500 (all winners added up)
- Gross loss: $4,200 (all losers added up)
- Profit factor: 12,500 / 4,200 = 2.98

Interpretation:
- < 1.0 = Losing strategy
- 1.0 - 1.5 = Breakeven to marginal
- 1.5 - 2.0 = Good
- 2.0 - 3.0 = Very good
- > 3.0 = Excellent
```

### Sharpe Ratio
```python
sharpe_ratio = (average_return - risk_free_rate) / std_dev_returns

Example:
- Average daily return: 1.2%
- Risk-free rate: 0.01% (daily)
- Std dev of returns: 0.8%
- Sharpe: (1.2 - 0.01) / 0.8 = 1.49

Interpretation:
- < 1.0 = Poor
- 1.0 - 2.0 = Good
- 2.0 - 3.0 = Very good
- > 3.0 = Excellent
```

### Maximum Drawdown
```python
max_drawdown = (peak_equity - trough_equity) / peak_equity * 100

Example:
- Peak equity: $30,000
- Lowest point after peak: $25,500
- Max drawdown: (30,000 - 25,500) / 30,000 = 15%

Acceptable levels:
- < 10% = Very conservative
- 10-20% = Moderate
- 20-30% = Aggressive
- > 30% = High risk
```

---

## Weekly Review

### Every Sunday

#### 1. Calculate Weekly Performance
```bash
python weekend_risk_analysis.py
```

Record:
- Starting equity (Monday): $______
- Ending equity (Friday): $______
- Weekly P&L: $______ (____%)
- Total trades: ______
- Win rate: _____%
- Best trade: $______
- Worst trade: $______

#### 2. Review Each Trade
Open: `trades/forex_ema_balanced_trades.json`

For each trade, ask:
- ✅ Did I follow the entry rules?
- ✅ Did I follow the exit rules?
- ✅ Was position sizing correct?
- ✅ What did I learn?

#### 3. Identify Patterns

**Winning trades - What worked?**
- Time of day?
- Market conditions?
- Specific setups?
- Position sizing?

**Losing trades - What didn't work?**
- Broke rules?
- Wrong market conditions?
- Poor entry/exit?
- Emotional decision?

#### 4. Adjust Plan for Next Week
- Increase/decrease position sizing?
- Add/remove systems?
- Change risk parameters?
- Focus on specific setups?

---

## Monthly Review

### First Sunday of Month

#### 1. Calculate Monthly Returns
```bash
python calculate_monthly_performance.py
```

Record:
- Starting balance: $______
- Ending balance: $______
- Monthly return: _____%
- Total trades: ______
- Profitable trades: ______
- Win rate: _____%
- Profit factor: ______
- Sharpe ratio: ______
- Max drawdown: _____%

#### 2. Compare to Goals

| Goal | Target | Actual | Status |
|------|--------|--------|--------|
| Monthly return | 20% | 22.5% | ✅ Exceeded |
| Win rate | 65% | 68% | ✅ Exceeded |
| Max drawdown | <15% | 12% | ✅ Good |
| Profit factor | >2.0 | 2.4 | ✅ Good |

#### 3. System Performance Review

| System | Started | Trades | Win Rate | P&L | Keep/Drop |
|--------|---------|--------|----------|-----|-----------|
| Forex EMA Balanced | Week 1 | 45 | 74% | +$4,200 | ✅ KEEP |
| Adaptive Dual Options | Week 2 | 28 | 71% | +$5,800 | ✅ KEEP |
| Week3 Scanner | Week 3 | 32 | 69% | +$3,400 | ✅ KEEP |
| Butterfly Spreads | Week 4 | 8 | 38% | -$420 | ❌ DROP |

**Decision:** Drop underperforming systems, scale up winners

---

## Position Sizing Adjustment

### When to Scale Up

✅ **Increase position size by 25% if:**
- Win rate > target for 20+ trades
- Max drawdown < 10%
- Profit factor > 2.5
- Feeling confident and disciplined

Example:
- Current risk: 1.0% per trade
- New risk: 1.25% per trade
- Account $30,000
- Trade risk: $375 instead of $300

### When to Scale Down

❌ **Decrease position size by 50% if:**
- Win rate < 50% for 10+ trades
- Max drawdown > 20%
- Broke risk rules
- Feeling emotional or stressed

Example:
- Current risk: 1.5% per trade
- New risk: 0.75% per trade
- Account $30,000
- Trade risk: $225 instead of $450

### When to Stop Trading

🛑 **STOP ALL TRADING if:**
- Down 25% from peak
- Win rate < 40% for 20+ trades
- Breaking rules consistently
- Health/mental affected

**Action:** Paper trade for 2+ weeks before resuming

---

## Real-Time Monitoring

### Monitor Script
```bash
# Run in separate terminal
watch -n 60 python MONITOR_ALL_SYSTEMS.py

# Or create dashboard
python live_status_dashboard.py
```

### What to Watch

#### Green Flags (Good)
- ✅ Win rate tracking target
- ✅ P&L positive for day
- ✅ Following risk rules
- ✅ No emotional trades
- ✅ Systems running smoothly

#### Yellow Flags (Caution)
- ⚠ Win rate slightly below target
- ⚠ One larger than normal loss
- ⚠ High correlation in positions
- ⚠ Near daily loss limit

**Action:** Reduce risk, no new positions until back on track

#### Red Flags (Danger)
- 🚨 Win rate < 50%
- 🚨 Down > 3% today
- 🚨 Multiple system failures
- 🚨 Breaking risk rules
- 🚨 Revenge trading

**Action:** STOP trading immediately, review, paper trade

---

## Performance Reports

### Generate Reports

```bash
# Daily report
python generate_daily_report.py

# Weekly report
python generate_weekly_report.py

# Monthly report
python generate_monthly_report.py

# Custom date range
python generate_report.py --start 2025-10-01 --end 2025-10-31
```

### Report Contents

**Daily Report:**
- Today's P&L
- Trades taken
- Win rate
- Current positions
- Tomorrow's plan

**Weekly Report:**
- Week's P&L
- All trades with notes
- Best/worst trades
- System performance
- Lessons learned
- Next week's plan

**Monthly Report:**
- Month's P&L
- All metrics (win rate, Sharpe, etc.)
- System comparison
- Equity curve chart
- Achievements/misses
- Goals for next month

---

## Advanced Tracking

### Trade Journal

For EVERY trade, record:

```markdown
## Trade #127 - AAPL Bull Put Spread

**Date:** 2025-10-16
**System:** Week3 Scanner
**Setup:** Low momentum (2.1%), good IV

**Entry:**
- Sold $170 put
- Bought $165 put
- Credit: $145 ($0.48 - $0.17 = $0.31)

**Plan:**
- Max profit: $145
- Max loss: $355
- Breakeven: $168.55
- Exit: 50% profit or expiration

**Outcome:**
- Closed at 60% profit: $87
- Held 2 days
- AAPL stayed above $172
- ✅ Followed plan

**Lessons:**
- Good entry timing
- Risk management worked
- Could have held to expiration
```

### Equity Curve

Track and plot:

```python
# equity_curve.py
import matplotlib.pyplot as plt
import pandas as pd

dates = pd.date_range('2025-10-01', '2025-10-31')
equity = [25000, 25420, 26105, 25890, ...]  # Daily equity

plt.plot(dates, equity)
plt.axhline(y=25000, color='r', linestyle='--', label='Starting Balance')
plt.title('Equity Curve - October 2025')
plt.xlabel('Date')
plt.ylabel('Account Equity ($)')
plt.legend()
plt.grid(True)
plt.savefig('equity_curve_oct_2025.png')
```

**Healthy equity curve:**
- Upward trending
- Small drawdowns
- Quick recovery
- Smooth growth

**Unhealthy equity curve:**
- Flat or downward
- Large drawdowns
- Slow recovery
- Volatile

---

## Compare to Benchmarks

### Your Performance vs S&P 500

| Period | Your Return | S&P 500 | Beat Market? |
|--------|-------------|---------|--------------|
| Week 1 | +8.2% | +1.1% | ✅ Yes (+7.1%) |
| Week 2 | +6.5% | -0.3% | ✅ Yes (+6.8%) |
| Week 3 | +4.1% | +2.2% | ✅ Yes (+1.9%) |
| Week 4 | +3.8% | +0.8% | ✅ Yes (+3.0%) |
| **Month** | **+24.2%** | **+3.9%** | **✅ Yes (+20.3%)** |

### Your Performance vs Target

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Monthly return | 20% | 24.2% | ✅ Beat |
| Win rate | 65% | 68% | ✅ Beat |
| Profit factor | 2.0 | 2.4 | ✅ Beat |
| Max drawdown | <15% | 12% | ✅ Good |
| Sharpe ratio | >1.5 | 2.1 | ✅ Great |

---

## Data Export

### Export for Analysis

```bash
# Export all trades to CSV
python export_trades.py --format csv --output trades_oct_2025.csv

# Export to Excel
python export_trades.py --format excel --output trades_oct_2025.xlsx

# Export specific system
python export_trades.py --system forex_ema_balanced --format csv
```

### Import to TradingView / Google Sheets

```bash
# Generate TradingView format
python export_for_tradingview.py

# Generate Google Sheets import
python export_for_sheets.py
```

---

## Automated Tracking

### Set Up Auto-Tracking

Create cron jobs (Linux/Mac):
```bash
crontab -e

# Daily report at 1:30 PM PDT (after market close)
30 13 * * 1-5 cd /path/to/PC-HIVE-TRADING && python generate_daily_report.py

# Weekly report every Sunday at 9 AM
0 9 * * 0 cd /path/to/PC-HIVE-TRADING && python generate_weekly_report.py

# Monthly report on 1st of month at 9 AM
0 9 1 * * cd /path/to/PC-HIVE-TRADING && python generate_monthly_report.py
```

Windows Task Scheduler:
```powershell
# Create task for daily report
schtasks /create /tn "Trading Daily Report" /tr "python C:\path\to\generate_daily_report.py" /sc daily /st 13:30
```

---

## Success Criteria

### You're Doing Well If:

✅ **Consistency:**
- Profitable 70%+ of days
- Win rate within 5% of target
- Following all rules

✅ **Growth:**
- Account growing steadily
- Equity curve trending up
- Beating benchmarks

✅ **Risk Management:**
- Max drawdown < 15%
- No daily loss > 3%
- Position sizing correct

✅ **Emotional Control:**
- No revenge trading
- Sleeping well
- Enjoying the process

### Warning Signs:

❌ **Inconsistency:**
- Wild swings day-to-day
- Breaking rules frequently
- Emotional decisions

❌ **Decline:**
- Account shrinking
- Equity curve downward
- Underperforming benchmarks

❌ **Poor Risk:**
- Max drawdown > 25%
- Daily losses > 5%
- Position sizing too large

❌ **Emotional Distress:**
- Revenge trading
- Can't sleep
- Affecting health

**If you see warning signs:** Stop, paper trade, reassess

---

## Tools & Resources

### Tracking Tools

**Free:**
- Google Sheets (use template)
- Excel
- Python scripts (included)

**Paid:**
- Edgewonk ($99/year) - Advanced analytics
- TraderSync ($49/month) - Trade journal
- TradingView ($14.95/month) - Charts & analysis

### Recommended Tracking Stack

1. **Daily:** Python scripts → CSV export
2. **Weekly:** Google Sheets with charts
3. **Monthly:** Detailed Excel analysis
4. **Real-time:** `MONITOR_ALL_SYSTEMS.py` dashboard

---

## Final Thoughts

**Tracking is NOT optional.** You can't improve what you don't measure.

**Key metrics to never skip:**
1. Daily P&L
2. Win rate
3. Profit factor
4. Max drawdown
5. Trade journal

**Review frequency:**
- Daily: 5 minutes
- Weekly: 30 minutes
- Monthly: 2 hours

**The traders who succeed are the ones who track everything and learn from every trade.**

---

## Quick Reference Card

```
DAILY:
☐ Morning: Check account status
☐ Evening: Calculate P&L
☐ Log: Record all trades

WEEKLY:
☐ Sunday: Weekend analysis
☐ Review: All trades
☐ Plan: Next week

MONTHLY:
☐ Calculate: All metrics
☐ Compare: To targets
☐ Adjust: Position sizing
☐ Celebrate: Wins
```

**Remember:** Data-driven decisions beat emotional decisions every time.
