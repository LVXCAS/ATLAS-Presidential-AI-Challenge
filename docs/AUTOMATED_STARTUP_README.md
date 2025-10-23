# PC-HIVE Auto Options Scanner - Automated Startup Guide

## Overview

This automated startup system ensures your options scanner runs **every day at 6:30 AM PT** without manual intervention. You'll never miss a trading opportunity because you forgot to start it!

## Quick Start (3 Steps)

### 1. Initial Setup (One-Time)
Right-click `SETUP_AUTOMATED_STARTUP.bat` and select **"Run as administrator"**

This will:
- Create a scheduled task to run daily at 6:30 AM PT
- Create a startup task to run when Windows boots
- Configure automatic recovery if the scanner crashes

### 2. Verify It's Working
Double-click `CHECK_SCANNER_STATUS.bat`

This shows:
- Whether the scanner is currently running
- How many trades executed today
- Next scheduled run time
- Recent log files

### 3. Manual Start (Optional)
Double-click `START_TRADING.bat` to start the scanner immediately

---

## File Reference

### Main Scripts

| File | Purpose | When to Use |
|------|---------|-------------|
| `START_TRADING.bat` | Start scanner with visible window | Manual testing, monitoring |
| `START_TRADING_BACKGROUND.bat` | Start scanner in background | Used by Task Scheduler |
| `CHECK_SCANNER_STATUS.bat` | Check if scanner is running | Daily verification |
| `STOP_SCANNER.bat` | Stop the scanner gracefully | End of day, maintenance |
| `EMERGENCY_STOP.bat` | Kill all trading immediately | Emergency situations |
| `VIEW_LATEST_LOG.bat` | View most recent log file | Troubleshooting |
| `SETUP_AUTOMATED_STARTUP.bat` | Configure Task Scheduler | Initial setup |

### How It Works

```
6:30 AM PT Daily
     |
     v
Windows Task Scheduler
     |
     v
START_TRADING_BACKGROUND.bat
     |
     v
auto_options_scanner.py --daily
     |
     v
Scans market, executes trades
     |
     v
Logs to logs/scanner_YYYYMMDD_HHMMSS.log
```

---

## Daily Workflow

### Morning (Automated)
1. **6:30 AM PT** - Scanner starts automatically
2. Scanner checks market conditions
3. If good opportunities found (score ≥ 8.0), executes trades
4. Logs all activity to `logs/` directory

### Your Part
1. **Check status** - Run `CHECK_SCANNER_STATUS.bat`
2. **Review trades** - Check your Alpaca dashboard
3. **Monitor logs** - Run `VIEW_LATEST_LOG.bat` if needed

### Evening (Optional)
- Run `STOP_SCANNER.bat` if you want to stop it
- Otherwise, it keeps running and will scan again tomorrow

---

## Scanner Behavior

### Default Settings
- **Schedule**: Daily at 6:30 AM PT
- **Max Trades/Day**: 4
- **Min Score**: 8.0
- **Market Hours**: 6:30 AM - 1:00 PM PT (M-F)

### Smart Features
1. **Market Hours Check** - Only trades during market hours
2. **Daily Limit** - Max 4 trades per day to manage risk
3. **Weekend Aware** - Automatically skips weekends
4. **Auto-Recovery** - Resumes if it crashes
5. **Comprehensive Logging** - Every action is logged

### What It Does Automatically
- Scans S&P 500 stocks for options opportunities
- Calculates AI confidence scores
- Executes high-scoring trades (≥ 8.0)
- Tracks positions and P&L
- Saves status to `auto_scanner_status.json`

---

## Monitoring & Logs

### Status File
`auto_scanner_status.json` contains:
```json
{
  "total_trades": 12,
  "trades_today": 2,
  "last_scan_date": "2025-10-15",
  "last_scan_time": "2025-10-15T06:30:15",
  "scan_count": 45
}
```

### Log Files
Location: `logs/scanner_YYYYMMDD_HHMMSS.log`

Each log contains:
- Scan start/end times
- Opportunities found
- Trades executed
- Errors encountered
- Daily statistics

### View Logs
```batch
# Latest log
VIEW_LATEST_LOG.bat

# Specific log
type logs\scanner_20251015_063000.log

# All logs
dir logs\*.log
```

---

## Task Scheduler Details

### Tasks Created

#### 1. Daily Task: "PC-HIVE Auto Scanner"
- **Trigger**: Daily at 6:30 AM PT
- **Action**: Run `START_TRADING_BACKGROUND.bat`
- **User**: Your Windows account
- **Privileges**: Highest
- **Wake computer**: No (must be on)

#### 2. Startup Task: "PC-HIVE Auto Scanner - Startup"
- **Trigger**: System startup (1 min delay)
- **Action**: Run `START_TRADING_BACKGROUND.bat`
- **User**: Your Windows account
- **Privileges**: Highest

### Manage Tasks
```batch
# View all scheduled tasks
taskschd.msc

# Query task status
schtasks /Query /TN "PC-HIVE Auto Scanner"

# Disable task
schtasks /Change /TN "PC-HIVE Auto Scanner" /DISABLE

# Enable task
schtasks /Change /TN "PC-HIVE Auto Scanner" /ENABLE

# Delete task
schtasks /Delete /TN "PC-HIVE Auto Scanner" /F
```

---

## Troubleshooting

### Scanner Not Running

**Check Process**
```batch
CHECK_SCANNER_STATUS.bat
```

**Common Causes**:
1. Task disabled - Run `SETUP_AUTOMATED_STARTUP.bat`
2. Python error - Check `VIEW_LATEST_LOG.bat`
3. PC was off at 6:30 AM - Run `START_TRADING.bat` manually
4. Emergency stop active - Delete `emergency_stop.flag`

### No Trades Executed

**Possible Reasons**:
1. **No good opportunities** - Market conditions didn't meet criteria (score < 8.0)
2. **Daily limit reached** - Already executed 4 trades today
3. **Outside market hours** - Scanner only trades 6:30 AM - 1:00 PM PT
4. **Weekend** - No trading on weekends
5. **API error** - Check logs for Alpaca API errors

### Task Scheduler Not Working

**Verify Task Exists**:
```batch
schtasks /Query /TN "PC-HIVE Auto Scanner"
```

**Recreate Task**:
1. Right-click `SETUP_AUTOMATED_STARTUP.bat`
2. Run as administrator
3. Check `CHECK_SCANNER_STATUS.bat`

**Manual Run Test**:
```batch
START_TRADING.bat
```

### Python Errors

**Check Python Installation**:
```batch
python --version
python -c "import schedule; import pandas; print('OK')"
```

**Reinstall Dependencies**:
```batch
pip install -r requirements.txt
```

---

## Advanced Configuration

### Change Scan Time

Edit scheduled task:
1. Open Task Scheduler (`taskschd.msc`)
2. Find "PC-HIVE Auto Scanner"
3. Right-click → Properties
4. Triggers tab → Edit
5. Change start time
6. OK → OK

### Change Max Trades

Edit `START_TRADING_BACKGROUND.bat`:
```batch
REM Change this line:
start /B pythonw auto_options_scanner.py --daily --max-trades 6
```

### Run Continuously (Every 4 Hours)

Edit `START_TRADING_BACKGROUND.bat`:
```batch
REM Change --daily to --continuous:
start /B pythonw auto_options_scanner.py --continuous --interval 4
```

### Custom Scan Interval

```batch
REM Every 2 hours:
python auto_options_scanner.py --continuous --interval 2

REM Every 6 hours:
python auto_options_scanner.py --continuous --interval 6
```

---

## Safety Features

### Emergency Stop
Run `EMERGENCY_STOP.bat` to:
- Kill all scanner processes immediately
- Disable all scheduled tasks
- Create emergency stop flag
- Prevent new trades

### Daily Limits
- Max 4 trades per day (configurable)
- Prevents over-trading
- Manages risk exposure

### Market Hours Only
- Only trades 6:30 AM - 1:00 PM PT
- Skips weekends automatically
- Respects market calendar

### Error Recovery
- Logs all errors
- Continues on next scheduled run
- Doesn't crash on single failure

---

## Maintenance

### Weekly
1. Run `CHECK_SCANNER_STATUS.bat`
2. Review `auto_scanner_status.json`
3. Check total trades executed

### Monthly
1. Review all logs in `logs/` directory
2. Verify Task Scheduler is active
3. Check Alpaca account status
4. Review P&L and performance

### As Needed
1. Update max trades limit
2. Adjust scan interval
3. Change scan time
4. Review and delete old logs

---

## FAQ

### Q: What if my PC is off at 6:30 AM?
**A**: The scanner won't run. It only runs when PC is on. Consider:
- Leaving PC on overnight
- Using "Wake on LAN" if your PC supports it
- Running `START_TRADING.bat` manually when you start your PC

### Q: Can I run the scanner 24/7?
**A**: Yes! Use continuous mode:
```batch
python auto_options_scanner.py --continuous --interval 4
```

### Q: How do I know if a trade was executed?
**A**: Three ways:
1. Check `auto_scanner_status.json` - shows `trades_today`
2. Run `CHECK_SCANNER_STATUS.bat`
3. Check your Alpaca dashboard

### Q: What if I want to trade manually too?
**A**: You can! The scanner just adds automated trades. You can still:
- Run `MONDAY_AI_TRADING.py` manually
- Trade directly in Alpaca
- Use other trading tools

### Q: How do I disable automatic startup?
**A**: Run as administrator:
```batch
schtasks /Change /TN "PC-HIVE Auto Scanner" /DISABLE
schtasks /Change /TN "PC-HIVE Auto Scanner - Startup" /DISABLE
```

### Q: Where are my logs stored?
**A**: `C:\Users\lucas\PC-HIVE-TRADING\logs\scanner_*.log`

### Q: Can I change the minimum score threshold?
**A**: Yes, edit `auto_options_scanner.py`:
```python
scanner = AutoOptionsScanner(
    min_score=9.0  # Changed from 8.0 to 9.0
)
```

---

## System Requirements

### Software
- Windows 10 or 11
- Python 3.8+
- All dependencies from `requirements.txt`
- Administrator access (for Task Scheduler setup)

### Hardware
- PC must be ON at scheduled time
- Internet connection
- Sufficient disk space for logs

### Alpaca Account
- Active Alpaca account
- Paper or live trading enabled
- API keys configured
- Options trading approved

---

## Support

### Getting Help
1. Check `VIEW_LATEST_LOG.bat` for error messages
2. Review this README
3. Check `auto_scanner_status.json`
4. Verify Task Scheduler setup

### Common Issues
- **No trades**: Check market conditions, daily limit, score threshold
- **Scanner not starting**: Verify Task Scheduler, check permissions
- **Python errors**: Update dependencies, check Python version
- **API errors**: Verify Alpaca keys, check account status

---

## Success Checklist

After setup, verify:
- [ ] `SETUP_AUTOMATED_STARTUP.bat` ran successfully
- [ ] `CHECK_SCANNER_STATUS.bat` shows task is scheduled
- [ ] `START_TRADING.bat` starts scanner without errors
- [ ] `auto_scanner_status.json` is created
- [ ] Logs directory exists with log files
- [ ] Task Scheduler shows both tasks enabled
- [ ] Scanner respects daily trade limits
- [ ] Emergency stop works

---

## Summary

You now have a **fully automated** options trading system that:
- ✅ Runs every day at 6:30 AM PT
- ✅ Survives system reboots
- ✅ Logs all activity
- ✅ Handles errors gracefully
- ✅ Enforces daily trade limits
- ✅ Respects market hours
- ✅ Can be monitored easily
- ✅ Can be stopped instantly

**You never have to manually start it again!**

Just keep your PC on and check the status periodically.

---

*Last Updated: 2025-10-15*
*Version: 1.0*
