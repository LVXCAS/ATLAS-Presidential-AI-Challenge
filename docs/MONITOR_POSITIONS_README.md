# Position Monitor - Quick Start Guide

## Quick Commands

```bash
# Single snapshot
python monitor_positions.py

# Watch mode (auto-refresh every 30 seconds)
python monitor_positions.py --watch

# Watch mode with 10-second intervals
python monitor_positions.py --watch --interval 10

# Export to JSON
python monitor_positions.py --json > positions.json
```

## Windows Quick Launch

**Double-click:** `MONITOR_POSITIONS.bat`

Interactive menu with options:
1. Single snapshot view
2. Watch mode (30 sec refresh)
3. Custom refresh interval
4. Export to JSON
5. Fast watch mode (10 sec)

## What You'll See

### OPTIONS (Alpaca)
- All options positions (calls, puts, spreads)
- Entry price vs current price
- Unrealized P&L ($ and %)
- Automatic spread detection
- Underlying stock price

### FOREX (OANDA)
- All forex positions (long/short)
- Entry price and current price
- Unrealized P&L
- Separate tracking for long/short legs

### FUTURES (Alpaca)
- All futures contracts
- Entry and current prices
- Market value
- Unrealized P&L

## Color Legend

- 🟢 **Green** = Winning position (positive P&L)
- 🔴 **Red** = Losing position (negative P&L)
- 🟡 **Yellow** = Status/account labels
- 🔵 **Blue** = Headers

## Example Output

```
================================================================================
POSITION MONITOR - All Active Trades
================================================================================
Time: 09:23:45 AM
Account: Alpaca Paper | OANDA Practice

OPTIONS POSITIONS (Alpaca Paper):
--------------------------------------------------------------------------------
1. META Bull Put Spread
   Sell: META251114P00570000 x1 @ $5.70
   Buy:  META251114P00560000 x1 @ $2.30
   Net Credit: $340.00
   Current: META @ $582.66
   Status: [+] WINNING
   Unrealized P&L: +$161.00 (+47.4%)

OPTIONS TOTAL: +$161.00

FOREX POSITIONS (OANDA):
--------------------------------------------------------------------------------
No open positions

FUTURES POSITIONS (Alpaca):
--------------------------------------------------------------------------------
No open positions

================================================================================
OVERALL P&L: +$161.00
================================================================================
```

## Setup Requirements

### 1. Install Dependencies
```bash
pip install alpaca-py python-dotenv oandapyV20
```

### 2. Configure .env File
```env
# Alpaca (Options + Futures)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# OANDA (Forex)
OANDA_API_KEY=your_key_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_BASE_URL=https://api-fxpractice.oanda.com
```

## Features

### Multi-Asset Support
- ✅ Options (all strategies: spreads, single legs)
- ✅ Forex (long/short, hedged positions)
- ✅ Futures (all contracts)

### Real-Time Data
- Live price updates
- Current P&L calculations
- Entry vs. current price comparison
- Intraday changes

### Smart Grouping
- Automatically groups options spreads
- Identifies strategy types (Bull Put, Credit Spread, etc.)
- Calculates net credit/debit for spreads
- Shows underlying stock price

### Export Capabilities
- JSON export for integration
- Historical snapshot capability
- Programmatic access via Python

## Integration Examples

### Alert on Loss Threshold
```python
import subprocess
import json

result = subprocess.run(['python', 'monitor_positions.py', '--json'],
                       capture_output=True, text=True)
data = json.loads(result.stdout)

if data['summary']['total_pl'] < -500:
    print(f"ALERT: Portfolio down ${abs(data['summary']['total_pl']):.2f}")
```

### Daily Snapshot
```bash
# Windows Task Scheduler
python monitor_positions.py --json > logs\positions_%date:~-4,4%%date:~-10,2%%date:~-7,2%.json

# Linux/Mac Cron (every hour during market hours)
0 9-16 * * 1-5 python /path/to/monitor_positions.py --json > /path/to/logs/positions_$(date +\%Y\%m\%d_\%H).json
```

### Slack Integration
```python
import requests
import subprocess
import json

result = subprocess.run(['python', 'monitor_positions.py', '--json'],
                       capture_output=True, text=True)
data = json.loads(result.stdout)

webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK"
message = {
    "text": f"Portfolio P&L: ${data['summary']['total_pl']:.2f}"
}
requests.post(webhook_url, json=message)
```

## Troubleshooting

### Issue: Colors not showing
**Solution:** Use Windows Terminal or enable ANSI in Command Prompt
```bash
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1
```

### Issue: API connection failed
**Solution:** Check .env file for correct credentials
```bash
# Verify credentials are set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Alpaca:', os.getenv('ALPACA_API_KEY')[:10] if os.getenv('ALPACA_API_KEY') else 'NOT SET')"
```

### Issue: OANDA not available
**Solution:** Install oandapyV20
```bash
pip install oandapyV20
```

### Issue: Positions not showing
**Solution:** Verify you have open positions in your account
```bash
python monitor_positions.py --json
```

## Best Practices

### Regular Monitoring Schedule
- **9:30 AM** - Market open check
- **12:00 PM** - Mid-day review
- **3:45 PM** - Pre-close check
- **4:00 PM** - End-of-day snapshot

### Risk Management
- Set alert thresholds (-$500, -$1000)
- Export daily snapshots
- Review P&L trends weekly
- Monitor spread integrity (both legs present)

### Data Management
- Save JSON snapshots before major trades
- Keep historical data for analysis
- Track win rates by asset type
- Review average hold times

## Advanced Usage

### Custom Python Integration
```python
from monitor_positions import PositionMonitor

# Initialize monitor
monitor = PositionMonitor()

# Get specific asset types
options = monitor.get_alpaca_options_positions()
forex = monitor.get_oanda_forex_positions()
futures = monitor.get_alpaca_futures_positions()

# Calculate custom metrics
total_risk = sum(abs(pos['cost_basis']) for pos in options)
winning = sum(1 for pos in options if pos['unrealized_pl'] > 0)
win_rate = winning / len(options) * 100 if options else 0

print(f"Total risk: ${total_risk:.2f}")
print(f"Win rate: {win_rate:.1f}%")
```

### Automated Monitoring Loop
```python
import time
from monitor_positions import PositionMonitor

monitor = PositionMonitor()

while True:
    data = monitor.get_positions_json()

    # Check stop loss threshold
    if data['summary']['total_pl'] < -1000:
        print("STOP LOSS THRESHOLD REACHED!")
        # send_alert()
        break

    time.sleep(60)  # Check every minute
```

## Files

- `monitor_positions.py` - Main script
- `MONITOR_POSITIONS.bat` - Windows launcher
- `POSITION_MONITOR_GUIDE.md` - Detailed documentation
- `MONITOR_POSITIONS_README.md` - This file
- `.gitignore` - Excludes position logs from git

## Support

For detailed documentation, see: `POSITION_MONITOR_GUIDE.md`

For issues or questions:
1. Check the troubleshooting section
2. Verify API credentials in .env
3. Test with `python monitor_positions.py --json`
4. Review logs for error messages

---

**Version:** 1.0.0
**Last Updated:** October 14, 2025
**Platform:** Windows, Linux, Mac
**Python:** 3.8+
