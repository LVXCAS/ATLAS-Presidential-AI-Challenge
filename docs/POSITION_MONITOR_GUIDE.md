# Position Monitoring System Guide

## Overview

The `monitor_positions.py` script provides comprehensive real-time monitoring of all active trades across:
- **OPTIONS** (Alpaca)
- **FOREX** (OANDA)
- **FUTURES** (Alpaca)

Features:
- Real-time P&L calculations
- Color-coded output (green for winning, red for losing)
- Automatic spread detection and grouping
- Support for multi-leg options strategies
- Live refresh mode
- JSON export capability

---

## Installation

### Required Dependencies

```bash
pip install alpaca-py python-dotenv oandapyV20
```

### Environment Variables

Ensure your `.env` file contains:

```env
# Alpaca (Options + Futures)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# OANDA (Forex)
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_oanda_account_id
OANDA_BASE_URL=https://api-fxpractice.oanda.com
```

---

## Usage

### Basic Usage

**Single Snapshot:**
```bash
python monitor_positions.py
```

**Watch Mode (Auto-refresh every 30 seconds):**
```bash
python monitor_positions.py --watch
```

**Custom Refresh Interval:**
```bash
python monitor_positions.py --watch --interval 10
```

**JSON Output:**
```bash
python monitor_positions.py --json
```

**Save to File:**
```bash
python monitor_positions.py --json > positions_snapshot.json
```

---

## Output Example

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
   Status: ✓ WINNING
   Unrealized P&L: +$161.00 (+47.4%)

2. NVDA Bull Put Spread
   Sell: NVDA251114P00178900 x1 @ $4.50
   Buy:  NVDA251114P00169000 x1 @ $1.80
   Net Credit: $270.00
   Current: NVDA @ $182.66
   Status: ✓ WINNING
   Unrealized P&L: +$17.48 (+6.5%)

OPTIONS TOTAL: +$178.48

FOREX POSITIONS (OANDA):
--------------------------------------------------------------------------------
1. EUR_USD
   Long: 50,000 units @ 1.09245
   Long P&L: +$125.50
   Current Price: 1.09496
   Total P&L: +$125.50

2. GBP_USD
   Short: 30,000 units @ 1.26750
   Short P&L: -$45.00
   Current Price: 1.26900
   Total P&L: -$45.00

FOREX TOTAL: +$80.50

FUTURES POSITIONS (Alpaca):
--------------------------------------------------------------------------------
1. ESZ2025
   Quantity: 2 contracts
   Entry Price: $5750.00
   Current Price: $5775.00
   Market Value: $57,750.00
   Unrealized P&L: +$1,250.00 (+2.2%)

FUTURES TOTAL: +$1,250.00

================================================================================
OVERALL P&L: +$1,508.98
================================================================================
```

---

## Features in Detail

### 1. Options Position Tracking

**Automatic Spread Detection:**
- Automatically groups multi-leg options strategies
- Identifies Bull Put Spreads, Credit Spreads, Debit Spreads
- Shows net credit/debit for spread positions
- Displays current underlying price

**Supported Options Formats:**
- OCC Symbol Format: `SYMBOL + YYMMDD + C/P + STRIKE`
- Example: `META251114P00570000`
  - Underlying: META
  - Expiration: 2025-11-14
  - Type: Put (P)
  - Strike: $570.00

**Position Data:**
- Entry price vs. current price
- Unrealized P&L ($ and %)
- Today's intraday P&L
- Market value and cost basis
- Win/loss status indicator

### 2. Forex Position Tracking

**OANDA Integration:**
- Tracks both long and short positions
- Shows separate P&L for long/short legs
- Displays average entry price
- Real-time pricing via OANDA API

**Multi-Leg Support:**
- Handles hedged positions (long + short on same pair)
- Calculates net P&L across all legs

### 3. Futures Position Tracking

**Alpaca Futures:**
- Contract-based position sizing
- Entry and current prices
- Market value tracking
- P&L in both $ and %

**Supported Formats:**
- Standard futures month codes (F, G, H, J, K, M, N, Q, U, V, X, Z)
- Example: `ESZ2025` (E-mini S&P 500, December 2025)

### 4. Color-Coded Output

**Terminal Colors:**
- 🟢 **GREEN**: Winning positions (positive P&L)
- 🔴 **RED**: Losing positions (negative P&L)
- 🟡 **YELLOW**: Neutral/account labels
- 🔵 **BLUE**: Headers and titles
- 🟣 **MAGENTA**: Section headers

**Automatic Disable:**
- Colors automatically disabled for JSON output
- Colors disabled for non-TTY environments (pipes, redirects)

### 5. Watch Mode

**Real-Time Monitoring:**
```bash
python monitor_positions.py --watch
```

- Auto-refreshes every 30 seconds (default)
- Clears screen before each update
- Shows countdown timer
- Press `Ctrl+C` to stop

**Custom Intervals:**
```bash
# Refresh every 10 seconds
python monitor_positions.py --watch --interval 10

# Refresh every 60 seconds
python monitor_positions.py --watch --interval 60
```

### 6. JSON Export

**Export Current Positions:**
```bash
python monitor_positions.py --json > snapshot.json
```

**JSON Structure:**
```json
{
  "timestamp": "2025-10-14T09:23:45.123456",
  "options": [
    {
      "symbol": "META251114P00570000",
      "underlying": "META",
      "qty": -1.0,
      "side": "short",
      "entry_price": 5.70,
      "current_price": 4.09,
      "unrealized_pl": 161.00,
      "unrealized_plpc": 47.4
    }
  ],
  "forex": [
    {
      "instrument": "EUR_USD",
      "units": 50000.0,
      "unrealized_pl": 125.50,
      "current_price": 1.09496
    }
  ],
  "futures": [],
  "summary": {
    "options_pl": 178.48,
    "forex_pl": 80.50,
    "futures_pl": 0.0,
    "total_pl": 258.98
  }
}
```

---

## Integration with Trading Systems

### 1. Scheduled Monitoring

**Windows Task Scheduler:**
```batch
@echo off
cd C:\Users\lucas\PC-HIVE-TRADING
python monitor_positions.py --json > logs\positions_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%.json
```

**Linux/Mac Cron:**
```bash
# Monitor every 5 minutes during market hours
*/5 9-16 * * 1-5 cd /path/to/PC-HIVE-TRADING && python monitor_positions.py --json > logs/positions_$(date +\%Y\%m\%d_\%H\%M).json
```

### 2. Alert Integration

**Email Alerts on Loss:**
```python
import subprocess
import json

result = subprocess.run(['python', 'monitor_positions.py', '--json'], capture_output=True, text=True)
data = json.loads(result.stdout)

if data['summary']['total_pl'] < -500:
    send_email_alert(f"Portfolio down ${abs(data['summary']['total_pl']):.2f}")
```

**Slack/Discord Integration:**
```python
import requests

result = subprocess.run(['python', 'monitor_positions.py', '--json'], capture_output=True, text=True)
data = json.loads(result.stdout)

webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
message = {
    "text": f"Portfolio P&L: ${data['summary']['total_pl']:.2f}\n"
            f"Options: ${data['summary']['options_pl']:.2f}\n"
            f"Forex: ${data['summary']['forex_pl']:.2f}\n"
            f"Futures: ${data['summary']['futures_pl']:.2f}"
}
requests.post(webhook_url, json=message)
```

### 3. Risk Management Integration

**Stop-Loss Monitor:**
```python
import json
import subprocess

# Get current positions
result = subprocess.run(['python', 'monitor_positions.py', '--json'], capture_output=True, text=True)
positions = json.loads(result.stdout)

# Check for positions exceeding loss threshold
STOP_LOSS_THRESHOLD = -200  # $200 loss

for position in positions['options']:
    if position['unrealized_pl'] < STOP_LOSS_THRESHOLD:
        print(f"ALERT: {position['symbol']} exceeds stop loss: ${position['unrealized_pl']:.2f}")
        # Trigger emergency close
        # close_position(position['symbol'])
```

---

## Troubleshooting

### Connection Issues

**Problem: "Failed to initialize Alpaca"**
```
Solution: Check your .env file for correct API keys and base URL
```

**Problem: "Failed to get OANDA forex positions"**
```
Solution: Verify OANDA_API_KEY and OANDA_ACCOUNT_ID are set correctly
```

### Data Issues

**Problem: Options positions not showing**
```
Solution:
1. Verify you have open options positions in your Alpaca account
2. Check that positions are in OCC format (SYMBOL+YYMMDD+C/P+STRIKE)
3. Run: python monitor_positions.py --json to see raw data
```

**Problem: Spreads not grouping correctly**
```
Solution: Spreads are grouped by underlying + expiration date
- Ensure positions have same expiration
- Check position symbols are correctly formatted
```

### Display Issues

**Problem: Colors not showing**
```
Solution:
1. Ensure you're running in a terminal that supports ANSI colors
2. Windows: Use Windows Terminal or enable ANSI in Command Prompt
3. Check that output is not being piped (colors auto-disable)
```

**Problem: Watch mode not clearing screen**
```
Solution: Use appropriate terminal:
- Windows: Command Prompt, PowerShell, or Windows Terminal
- Mac/Linux: Any standard terminal
```

---

## API Rate Limits

### Alpaca
- Market data: 200 requests/minute
- Trading: 200 requests/minute
- Position monitoring uses 1-2 requests per refresh

### OANDA
- Practice account: 120 requests/second
- Live account: 100 requests/second
- Position monitoring uses 2 requests per refresh (positions + pricing)

**Recommended Refresh Intervals:**
- Real-time trading: 10-30 seconds
- Position monitoring: 60 seconds
- End-of-day checks: On-demand only

---

## Advanced Usage

### Custom Position Filtering

**Filter by Asset Type:**
```python
from monitor_positions import PositionMonitor

monitor = PositionMonitor()

# Get only options
options = monitor.get_alpaca_options_positions()
print(f"Options positions: {len(options)}")

# Get only forex
forex = monitor.get_oanda_forex_positions()
print(f"Forex positions: {len(forex)}")

# Get only futures
futures = monitor.get_alpaca_futures_positions()
print(f"Futures positions: {len(futures)}")
```

### Custom P&L Calculations

**Calculate Total Risk:**
```python
monitor = PositionMonitor()
options = monitor.get_alpaca_options_positions()

total_risk = sum(abs(pos['cost_basis']) for pos in options)
print(f"Total capital at risk: ${total_risk:.2f}")
```

**Win Rate Analysis:**
```python
monitor = PositionMonitor()
data = monitor.get_positions_json()

total_positions = len(data['options']) + len(data['forex']) + len(data['futures'])
winning_positions = sum(1 for pos in data['options'] if pos['unrealized_pl'] > 0)
winning_positions += sum(1 for pos in data['forex'] if pos['unrealized_pl'] > 0)
winning_positions += sum(1 for pos in data['futures'] if pos['unrealized_pl'] > 0)

win_rate = (winning_positions / total_positions * 100) if total_positions > 0 else 0
print(f"Current win rate: {win_rate:.1f}%")
```

---

## Best Practices

### 1. Regular Monitoring
- Check positions at market open (9:30 AM ET)
- Monitor mid-day (12:00 PM ET)
- Review at market close (4:00 PM ET)
- Use watch mode during active trading

### 2. Risk Management
- Set alerts for portfolio-wide loss thresholds
- Monitor individual position P&L percentages
- Track days to expiration for options
- Review spread integrity (both legs present)

### 3. Performance Tracking
- Export daily snapshots to JSON
- Calculate daily/weekly P&L trends
- Track win rates by strategy type
- Monitor average hold time

### 4. Data Backup
- Save position snapshots before major trades
- Export JSON after significant market events
- Keep historical records for analysis

---

## Support & Maintenance

### Logs Location
```
logs/position_*.log         # Position monitoring logs
logs/positions_*.json       # Historical snapshots
```

### Error Handling
The script gracefully handles:
- Missing API credentials
- Network timeouts
- Invalid position data
- Broker API downtime

All errors are logged to console with descriptive messages.

### Updates
To update the position monitor:
```bash
git pull origin main
pip install --upgrade alpaca-py oandapyV20
```

---

## Quick Reference

```bash
# Basic commands
python monitor_positions.py                      # Single snapshot
python monitor_positions.py --watch              # Live monitoring
python monitor_positions.py --watch --interval 10  # Custom interval
python monitor_positions.py --json               # JSON export

# With output
python monitor_positions.py > positions.txt      # Save to file
python monitor_positions.py --json > data.json   # Export JSON

# Background monitoring (Linux/Mac)
nohup python monitor_positions.py --watch &      # Run in background

# Windows scheduled task
schtasks /create /tn "Position Monitor" /tr "python C:\path\to\monitor_positions.py --json > C:\path\to\output.json" /sc minute /mo 5
```

---

## FAQ

**Q: Can I monitor live and paper accounts simultaneously?**
A: Yes, modify the script to initialize two separate TradingClient instances with different base URLs.

**Q: Does this work with other brokers?**
A: Currently supports Alpaca and OANDA. Integration with other brokers requires adding broker-specific API calls.

**Q: Can I export historical P&L?**
A: Use scheduled JSON exports and analyze the historical data. See "Integration with Trading Systems" section.

**Q: How do I monitor from my phone?**
A: Deploy the script on a VPS and access via SSH, or integrate with Slack/Discord for mobile alerts.

**Q: What if I have positions with different expirations?**
A: The script handles this automatically - each expiration date creates a separate group.

---

## License

This position monitoring system is part of the PC-HIVE-TRADING project.
For questions or support, refer to the main project documentation.

---

**Last Updated:** October 14, 2025
**Version:** 1.0.0
