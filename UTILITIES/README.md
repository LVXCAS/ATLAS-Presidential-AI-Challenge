# UTILITIES - Monitoring & Diagnostic Tools

This folder contains helper scripts for monitoring, analyzing, and managing the trading systems.

## Position Monitoring

### check_oanda_positions.py
Check current OANDA forex positions and P/L in real-time.

**Usage:**
```bash
python UTILITIES/check_oanda_positions.py
```

**Output:**
- Current open positions
- Unrealized P/L
- Account balance
- Position details (entry price, current price, units)

### monitor_new_bot.py
Monitor the health and status of running trading bots.

**Usage:**
```bash
python UTILITIES/monitor_new_bot.py
```

### quick_status.py
Quick snapshot of system status across all markets.

## Trade Analysis

### analyze_current_trades.py
Deep analysis of current open positions:
- Trade performance metrics
- Risk/reward analysis
- Time in trade
- Unrealized P/L breakdown

### performance_analytics.py
Comprehensive performance analytics:
- Win rate calculations
- Profit factor
- Sharpe ratio
- Maximum drawdown
- Trade distribution analysis

### trade_journal.py
Trade journaling and record keeping:
- Log all trades with entry/exit details
- Performance over time
- Trade tagging and categorization

## Market Data

### forex_calendar.py
Economic calendar integration:
- Major forex news events (Fed, ECB, NFP, CPI)
- High-impact event filtering
- News-based trade avoidance

## Position Management

### close_all_positions.py
Emergency: Close all open positions across all markets.

**WARNING**: Use only in emergencies!

**Usage:**
```bash
python UTILITIES/close_all_positions.py
```

### close_position_now.py
Close a specific position manually.

**Usage:**
```bash
python UTILITIES/close_position_now.py --symbol EUR_USD
```

### force_scan_now.py
Trigger an immediate market scan (bypasses scan interval timer).

**Usage:**
```bash
python UTILITIES/force_scan_now.py
```

## Performance Tracking

### calculate_pnl.py
Calculate profit/loss for current session.

### calculate_correct_pnl.py
Corrected P/L calculation accounting for commissions and slippage.

### check_trade_peak.py
Analyze peak unrealized P/L vs actual exit P/L:
- Identifies if trades are held too long
- Measures profit give-back
- Suggests optimal exit timing

### check_why_no_trades.py
Diagnostic tool to understand why no trades are being generated:
- Checks TA signal scores
- Reviews time filters
- Analyzes threshold settings
- Suggests parameter adjustments

## Usage Notes

**From Root Directory:**
```bash
# Run utility from anywhere
python UTILITIES/check_oanda_positions.py
```

**Import in Code:**
```python
import sys
sys.path.append('UTILITIES')
from check_oanda_positions import get_current_positions
```

## Production Integration

These utilities work with:
- `WORKING_FOREX_OANDA.py` (production forex bot)
- `MULTI_MARKET_TRADER.py` (unified multi-market system)
- All market handlers in `FOREX/`, `FUTURES/`, `CRYPTO/`

## Maintenance

- **Keep utilities simple**: Single-purpose, easy to understand
- **Avoid side effects**: Read-only operations preferred
- **Clear outputs**: Human-readable results
- **Error handling**: Graceful failures with helpful error messages
