# ATLAS Trade Logging System - Quick Reference

**Status:** ✅ ACTIVE (Deployed 2025-12-02)

---

## What's New

### Comprehensive Trade Tracking
Every trade is now logged with full details:
- Entry/exit prices and timestamps
- Position size (Kelly Criterion calculated)
- All 16 agent votes and reasoning
- P/L, pips, R-multiple, duration
- Exit reason (stop loss, take profit, manual, etc.)
- Account balance before/after

### Log Files Location
```
BOTS/ATLAS_HYBRID/logs/trades/
├── trades_2025-12-02.json          # Daily log (all trades for the day)
├── session_20251202_120000.json     # Session log (this run only)
└── summary_2025-12-02.json          # Daily performance summary
```

---

## Quick Commands

### View Today's Trades
```bash
cd BOTS/ATLAS_HYBRID
python view_trades.py
```

**Shows:**
- Open positions
- Closed trades (with P/L, pips, duration)
- Failed executions
- Win rate and total P/L

### View Performance Summary
```bash
python view_trades.py summary
```

**Shows:**
- Total trades, wins, losses
- Win rate, profit factor
- Average win/loss
- Expectancy per trade
- Largest win/loss

### View Agent Performance
```bash
python view_trades.py agents
```

**Shows:**
- Which agents voted for winning trades
- Win rate per agent
- P/L contribution per agent
- Helps optimize agent weights

---

## Trade Log Format

Each trade is stored as JSON with this structure:

```json
{
  "trade_id": "ATLAS_20251202_0001",
  "oanda_trade_ids": ["445"],

  "timestamp_decision": "2025-12-02T12:30:45.123456",
  "timestamp_entry": "2025-12-02T12:30:46.789012",
  "timestamp_exit": "2025-12-02T13:15:22.345678",
  "duration_minutes": 44.6,

  "pair": "EUR_USD",
  "direction": "SELL",
  "units": 2500000,
  "lots": 25.0,

  "entry_price": 1.15950,
  "exit_price": 1.15880,
  "stop_loss": 1.16090,
  "take_profit": 1.15740,

  "pnl": 1750.00,
  "pnl_pct": 0.95,
  "pips": 7.0,
  "r_multiple": 0.5,

  "atlas_score": 2.5,
  "atlas_threshold": 1.0,
  "agent_votes": {
    "TechnicalAgent": {"vote": "SELL", "confidence": 0.75, "weight": 1.5},
    "GSQuantAgent": {"vote": "ALLOW", "confidence": 0.90, "weight": 2.0},
    ...all 16 agents...
  },

  "account_balance_before": 184961.16,
  "account_balance_after": 186711.16,

  "kelly_calculation": {
    "balance": 184961.16,
    "kelly_fraction": 0.10,
    "risk_amount": 18496.12,
    "stop_loss_pips": 14,
    "calculated_lots": 25.0,
    "units": 2500000
  },

  "exit_reason": "take_profit",
  "status": "closed",
  "notes": ""
}
```

---

## Key Metrics Explained

### R-Multiple
How many times your risk you made/lost:
- **+2.0R** = Made 2x your risk (hit TP at 2:1 ratio)
- **+0.5R** = Made half your risk
- **-1.0R** = Lost full stop loss amount

### Profit Factor
Gross profit / Gross loss:
- **2.0** = Make $2 for every $1 lost (good)
- **1.5** = Make $1.50 for every $1 lost (acceptable)
- **<1.0** = Losing money overall (bad)

### Expectancy
Average profit per trade:
- **$250** = Expect to make $250 per trade on average
- Combines win rate and win/loss size
- **Higher is better**

---

## Example Usage

### Check if System is Logging
```bash
cd BOTS/ATLAS_HYBRID
ls -lh logs/trades/
```

You should see files like:
- `trades_2025-12-02.json`
- `session_20251202_120000.json`

### View Latest Trade
```bash
python -c "
import json
with open('logs/trades/trades_2025-12-02.json') as f:
    trades = json.load(f)
    if trades:
        latest = trades[-1]
        print(f'{latest[\"trade_id\"]}: {latest[\"pair\"]} {latest[\"direction\"]}')
        print(f'Status: {latest[\"status\"]}')
        if latest[\"status\"] == \"closed\":
            print(f'P/L: \${latest[\"pnl\"]:+,.2f}')
"
```

### Export Trades to CSV (for Excel/analysis)
```bash
python -c "
import json
import csv

with open('logs/trades/trades_2025-12-02.json') as f:
    trades = json.load(f)

closed = [t for t in trades if t['status'] == 'closed']

with open('trades_export.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Trade ID', 'Pair', 'Direction', 'Entry', 'Exit', 'P/L', 'Pips', 'Duration'])

    for t in closed:
        writer.writerow([
            t['trade_id'],
            t['pair'],
            t['direction'],
            t['entry_price'],
            t['exit_price'],
            t['pnl'],
            t['pips'],
            t['duration_minutes']
        ])

print(f'Exported {len(closed)} trades to trades_export.csv')
"
```

---

## Benefits

### For Performance Analysis
- **Track Kelly Criterion effectiveness** - See if 1/10 Kelly is optimal
- **Identify best market conditions** - When does system win most?
- **Agent optimization** - Which agents contribute to wins?

### For Prop Firm Applications
- **Proof of performance** - Complete trade history
- **Compliance evidence** - Show all risk management rules followed
- **No EA/bot violations** - Every decision logged with reasoning

### For Learning & Improvement
- **Pattern recognition** - What setups work best?
- **Risk management validation** - Are stop losses appropriate?
- **Agent weight tuning** - Increase weights of best performers

---

## Next Steps

### After First Trade Executes:
1. Run `python view_trades.py` to see the logged trade
2. Verify all data is captured correctly
3. Check `logs/trades/` directory for JSON files

### After First Day of Trading:
1. Run `python view_trades.py summary` for daily stats
2. Run `python view_trades.py agents` to see agent performance
3. Review log files to identify improvement opportunities

### For Prop Firm Applications:
1. Export trades to CSV
2. Calculate monthly ROI
3. Verify compliance with drawdown limits
4. Show consistent profitability over 60+ day period

---

## Troubleshooting

### No log files created?
Check if `logs/trades/` directory exists:
```bash
cd BOTS/ATLAS_HYBRID
ls -la logs/
```

### Trades not being logged?
Check console output for errors:
```bash
tasklist | findstr pythonw
```

System should print: `[OK] Trade Logger initialized`

### Want to see raw JSON?
```bash
cat logs/trades/trades_2025-12-02.json | python -m json.tool
```

---

**Generated:** 2025-12-02 18:00 UTC
**System:** ATLAS Hybrid Trading System v2.0
**Author:** Claude Code + Lucas
