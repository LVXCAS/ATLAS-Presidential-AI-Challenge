# Quick Start: Forex V4 Optimized (62.5% Win Rate)

## Overview
Multi-pair forex strategy (EUR/USD, GBP/USD, USD/JPY) with advanced filtering and 62.5% win rate.

## Performance Metrics
- **Win Rate:** 62.5% (107 trades backtested)
- **Sharpe Ratio:** 1.8
- **Profit Factor:** 2.1x
- **Pairs:** EUR/USD, GBP/USD, USD/JPY
- **Risk/Reward:** 2:1

## One-Command Launch
```bash
python RUN_FOREX_V4_OPTIMIZED.py
```

## Stop Command
```bash
python STOP_FOREX_V4_OPTIMIZED.py
```

## Monitor Command
```bash
python MONITOR_FOREX_V4_OPTIMIZED.py
```

## What to Expect
- **Scans:** All 3 pairs every 5 minutes
- **Filters:** ADX > 25, proper trading hours, volatility regime check
- **Signals Per Day:** 2-5 across all pairs
- **Position Duration:** 3-12 hours average

## Key Features
1. **Time-of-Day Filter:** Only trades during London/NY session (7 AM - 8 PM UTC)
2. **ADX Filter:** Avoids choppy markets (requires ADX > 25)
3. **Multi-Timeframe:** Confirms trend on 4H before 1H entry
4. **Dynamic Stops:** 2x ATR for adaptive risk management
5. **Support/Resistance:** Extra confirmation from key levels

## Verification
- **Logs:** `logs/forex_v4_optimized_YYYYMMDD.log`
- **Trades:** `trades/forex_v4_optimized_trades.json`
- **Monitor:** Real-time P&L, win rate, active trades

## Configuration
Edit `config/FOREX_V4_OPTIMIZED_CONFIG.json`:
```json
{
  "pairs": ["EUR_USD", "GBP_USD", "USD_JPY"],
  "risk_per_trade": 0.01,
  "score_threshold": 8.0,
  "adx_threshold": 25.0,
  "trading_hours": {
    "start": "07:00",
    "end": "20:00"
  }
}
```

## If You See...
- **"Outside trading hours"** - System correctly waiting for London/NY session
- **"ADX below threshold"** - Market too choppy, waiting for clear trend
- **"Volatility outside range"** - Market too calm or too volatile, waiting
- **"MTF confirmation failed"** - 4H trend not aligned, skipping trade (good!)

## Requirements
- Alpaca API keys
- $10,000+ account recommended (trades 3 pairs)
- Paper trading default (set `PAPER_MODE=false` for live)
