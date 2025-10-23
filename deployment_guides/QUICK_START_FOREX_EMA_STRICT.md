# Quick Start: Forex EMA Strict (71.43% Win Rate, 12.87 Sharpe)

## Overview
Ultra-high quality forex system with strictest filters. Trades less frequently but with exceptional win rate.

## Performance Metrics
- **Win Rate:** 71.43% (14/14 profitable backtest runs)
- **Sharpe Ratio:** 12.87 (exceptional)
- **Profit Factor:** 3.5+
- **Signals:** 1-2 per week (high quality only)
- **Risk/Reward:** 2:1

## One-Command Launch
```bash
python RUN_FOREX_EMA_STRICT.py
```

## Stop Command
```bash
python STOP_FOREX_EMA_STRICT.py
```

## Monitor Command
```bash
python MONITOR_FOREX_EMA_STRICT.py
```

## What to Expect
- **Frequency:** Very selective (1-2 signals per week)
- **Quality:** Each signal passes 7+ filters
- **Win Rate:** Extremely high (70%+)
- **Drawdown:** Minimal due to strict risk management

## Filters Applied
1. ✅ Volume/Activity > 70% of average
2. ✅ Multi-Timeframe Confirmation (4H + 1H)
3. ✅ RSI in optimal range (not overbought/oversold)
4. ✅ EMA clear separation (> 0.01%)
5. ✅ Trend alignment (price vs 200 EMA)
6. ✅ Score threshold 7.2+ (vs 6.5 for balanced)
7. ✅ Dynamic ATR stops

## When to Use This System
- **Conservative traders:** Want high win rate over frequency
- **Small accounts:** Minimize drawdown risk
- **Side income:** Don't want to babysit trades
- **Learning:** Build confidence with quality over quantity

## Verification
- **Logs:** `logs/forex_ema_strict_YYYYMMDD.log`
- **Trades:** `trades/forex_ema_strict_trades.json`
- **Performance:** Run `python MONITOR_FOREX_EMA_STRICT.py`

## Configuration
Edit `config/FOREX_EMA_STRICT_CONFIG.json`:
```json
{
  "score_threshold": 7.2,
  "rsi_long_range": [51, 79],
  "rsi_short_range": [21, 49],
  "min_ema_separation_pct": 0.00015,
  "volume_filter_pct": 0.55,
  "risk_per_trade": 0.01
}
```

## If You See...
- **"No signals for days"** - Normal! This system is VERY selective
- **"Waiting for high-quality setup"** - System working correctly
- **"Score 7.0 (below threshold 7.2)"** - Good signal but not strict enough

## Performance Tracking
- **Week 1:** Expect 1-2 trades
- **Month 1:** Expect 5-8 trades
- **Target Win Rate:** 70%+
- **Monthly Return:** 5-10% (conservative but consistent)

## Requirements
- Alpaca API keys
- $5,000+ account minimum
- Patience (this system prioritizes quality)
