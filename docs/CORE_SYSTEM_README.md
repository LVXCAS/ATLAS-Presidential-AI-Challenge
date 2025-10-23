# Core Trading System - Production Ready

> Autonomous AI-powered trading system for Options, Forex, and Futures

## Overview

The Core Trading System is a production-ready autonomous trading platform that combines traditional technical strategies with AI enhancement for Options, Forex, and Futures markets.

### Key Features

- **Multi-Asset Support**: Options (Bull Put Spreads), Forex (EMA Crossover), Futures (Micro E-mini)
- **AI Enhancement**: Meta-learning system that improves from trade outcomes
- **Autonomous Execution**: 24/7 operation with smart risk management
- **Paper Trading**: Test strategies safely before going live
- **Position Monitoring**: Real-time P&L tracking across all assets

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   MONDAY_AI_TRADING.py                      │
│              (Master Orchestrator)                          │
└────────────┬────────────────────────────────────┬───────────┘
             │                                    │
    ┌────────▼────────┐                  ┌────────▼────────┐
    │  AI Scanners    │                  │  Execution      │
    ├─────────────────┤                  │  Engine         │
    │ - Options       │                  ├─────────────────┤
    │ - Forex         │◄─────────────────┤ - Alpaca        │
    │ - Futures       │                  │ - OANDA         │
    └─────────────────┘                  └─────────────────┘
             │
    ┌────────▼────────────────────────────────────────────┐
    │            Strategy Engines                         │
    ├─────────────────────────────────────────────────────┤
    │ - forex_ema_strategy.py (63.6% WR)                 │
    │ - futures_ema_strategy.py (60%+ WR target)         │
    │ - Bull Put Spread logic (Options)                  │
    └─────────────────────────────────────────────────────┘
```

## File Structure

```
PC-HIVE-TRADING/
├── config/
│   └── trading_config.py              # Centralized configuration
│
├── execution/
│   └── auto_execution_engine.py       # Trade execution engine
│
├── strategies/
│   ├── forex_ema_strategy.py          # Forex EMA strategy
│   └── futures_ema_strategy.py        # Futures EMA strategy
│
├── tests/
│   └── test_core_system.py            # Unit tests
│
├── MONDAY_AI_TRADING.py               # Main AI trading orchestrator
├── auto_options_scanner.py            # Automated options trading
├── forex_paper_trader.py              # Forex paper trading
├── futures_live_validation.py         # Futures validation
├── monitor_positions.py               # Position monitoring
├── ai_enhanced_forex_scanner.py       # AI-enhanced forex scanner
├── ai_enhanced_options_scanner.py     # AI-enhanced options scanner
│
├── CODE_QUALITY_IMPROVEMENTS_SUMMARY.md
└── CORE_SYSTEM_README.md              # This file
```

## Quick Start

### 1. Full AI Trading System (Recommended)

Run the complete AI-enhanced trading system:

```bash
# Autonomous mode (default)
python MONDAY_AI_TRADING.py

# Enable futures
python MONDAY_AI_TRADING.py --futures

# Manual mode (no auto-execution)
python MONDAY_AI_TRADING.py --manual

# Customize max trades
python MONDAY_AI_TRADING.py --max-trades 3
```

### 2. Options Auto-Trading

Continuously scan and execute options trades:

```bash
# Daily mode (runs at market open)
python auto_options_scanner.py --daily

# Continuous mode (every 4 hours)
python auto_options_scanner.py --continuous

# Run once and exit
python auto_options_scanner.py --once

# Custom interval and max trades
python auto_options_scanner.py --continuous --interval 2 --max-trades 3
```

### 3. Forex Paper Trading

Paper trade EUR/USD with optimized EMA strategy:

```bash
# Default: scan every 15 minutes
python forex_paper_trader.py

# Custom scan interval
python forex_paper_trader.py --interval 30  # 30 minutes
```

### 4. Futures Live Validation

48-hour observation mode before enabling live futures trading:

```bash
# Full 48-hour validation
python futures_live_validation.py

# Quick 24-hour validation
python futures_live_validation.py --duration 24

# 1-hour test mode
python futures_live_validation.py --quick-test
```

### 5. Monitor All Positions

Real-time position monitoring across all assets:

```bash
# Single snapshot
python monitor_positions.py

# Auto-refresh every 30 seconds
python monitor_positions.py --watch

# JSON output
python monitor_positions.py --json

# Custom refresh interval
python monitor_positions.py --watch --interval 60
```

## Configuration

All system configuration is centralized in `config/trading_config.py`.

### Key Parameters

```python
# Options
OPTIONS_SCAN_INTERVAL_HOURS = 4         # Scan every 4 hours
OPTIONS_MAX_TRADES_PER_DAY = 4          # Max 4 trades per day
OPTIONS_MIN_SCORE_THRESHOLD = 8.0       # Min score to execute
OPTIONS_MAX_RISK_PER_TRADE = 500.0      # Max $500 risk per trade

# Forex
FOREX_SCAN_INTERVAL_MINUTES = 15        # Scan every 15 minutes
FOREX_TARGET_WIN_RATE = 0.636           # 63.6% target win rate
FOREX_MIN_SCORE_THRESHOLD = 9.0         # Min score to execute

# Futures
FUTURES_OBSERVATION_DURATION_HOURS = 48 # 48-hour validation
FUTURES_MIN_WIN_RATE = 0.60             # 60% min win rate
FUTURES_MAX_RISK_PER_TRADE = 100.0      # Max $100 risk per trade

# Risk Management
MAX_POSITIONS = 5                       # Max 5 positions total
MAX_DAILY_LOSS = 1000.0                 # $1000 daily loss limit
```

To modify parameters, edit `config/trading_config.py` directly.

## Trading Strategies

### Options: Bull Put Spreads

**Strategy:**
- Sell put at 95% of current price (collect premium)
- Buy put at 90% of current price (protection)
- 30-day expiration
- Expected credit: 30% of spread width

**Entry Criteria:**
- AI score ≥ 8.0
- Neutral/low volatility market regime
- High liquidity stocks
- During market hours (6:30 AM - 1:00 PM PT)

### Forex: Enhanced EMA Crossover v3.0

**Strategy:**
- Fast EMA (8) crosses Slow EMA (21)
- Trend filter: 200 EMA
- RSI momentum confirmation
- Multi-timeframe confirmation (4H)
- ATR-based stops (2x) and targets (1.5x)

**Performance:**
- Target: 63.6% win rate
- Risk/Reward: 1:1.5
- Timeframe: 1-hour entries, 4-hour trend

**Entry Criteria:**
- Long: Fast EMA > Slow EMA, price > 200 EMA, 51 < RSI < 79
- Short: Fast EMA < Slow EMA, price < 200 EMA, 21 < RSI < 49
- Volume filter: Activity > 55% of average
- Score ≥ 7.2

### Futures: EMA Crossover (MES/MNQ)

**Strategy:**
- Fast EMA (10) crosses Slow EMA (20)
- Trend filter: 200 EMA
- RSI momentum validation
- ATR-based stops (2x) and targets (3x)

**Contracts:**
- MES (Micro E-mini S&P 500): $5 per point
- MNQ (Micro E-mini Nasdaq-100): $2 per point

**Conservative Mode:**
- Max risk: $100 per trade
- Max positions: 2
- Max total risk: $500

## Risk Management

### Position Limits

- **Options**: Max $500 risk per trade, 4 trades per day
- **Forex**: 5,000 units paper trading
- **Futures**: 1-2 contracts max, $100 risk per trade
- **Global**: Max 5 positions total, $1,000 daily loss limit

### Safety Features

1. **Market Hours**: Only trade during market hours (automated)
2. **Daily Limits**: Automatic daily trade counter reset
3. **Score Thresholds**: Only execute high-scoring opportunities
4. **Stop Losses**: All trades have automatic stop losses
5. **Position Tracking**: Real-time position monitoring
6. **Paper Trading**: Default mode is paper trading

## AI Enhancement

The system uses meta-learning to improve from trade outcomes:

### How It Works

1. **Generate Signal**: Traditional strategy generates base signal
2. **AI Scoring**: AI analyzes and scores the opportunity
3. **Execution**: High-scoring trades executed automatically
4. **Outcome Tracking**: Record win/loss and returns
5. **Learning**: AI adjusts scoring for future trades

### Recording Outcomes

After each trading session, record outcomes for AI learning:

```python
from MONDAY_AI_TRADING import MondayAITrading

system = MondayAITrading()

# Record options trade
system.record_trade('AAPL', 'OPTIONS', True, 0.087)  # 8.7% return

# Record forex trade
system.record_trade('EUR_USD', 'FOREX', False, -0.015)  # -1.5% loss

# View AI performance
system.show_ai_performance()
```

## Development

### Code Quality Standards

All code follows these standards:

- **Type Hints**: Required on all functions
- **Docstrings**: Google style for all classes/functions
- **Logging**: Structured logging (no print statements)
- **Constants**: All in `config/trading_config.py`
- **Error Handling**: Specific exceptions with context
- **Input Validation**: All public methods
- **Unit Tests**: Required for new features

### Running Tests

```bash
# Run all tests
python -m pytest tests/test_core_system.py -v

# Run with coverage
python -m pytest tests/test_core_system.py -v --cov=. --cov-report=html

# Run specific test
python -m pytest tests/test_core_system.py::TestForexEMAStrategy -v
```

### Adding New Strategies

1. Create strategy file in `strategies/`
2. Implement required methods:
   - `calculate_indicators(df)`
   - `analyze_opportunity(df, symbol)`
   - `validate_rules(opportunity)`
3. Add to appropriate scanner
4. Add configuration constants
5. Write unit tests

## Monitoring and Maintenance

### Daily Routine

1. **Morning** (6:30 AM PT):
   - Review AI scan results
   - Check executed trades
   - Verify systems running

2. **During Trading Hours**:
   - Monitor positions: `python monitor_positions.py --watch`
   - Check for alerts/errors in logs

3. **End of Day** (1:00 PM PT):
   - Close/review positions
   - Record outcomes for AI learning
   - Review performance metrics

### Log Files

```
logs/
├── monday_ai_scan_YYYYMMDD_HHMMSS.json
├── execution_log_YYYYMMDD.json
└── auto_scanner_status.json
```

### Performance Tracking

The system automatically tracks:
- Win rate by strategy
- Average return per trade
- AI confidence vs. outcome
- Daily/weekly/monthly P&L

## Troubleshooting

### Common Issues

**Issue**: "OANDA API not available"
- **Solution**: Check `.env` file has correct OANDA credentials
- **Solution**: Verify `oandapyV20` package installed

**Issue**: "Max positions reached"
- **Solution**: Close existing positions or increase `MAX_POSITIONS`
- **Solution**: Check `monitor_positions.py` for current positions

**Issue**: "Outside market hours"
- **Solution**: Options only trade 6:30 AM - 1:00 PM PT weekdays
- **Solution**: Forex trades 24/5, check if weekend

**Issue**: "No signals found"
- **Solution**: Market conditions may not meet entry criteria
- **Solution**: Lower score threshold in config (not recommended)
- **Solution**: Wait for better setups

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Safety and Disclaimers

### Default: Paper Trading

The system defaults to paper trading mode. To enable live trading:

1. Complete 48-hour futures validation
2. Verify win rates meet targets
3. Start with minimum position sizes
4. Monitor closely for first week

### Risk Warnings

- **Past performance does not guarantee future results**
- **All trading involves risk of loss**
- **Start with paper trading**
- **Test thoroughly before live trading**
- **Never risk more than you can afford to lose**

## Support and Resources

### Documentation

- `CODE_QUALITY_IMPROVEMENTS_SUMMARY.md` - Code quality guide
- `FULL_AUTO_MODE_GUIDE.md` - Autonomous trading guide
- `WEEK3_MULTI_STRATEGY_ACTIVATION.md` - Strategy activation guide

### Configuration

- `config/trading_config.py` - All system parameters

### Testing

- `tests/test_core_system.py` - Unit tests

## Version History

### v1.0 (Current)
- Centralized configuration
- Type hints throughout
- Comprehensive logging
- Unit tests
- Code quality improvements
- Production-ready

### v0.2
- Multi-strategy support
- AI enhancement
- Auto-execution
- Position monitoring

### v0.1
- Initial release
- Basic strategies
- Manual execution

---

## License

Proprietary - PC HIVE Trading System

---

## Contact

For issues, questions, or contributions, create an issue in the repository.

---

**Remember**: The key to successful automated trading is:
1. Start with paper trading
2. Validate strategies thoroughly
3. Monitor continuously
4. Record and learn from outcomes
5. Never stop improving

Happy trading!
