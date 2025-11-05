# DEPRECATED Code Archive

This folder contains old versions of trading bots and systems that are no longer in active use but preserved for reference.

## Purpose
- **Reference**: Old code may contain useful logic or ideas
- **Rollback**: Can revert to previous versions if needed
- **History**: Documents evolution of the trading system

## Folder Structure

### old_bots/
Old trading bot versions from development phase:
- `ACTUALLY_WORKING_TRADER.py` - Early working version
- `SIMPLE_WORKING_TRADER.py` - Simplified bot
- `PROP_FIRM_TRADER.py` - Old prop firm version
- Various scanners and old market-specific bots

**Status**: Superseded by `MULTI_MARKET_TRADER.py` and `WORKING_FOREX_OANDA.py`

### backtesting/
Parameter optimization and backtesting scripts:
- `forex_v4_optimized.py` - Optimized forex parameters
- `forex_parameter_optimizer.py` - Parameter tuning tool
- `prop_firm_eval_simulator.py` - Prop firm evaluation simulator

**Status**: Used during development, not needed for production

### empire_versions/
Old "trading empire" multi-system launchers:
- `FIXED_AUTONOMOUS_EMPIRE.py` - Old empire system
- `TRADING_EMPIRE_MASTER.py` - Master launcher
- Various empire and launcher versions

**Status**: Replaced by unified `MULTI_MARKET_TRADER.py`

### testing/
One-off test files and diagnostics:
- `test_bot_execution.py` - Execution testing
- `test_kelly_sizing.py` - Kelly Criterion tests
- `DIAGNOSTIC_TEST.py` - System diagnostics

**Status**: Testing complete, scripts no longer needed

## Do NOT Delete
Files are preserved for:
1. Code archaeology (understanding past decisions)
2. Emergency rollback (if new system fails)
3. Extracting useful patterns for future development

## Current Production System
Use these instead:
- **Production Forex**: `WORKING_FOREX_OANDA.py` (root directory)
- **Unified Multi-Market**: `MULTI_MARKET_TRADER.py` (root directory)
- **Shared Libraries**: `SHARED/` (technical_analysis, kelly_criterion, etc.)
