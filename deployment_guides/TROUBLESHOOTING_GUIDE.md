# Troubleshooting Guide - Fix Any System Issue

## Quick Diagnostics

### System Won't Start
```bash
python verify_day3_ready.py
```

This checks:
- ✅ API keys configured
- ✅ Dependencies installed
- ✅ Account accessible
- ✅ Market data working

---

## Common Errors By System

### FOREX SYSTEMS

#### Error: "Cannot fetch USD/JPY data"
**Cause:** Data source unavailable or rate limited

**Fix:**
```bash
# Check data fetcher
python test_market_data.py

# Try alternative data source
# Edit config/FOREX_USD_JPY_CONFIG.json
# Change "data_source": "alpaca" to "data_source": "yfinance"
```

#### Error: "Outside trading hours"
**Cause:** System correctly waiting for London/NY session

**Fix:** This is NORMAL. System only trades 7 AM - 8 PM UTC.
- Check current time: `python -c "from datetime import datetime; import pytz; print(datetime.now(pytz.UTC))"`
- Verify trading hours in config
- System will auto-start when market opens

#### Error: "ADX below threshold"
**Cause:** Market too choppy, no clear trend

**Fix:** This is GOOD! System avoiding bad conditions.
- Lower ADX threshold (NOT recommended): Edit config `"adx_threshold": 20`
- Wait for trending market (recommended)

#### Error: "No signals for days"
**Cause:** Strict filters working correctly (EMA Strict system)

**Fix:** This is NORMAL for strict system!
- Check last signal time: `tail logs/forex_ema_strict_*.log`
- Verify system is scanning: Check log updates every 5 minutes
- Consider using EMA Balanced for more signals

#### Error: "Broker rejected order"
**Possible Causes:**
1. Insufficient buying power
2. Position limits reached
3. Symbol not tradeable
4. Outside market hours

**Fix:**
```bash
# Check account
python check_account_status.py

# Verify buying power
python -c "from alpaca_trade_api import REST; api = REST(); print(api.get_account().buying_power)"

# Check positions
python -c "from alpaca_trade_api import REST; api = REST(); print(len(api.list_positions()))"
```

---

### OPTIONS SYSTEMS

#### Error: "Options not tradeable on this account"
**Cause:** Options trading not enabled on Alpaca account

**Fix:**
1. Log into Alpaca dashboard
2. Go to Account Settings
3. Enable options trading
4. Wait 24-48 hours for approval
5. Verify: `python check_options_capability.py`

#### Error: "No options found for strike $X"
**Cause:** Strike not available or options chain not loaded

**Fix:**
```bash
# Check options availability
python enhanced_options_validator.py AAPL

# Try different strike (system should auto-adjust)
# If persistent, check Alpaca options availability
```

#### Error: "Insufficient collateral for cash-secured put"
**Cause:** Not enough buying power for put collateral

**Fix:**
- Cash-secured put needs: `strike × 100 × contracts`
- Example: $50 strike needs $5,000 per contract
- Reduce contracts or increase account balance
- Check: `python calculate_options_collateral.py`

#### Error: "Bull Put Spread rejected"
**Cause:** Leg order issues or spread not recognized

**Fix:**
```bash
# Verify spread is supported
python test_bull_put_spread.py

# Check if need to enable spreads in Alpaca
# Some accounts default to "long only" options
```

#### Error: "Week3 Scanner finds 0 candidates"
**Cause:** Market regime (VERY_BULLISH = no Bull Put Spread candidates)

**Fix:** This is CORRECT behavior!
```bash
# Check market regime
python market_regime_detector.py

# If VERY_BULLISH:
#   - Bull Put Spreads NOT viable (all stocks have high momentum)
#   - System correctly waiting OR using Dual Options instead
#   - This is GOOD risk management

# Solution: Wait for market to cool OR use Dual Options
```

#### Error: "Adaptive Dual Options - QuantLib Greeks failed"
**Cause:** QuantLib pricing library issue

**Fix:** System will auto-fallback to percentage-based strikes
```bash
# This is NORMAL and safe
# QuantLib is optional enhancement
# System uses proven percentage method as backup

# To fix QuantLib (optional):
pip install QuantLib-Python
```

---

### GPU/AI SYSTEMS

#### Error: "CUDA not available"
**Cause:** No GPU or GPU drivers not installed

**Fix:**
```bash
# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install CUDA drivers:
# NVIDIA: https://developer.nvidia.com/cuda-downloads

# System will auto-fallback to CPU (slower but works)
```

#### Error: "GPU out of memory"
**Cause:** Batch size too large for GPU

**Fix:**
```bash
# Edit RUN_GPU_AI_AGENT.py
# Find: batch_size = 256
# Change to: batch_size = 128 (or 64)

# Or reduce model size in config
```

#### Error: "Training not converging"
**Cause:** Learning rate too high or data quality issue

**Fix:**
```bash
# Check training data
python test_gpu_training_data.py

# Reduce learning rate in config:
# "learning_rate": 0.001 → 0.0005

# Increase training episodes
```

---

### GENERAL ISSUES

#### Error: "ModuleNotFoundError: No module named 'X'"
**Fix:**
```bash
# Reinstall requirements
pip install -r requirements.txt

# If specific module:
pip install [module-name]

# Common missing modules:
pip install alpaca-trade-api yfinance pandas numpy torch
```

#### Error: "API authentication failed"
**Fix:**
```bash
# Check .env file exists
ls -la .env

# Verify keys are set
cat .env | grep ALPACA

# Should see:
# ALPACA_API_KEY=...
# ALPACA_SECRET_KEY=...
# ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Get new keys from:
# https://app.alpaca.markets/paper/dashboard/overview
```

#### Error: "Rate limit exceeded"
**Cause:** Too many API calls to data provider

**Fix:**
```bash
# System should handle this automatically
# If persistent, increase scan interval:

# Edit config:
# "scan_interval_seconds": 300 → 600 (10 minutes instead of 5)

# Or use Multi-Source Data Fetcher (no rate limits):
# Already default in Week3+ systems
```

#### Error: "Connection timeout"
**Fix:**
```bash
# Check internet
ping google.com

# Check Alpaca status
curl https://status.alpaca.markets

# Restart system
python STOP_[SYSTEM].py
python RUN_[SYSTEM].py
```

#### Error: "Cannot write to logs/"
**Fix:**
```bash
# Check directory permissions
ls -la logs/

# Create if missing
mkdir -p logs trades config

# Fix permissions (Linux/Mac)
chmod 755 logs trades config
```

---

## System-Specific Diagnostics

### Forex EMA Balanced Not Trading
**Check:**
```bash
# 1. Is it scanning?
tail -f logs/forex_ema_balanced_$(date +%Y%m%d).log

# 2. Are there signals?
grep "SIGNAL" logs/forex_ema_balanced_*.log | tail -20

# 3. What's the last score?
grep "Score:" logs/forex_ema_balanced_*.log | tail -10

# 4. Check filters
python -c "
from strategies.forex_ema_strategy import ForexEMAStrategy
s = ForexEMAStrategy()
print(f'Threshold: {s.score_threshold}')
print(f'RSI bounds: [{s.rsi_long_lower}, {s.rsi_long_upper}]')
"
```

### Week3 Scanner Not Executing
**Check:**
```bash
# 1. Is it scanning?
tail -f logs/week3_scanner_*.log

# 2. Any opportunities found?
grep "qualified opportunities" logs/week3_scanner_*.log | tail -5

# 3. Check market regime
python market_regime_detector.py

# 4. Verify account
python check_account_status.py
```

### Options System Not Finding Trades
**Check:**
```bash
# 1. Options enabled?
python check_options_capability.py

# 2. Check specific symbol
python enhanced_options_validator.py AAPL

# 3. Review strike selection
python test_strike_selection.py

# 4. Market regime compatible?
python market_regime_detector.py
```

---

## Performance Issues

### System Running Slow
**Diagnose:**
```bash
# Check CPU usage
top  # Linux/Mac
# or
taskmgr  # Windows

# Check memory
free -h  # Linux
# or
wmic OS get FreePhysicalMemory  # Windows
```

**Fix:**
1. Close unnecessary programs
2. Reduce scan frequency (5 min → 10 min)
3. Reduce number of concurrent systems
4. Upgrade RAM (8GB → 16GB)

### High Memory Usage
**Fix:**
```bash
# Clear old logs
find logs/ -name "*.log" -mtime +30 -delete

# Reduce data lookback
# Edit config: "lookback_periods": 500 → 250

# Restart system daily
crontab -e
# Add: 0 0 * * * /path/to/restart_systems.sh
```

---

## Emergency Procedures

### Account Losing Money Fast
```bash
# IMMEDIATE ACTION
python scripts/emergency_stop.py

# This will:
# 1. Stop all running systems
# 2. Close all open positions at market
# 3. Cancel all pending orders
# 4. Lock account (manual unlock required)

# After emergency stop:
# 1. Review what went wrong
# 2. Check logs for the day
# 3. Verify no system bugs
# 4. Paper trade for 3-5 days before resuming
```

### System Acting Strange
```bash
# Quick restart
python STOP_ALL_SYSTEMS.py
python RUN_FOREX_EMA_BALANCED.py  # Or your preferred system

# Full reset
rm logs/*.log
rm trades/*_temp.json
python verify_day3_ready.py
python RUN_[SYSTEM].py
```

### Can't Stop a System
```bash
# Find process
ps aux | grep python  # Linux/Mac
# or
tasklist | findstr python  # Windows

# Kill process
kill -9 [PID]  # Linux/Mac
# or
taskkill /F /PID [PID]  # Windows

# Or nuclear option (kills ALL Python)
pkill -9 python  # Linux/Mac
# or
taskkill /F /IM python.exe  # Windows
```

---

## Validation Commands

### Verify Everything is Working
```bash
# Run full validation
python verify_day3_ready.py

# Test each component
python test_market_data.py
python test_options_capability.py
python test_database_schema.py
python test_broker_integration.py

# Should see all ✓ checks pass
```

### Check System Health
```bash
# Monitor all systems
python MONITOR_ALL_SYSTEMS.py

# Check specific system
python MONITOR_FOREX_EMA_BALANCED.py

# Review performance
python calculate_pnl.py
```

---

## Getting Help

### Before Asking for Help

1. **Check the logs:**
```bash
tail -100 logs/[system]_$(date +%Y%m%d).log
```

2. **Run diagnostics:**
```bash
python verify_day3_ready.py
python test_market_data.py
```

3. **Search this guide** (Ctrl+F)

4. **Check recent changes:**
```bash
git log --oneline -10
```

### When Asking for Help, Provide:

1. **System name:** (e.g., "Forex EMA Balanced")
2. **Error message:** (exact text from logs)
3. **What you tried:** (list troubleshooting steps)
4. **Log snippet:**
```bash
tail -50 logs/[system].log > error_log.txt
# Attach error_log.txt
```

5. **System info:**
```bash
python --version
pip list | grep alpaca
python -c "import sys; print(sys.platform)"
```

---

## Prevention Tips

### Daily Maintenance
```bash
# Check system health
python MONITOR_ALL_SYSTEMS.py

# Review logs for warnings
grep -i "warning\|error" logs/*.log | tail -20

# Backup trades
cp trades/*.json backups/trades_$(date +%Y%m%d)/
```

### Weekly Maintenance
```bash
# Update systems
git pull origin main
pip install -r requirements.txt --upgrade

# Clean old logs (keep 30 days)
find logs/ -name "*.log" -mtime +30 -delete

# Review performance
python weekend_risk_analysis.py
```

### Monthly Maintenance
```bash
# Full system review
python verify_day3_ready.py

# Update API keys if needed
# Review all config files
# Analyze monthly performance
# Adjust strategies based on results
```

---

## Still Stuck?

### Last Resort Debugging
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG  # Linux/Mac
# or
set LOG_LEVEL=DEBUG  # Windows

# Run with full debug output
python RUN_[SYSTEM].py 2>&1 | tee debug_output.txt

# Review debug_output.txt for clues
```

### Nuclear Reset (Fresh Start)
```bash
# ⚠ WARNING: This deletes all logs and trades!

# Backup first
tar -czf backup_$(date +%Y%m%d).tar.gz logs/ trades/ config/

# Clean everything
rm -rf logs/* trades/*

# Reinstall
pip install -r requirements.txt --force-reinstall

# Verify setup
python verify_day3_ready.py

# Start fresh
python RUN_[SYSTEM].py
```

---

## Common "Not Actually Errors"

### "Waiting for signal" (appears for hours)
✅ **NORMAL** - System is selective, waiting for quality setup

### "Score 7.8 (below threshold 8.0)"
✅ **NORMAL** - Close but not quite, system correctly skipping

### "Outside trading hours"
✅ **NORMAL** - Forex only trades London/NY session

### "Market regime: VERY_BULLISH - Bull Put Spreads NOT viable"
✅ **NORMAL** - Correct risk management, waiting for better conditions

### "ADX 22 (below threshold 25)"
✅ **NORMAL** - Market too choppy, avoiding poor conditions

### "QuantLib Greeks failed, using percentage method"
✅ **NORMAL** - Backup method works perfectly fine

---

**Remember:** Most "errors" are actually the system correctly waiting for optimal conditions. Patience is a feature, not a bug.
