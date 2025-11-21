# Hybrid TA + AI Multi-Market Trading System

## Overview

Unified trading system that combines **proven quantitative analysis** (TA-Lib + Kelly Criterion) with **intelligent AI confirmation** (DeepSeek V3.1 + MiniMax) across forex, futures, and crypto markets.

### Architecture

```
MULTI_MARKET_TRADER.py (Unified Program)
├── ForexMarketHandler (E8 Markets)
│   ├── EUR_USD, USD_JPY, GBP_USD, GBP_JPY
│   └── OANDA API integration
├── FuturesMarketHandler (Apex Trader Funding)
│   ├── ES, NQ, CL, GC
│   └── Futures data integration (TODO)
├── CryptoMarketHandler (Crypto Fund Trader)
│   ├── BTC/USD, ETH/USD
│   └── Crypto exchange API (TODO)
└── UnifiedTradingEngine
    ├── Scans all 3 markets every 30 minutes
    ├── Applies TA-Lib scoring (PRIMARY)
    ├── AI confirmation layer (SECONDARY)
    └── Executes validated trades

SHARED/ Libraries:
├── technical_analysis.py - RSI, MACD, EMA, ADX, ATR, Bollinger
├── kelly_criterion.py - Position sizing across all markets
├── multi_timeframe.py - 4H trend confirmation
├── ai_confirmation.py - DeepSeek + MiniMax validation
└── trade_logger.py - A/B testing performance tracking
```

## How It Works

### 1. TA-Lib Signal Generation (PRIMARY)
```python
# Scan market and calculate technical score
rsi = ta.calculate_rsi(closes)          # Oversold/overbought
macd = ta.calculate_macd(closes)        # Momentum
ema_fast/slow = ta.calculate_ema()      # Trend direction
adx = ta.calculate_adx()                # Trend strength
trend_4h = mtf.get_higher_timeframe()   # 4H confirmation

# Score LONG signals (0-10 scale)
if rsi < 30: score += 2                 # RSI oversold
if macd > signal: score += 2            # MACD bullish cross
if trend_4h == 'bullish': score += 2    # 4H trend confirms
# ... more signals

# If score >= min_score threshold → TRADE CANDIDATE
```

### 2. AI Confirmation Layer (SECONDARY)
```python
# For high-confidence candidates (score >= 6.0):
trade_data = {
    'symbol': 'EUR_USD',
    'direction': 'long',
    'score': 7.5,
    'rsi': 28.5,
    'macd': {...},
    'trend_4h': 'bullish'
}

# Multi-model AI analysis
deepseek_response = ai_agent.call_llm('deepseek', prompt)
minimax_response = ai_agent.call_llm('minimax', prompt)

# Both models must agree (consensus voting)
if deepseek == 'APPROVE' and minimax == 'APPROVE':
    EXECUTE TRADE
elif deepseek == 'REJECT' or minimax == 'REJECT':
    SKIP TRADE
else:
    REDUCE POSITION SIZE (conservative default)
```

### 3. Kelly Criterion Position Sizing
```python
# Calculate optimal position size
position_data = kelly.calculate_position_size(
    technical_score=7.5,
    fundamental_score=0,  # Not used yet
    account_balance=190000,
    risk_per_trade=0.01  # 1%
)

# Adjust size based on AI confidence
if ai_decision['action'] == 'APPROVE' and ai_confidence > 80:
    final_size = position_data['final_size'] * 1.0
elif ai_decision['action'] == 'REDUCE_SIZE':
    final_size = position_data['final_size'] * 0.5
elif ai_decision['action'] == 'REJECT':
    final_size = 0  # Skip trade
```

### 4. Comprehensive Logging
```python
# All signals logged for A/B testing
trade_logger.log_ta_signal(market, signal_data)
trade_logger.log_ai_decision(market, symbol, ta_score, ai_decision)
trade_logger.log_execution(market, execution_data)

# On shutdown: Export JSON logs
trade_logger.export_session_logs()
# → logs/signals_20251031_101530.json
# → logs/ai_decisions_20251031_101530.json
# → logs/executions_20251031_101530.json
# → logs/summary_20251031_101530.json
```

## Installation

### 1. Install Dependencies
```bash
pip install numpy pandas talib requests python-dotenv oandapyV20
```

### 2. Configure Environment Variables
Create `.env` file:
```env
# OANDA (Forex)
OANDA_API_KEY=your_oanda_api_key_here
OANDA_ACCOUNT_ID=101-004-12345678-001

# OpenRouter (Free AI APIs)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional: Other prop firm APIs (TODO)
# APEX_API_KEY=...
# CFT_API_KEY=...
```

Get free OpenRouter API key: https://openrouter.ai

### 3. Test Installation
```bash
# Test TA-Lib
python -c "import talib; print('TA-Lib OK')"

# Test OANDA connection
python -c "from SHARED.technical_analysis import ta; print('Shared libs OK')"
```

## Usage

### Run TA+AI Hybrid Mode (Default)
```bash
# Terminal (foreground)
python MULTI_MARKET_TRADER.py

# Background (Windows)
pythonw MULTI_MARKET_TRADER.py

# Background (Linux/Mac)
nohup python MULTI_MARKET_TRADER.py &
```

### Run TA-Only Mode (Disable AI)
Edit [MULTI_MARKET_TRADER.py:779](MULTI_MARKET_TRADER.py#L779):
```python
# Change this line:
self.use_ai_confirmation = True   # AI enabled

# To:
self.use_ai_confirmation = False  # AI disabled (pure TA-Lib mode)
```

Then run:
```bash
python MULTI_MARKET_TRADER.py
```

### Monitor Performance
```bash
# View logs directory
dir logs  # Windows
ls logs/  # Linux/Mac

# View latest AI decisions
type logs\ai_decisions_*.json | tail -50  # Windows
tail -50 logs/ai_decisions_*.json  # Linux/Mac

# View session summary
type logs\summary_*.json
```

### Stop System
```
Press Ctrl+C

# System will:
1. Print session performance summary
2. Export all logs to JSON
3. Shutdown gracefully
```

## A/B Testing: TA-Only vs TA+AI

See [AB_TEST_CONFIG.md](AB_TEST_CONFIG.md) for full testing protocol.

### Quick Start
**Week 1: Run both modes in parallel**
- System A: Keep WORKING_FOREX_OANDA.py running (TA-only baseline)
- System B: Launch MULTI_MARKET_TRADER.py with `use_ai_confirmation = True`

**Compare after 1 week:**
```python
# Analyze logs
mode_a_summary = json.load(open('forex_trading.log'))  # TA-only
mode_b_summary = json.load(open('logs/summary_*.json'))  # TA+AI

# Compare win rates, drawdown, profit factor
```

## Configuration Options

### Market Handlers
Edit [MULTI_MARKET_TRADER.py](MULTI_MARKET_TRADER.py):

**Forex (ForexMarketHandler:62-82)**
```python
self.pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'GBP_JPY']
self.min_score = 2.5        # Minimum TA score to trade
self.risk_per_trade = 0.01  # 1% risk per trade
self.leverage = 5           # 5x leverage
```

**Futures (FuturesMarketHandler:259-294)**
```python
self.contracts = {'ES': {...}, 'NQ': {...}, 'CL': {...}, 'GC': {...}}
self.account_size = 150000
self.min_score = 3.0        # Higher threshold for futures
self.risk_per_trade = 0.01
```

**Crypto (CryptoMarketHandler:507-518)**
```python
self.pairs = {'BTCUSD': {...}, 'ETHUSD': {...}}
self.account_size = 200000
self.min_score = 2.5
self.risk_per_trade = 0.008  # 0.8% (crypto more volatile)
```

### AI Confirmation Settings
Edit [SHARED/ai_confirmation.py:45-58](SHARED/ai_confirmation.py#L45-58):

```python
# Free models on OpenRouter
self.models = {
    'deepseek': 'deepseek/deepseek-chat',  # DeepSeek V3.1
    'minimax': 'minimax/minimax-01'        # MiniMax
}

# Voting threshold
self.consensus_required = True  # Both models must agree

# Add more free models:
# 'gemini': 'google/gemini-2.0-flash-exp:free'
# 'qwen': 'qwen/qwen-2.5-72b-instruct:free'
```

### Scanning Intervals
Edit [MULTI_MARKET_TRADER.py:759](MULTI_MARKET_TRADER.py#L759):

```python
# Unified scanning interval
self.scan_interval = 1800  # 30 minutes (default)

# Options:
# 3600 = 1 hour (slower, fewer API calls)
# 1800 = 30 minutes (balanced)
# 900 = 15 minutes (aggressive, more trades)
```

## Prop Firm Deployment

### Current Status
- ✅ **FOREX (E8 Markets)**: OANDA API integrated, ready for E8 $500K account
- 🟡 **FUTURES (Apex)**: Placeholder code, needs Rithmic/CQG API integration
- 🟡 **CRYPTO (CFT)**: Placeholder code, needs exchange API (ccxt library)

### Next Steps for Full Deployment
1. **Validate FOREX bot**: Run TA+AI hybrid on OANDA practice for 1 week
2. **Integrate futures API**: Connect to Apex Trader Funding data feed
3. **Integrate crypto API**: Use ccxt library for Binance/Bybit/etc.
4. **Purchase prop firm accounts**:
   - E8 One $500K ($1,627)
   - Apex $150K PA ($167)
   - Crypto Fund Trader $200K ($499)
5. **Deploy on funded capital**: Run MULTI_MARKET_TRADER.py on all 3 accounts

## Troubleshooting

### AI Confirmation Not Working
```bash
# Check OpenRouter API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENROUTER_API_KEY'))"

# Should output: sk-or-v1-...
# If None: Add OPENROUTER_API_KEY to .env file
```

### TA-Lib Import Error
```bash
# Windows: Install from wheel
pip install TA_Lib-0.4.24-cp310-cp310-win_amd64.whl

# Linux: Install via apt
sudo apt-get install ta-lib

# Mac: Install via brew
brew install ta-lib
```

### OANDA API Errors
```bash
# Verify API key and account ID
python check_oanda_positions.py

# Check if practice/live environment matches
# Practice: environment='practice'
# Live: environment='live' (only use after validation!)
```

### No Opportunities Found
```bash
# Lower min_score threshold temporarily
# Edit MULTI_MARKET_TRADER.py:
self.min_score = 1.5  # Lower from 2.5 to see more signals

# Or disable time filters
# Comment out AVOID_HOURS checks
```

## Performance Expectations

### Conservative Estimates (After Validation)
**Monthly ROI by Market:**
- FOREX (E8 $500K): 6% = $30,000/month
- FUTURES (Apex $150K): 6% = $9,000/month
- CRYPTO (CFT $200K): 6% = $12,000/month

**Total: $51,000/month across 3 markets**

**Cost: $2,293 one-time (all 3 prop firm accounts)**

**Porsche 911 Turbo S ($240k):**
- Down payment ($60k): Month 2
- Full purchase: Month 5-6

## Safety Features

1. **Fallback to TA-Only**: If AI APIs fail, system auto-reverts to pure TA-Lib mode
2. **Multi-Model Consensus**: Requires both DeepSeek + MiniMax to agree (reduces AI hallucination risk)
3. **Kelly Criterion**: Position sizing prevents over-leveraging
4. **Multi-Timeframe Confirmation**: 4H trend must align with 1H entry
5. **Time Filters**: Avoids choppy hours (London close, NY afternoon, etc.)
6. **Comprehensive Logging**: Every signal/decision tracked for audit

## License

Proprietary - Lucas Trading System v0.1

## Support

For issues:
1. Check logs: `logs/summary_*.json`
2. Review configuration: `.env` file and handler settings
3. Test components individually: `python -c "from SHARED.ai_confirmation import ai_agent; print(ai_agent.enabled)"`
