# HIVE TRADE - Complete System Architecture

## 🔄 PARALLEL SYSTEMS OVERVIEW

### System 1: LIVE CRYPTO TRADING 🚀
**File:** `live_crypto_final.py` (Background Process)
**Purpose:** Real money trading 24/7
**How it works:**
1. **Every 2 minutes:**
   - Connects to Alpaca API
   - Gets account info + positions
   - Generates trading signals (40% BUY, 30% SELL, 30% HOLD)
   - Places real trades if confidence > 70%
   - Logs all trades to `live_crypto_trades.log`

2. **Risk Management:**
   - Max $75 per trade
   - 2% of buying power maximum
   - Only trades BTC/ETH pairs
   - Requires minimum $25 buying power

3. **Current Results:**
   - 9 trades executed automatically
   - BTC position: +$0.12 profit
   - ETH position: -$1.61 unrealized
   - Portfolio: $99,997.05

### System 2: AI TRAINING SYSTEM 🤖
**File:** `ai_training_system.py` (Background Process) 
**Purpose:** Learn from market data to improve predictions
**How it works:**
1. **Every 3 minutes:**
   - Fetches real data from yfinance (AAPL, MSFT, GOOGL, BTC-USD, ETH-USD, etc.)
   - Calculates technical indicators: SMA, RSI, volatility, momentum
   - Performs sentiment analysis (price-based)
   - Creates training features [8 features per symbol]

2. **Every 15 minutes (5 cycles):**
   - Trains Random Forest models (Stock + Crypto)
   - Calculates model accuracy
   - Generates AI predictions
   - Keeps 500 most recent training samples

3. **Features Used:**
   - SMA(10)/Current Price ratio
   - SMA(20)/Current Price ratio  
   - Normalized RSI (0-1)
   - Price volatility
   - 5-day momentum
   - Sentiment score (-1 to +1)
   - Log volume
   - Daily price change

### System 3: BACKEND API 🔗
**File:** `backend/main.py` (Background Process)
**Purpose:** Serve real-time data to dashboards
**Endpoints:**
- `/api/dashboard/live-feed` - Simulated Bloomberg data
- `/api/crypto/status` - Real crypto trading data
- `/api/crypto/positions` - Live positions
- `/api/crypto/trades` - Trade history

### System 4: DASHBOARDS 📊
**Files:** `*.html` (Web interfaces)
1. **Bloomberg Terminal** - Main trading interface
2. **Crypto Dashboard** - Live crypto monitoring  
3. **Simple Dashboard** - Clean overview
4. **AI Training Dashboard** - ML progress

## 🔄 DATA FLOW

```
Real Market Data (yfinance) → AI Training System → ML Models → Predictions
                                     ↓
Live Crypto Trading ← Trading Signals ← Risk Management ← Account Data (Alpaca)
        ↓
Trade Execution → Trade Log → Backend API → Dashboards → User Interface
```

## 🧠 CURRENT AI APPROACH (Supervised Learning)

**What it does:**
- Learns patterns from historical price data
- Predicts BUY/SELL based on technical indicators
- Updates models with new market data

**What it DOESN'T do (yet):**
- Learn from actual trading results
- Adapt based on profit/loss outcomes
- Meta-learning (learning how to learn better)
- Reinforcement learning from environment feedback

## 🚀 UPGRADE TO META-LEARNING + RL

**We can enhance it to:**
1. **Reinforcement Learning:** Learn from actual trade results
2. **Meta-Learning:** Learn optimal learning strategies
3. **Feedback Loop:** Use P&L to improve signal generation
4. **Adaptive Models:** Change strategy based on market conditions

**Would you like me to implement the RL/Meta-learning upgrade?**