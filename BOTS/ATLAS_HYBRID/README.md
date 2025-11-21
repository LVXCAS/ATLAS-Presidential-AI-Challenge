# ATLAS HYBRID - Multi-Agent Forex Trading System

## Overview

ATLAS (Adaptive Trading & Learning Agent System) is a multi-agent forex trading system designed for E8 prop firm challenges with built-in learning capabilities.

## Key Features

✅ **10 Specialized Agents** - Each votes independently on trade decisions
✅ **Learning Engine** - Improves performance over time through reinforcement learning
✅ **Paper Trading Mode** - Train agents for 60 days before risking real capital
✅ **Dual Strategy Modes** - Hybrid-Optimized (3-5 lots) + Ultra-Aggressive (5-7 lots)
✅ **News Protection** - Auto-close positions before major economic events
✅ **Performance Dashboard** - Real-time monitoring of agent performance
✅ **E8 Compliance** - Built-in daily DD tracking and circuit breakers

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      COORDINATOR                            │
│  - Collects votes from all agents                          │
│  - Applies learned weights                                  │
│  - Makes final BUY/SELL/HOLD decision                       │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │ Agent 1 │       │ Agent 2 │  ...  │ Agent 10│
   │Technical│       │ Pattern │       │  News   │
   └─────────┘       └─────────┘       └─────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │   LEARNING  │
                    │   ENGINE    │
                    │ - Tracks    │
                    │ - Adjusts   │
                    │ - Learns    │
                    └─────────────┘
```

## Training Pipeline

### Phase 1: Exploration (Days 1-20)
- **Goal:** Generate maximum training data
- **Threshold:** 3.5 score
- **Position Size:** 1-2 lots
- **Learning Rate:** HIGH
- **Expected Trades:** 100-150

### Phase 2: Refinement (Days 21-40)
- **Goal:** Optimize winning patterns
- **Threshold:** 4.0 score
- **Position Size:** 2-3 lots
- **Learning Rate:** MEDIUM
- **Expected Trades:** 80-120

### Phase 3: Validation (Days 41-60)
- **Goal:** Prove E8 readiness
- **Threshold:** 4.5 score
- **Position Size:** 3-5 lots (E8 sizing)
- **Learning Rate:** LOW
- **Expected Trades:** 60-100

### Phase 4: E8 Deployment (After validation)
- **Requirement:** 25%+ monthly ROI, 0 daily DD violations
- **Mode:** Live trading with learned weights
- **Expected Pass Rate:** 50-60%

## Agent Descriptions

### 1. TechnicalAgent
- **Signals:** RSI, MACD, EMAs, Bollinger Bands, ADX
- **Weight:** 1.5 (initial)
- **Vote:** BUY/SELL/NEUTRAL

### 2. PatternRecognitionAgent
- **Learns:** Winning setups from historical trades
- **Examples:** "EUR/USD + RSI 38-42 during London = 78% win rate"
- **Weight:** Starts at 1.0, increases as patterns discovered
- **Vote:** BUY/SELL/NEUTRAL + confidence based on pattern match

### 3. VolumeAgent
- **Signals:** Volume spikes, volume profile, liquidity sweeps
- **Weight:** 1.0 (initial)
- **Vote:** BUY/SELL/NEUTRAL

### 4. MarketRegimeAgent
- **Detection:** Trending vs Ranging vs Choppy
- **Weight:** 1.2 (initial)
- **Vote:** ALLOW/BLOCK trade based on regime

### 5. NewsFilterAgent
- **Function:** Blocks trades 60 min before NFP/FOMC/CPI
- **Auto-Close:** Closes positions 30 min before major news
- **Weight:** 2.0 (VETO power)
- **Vote:** ALLOW/BLOCK

### 6. RiskManagementAgent
- **Function:** Position sizing, Kelly Criterion, DD checks
- **Weight:** 1.5
- **Vote:** APPROVE size or REDUCE size

### 7. E8ComplianceAgent
- **Tracks:** Daily DD, trailing DD, profit target
- **Circuit Breaker:** Stops trading if down -$2,500 in single day
- **Weight:** 2.0 (VETO power)
- **Vote:** ALLOW/BLOCK

### 8. SessionTimingAgent
- **Optimizes:** Trade timing (London open, NY session, etc.)
- **Avoids:** Low-liquidity Asian session (unless high conviction)
- **Weight:** 1.2
- **Vote:** BOOST score during optimal hours

### 9. CorrelationAgent
- **Monitors:** Pair correlations (EUR/USD vs GBP/USD)
- **Prevents:** Over-exposure to same currency
- **Weight:** 1.0
- **Vote:** ALLOW/BLOCK based on existing positions

### 10. SentimentAgent (Future)
- **Data:** News headlines, social sentiment
- **Weight:** 0.8 (initially low until proven)
- **Vote:** BUY/SELL/NEUTRAL

## Scoring System

Each agent votes with confidence (0.0 - 1.0):

```python
agent_votes = {
    "TechnicalAgent": {"vote": "BUY", "confidence": 0.85},
    "PatternRecognitionAgent": {"vote": "BUY", "confidence": 0.92},
    "VolumeAgent": {"vote": "BUY", "confidence": 0.70},
    "MarketRegimeAgent": {"vote": "ALLOW", "confidence": 0.80},
    "NewsFilterAgent": {"vote": "ALLOW", "confidence": 1.0},
    "RiskManagementAgent": {"vote": "APPROVE", "confidence": 0.75},
    "E8ComplianceAgent": {"vote": "ALLOW", "confidence": 1.0},
    "SessionTimingAgent": {"vote": "BOOST", "confidence": 0.65},
    "CorrelationAgent": {"vote": "ALLOW", "confidence": 0.90}
}

# Calculate final score
final_score = 0
for agent, data in agent_votes.items():
    weight = agent_weights[agent]  # Learned over time
    confidence = data["confidence"]

    if data["vote"] == "BUY":
        final_score += confidence * weight
    elif data["vote"] == "BLOCK":
        final_score = 0  # VETO
        break

# Decision
if final_score >= threshold:  # 4.0 or 4.5 depending on mode
    execute_trade()
```

## Learning Engine

### How Agents Learn

After each trade closes:

```python
trade_result = {
    "pair": "EUR_USD",
    "entry_time": "2025-11-20 08:30",
    "exit_time": "2025-11-20 10:15",
    "pnl": +1200,
    "r_multiple": 2.3,
    "outcome": "WIN",
    "agent_votes": {...}  # All agent votes at entry
}

learning_engine.process_trade(trade_result)
```

**Learning Engine Actions:**
1. **Update Agent Weights:**
   - If TechnicalAgent voted BUY and trade won → increase weight
   - If MarketRegimeAgent voted BUY and trade lost → decrease weight

2. **Discover Patterns:**
   - Track conditions when trades win vs lose
   - Example: "RSI 35-40 + London open = 72% win rate"
   - Store in pattern library

3. **Adjust Thresholds:**
   - If win rate < 55% → raise score threshold
   - If trade frequency < target → lower score threshold

4. **Agent Leaderboard:**
   - Rank agents by performance
   - Visualize which agents contribute most to wins

### Weight Adjustment Formula

```python
# After every 50 trades
for agent in agents:
    win_rate = agent.wins / agent.total_votes
    avg_r_multiple = agent.total_r / agent.wins

    # Performance score
    performance = (win_rate * 0.6) + (avg_r_multiple / 5 * 0.4)

    # Adjust weight
    if performance > 0.7:
        agent.weight *= 1.15  # Boost high performers
    elif performance < 0.4:
        agent.weight *= 0.85  # Reduce poor performers

    # Clip weights
    agent.weight = max(0.5, min(2.5, agent.weight))
```

## Configuration Modes

### Hybrid-Optimized (Recommended for Training)
```json
{
  "mode": "hybrid_optimized",
  "position_size_lots": [3, 5],
  "score_threshold": 4.0,
  "max_trades_per_week": 8-12,
  "stop_loss_pips": 12-15,
  "take_profit_r": [1.5, 3.0],
  "daily_dd_limit": 2500,
  "target_monthly_roi": 0.25-0.40
}
```

### Ultra-Aggressive (High Risk / High Reward)
```json
{
  "mode": "ultra_aggressive",
  "position_size_lots": [5, 7],
  "score_threshold": 3.5,
  "max_trades_per_week": 15-20,
  "stop_loss_pips": 10-12,
  "take_profit_r": [1.5, 4.0],
  "daily_dd_limit": 2500,
  "target_monthly_roi": 0.40-0.60
}
```

## Paper Trading Setup

### OANDA Paper Account
1. Create free OANDA paper trading account
2. Get API credentials (same as live account process)
3. Set initial balance to $200,000 (E8 starting capital)

### Running Paper Trading

```bash
# Start 60-day training
python BOTS/ATLAS_HYBRID/run_paper_training.py --mode exploration --days 20

# Continue with refinement
python BOTS/ATLAS_HYBRID/run_paper_training.py --mode refinement --days 20

# Final validation
python BOTS/ATLAS_HYBRID/run_paper_training.py --mode validation --days 20

# Check if ready for E8
python BOTS/ATLAS_HYBRID/check_deployment_readiness.py
```

### A/B Testing (Parallel Strategies)

```bash
# Run both strategies simultaneously
python BOTS/ATLAS_HYBRID/run_ab_test.py --account_a hybrid --account_b ultra --days 60

# Compare results
python BOTS/ATLAS_HYBRID/compare_strategies.py
```

## Performance Dashboard

### Real-Time Monitoring

```bash
# Launch dashboard
python BOTS/ATLAS_HYBRID/dashboard/live_dashboard.py
```

**Dashboard shows:**
- Current account balance and ROI
- Win rate, profit factor, R-multiples
- Agent performance leaderboard
- Recent trades with agent votes
- Discovered patterns
- Learning progress metrics
- Deployment readiness checklist

## Deployment Criteria

System must meet ALL criteria before E8 deployment:

✅ **60 days of paper trading completed**
✅ **Monthly ROI ≥ 25%**
✅ **Win rate ≥ 55%**
✅ **Zero daily DD violations** (never lost more than $3k in single day)
✅ **Max drawdown < 6%**
✅ **Profit factor ≥ 1.5**
✅ **At least 150 trades executed** (sufficient training data)

## File Structure

```
BOTS/ATLAS_HYBRID/
├── README.md (this file)
├── core/
│   ├── coordinator.py
│   ├── learning_engine.py
│   ├── performance_tracker.py
│   └── deployment_gatekeeper.py
├── agents/
│   ├── base_agent.py
│   ├── technical_agent.py
│   ├── pattern_recognition_agent.py
│   ├── volume_agent.py
│   ├── market_regime_agent.py
│   ├── news_filter_agent.py
│   ├── risk_management_agent.py
│   ├── e8_compliance_agent.py
│   ├── session_timing_agent.py
│   └── correlation_agent.py
├── modes/
│   ├── paper_trading.py
│   ├── live_trading.py
│   └── training_phases.py
├── learning/
│   ├── trade_database.json
│   ├── agent_weights.json
│   ├── pattern_library.json
│   └── performance_history.json
├── dashboard/
│   ├── live_dashboard.py
│   └── agent_leaderboard.py
├── config/
│   ├── paper_exploration.json
│   ├── paper_refinement.json
│   ├── paper_validation.json
│   ├── e8_hybrid_optimized.json
│   └── e8_ultra_aggressive.json
├── run_paper_training.py
├── run_ab_test.py
├── check_deployment_readiness.py
└── deploy_to_e8.py
```

## Next Steps

1. **Build Core Architecture** (coordinator, learning engine)
2. **Implement All 10 Agents**
3. **Create Paper Trading Mode**
4. **Build Performance Dashboard**
5. **Run 60-Day Training**
6. **Deploy to E8**

---

**Status:** Building now...
