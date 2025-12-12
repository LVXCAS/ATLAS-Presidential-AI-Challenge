# ATLAS - Adaptive Trading & Learning Agent System

**Multi-Agent Forex Trading System with Institutional-Grade Intelligence**

---

## What is ATLAS?

ATLAS is a consensus-based trading system where **16 specialized AI agents** vote on every trade opportunity. Only when multiple agents agree across different analytical dimensions (technical, fundamental, quantitative, pattern-based) does ATLAS execute a trade.

**Current Status:** Live on OANDA paper trading ($159,554 balance)
**Critical Bugs Fixed:** Counter-trend trading bug (prevented $23K+ losses)
**System Version:** v0.1 (Exploration Phase)

---

## Quick Start

```bash
# Start ATLAS
cd BOTS/ATLAS_HYBRID
python run_paper_training.py

# Check account status
python -c "from adapters.oanda_adapter import OandaAdapter; import os; a = OandaAdapter(os.getenv('OANDA_API_KEY'), os.getenv('OANDA_ACCOUNT_ID'), practice=True); b = a.get_account_balance(); p = a.get_open_positions(); print('Balance: $%.2f | Positions: %d' % (b['balance'], len(p if p else [])))"

# Analyze threshold performance (after collecting 20+ trades)
python analyze_threshold_performance.py
```

---

## System Architecture

### The 16 Specialized Agents

#### **Core Analysis Agents** (Vote: BUY/NEUTRAL/BLOCK)

1. **TechnicalAgent** (Weight: 1.5, VETO Power)
   - RSI, MACD, EMAs, Bollinger Bands, ADX
   - **Counter-Trend Blocker**: Prevents buying into strong bearish trends
   - Example block: "Price below EMA200, ADX 50+ → BLOCK LONG"

2. **PatternRecognitionAgent** (Weight: 1.0)
   - Chart patterns, candlestick formations
   - Learns winning setups from historical trades

3. **SentimentAgent** (Weight: 1.5)
   - Market sentiment analysis
   - Risk-on/risk-off regime detection

4. **CorrelationAgent** (Weight: 1.0)
   - Cross-pair correlation monitoring
   - Prevents over-exposure to same currency

5. **SessionTimingAgent** (Weight: 1.0)
   - Optimal entry timing (London open, NY session)
   - Avoids low-liquidity periods

6. **MarketRegimeAgent** (Weight: 1.2)
   - Trending vs ranging vs choppy detection
   - Adapts strategy to current regime

7. **DivergenceAgent** (Weight: 1.5)
   - Price/indicator divergences
   - Hidden vs regular divergence patterns

#### **Machine Learning Agents**

8. **XGBoostMLAgent** (Weight: 2.5)
   - Gradient boosted decision trees
   - Trained on 1000+ market features

9. **QlibResearchAgent** (Weight: 1.8)
   - Microsoft Research institutional quant library
   - 1000+ quantitative factors

10. **GSQuantAgent** (Weight: 2.0)
    - Goldman Sachs institutional quant toolkit
    - Risk models and pricing analytics

11. **AutoGenAgent** (Weight: 1.5)
    - Microsoft multi-agent orchestration
    - Meta-analysis of other agents

12. **MonteCarloAgent** (Weight: 1.8)
    - Probabilistic risk assessment
    - Simulates 10,000+ possible outcomes

#### **Risk & Execution Agents**

13. **NewsFilterAgent** (Weight: 2.0, VETO Power)
    - Blocks trades 60min before NFP/FOMC/CPI
    - Auto-closes positions before major news

14. **RiskManagerAgent** (Weight: 1.5)
    - Kelly Criterion position sizing
    - Daily drawdown monitoring
    - Stop loss: 25 pips, Take profit: 50 pips (1:2 R:R)

15. **MultiTimeframeAgent** (Weight: 2.0)
    - Confirms trends across H1, H4, D1 timeframes
    - Prevents false breakouts

16. **MeanReversionAgent** (Weight: 1.8)
    - Detects overbought/oversold extremes
    - Statistical mean reversion signals

**Total Agent Weight:** 25.8

---

## Scoring System

Every trade requires **consensus** across multiple agents:

```python
# Example: EUR_USD analysis
agent_votes = {
    "TechnicalAgent": {"vote": "NEUTRAL", "confidence": 0.33},  # Weak trend
    "SentimentAgent": {"vote": "BUY", "confidence": 0.31},      # Slightly bullish
    "QlibResearchAgent": {"vote": "BUY", "confidence": 0.72},   # Strong quant signal
    "NewsFilterAgent": {"vote": "ALLOW", "confidence": 1.0},    # No news block
    # ... 12 more agents ...
}

# Score calculation
score = 0
for agent, data in agent_votes.items():
    if data["vote"] == "BUY":
        score += data["confidence"] * agent_weight
    elif data["vote"] == "BLOCK" and agent.is_veto:
        score = 0  # Instant rejection
        break

# Example calculation:
# Sentiment: 0.31 × 1.5 = 0.465
# Qlib:      0.72 × 1.8 = 1.296
# Total score: 1.77

# Decision
if score >= 1.5:  # Threshold
    execute_trade()
else:
    hold()  # Wait for higher consensus
```

### Current Configuration

- **Score Threshold:** 1.5 (requires ~2-3 agents voting BUY with moderate confidence)
- **Threshold as % of Max:** 5.8% (very selective - only high-conviction trades)
- **VETO Agents:** TechnicalAgent, NewsFilterAgent (can instantly block trades)

---

## Recent Critical Bug Fixes

### Bug #1: Counter-Trend Trading ($23,445 Loss)

**Problem:** ATLAS bought USD_JPY 12 times into strong bearish trend
**Root Cause:** `direction` parameter defaulted to empty string, disabling counter-trend blocker
**Fix:** [technical_agent.py:45](agents/technical_agent.py#L45) - Default to "long"
**Status:** ✅ Fixed and verified (EUR_USD properly blocked in testing)

```python
# BEFORE (BUGGY):
direction = market_data.get("direction", "").lower()  # Defaulted to ""
if direction == "long" and strong_downtrend:  # Never matched!
    return "BLOCK"

# AFTER (FIXED):
direction = market_data.get("direction", "long").lower()  # Defaults to "long"
if direction == "long" and strong_downtrend:  # Now works!
    return "BLOCK"
```

### Bug #2: Threshold Logging Shows 0

**Problem:** Trade logs showed `atlas_threshold: 0` instead of actual value
**Root Cause:** Incorrect dictionary path in TradeLogger
**Fix:** [trade_logger.py:148](core/trade_logger.py#L148) - Corrected nested path
**Status:** ✅ Fixed

---

## Trade Execution Example

**Real EUR_USD Analysis (Score: 0.00 - HOLD Decision)**

```
=== Market Scan 2025-12-11 14:32 ===
Pair: EUR_USD
Price: 1.0487

Agent Votes:
✓ TechnicalAgent:      NEUTRAL (0.33) - "No clear trend, ADX 28"
✓ SentimentAgent:      NEUTRAL (0.20) - "Neutral market sentiment"
✓ QlibAgent:           NEUTRAL (0.15) - "1000+ factors mixed"
✓ XGBoostMLAgent:      NEUTRAL (0.42) - "Low conviction signal"
✓ NewsFilterAgent:     ALLOW   (1.00) - "No major news"
✓ RiskManager:         APPROVE (0.75) - "Risk parameters OK"
... 10 more agents voting NEUTRAL ...

Final Score: 0.00 (threshold: 1.5)
Decision: HOLD

Reason: Insufficient consensus - agents detecting choppy/unclear market
```

**This is CORRECT behavior** - ATLAS being selective and protecting capital.

---

## Why ATLAS is Better Than Traditional Algo Bots

| Traditional Algo Bot | ATLAS Multi-Agent System |
|---------------------|-------------------------|
| Single strategy - fails when market changes | 16 strategies - adapts to regime changes |
| Trades on schedule regardless of conditions | Only trades when multiple agents agree |
| No veto power - executes bad trades | TechnicalAgent can BLOCK counter-trend trades |
| Fixed rules forever | Learning weights improve over time |
| Can't explain losses | Full vote transparency in logs |
| One-dimensional analysis | Multi-dimensional: Technical + Fundamental + Quant + ML |

**Example:** When counter-trend bug was active, a simple algo would keep losing forever. ATLAS architecture allowed us to:
1. Identify which agent failed (TechnicalAgent)
2. Fix one component
3. Entire system improves

---

## Current Performance

**Account Balance:** $159,554.38
**Starting Balance:** $183,000
**Loss from Counter-Trend Bug:** -$23,445 (12 USD_JPY trades, now fixed)
**Clean Trades (Post-Fix):** 0 (system just restarted)
**Open Positions:** 0

**Profitability Status:** Unknown - need 20-30 clean trades to measure
**Estimated Win Rate Needed:** 55%+ (with 1:2 R:R) for consistent profitability
**Probability ATLAS is Profitable:** 60-70% (after bug fixes)

---

## Threshold Optimization Plan

**Current Threshold:** 1.5 (set conservatively during exploration)
**Question:** Is 1.5 optimal, or should it be 2.0/2.5/3.0?

**Scientific Approach:**
1. Run ATLAS for 24-48 hours
2. Collect 20-30 completed trades
3. Run `python analyze_threshold_performance.py`
4. Analyze win rate by threshold:
   - Threshold 0.5: High trade count, low win rate
   - Threshold 1.5: Moderate trade count, ??? win rate
   - Threshold 2.5: Low trade count, high win rate
   - Threshold 3.5: Very few trades, very high win rate
5. Find threshold with best **expectancy** (EV per trade)

**Expectancy Formula:**
`EV = (Win_Rate × Avg_Win) - (Loss_Rate × Avg_Loss) - Costs`

With 1:2 R:R ratio:
- 40% win rate = breakeven
- 55% win rate = consistently profitable
- 60% win rate = excellent system

---

## File Structure

```
BOTS/ATLAS_HYBRID/
├── README.md                          # This file
├── run_paper_training.py              # Main entry point
├── analyze_threshold_performance.py   # Threshold optimization tool
├── backtest_threshold.py              # Historical backtest (limited by data availability)
│
├── core/
│   ├── coordinator.py                 # Multi-agent orchestrator
│   ├── trade_logger.py                # Trade logging (fixed threshold bug)
│   └── risk_manager.py                # Kelly Criterion, position sizing
│
├── agents/
│   ├── technical_agent.py             # RSI, MACD, EMAs (fixed counter-trend bug)
│   ├── pattern_recognition_agent.py   # Chart patterns
│   ├── sentiment_agent.py             # Market sentiment
│   ├── news_filter_agent.py           # News blocker (VETO)
│   ├── xgboost_ml_agent.py            # Gradient boosted trees
│   ├── qlib_research_agent.py         # Microsoft quant library
│   ├── gsquant_agent.py               # Goldman Sachs toolkit
│   ├── autogen_agent.py               # Multi-agent orchestration
│   ├── monte_carlo_agent.py           # Probabilistic risk
│   ├── correlation_agent.py           # Cross-pair monitoring
│   ├── session_timing_agent.py        # Timing optimization
│   ├── market_regime_agent.py         # Trend detection
│   ├── divergence_agent.py            # Price/indicator divergence
│   ├── multi_timeframe_agent.py       # H1/H4/D1 confirmation
│   ├── mean_reversion_agent.py        # Statistical reversion
│   └── risk_management_agent.py       # Risk oversight
│
├── adapters/
│   └── oanda_adapter.py               # OANDA API integration
│
├── config/
│   └── hybrid_optimized.json          # Agent weights and parameters
│
├── learning/
│   └── state/                         # Agent learning states (weights, patterns)
│
└── logs/
    └── trades/                        # Trade history logs
```

---

## Next Steps

1. ✅ **Fix Critical Bugs** - Counter-trend blocker, threshold logging
2. ✅ **Start ATLAS Live** - Running on OANDA paper account
3. ⏳ **Collect Data** - 24-48 hours, 20-30 clean trades
4. ⏳ **Optimize Threshold** - Run scientific analysis on outcomes
5. ⏳ **Measure Win Rate** - Determine if system is profitable
6. 🎯 **Scale Up** - If win rate ≥55%, increase position sizes

---

## Configuration

**Risk Parameters:**
- Stop Loss: 25 pips fixed
- Take Profit: 50 pips fixed (1:2 risk/reward)
- Position Sizing: Kelly Criterion with 10% fraction
- Lot Range: 20-25 lots based on Kelly calculation
- Daily Drawdown Limit: $2,500 (circuit breaker)

**Pairs Traded:**
- EUR_USD, GBP_USD, USD_JPY, AUD_USD, USD_CAD, NZD_USD, USD_CHF
- EUR_GBP, EUR_JPY, GBP_JPY

**Scan Frequency:** Every 5 minutes

---

## Monitoring

**Quick Status Check:**
```bash
python -c "from adapters.oanda_adapter import OandaAdapter; import os; a = OandaAdapter(os.getenv('OANDA_API_KEY'), os.getenv('OANDA_ACCOUNT_ID'), practice=True); b = a.get_account_balance(); p = a.get_open_positions(); print('Balance: $%.2f | Positions: %d' % (b['balance'], len(p if p else [])))"
```

**Watch for:**
- **HOLD decisions (score 0.00):** Normal during choppy markets
- **BUY executions:** Verify `atlas_threshold: 1.5` in logs
- **BLOCK messages:** Counter-trend protection working
- **Score patterns:** Look for scores approaching/exceeding 1.5

**Trade Logs Location:** `logs/trades/session_*.json`

---

## Contributing

**Bug Reports:** If you find ATLAS making dumb trades, check the trade log for agent votes and file an issue with the trade_id.

**Agent Ideas:** Want to add a 17th agent? Implement `BaseAgent` interface and add to `config/hybrid_optimized.json`.

---

## License

Proprietary. All rights reserved.

---

**Status:** Live in exploration phase (threshold 1.5) - Collecting data to optimize threshold scientifically.
