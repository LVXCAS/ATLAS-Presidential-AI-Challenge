# A/B Testing Framework - TA-Only vs TA+AI Hybrid

## Purpose
Compare performance of pure TA-Lib signals vs AI-enhanced hybrid system over 1 week.

## Test Modes

### Mode A: TA-Only (Baseline)
```python
# In MULTI_MARKET_TRADER.py, set:
self.use_ai_confirmation = False
```
**Characteristics:**
- Uses only TA-Lib indicators (RSI, MACD, EMA, ADX, ATR, Bollinger)
- Kelly Criterion position sizing based on TA confidence
- Multi-timeframe confirmation (4H trend filter)
- Executes all trades with score >= min_score threshold

**Pros:**
- Faster execution (no API calls)
- Zero external dependencies
- Deterministic signals
- Proven quantitative approach

**Cons:**
- No context awareness (news, volatility spikes)
- May trade during unfavorable conditions
- No intelligent filtering of edge cases

### Mode B: TA+AI Hybrid (Experimental)
```python
# In MULTI_MARKET_TRADER.py, set:
self.use_ai_confirmation = True
```
**Characteristics:**
- TA-Lib generates trade candidates (PRIMARY)
- AI analyzes high-confidence trades (score >= 6.0) for context
- DeepSeek V3.1 + MiniMax multi-model voting (SECONDARY)
- AI can APPROVE / REJECT / REDUCE_SIZE
- Executes only AI-validated trades

**Pros:**
- Context-aware filtering (ECB meetings, Fed announcements, high volatility)
- Intelligent risk reduction during uncertain conditions
- Multi-model consensus reduces AI hallucination risk
- Free APIs (zero cost)

**Cons:**
- Adds latency (2-3 seconds per AI call)
- Requires OPENROUTER_API_KEY
- Non-deterministic (AI responses vary slightly)
- May reject profitable trades (false negatives)

## Metrics to Track

### Performance Metrics
| Metric | Mode A (TA-Only) | Mode B (TA+AI) | Winner |
|--------|------------------|----------------|--------|
| **Total Signals Generated** | | | |
| **Trades Executed** | | | |
| **Win Rate** | | | |
| **Profit Factor** | | | |
| **Max Drawdown** | | | |
| **Sharpe Ratio** | | | |
| **Average Trade Duration** | | | |
| **Total P/L** | | | |

### AI-Specific Metrics (Mode B Only)
- **AI Approval Rate**: X% of high-confidence signals approved
- **AI Rejection Rate**: Y% of high-confidence signals rejected
- **Model Consensus Rate**: Z% both DeepSeek + MiniMax agreed
- **AI Latency**: Average time per AI call
- **False Rejections**: Trades AI rejected that would have been winners
- **True Rejections**: Trades AI rejected that would have been losers

## Testing Protocol

### Week 1: Parallel Testing (Recommended)
**Run both modes simultaneously on different accounts:**
- **TA-Only**: OANDA practice account (existing WORKING_FOREX_OANDA.py)
- **TA+AI Hybrid**: MULTI_MARKET_TRADER.py on separate practice account

**Benefits:**
- Direct comparison under identical market conditions
- No bias from different time periods
- Real-world validation of AI value-add

**Setup:**
1. Keep WORKING_FOREX_OANDA.py running (TA-only baseline)
2. Launch MULTI_MARKET_TRADER.py with `use_ai_confirmation = True`
3. Run for 7 days (Mon-Fri, 2 weeks if needed)
4. Compare logs: `logs/summary_*.json` vs OANDA performance

### Alternative: Sequential Testing
**Run Mode A for 1 week, then Mode B for 1 week:**
- Week 1: `use_ai_confirmation = False`
- Week 2: `use_ai_confirmation = True`

**Benefits:**
- Simpler setup (one system at a time)
- No separate accounts needed

**Drawbacks:**
- Market conditions may differ between weeks
- Harder to compare apples-to-apples

## Data Collection

### Automated Logging
All data automatically logged to `logs/` directory:

**Signals Log** (`logs/signals_{session_id}.json`):
```json
[
  {
    "timestamp": "2025-01-31T10:15:30",
    "market": "forex",
    "symbol": "EUR_USD",
    "direction": "long",
    "score": 7.5,
    "rsi": 28.5,
    "signals": ["RSI_OVERSOLD", "MACD_BULL_CROSS", "4H_BULLISH_TREND"]
  }
]
```

**AI Decisions Log** (`logs/ai_decisions_{session_id}.json`):
```json
[
  {
    "timestamp": "2025-01-31T10:15:35",
    "market": "forex",
    "symbol": "EUR_USD",
    "ta_score": 7.5,
    "action": "APPROVE",
    "confidence": 85,
    "consensus": true,
    "reason": "Strong technical setup, no news conflicts"
  }
]
```

**Summary Log** (`logs/summary_{session_id}.json`):
```json
{
  "session_id": "20251031_101530",
  "duration_hours": 168.0,
  "total_signals": 47,
  "ta_only_signals": 32,
  "ai_analyzed": 18,
  "ai_approved": 12,
  "ai_rejected": 6,
  "consensus_rate": 83.3,
  "rejection_rate": 33.3
}
```

## Analysis After 1 Week

### Key Questions to Answer:
1. **Did AI improve win rate?**
   - Compare Mode B win% vs Mode A win%

2. **Did AI reduce drawdown?**
   - Compare max drawdown between modes

3. **Did AI save from bad trades?**
   - Analyze rejected trades: Would they have lost money?
   - Use `trade_logger.compare_ai_rejected_outcomes()`

4. **Was AI consensus valuable?**
   - Did trades with 2-model consensus perform better than single-model approvals?

5. **What was the cost/benefit?**
   - Mode A: More trades, faster execution
   - Mode B: Fewer trades, but higher quality?

### Decision Criteria

**Keep AI if:**
- ✅ Mode B win rate >= Mode A win rate + 5%
- ✅ Mode B max drawdown < Mode A max drawdown
- ✅ AI rejected >50% losers (true rejections)
- ✅ Model consensus rate > 70%

**Disable AI if:**
- ❌ Mode B win rate < Mode A win rate
- ❌ AI rejected >30% winners (false rejections)
- ❌ API latency causes missed opportunities
- ❌ Model consensus rate < 50% (unreliable)

**Hybrid Approach if:**
- 🟡 Mixed results - use AI only for specific markets (e.g., forex yes, crypto no)
- 🟡 Use AI only during high-volatility periods
- 🟡 Require 2-model consensus for AI to override TA

## Running the Test

### Setup OpenRouter API (Free)
1. Sign up at https://openrouter.ai
2. Get free API key
3. Add to `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   ```

### Launch TA+AI Hybrid Mode
```bash
# Windows
python MULTI_MARKET_TRADER.py

# Background (Windows)
pythonw MULTI_MARKET_TRADER.py
```

### Launch TA-Only Mode (Baseline)
```bash
# Already running: WORKING_FOREX_OANDA.py
# OR set use_ai_confirmation = False in MULTI_MARKET_TRADER.py
```

### Monitor Logs
```bash
# View AI decisions live
type logs\ai_decisions_*.json

# View summary
type logs\summary_*.json
```

### Stop and Analyze
```bash
# Press Ctrl+C to stop
# Logs auto-export on shutdown
# Compare: logs\summary_*.json files from both modes
```

## Expected Outcome

**Hypothesis:** AI confirmation will improve risk-adjusted returns by filtering 20-30% of signals that occur during:
- Major news events (Fed, ECB, NFP)
- Choppy/low-liquidity hours
- High volatility spikes
- Contra-trend setups that TA can't detect

**Success Metric:** Mode B (TA+AI) achieves higher Sharpe ratio than Mode A (TA-only), even if total profit is similar.

**Validation Period:** 1 week (minimum), 2-4 weeks (ideal)

## Next Steps After A/B Test

### If AI Wins:
1. Deploy TA+AI hybrid on prop firm accounts (E8, Apex, CFT)
2. Fine-tune AI prompts for each market
3. Consider adding more free models (Gemini 2.0 Flash, Qwen)

### If TA-Only Wins:
1. Disable AI confirmation layer
2. Keep pure TA-Lib + Kelly Criterion system
3. Focus on optimizing TA parameters instead

### If Inconclusive:
1. Extend test to 4 weeks
2. Try hybrid: AI for forex only, TA-only for futures/crypto
3. Require 100% consensus (both models must agree)
