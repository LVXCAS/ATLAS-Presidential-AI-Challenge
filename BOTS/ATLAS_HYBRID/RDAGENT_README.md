# RD-Agent Autonomous Strategy Discovery

**Microsoft RD-Agent** integrated with ATLAS for autonomous trading strategy evolution.

## What It Does

RD-Agent **autonomously discovers NEW trading factors** that outperform your current strategies:

1. **Analyzes** your current performance and identifies weaknesses
2. **Generates** factor hypotheses using AI (GPT-4/Claude)
3. **Codes** new factors in Python automatically
4. **Backtests** factors on 6 months of historical data
5. **Validates** using walk-forward analysis
6. **Ranks** by Sharpe ratio and deploys top performers

**Result:** System that **improves itself** every week without manual intervention.

## Quick Start

### 1. Set Up LLM API Key

RD-Agent needs an LLM to generate factor ideas. Choose one:

**Option A: OpenAI GPT-4** (Recommended for speed)
```bash
# Add to .env file
OPENAI_API_KEY=sk-your-key-here
```

**Option B: Anthropic Claude** (Recommended for quality)
```bash
# Add to .env file
ANTHROPIC_API_KEY=your-key-here
```

Then update `llm_model` in [run_rdagent_discovery.py:22](run_rdagent_discovery.py#L22):
```python
rd_agent = RDAgentFactorDiscovery(
    llm_model="gpt-4o-mini"  # or "claude-3-5-sonnet"
)
```

### 2. Run Discovery Cycle (Manual)

```bash
cd C:\Users\lucas\PC-HIVE-TRADING\BOTS\ATLAS_HYBRID
python run_rdagent_discovery.py
```

**What happens:**
- Fetches 6 months of EUR/GBP/JPY data from OANDA
- Loads your current ATLAS performance
- Generates 4-8 new factor hypotheses
- Backtests each factor
- Saves report to `rdagent_workspace/discovery_report_*.md`
- Recommends top factors for deployment

**Runtime:** 5-15 minutes depending on LLM speed

### 3. Review Discovered Factors

Check the generated report:
```
rdagent_workspace/discovery_report_20251201_140530.md
```

Example output:
```markdown
## Top Discovered Factors

### 1. VolumeMomentumCross
**Sharpe Ratio:** 2.10
**Win Rate:** 61.5%
**Description:** Detects when volume surge confirms price momentum
**Formula:** `(volume / avg_volume_20) * (price - ema_20) / atr_14`
```

### 4. Deploy Top Factors

**Manual Deployment** (Recommended for first factors):

1. Review factor code in report
2. Create new agent file: `agents/volume_momentum_agent.py`
3. Copy factor logic from report
4. Add to `run_paper_training.py` imports
5. Test in paper trading for 1 week
6. If Sharpe > 1.8, deploy to production

**Auto-Deployment** (Future feature):
- RD-Agent will auto-deploy factors with Sharpe > 2.0
- Monitors performance for 48 hours
- Reverts if underperforming

## Scheduling Weekly Discovery

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task: "ATLAS RD-Agent Discovery"
3. Trigger: Weekly, Sunday 2:00 AM
4. Action: Start a program
   ```
   Program: C:\Users\lucas\AppData\Local\Programs\Python\Python313\python.exe
   Arguments: run_rdagent_discovery.py
   Start in: C:\Users\lucas\PC-HIVE-TRADING\BOTS\ATLAS_HYBRID
   ```
5. Done! RD-Agent runs weekly automatically

### Verification

Check if it ran:
```bash
dir rdagent_workspace\discovery_report_*.md
```

You should see new reports every Sunday.

## Example Discovery Cycle

```
================================================================================
RD-AGENT FACTOR DISCOVERY CYCLE #1
================================================================================

[ANALYSIS] Identified 2 areas for improvement:
  - Zero trades executed - agents too conservative, need new signals
  - Need more consistent returns

[HYPOTHESIS] Generating factor ideas using gpt-4o-mini...
[HYPOTHESIS] Generated 4 factor ideas

[CODING] Auto-generating factor code...
[CODING] Successfully coded 4 factors

[BACKTEST] Running backtests on 4 factors...
[BACKTEST] ✓ VolumeMomentumCross: Sharpe 2.10, WR 61.5%
[BACKTEST] ✓ RegimeAdjustedRSI: Sharpe 1.85, WR 58.2%
[BACKTEST] ✗ MicrostructureImbalance: Failed validation
[BACKTEST] ✗ CorrelationDivergence: Failed validation

[COMPLETE] Discovery cycle finished
[COMPLETE] Discovered 2 high-quality factors
[COMPLETE] Top factor: VolumeMomentumCross (Sharpe 2.10)
```

## Cost Estimation

**LLM API Costs** (per discovery cycle):
- GPT-4o-mini: ~$0.10-0.30 per cycle
- GPT-4o: ~$1-3 per cycle
- Claude 3.5 Sonnet: ~$0.50-1.50 per cycle

**Weekly:** $1-12/month depending on model choice

**ROI Calculation:**
- If RD-Agent discovers 1 factor that adds +0.3 Sharpe
- On $200k account = ~$30k-60k extra annual profit
- API cost: $144/year (weekly runs)
- **ROI: 20,000%+**

## Advanced: Full RD-Agent Integration

The current system uses simplified backtesting. For **full Microsoft RD-Agent**:

### 1. Install Additional Dependencies

```bash
pip install qlib
pip install backtrader
```

### 2. Enable Full RD-Agent in Code

Uncomment lines in [rdagent_factor_discovery.py:38-42](agents/rdagent_factor_discovery.py#L38-42):

```python
# Currently using simplified mode
# To enable full RD-Agent, install:
# - qlib (Microsoft quantitative research platform)
# - Full RD-Agent scenarios

from rdagent.scenarios.qlib.experiment.factor_experiment import FactorFBWorkspace
# ... rest of imports
```

### 3. Configure RD-Agent Settings

Create `rdagent_config.yaml`:
```yaml
llm:
  model: "gpt-4o"  # or claude-3-5-sonnet
  temperature: 0.7
  max_tokens: 4000

backtest:
  start_date: "2024-06-01"
  end_date: "2024-12-01"
  initial_capital: 200000
  commission: 0.0001

validation:
  min_sharpe: 1.8
  min_win_rate: 0.58
  max_drawdown: 0.12
  min_trades: 100
```

## Troubleshooting

**"RD-Agent components not fully available"**
- Normal on first run - using simplified mode
- Full RD-Agent requires additional setup (optional)

**"No historical data available"**
- Check OANDA connection
- Verify .env has OANDA_API_KEY and OANDA_ACCOUNT_ID

**"LLM API error"**
- Check API key in .env
- Verify API credits/balance
- Try switching models (gpt-4o-mini → claude-3-5-sonnet)

**Discovery takes too long**
- Use faster model: gpt-4o-mini (60 seconds vs 5 minutes)
- Reduce backtest period in code (6 months → 3 months)

## Next Steps

1. **Week 1:** Run first manual discovery, review factors
2. **Week 2:** Deploy top factor to paper trading
3. **Week 3:** If profitable, move to production
4. **Week 4+:** Enable weekly auto-discovery, monitor reports

**Goal:** Fully autonomous system that evolves strategies without your intervention.

---

**Questions?**
- RD-Agent Docs: https://github.com/microsoft/RD-Agent
- ATLAS Integration: Check `rdagent_factor_discovery.py` code comments
