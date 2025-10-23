# Quick Start: Enable FOREX Continuous Learning

## TL;DR - 3 Steps to Enable Learning

### 1. Collect Baseline (Week 1-2)
```bash
# Run with learning DISABLED to collect baseline data
python forex_auto_trader.py
```
**Goal**: Get 50+ trades at ~60% win rate baseline

### 2. Enable Learning
```bash
# Edit forex_learning_config.json
# Change: "enabled": false  →  "enabled": true
```

### 3. Monitor & Profit
```bash
# Run with learning ENABLED
python forex_auto_trader.py

# Check progress
cat forex_learning_logs/parameters.json
cat forex_learning_logs/trade_outcomes.json
```
**Expected**: 60% → 68%+ win rate over 4-8 weeks

---

## Configuration File Edit

**File**: `forex_learning_config.json`

**Change ONE line**:
```json
{
  "enabled": true,  // ← Change from false to true
  "learning_frequency": "weekly",
  "min_feedback_samples": 50,
  "max_parameter_change": 0.30,
  "confidence_threshold": 0.80
}
```

---

## Safety Checklist

Before enabling learning on LIVE trader:

- [x] ✓ Collected 50+ baseline trades
- [x] ✓ Baseline win rate is stable (~60%)
- [x] ✓ Ran test suite: `python test_forex_learning_simple.py`
- [x] ✓ Reviewed `forex_learning_config.json`
- [x] ✓ Understand how to disable learning (set enabled: false)
- [x] ✓ Know where logs are: `forex_learning_logs/`
- [ ] Ready to enable learning!

---

## What Happens When Enabled?

### Week 1 (Baseline Collection)
- Trader runs normally
- Learning system collects trade data
- **No parameter changes yet** (need 50 trades first)

### Week 2-3 (First Optimization)
- After 50 trades: First optimization cycle runs
- System analyzes what worked/didn't work
- Proposes optimized parameters (e.g., EMA 10→12)
- **IF** confidence > 80%: Applies changes
- **ELSE**: Waits for more data

### Week 4-8 (Continuous Improvement)
- Weekly optimization cycles
- Gradual parameter refinement
- Win rate improves: 60% → 62% → 65% → 68%+
- Sharpe ratio improves
- Drawdown controlled

---

## Monitoring Commands

### Check Current Win Rate
```bash
python -c "import json; d=json.load(open('forex_learning_logs/trade_outcomes.json')); print(f\"Win Rate: {d['win_rate']:.1%}\")"
```

### Check Parameter Changes
```bash
cat forex_learning_logs/parameters.json | grep -A 20 "current_parameters"
```

### Check Optimization Count
```bash
python -c "import json; d=json.load(open('forex_learning_logs/parameters.json')); print(f\"Optimizations: {d['optimization_count']}\")"
```

### View Last Trade Outcome
```bash
tail -20 forex_learning_logs/trade_outcomes.json
```

---

## Disable Learning (Emergency)

### Method 1: Config File
```json
// forex_learning_config.json
{
  "enabled": false  // Set to false
}
```

### Method 2: Command Line
```bash
python forex_auto_trader.py --no-learning
```

### Method 3: Revert to Baseline
```python
# Restore original parameters from baseline
import json
with open('forex_learning_logs/parameters.json', 'r') as f:
    data = json.load(f)
    baseline = data['baseline_parameters']
    print("Baseline parameters:", baseline)
# Apply these back to your config
```

---

## Expected Results

| Metric | Baseline | Target | Timeline |
|--------|----------|--------|----------|
| Win Rate | 60% | 68%+ | 6-8 weeks |
| Sharpe Ratio | ~0.8 | >1.5 | 6-8 weeks |
| Avg Pips/Trade | 8-10 | >15 | 6-8 weeks |
| Max Drawdown | 15% | <10% | 6-8 weeks |

---

## Troubleshooting

### "Learning not starting"
- Check: Did you collect 50+ trades?
- Check: Is `enabled: true` in config?
- Check: Are dependencies installed? `pip install scikit-learn numpy pandas`

### "Parameters not changing"
- Check: Confidence might be < 80%
- Solution: Wait for more trades, or lower confidence_threshold (not recommended for live)

### "Win rate going down"
- Solution: DISABLE learning immediately
- Revert to baseline parameters
- Review logs to see what changed

---

## Contact & Support

**Files to check**:
1. `FOREX_LEARNING_INTEGRATION_SUMMARY.md` - Full documentation
2. `forex_learning_config.json` - Configuration settings
3. `forex_learning_logs/` - All logs and data
4. `test_forex_learning_simple.py` - Run tests

**Common Issues**:
- Dependencies: `pip install scikit-learn numpy pandas`
- Config syntax: Validate JSON at jsonlint.com
- Logs location: `forex_learning_logs/` created automatically

---

## Quick Reference

```bash
# Test integration
python test_forex_learning_simple.py

# Run with learning (default: enabled if config says so)
python forex_auto_trader.py

# Run WITHOUT learning (temporary disable)
python forex_auto_trader.py --no-learning

# Check current performance
cat forex_learning_logs/trade_outcomes.json

# Check parameter history
cat forex_learning_logs/parameters.json

# Enable learning
# Edit forex_learning_config.json → "enabled": true

# Disable learning
# Edit forex_learning_config.json → "enabled": false
```

---

*Quick Start Guide - FOREX Learning Integration*
*For detailed documentation, see FOREX_LEARNING_INTEGRATION_SUMMARY.md*
