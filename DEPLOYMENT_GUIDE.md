# ATLAS - Complete Deployment Guide for Presidential AI Challenge

## 🎯 TONIGHT'S DEADLINE CHECKLIST

### ✅ Step 1: Verify Agents Are Working (DONE)
Your 13 AI agents are already working perfectly! We tested them.

### ✅ Step 2: Deploy Website to GitHub Pages

#### 2a. Deploy the Website
```bash
cd /Users/lvxcas/ATLAS-Presidential-AI-Challenge/frontend
npm run deploy
```

This will:
- Build your React website
- Push it to the `gh-pages` branch
- Make it ready for GitHub Pages

**Expected output:** "Published" message after ~30 seconds

#### 2b. Enable GitHub Pages in Your Repo

1. Go to: https://github.com/lvxcas/ATLAS-Presidential-AI-Challenge/settings/pages
2. Under "Source", select: **Branch: gh-pages** → **/root**
3. Click **Save**
4. Wait 1-2 minutes

Your website will be live at:
**https://lvxcas.github.io/ATLAS-Presidential-AI-Challenge**

### ✅ Step 3: Test Your Live Website

Open https://lvxcas.github.io/ATLAS-Presidential-AI-Challenge and verify:

- [ ] Navigation works (Overview, Method, Findings, Demo, Run, Links)
- [ ] "Run the mini demo" button works
- [ ] Agent names are displayed (12+ agents listed)
- [ ] Evaluation results show (cached data vs synthetic)
- [ ] Candlestick charts render
- [ ] All sections scroll smoothly

### ✅ Step 4: Test AI Agents Locally

Run these commands to demonstrate your agents work:

```bash
cd /Users/lvxcas/ATLAS-Presidential-AI-Challenge

# Test 1: Full evaluation (generates metrics)
python3 Agents/ATLAS_HYBRID/quant_team_eval.py

# Test 2: Demo with regime-shift stress window
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --window regime-shift

# Test 3: Demo with volatility-spike stress window
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --window volatility-spike
```

**What you should see:**
- All 13 agents initialize successfully
- Risk posture: GREENLIGHT / WATCH / STAND_DOWN
- Agent breakdown with scores and reasoning
- Risk flags (e.g., REGIME_SHIFT, HIGH_VOLATILITY)
- Detailed explanations

## 📋 What to Submit for the Challenge

### Your Submission Package:

1. **GitHub Repository URL:**
   ```
   https://github.com/lvxcas/ATLAS-Presidential-AI-Challenge
   ```

2. **Live Demo Website:**
   ```
   https://lvxcas.github.io/ATLAS-Presidential-AI-Challenge
   ```

3. **Key Files to Highlight:**
   - `Agents/ATLAS_HYBRID/` - Your 13 AI agents (the AI part!)
   - `submission/evaluation_results.json` - Evaluation metrics
   - `Team Authorized Narrative.md` - Your team's story
   - `README.md` - Complete documentation

## 🔍 Quick Verification Commands

```bash
# Verify agents work
python3 Agents/ATLAS_HYBRID/quant_team_eval.py

# Verify frontend builds
cd frontend && npm run build

# Verify deployment is ready
cd frontend && npm run deploy
```

## 🚨 Troubleshooting

### Problem: "npm run deploy" fails
**Solution:**
```bash
cd frontend
rm -rf node_modules
npm install
npm run build
npm run deploy
```

### Problem: Website shows 404
**Solution:**
1. Go to GitHub repo → Settings → Pages
2. Make sure "Source" is set to: **gh-pages branch**
3. Wait 2-3 minutes, then refresh

### Problem: Agents don't run
**Solution:**
```bash
# Make sure you're in the repo root
cd /Users/lvxcas/ATLAS-Presidential-AI-Challenge

# Try running with Python 3 explicitly
python3 Agents/ATLAS_HYBRID/quant_team_eval.py
```

### Problem: "Module not found" error
**Solution:**
```bash
# Install Python dependencies
pip3 install pandas numpy scipy scikit-learn pyyaml matplotlib
```

## 🎯 Your System Architecture (For the Judges)

**ATLAS** = Educational AI Risk Literacy System

**The AI Part (What Makes This an AI Project):**
- 13 independent AI risk agents
- Multi-agent reasoning with weighted aggregation
- Veto-based safety logic (safety-first coordination)
- Offline ML models (Ridge Regression for volatility/drawdown forecasting)
- Deterministic, explainable outputs

**The Agents (These are your AI!):**
1. TechnicalAgent (veto) - RSI, EMA, Bollinger, ATR volatility
2. MarketRegimeAgent - Trend vs choppy detection (ADX proxy)
3. CorrelationAgent - Position overlap and concentration risk
4. GSQuantAgent - VaR-style risk proxy
5. MonteCarloAgent - Tail risk simulation
6. RiskManagementAgent - Account safety rules
7. NewsFilterAgent (veto) - Event risk from calendar
8. SessionTimingAgent - Time-of-day liquidity effects
9. MultiTimeframeAgent - Trend alignment checks
10. VolumeLiquidityAgent - Spread and liquidity proxy
11. SupportResistanceAgent - Key level proximity
12. DivergenceAgent - RSI vs price divergence
13. OfflineMLRiskAgent - ML forecasts (volatility/drawdown)

**The Coordination:**
- Weighted aggregation of agent scores
- Safety-first veto rule (any high-risk → STAND_DOWN)
- Risk posture: GREENLIGHT (< 0.25) / WATCH (0.25-0.36) / STAND_DOWN (≥ 0.36)

**The Output:**
- Risk posture label
- Aggregated risk score (0-1)
- Risk flags (REGIME_SHIFT, HIGH_VOLATILITY, etc.)
- Plain-English explanation
- Full agent traceability

**The Safety:**
- Simulation-only (no trading, no real money)
- Offline-first (cached CSVs + synthetic fallback)
- No price prediction
- No financial advice
- Educational disclaimers everywhere

## 🏆 Why This Wins

1. **Clear AI Component:** 13 specialized agents with transparent reasoning
2. **Educational Value:** Teaches risk literacy, not trading
3. **Safety-First:** No trading, no real money, no harmful outputs
4. **Explainability:** Every decision is traceable to specific agents
5. **Real Problem:** Addresses K-12 financial literacy gap
6. **Working Demo:** Agents actually run and produce outputs
7. **Professional Delivery:** Live website + comprehensive documentation

## ✅ Final Pre-Submission Checklist

Before you submit tonight:

- [ ] Website is live at https://lvxcas.github.io/ATLAS-Presidential-AI-Challenge
- [ ] All agents run successfully (test with `python3 Agents/ATLAS_HYBRID/quant_team_eval.py`)
- [ ] `Team Authorized Narrative.md` is complete and polished
- [ ] `README.md` has clear quick-start instructions
- [ ] `submission/evaluation_results.json` exists and has data
- [ ] GitHub repo is public and accessible
- [ ] All code is committed and pushed to GitHub

## 📧 What to Submit

**In your submission form:**

1. **GitHub Repo:** https://github.com/lvxcas/ATLAS-Presidential-AI-Challenge
2. **Live Demo:** https://lvxcas.github.io/ATLAS-Presidential-AI-Challenge
3. **Team Name:** [Your team name from Team Authorized Narrative.md]
4. **Project Title:** ATLAS AI Quant Team - Educational Risk Literacy System
5. **One-Sentence Description:**
   "A multi-agent AI system that teaches K-12 students market risk through transparent, explainable reasoning—simulation-only, no trading, offline-first."

---

## 🎉 YOU'VE GOT THIS!

Your agents work. Your website builds. You have a complete, professional submission ready to go.

Just run the deployment, verify it's live, and submit before tonight's deadline!

Good luck! 🚀
