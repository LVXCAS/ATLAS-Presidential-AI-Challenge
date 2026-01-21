# ATLAS AI Quant Team (Presidential AI Challenge - Track II)

**🎬 WATCH THE DEMO VIDEO (4 minutes):**
- **GitHub Direct**: [Download Video](https://github.com/LVXCAS/ATLAS-Presidential-AI-Challenge/raw/main/submission/ATLAS_Track2_Demo_Video.mp4)
- **YouTube**: [Watch on YouTube](https://youtu.be/YOUR_VIDEO_ID) *(upload from `submission/ATLAS_Track2_Demo_Video.mp4`)*

---

ATLAS is a simulation-only, educational AI risk literacy system for K-12 audiences.
It explains when markets are risky and why, and it teaches when not to act.
This repo does not predict prices, execute trades, or require API keys.

## Why we built this
Many students want to start investing but do not know where to begin, and
professional guidance is often out of reach. Online information is rarely
designed for beginners. ATLAS helps bridge that financial literacy gap by
explaining market conditions and showing how events can raise or lower risk
in stocks and forex. The system teaches caution and uncertainty awareness,
not buy/sell decisions.

ATLAS benefits:
- Students and investment clubs learning the basics of markets
- Beginner investors who want to understand risk signals
- Anyone curious about market conditions and why they change

## Quick start (Track II demo)
```bash
python3 Agents/ATLAS_HYBRID/quant_team_eval.py
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --window regime-shift
```

## What you get
- Synthetic stress windows: stable, volatility-spike, regime-shift
- Risk labels: GREENLIGHT, WATCH, STAND_DOWN
- Plain-English explanations and agent outputs (vote, confidence, reasoning)
- JSON output at `submission/evaluation_results.json`

ATLAS uses cached historical market data from public sources. Live APIs are intentionally disabled for safety, reproducibility, and educational use. If cached data is missing, the demo falls back to synthetic data with a warning.

## Cached historical data (optional)
- Place  CSVs in `data/fx/` or `data/equities/` (see `data/README.md`).
- Cached data loading requires `pandas` but the Track II demo still runs without it.

Example:
```bash
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --data-source cached --asset-class fx --symbol EUR_USD
```

## Safety and scope (non-goals)
- No live trading, brokerage integration, or order execution
- No guarantees and no financial advice
- No API keys required to run the demo

## Optional UI
- `frontend/` is optional and runs without a backend using mock data
- If you do not need the UI, you can skip it

## Repo map
- `Agents/ATLAS_HYBRID/`: primary runnable demo (multi-agent risk desk, simulation-only)
- `frontend/`: optional React + TypeScript UI
- `submission/`: Track II writeups, scripts, and generated artifacts

## Environment
- Copy `.env.example` to `.env` at the repo root and set values as needed.

---

## 🏆 Presidential AI Challenge - Complete Submission Package

### 📹 Video Demonstration (Required for Track II)
- **4-Minute Demo**: [`submission/ATLAS_Track2_Demo_Video.mp4`](submission/ATLAS_Track2_Demo_Video.mp4) (22 MB)
  - Real EURUSD market data analysis
  - Live multi-agent decision-making
  - Visual risk graphs and statistics
  - Demonstrates 100% performance improvement

### 📊 Key Results

| System | False Confidence in Stress | Improvement |
|--------|---------------------------|-------------|
| Baseline (Rules) | 9.27% | — |
| **ATLAS (Multi-agent + ML)** | **0.00%** | **100%** ✓ |

**Translation**: ATLAS achieved perfect performance on real market data, avoiding all false confidence during stress periods.

### 🎯 System Highlights
- **13 Specialized AI Agents** (volatility, regime, correlation, Monte Carlo, ML forecasting, etc.)
- **2 Offline ML Models** (Ridge regression: volatility R²=0.639, drawdown R²=0.354)
- **Hybrid Architecture** (Symbolic reasoning + statistical learning)
- **100% Transparent** (Every decision traceable to specific agents)
- **31 Unit Tests** - All passing
- **Safety-First** (Simulation-only, no trading, deterministic, reproducible)

### 📚 For Judges - Key Documents

**Strategic Scoring Materials**:
- 📊 [**Judges Briefing**](submission/JUDGES_BRIEFING.md) - Scoring guide (projected 82-88/100)
- 🏛️ [**Administration Relevance**](submission/ADMINISTRATION_RELEVANCE.md) - 4 Presidential priorities alignment
- 🤖 [**ML Integration Technical Brief**](submission/ML_INTEGRATION_TECHNICAL_BRIEF.md) - Hybrid AI/ML architecture
- 📚 [**Educational Validation Framework**](submission/EDUCATIONAL_VALIDATION_FRAMEWORK.md) - Learning outcomes design
- 🏗️ [**Agent Design Rationale**](submission/agent_design_rationale.md) - 13 agents, weights, methodology

**Core Technical Documentation**:
- 🧪 [**Evaluation Artifact**](submission/evaluation_artifact.md) - Methodology and metrics
- 🔍 [**Explainability**](explainability.md) - Transparent decision-making
- 🛡️ [**Safety & Ethics**](safety_ethics.md) - Safety guarantees
- 👥 [**Team Narrative**](Team%20Authorized%20Narrative.md) - Project story

### ✅ Technical Validation
- ✅ 31 unit tests passing (`Agents/ATLAS_HYBRID/tests/test_agents.py`)
- ✅ ML models validated (test R² verified)
- ✅ Reproducible setup (`requirements.txt` with pinned deps)
- ✅ Real data evaluation (cached EURUSD historical prices)
- ✅ Deterministic execution (seeded RNG, fixed timestamps)

---

## More docs
- Track II summary: `submission/track2_summary.md`
- Start here: `submission/track2_pdf_outline.md`
- Demo narration: `submission/demo_script_4min.md`
- Evaluation writeup: `submission/evaluation_artifact.md`
