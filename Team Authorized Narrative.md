# Team Authorized Narrative - ATLAS (Track II)

## Problem and Purpose
Students and beginner investors are exposed to financial markets and AI tools without understanding risk, uncertainty, or how decisions are made. Most tools either encourage trading, hide their reasoning, or depend on live data. ATLAS addresses this gap by teaching risk literacy in a safe, explainable, and simulation-only way. The goal is to help learners recognize when markets are risky and why, not to tell them what to buy or sell.

## What ATLAS Is (and Is Not)
ATLAS is an agent-based AI reasoning system that evaluates market risk conditions and produces a categorical risk posture: GREENLIGHT, WATCH, or STAND_DOWN.

ATLAS is not:
- A trading bot
- A price prediction engine
- Financial advice
- A live market system

This project is education-first and designed for K-12 Track II.

## How the System Works
ATLAS follows a transparent pipeline:
1. Load a cached historical scenario (CSV) or synthetic stress window.
2. Independent risk agents compute interpretable signals.
3. A coordinator aggregates agent votes with a safety-first veto rule.
4. The system outputs a risk posture and a plain-English explanation.

Each agent returns a structured result with a vote, confidence, and reasoning. If data is insufficient, the agent returns NEUTRAL and does not bias the aggregate.

## AI Components and Reasoning
ATLAS uses multiple small, interpretable agents rather than a black-box model. Example agent lenses include:
- Volatility: rolling volatility and spike detection
- Regime: trend stability and regime-shift proxy
- Correlation: correlation breakdown or convergence
- Liquidity/Volume: proxy stress signals
- Technical: explainable rule-based signals

The aggregation step prioritizes safety. Any high-risk veto can force STAND_DOWN. This mirrors real-world risk management and teaches caution over action.

For teams that want to demonstrate ML workflows, ATLAS includes a research-only lab that can use
Qlib factor models and an RD-Agent workflow (with an optional LLM) to propose and backtest
strategy parameters on cached data. These experiments are sandboxed, offline, and not part of
the Track II demo outputs.

## Data Sources and Caching
ATLAS runs offline by default using cached CSVs in `data/` (schema: `date,open,high,low,close,volume`). The demo uses local files only, so the project is deterministic and judge-safe.

We support optional data refresh via external APIs to create or update cached files (not required to run the demo):
- Polygon (market data)
- Alpha Vantage (market data)
- Alpaca (historical market data only)
- FRED (macro context)

API keys are never committed to the repository; they are supplied via `.env` when needed.

## Evaluation Method
We evaluate ATLAS on behavioral correctness rather than profit:
- Calm window should produce GREENLIGHT
- Transition window should produce WATCH
- Crisis window should produce STAND_DOWN

The primary metric is lowering false GREENLIGHT labels during stress. Results are written to `submission/evaluation_results.json` for review and reproducibility.

## User Experience and Explainability
The GitHub Pages site presents:
- Candlestick charts (from cached data)
- Risk posture summaries and explanations
- Matplotlib figures for trends and risk scores

This visual layer helps students connect risk signals to real market behavior without encouraging trading.

## Optional Research Extensions (Not Required for Track II)
We include an optional research sandbox that can integrate Qlib and an R&D agent for deeper exploration. This is clearly separated from the Track II demo and is disabled by default to preserve determinism and safety.

An optional LLM-based explainer is available for summarizing risk explanations, but it is off by default and not required to run the project.

## Challenges and What We Learned
The biggest challenges were building a reproducible data pipeline and maintaining explainability while keeping the system safe. We learned to prioritize clear reasoning, uncertainty reporting, and reproducible outputs over accuracy claims.

## Future Work
With more time and resources, we would expand the library of stress scenarios, add more global datasets, and create additional learning modules that teach students how to interpret risk signals across different market contexts.

## Summary
ATLAS is a deterministic, offline, agentic AI risk literacy system. It teaches students when not to act by explaining volatility, regime shifts, correlations, and liquidity stress in clear language. This approach aligns with Track II goals by prioritizing safety, interpretability, and educational impact.
