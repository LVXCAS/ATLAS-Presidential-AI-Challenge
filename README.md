# ATLAS AI Quant Team (Presidential AI Challenge - Track II)

ATLAS is a simulation-only, educational AI risk literacy system for K-12 audiences.
It explains when markets are risky and why, and it teaches when not to act.
This repo does not predict prices, execute trades, or require API keys.

## Problem statement
Students and beginners are exposed to markets and AI tools without understanding risk,
uncertainty, or how AI decisions are formed. Many tools push trading, hide reasoning,
or use real money, which creates harm rather than learning. ATLAS addresses this gap
by making risk transparent, explainable, and safe to explore offline.

## Codex Framing
ATLAS is an agent-based AI reasoning system for risk literacy. Given a cached market
scenario (CSV) or a synthetic stress window, independent risk agents (volatility,
regime, correlation, liquidity) compute interpretable signals and the coordinator
aggregates them into a categorical risk posture (GREENLIGHT, WATCH, STAND_DOWN)
plus a plain-English explanation.

ATLAS is deterministic, offline, simulation-only, and does not trade or predict prices.

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

## Codex execution overview (agentic reasoning)
ATLAS is an agent-based AI reasoning system in which independent risk agents analyze
different dimensions of uncertainty and collectively determine a categorical risk posture.

Input -> output contract:
- **Input:** cached CSV scenario (`date,open,high,low,close,volume`) or a synthetic scenario
- **Agents:** volatility, regime, correlation, and liquidity proxies (interpretable, rule-based)
- **Processing:** independent scoring -> weighted aggregation + risk flags
- **Output:** `posture`, `risk_score`, `risk_flags`, `agent_signals`, `explanation`
- **Guarantees:** deterministic, offline, no side effects, no live data

Minimal runnable skeleton (Codex-friendly):
```bash
python3 src/main.py --input src/tests/data/calm.csv
python3 src/main.py --scenario crisis
python3 -m unittest src/tests/test_scenarios.py
```

## What you get
- Cached historical CSVs (default if present) loaded offline
- Synthetic stress windows: stable, volatility-spike, regime-shift (fallback or `--data-source synthetic`)
- Risk labels: GREENLIGHT, WATCH, STAND_DOWN
- Plain-English explanations and agent outputs (vote, confidence, reasoning)
- JSON output at `submission/evaluation_results.json`

ATLAS uses cached historical market data from public sources. Live APIs are intentionally disabled for safety, reproducibility, and educational use. If cached data is missing, the demo falls back to synthetic data with a warning.

## Cached historical data (optional)
- Place OHLCV CSVs in `data/fx/` or `data/equities/` (see `data/README.md`).
- Required schema: `date,open,high,low,close,volume` (lowercase headers).
- Files should be sorted by date; FX volume may be 0.
- Cached data loading requires `pandas` but the Track II demo still runs without it.
Note: The repo ships small placeholder CSVs for format only; replace them with real historical data if desired.

Example:
```bash
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --data-source cached --asset-class fx --symbol EUR_USD
```

## Optional external APIs (disabled by default)
ATLAS can optionally use external data APIs to refresh cached CSVs. Keys belong in
`.env` and are never committed. The Track II demo remains offline and deterministic.

To refresh cached CSVs (example uses Alpha Vantage):
```bash
python3 scripts/cache_data.py --enable-live --provider alpha_vantage --asset-class equities --symbols SPY AAPL MSFT
python3 scripts/cache_data.py --enable-live --provider alpha_vantage --asset-class fx --symbols EURUSD GBPUSD
```

You can also set `USE_LIVE_DATA=true` in `.env` instead of `--enable-live`.
See `.env.example` for placeholders.

Other providers:
```bash
python3 scripts/cache_data.py --enable-live --provider polygon --asset-class equities --symbols SPY AAPL
python3 scripts/cache_data.py --enable-live --provider polygon --asset-class fx --symbols EURUSD
python3 scripts/cache_data.py --enable-live --provider alpaca --asset-class equities --symbols SPY AAPL
python3 scripts/cache_data.py --enable-live --provider fred --asset-class macro --symbols DGS10 CPIAUCSL
```
Notes: Alpaca support is equities only in this script; use Alpha Vantage or Polygon for FX.
FRED macro data is cached under `data/macro/` and is optional.

Automation helpers (optional):
```bash
python3 scripts/refresh_demo_data.py --enable-live
python3 scripts/run_agentic_pipeline.py --refresh --enable-live --asset-class equities --symbol SPY
python3 scripts/llm_explain.py --enable-live --input submission/evaluation_results.json --output submission/llm_summary.txt
```
The LLM summary is optional and not used by the Track II demo.
To enable the optional LLM agent, set `ENABLE_LLM_AGENTS=true` in `.env` and
enable `LLMTechnicalAgent` in `Agents/ATLAS_HYBRID/config/track2_quant_team.json`.
For OpenAI-compatible endpoints, set `LLM_API_BASE=https://api.openai.com` and
provide `LLM_API_KEY` or `OPENAI_API_KEY` in `.env` (never commit keys).

## What ATLAS is NOT
- Not a trading bot
- Not a prediction engine or price forecaster
- Not financial advice
- Not connected to live data, brokers, or real money
- Not an execution system

## Safety and ethics
See `safety_ethics.md` for formal safety, ethics, and age-appropriate design commitments.

## Optional UI
- `frontend/` is optional and runs without a backend using mock data
- If you do not need the UI, you can skip it
- Candlestick snapshots are sourced from `frontend/src/data/candles_spy.json`
  (regenerate from cached CSVs via `python3 scripts/export_frontend_candles.py`)
- Cached evaluation summaries come from `frontend/src/data/results_cached.json`
  (regenerate via `python3 scripts/export_frontend_results.py`)
- Matplotlib figures are generated under `frontend/public/figures/`
  (run `python3 scripts/generate_matplotlib_figures.py` after a cached eval)

## Optional research sandbox
- `research/` contains optional Qlib/R&D experiments (not required for Track II)
- Install extras with `python3 -m pip install -r requirements-research.txt`

## Repo map
- `Agents/ATLAS_HYBRID/`: primary runnable demo (multi-agent risk desk, simulation-only)
- `frontend/`: optional React + TypeScript UI
- `submission/`: Track II writeups, scripts, and generated artifacts

## Environment
- No `.env` file is required for the Track II demo.

## More docs
- Track II summary: `submission/track2_summary.md`
- Start here: `submission/track2_pdf_outline.md`
- Demo narration: `submission/demo_script_4min.md`
- Demo story + team roles: `submission/demo_script.md`
- Evaluation writeup: `submission/evaluation_artifact.md`
- Explainability artifact: `explainability.md`
- Safety & ethics statement: `safety_ethics.md`
