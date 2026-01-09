# Evaluation Artifact (Track II)

This project evaluates an **educational AI quant team** (GREENLIGHT/WATCH/STAND_DOWN) against a **simple baseline** across 3 stress windows.

## Primary evaluation idea
- **Metric:** `GREENLIGHT-in-stress rate` (lower is better)
- **Interpretation:** how often a tool incorrectly “greenlights” a stressful market condition, which can create false confidence for beginners.

## Baseline (simple rules)
The baseline is intentionally limited: it flags risk using static thresholds (ATR/volatility, RSI extremes, and low “ADX”/trend-strength proxy). This represents what a student might implement from a checklist or basic dashboard.

## AI quant team (multi-agent)
The multi-agent quant team:
- collects votes from multiple specialist agents (trend/regime, correlation/exposure, probabilistic risk, constraints)
- aggregates them into GREENLIGHT/WATCH/STAND_DOWN
- returns an interpretable rationale (“flagged by” agents)

It does **not** generate buy/sell signals and does **not** execute trades.

## How to reproduce
From the repo root:
```bash
python3 BOTS/ATLAS_HYBRID/quant_team_eval.py
```

Outputs:
- Human-readable summary to stdout (for screenshots in the PDF/video)
- `submission/evaluation_results.json` (machine-readable)

## Stress windows used
These are synthetic windows designed to be understandable and reproducible:
- `stable-range`: mostly normal conditions
- `volatility-spike`: whipsaw volatility, elevated risk
- `regime-shift`: calm → transition into trend + higher volatility

Generator and indicator proxies live in:
- `BOTS/ATLAS_HYBRID/quant_team_utils.py`

## What to include in the PDF (recommended)
- A 1-row summary table per window with baseline vs quant-team GREENLIGHT-in-stress rates.
- 1–2 sentences explaining why false GREENLIGHT labels are dangerous for beginners.
