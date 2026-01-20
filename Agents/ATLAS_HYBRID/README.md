# ATLAS AI Quant Team (Track II)

This folder contains the core multi-agent “AI quant team” demo used for the Presidential AI Challenge Track II submission.

## What this does
- Produces an educational desk stance label: `GREENLIGHT` / `WATCH` / `STAND_DOWN`
- Explains *why* using traceable multi-agent rationale
- Demonstrates how static rule baselines can give false confidence during stress windows

## What this does not do
- No automated execution
- No buy/sell signals
- Not financial advice

## Run the Track II demo
From the repo root:

```bash
python3 Agents/ATLAS_HYBRID/quant_team_eval.py
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --window regime-shift
```

## Cached historical data (optional)
ATLAS uses cached historical market data from public sources. Live APIs are intentionally disabled for safety, reproducibility, and educational use.
Place OHLCV CSVs under `data/fx/` or `data/equities/` and run:

```bash
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --data-source cached --asset-class fx --symbol EUR_USD
```

If cached data is missing, the demo falls back to synthetic data with a warning.

Optional: refresh cached CSVs with `scripts/cache_data.py` (requires API keys and `--enable-live`).
Optional: enable the LLM risk agent by setting `ENABLE_LLM_AGENTS=true` in `.env`
and toggling `LLMTechnicalAgent` in `Agents/ATLAS_HYBRID/config/track2_quant_team.json`.

## Key files
- `Agents/ATLAS_HYBRID/quant_team_demo.py`: prints baseline vs multi-agent summaries
- `Agents/ATLAS_HYBRID/quant_team_eval.py`: writes `submission/evaluation_results.json`
- `Agents/ATLAS_HYBRID/quant_team_utils.py`: stress windows + lightweight indicators
- `Agents/ATLAS_HYBRID/config/track2_quant_team.json`: Track II demo configuration
