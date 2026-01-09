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
python3 BOTS/ATLAS_HYBRID/quant_team_eval.py
python3 BOTS/ATLAS_HYBRID/quant_team_demo.py --window regime-shift
```

## Key files
- `BOTS/ATLAS_HYBRID/quant_team_demo.py`: prints baseline vs multi-agent summaries
- `BOTS/ATLAS_HYBRID/quant_team_eval.py`: writes `submission/evaluation_results.json`
- `BOTS/ATLAS_HYBRID/quant_team_utils.py`: stress windows + lightweight indicators
- `BOTS/ATLAS_HYBRID/config/track2_quant_team.json`: Track II demo configuration
