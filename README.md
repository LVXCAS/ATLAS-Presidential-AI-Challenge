# ATLAS AI Quant Team (Presidential AI Challenge — Track II)

This repo is a cleaned, submission-focused codebase for a Track II technology demo.
It is intentionally framed as an educational **AI quant team simulator** (multi-agent analysis), not an execution bot.

## What’s in here
- `BOTS/ATLAS_HYBRID/`: primary runnable demo (multi-agent “quant desk”, simulation-only)
- `frontend/`: optional UI (install deps yourself; `node_modules/` is intentionally not included)
- `submission/`: Track II PDF outline, demo script, and evaluation/pilot materials

## Quick start (CLI demo)
```bash
python3 BOTS/ATLAS_HYBRID/quant_team_eval.py
python3 BOTS/ATLAS_HYBRID/quant_team_demo.py --window regime-shift
```

## Environment
- Copy `.env.example` to `.env` at the repo root and set values as needed.

## More docs
- Start here: `submission/track2_pdf_outline.md`
- Demo narration: `submission/demo_script_4min.md`
- Evaluation writeup: `submission/evaluation_artifact.md`
