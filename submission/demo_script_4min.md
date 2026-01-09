# Track II Demo Script (≤4 minutes)

Goal: make the demo look like an **AI quant desk** (decision discipline + market context), not a trading system.

## Setup (before recording)
- Have a terminal ready in the repo root.
- Optional: open `submission/evaluation_results.json` after running the eval.
- Do not show broker accounts, balances, API keys, or “execution” features.

## Script (suggested timing)

### 0:00–0:20 — Hook (problem)
“Student investment clubs rarely have access to institutional-style quantitative research. We built an AI ‘quant desk’ made of specialist agents that explains market regime and uncertainty so students can practice better decision discipline.”

### 0:20–0:40 — What this is / is not
“This is not financial advice and it does not place trades. It outputs a simple desk stance—GREENLIGHT, WATCH, or STAND_DOWN—plus an explanation and constraints like ‘volatility spike’ or ‘regime shift.’”

### 0:40–1:20 — Baseline (show how static rules can fail)
On screen:
- Briefly describe baseline: “static RSI/ATR rules.”
- Run the evaluation:
  - `python3 BOTS/ATLAS_HYBRID/quant_team_eval.py`

Say:
“The baseline is intentionally simple. It can look confident even during stress windows, which is exactly when beginners make catastrophic mistakes.”

### 1:20–2:40 — Quant desk walkthrough (2 stress windows)
On screen:
- Run the demo and show the summary lines:
  - `python3 BOTS/ATLAS_HYBRID/quant_team_demo.py --window volatility-spike`
  - `python3 BOTS/ATLAS_HYBRID/quant_team_demo.py --window regime-shift`

Say:
“Instead of giving a buy/sell signal, it summarizes multiple specialist perspectives into a single stance and explains why. The point is to teach decision discipline: wait, reduce exposure, or stand down when conditions are unstable.”

Call out 1–2 concrete “constraints” the tool outputs:
- “One or more agents flagged high risk.”
- “Mixed signals / elevated uncertainty.”

### 2:40–3:20 — Evidence (one headline metric)
On screen:
- Open `submission/evaluation_results.json` or read the printed summary.

Say:
“Our headline evaluation metric is the GREENLIGHT‑in‑stress rate. Lower is better because ‘GREENLIGHT’ during stress is false confidence. Across three stress windows, the AI quant team reduces GREENLIGHT‑in‑stress compared to the baseline.”

### 3:20–3:50 — Pilot plan / learning outcome
Say:
“Our primary outcome for the challenge is learning and behavior: we measure improvement in a scenario‑based risk decision quiz before and after using the tool. We also capture feedback and iterate based on what students misunderstand.”

### 3:50–4:00 — Responsibility
Say:
“We designed this responsibly: no personal data, no automated execution, and clear warnings. The goal is safer participation and better risk literacy.”
