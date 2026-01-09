# Presidential AI Challenge — Track II Submission (PDF Outline)

This file is a page-by-page outline you can paste into a document editor and export as a single PDF (≤10 pages, ≤50MB, 12pt+ font). Replace placeholders marked `[[...]]`.

## Locked framing (edit only if you want to re-scope)

**Community problem statement (1–2 sentences)**  
Student-run investment clubs and retail investors rarely have access to institutional-style quantitative research, so they often rely on hype or single-indicator tools and misread regime shifts and uncertainty. Our project simulates an AI-powered “quant desk” made of specialist agents that generates interpretable market context (regime, volatility, correlation, uncertainty) to support better research and decision discipline.

**Primary metric (single headline number)**  
`Decision Quality Score` improvement on a short scenario-based assessment (pre vs post), measured as `% correct regime/risk decisions` across `[[N]]` scenarios.

---

## Page 1 — Title + Team + Summary

- **Project title:** `ATLAS AI Quant Team: Multi‑Agent Market Analysis for Student Investment Clubs`
- **Track:** Track II (technology solution demo)
- **Team:** `[[team name]]` — `[[members]]` — `[[school/club]]` — `[[location]]`
- **Demo video link (required):** `[[public link, no password]]` (≤4 minutes)
- **Optional repo link:** `[[link]]`
- **One-paragraph abstract (5–7 sentences):**
  - Problem → Users → What you built → What “good” looks like → Evidence snapshot → Safety/limitations.

---

## Page 2 — Community Challenge (Who, Why, Impact)

- Who is affected (students/retail investors, educators).
- Why it matters (avoid catastrophic mistakes; build risk literacy; safer participation).
- Concrete example story (1 short vignette).
- What success looks like (behavior and understanding, not returns).

---

## Page 3 — What the Technology Does (Not a Trading Tool)

**Core promise:** “Explain regime + uncertainty; recommend constraints; do not issue buy/sell signals.”

- Inputs (price history, volatility proxy, correlation proxy, basic macro/news proxies if used).
- Outputs:
  - Desk stance label: GREENLIGHT / WATCH / STAND_DOWN
  - Regime label: range / trend / transition (with confidence)
  - Plain-language rationale + “do‑not‑trade” conditions
  - Uncertainty flag (when signals disagree)
- UI modality (CLI or UI) and how educators/clubs use it in a lesson.

Add a clear disclaimer block:
> Educational use only. Not financial advice. No automated execution.

---

## Page 4 — Why AI / Multi-Agent Reasoning (vs Static Rules)

- Baseline tool description (simple RSI/ATR/volatility threshold rules).
- Why baseline fails: “false confidence” during regime shifts and volatility spikes.
- Why multi-agent helps:
  - multiple perspectives (trend, regime, correlation, risk limits, uncertainty)
  - interpretable *reasons* for risk calls
  - explicit “STAND_DOWN” conditions

---

## Page 5 — System Overview (High-Level Architecture)

- Data pipeline (even if simulated for demo): windowed price series → indicator features.
- Agent layer: each agent produces vote + confidence + rationale.
- Quant desk layer: consolidates votes into GREENLIGHT/WATCH/STAND_DOWN with traceability (“flagged by” agents).
- No trade execution layer (explicitly removed/disabled for Track II).

Include a small diagram (box/arrow) if you can.

---

## Page 6 — Evidence: Stress-Window Evaluation (Baseline vs AI Quant Team)

Use 2–3 “stress windows” (historical or simulated) and show:
- Baseline label vs Quant-team label
- What the user should do (avoid / reduce exposure / wait)
- One screenshot or table per window

Recommended headline metric for this section (secondary metric):
- `GREENLIGHT-in-stress rate` (lower is better): how often the tool incorrectly “greenlights” during stress conditions.

Repro instructions:
- `python3 BOTS/ATLAS_HYBRID/quant_team_eval.py`

---

## Page 7 — Evidence: Micro Pilot (Learning + Behavior)

Describe the pilot:
- Participants: `[[N]]` students, `[[club/school]]`
- Task: `[[short scenario decision task]]`
- Measures:
  - Pre/post quiz: `% correct risk decisions`
  - Qualitative feedback: 3 quotes + common confusion points
  - One concrete iteration you made based on feedback

Show results (even small):
- Pre score → Post score
- One example of a corrected misconception

---

## Page 8 — Safety, Responsibility, and Limitations

- Not financial advice; no automatic execution; no personal data.
- How you prevent misinterpretation:
  - never output “BUY/SELL”
  - always show uncertainty and constraints
  - “when not to use” section (thin data, illiquid assets, etc.)
- Limitations (data quality, simulated windows, generalization).
- Future work (more user testing, better datasets, educator materials).

---

## Page 9 — Team Reflection (500+ words requirement)

You can use this draft and expand/modify. It is written to be team-authored; replace `[[...]]` and add your lived experience.

**Narrative draft (expand to match your real build + pilot):**

We built ATLAS AI Quant Team because we noticed a consistent problem in our community: students and first‑time investors often treat markets like a guessing game and rely on “prediction tools” that feel confident even when the situation is unstable. In our investment club, we saw that people can memorize indicators like RSI, but they still struggle to answer basic questions like “What market regime are we in?” and “What would make this idea unsafe even if my indicator looks good?” During volatile periods, this gap leads to overconfidence and mistakes that are avoidable with better decision discipline and a clearer understanding of uncertainty.

Our project is not a trading bot and does not tell users what to buy or sell. Instead, it simulates a quant desk: multiple specialist agents analyze the same market window and produce an interpretable “desk stance” that supports research and decision discipline. The output is GREENLIGHT, WATCH, or STAND_DOWN, plus an explanation of “why,” including constraints like “volatility spike,” “signals disagree,” or “regime shift/transition.”

We used a multi‑agent AI approach because market interpretation is multi‑dimensional. A single rule can be helpful, but it can also give false confidence—especially during regime changes and volatility spikes. Our agents represent different perspectives (technical trend/regime context, correlation and exposure awareness, probabilistic risk simulation, and conservative constraints). Each agent produces a vote and rationale, and the quant desk layer aggregates these into a single educational recommendation that remains traceable to the underlying reasons.

To evaluate the system, we compare it against a simple baseline that uses static rules (like volatility and RSI thresholds). We then test both approaches across multiple stress windows. Our focus is not returns or profitability. Instead, we look for situations where a tool incorrectly “greenlights” a stressful period. Reducing these false-greenlight moments is important because they are when beginners are most likely to take catastrophic risk. We also plan to run a micro pilot with a student investment club. Our main outcome metric is a scenario‑based decision assessment: students answer short questions about whether a situation is stable, uncertain, or unsafe and why. We will measure improvement from pre to post and collect qualitative feedback about what explanations helped or confused them.

We are building this responsibly. The tool includes clear warnings that it is not financial advice, it does not execute trades, and it should not be used as a prediction engine. It avoids personal data, and it emphasizes uncertainty instead of certainty. Through this project, we learned that a “cool AI system” is not enough—impact requires clarity, safety, and evidence. We also learned that the hardest part is communication: we must present risk information in a way that is understandable to non‑experts and reduces overconfidence rather than increasing it. Our next steps are to expand user testing, improve the educational materials, and continue iterating based on how real students interpret the outputs.

---

## Page 10 — Certification + Citations

### Certification of Originality (required)
Include the signed/affirmed statement (see `submission/certification_of_originality.md`).

### AI tools and sources (required)
- AI tools used for development: `[[list]]`
- AI tools used for writing/narrative: `[[list]]`
- Primary sources / references: `[[list]]`

EOF
