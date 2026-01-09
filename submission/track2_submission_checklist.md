# Track II Submission Checklist (Nationals-Level, Evidence-Driven)

Use this as a step-by-step checklist to ensure your submission is **rubric-aligned**, **measurable**, and **easy to review**. It’s written to match this repo’s framing: an **educational, simulation-only AI risk coach** (multi-agent “quant desk”), not a trading system.

## 0) Non-negotiables (must be true everywhere)
- [ ] **No buy/sell signals** (use `GREENLIGHT / WATCH / STAND_DOWN` language).
- [ ] **No live trading / no execution** (simulation only).
- [ ] **Not financial advice** disclaimer shown in demo + PDF + README.
- [ ] **Uncertainty + “when to stand down”** is explicit, not implied.
- [ ] **Success metrics are about learning/discipline**, not profit or returns.

## 1) Required artifacts (submission package)
- [ ] **PDF (≤10 pages, 12pt+, ≤50MB)** following `submission/track2_pdf_outline.md`.
- [ ] **4-minute demo video (public link, no password)** following `submission/demo_script_4min.md`.
- [ ] **Certification of originality**: include `submission/certification_of_originality.md`.
- [ ] **Citations / AI tool disclosure**: use `submission/citations_template.md`.
- [ ] **Repro instructions** in `README.md` (already present; verify still accurate).

## 2) Community challenge framing (Page 1–2 of PDF)
- [ ] **Problem (1–2 sentences):** students/retail investors take high-risk actions because they can’t recognize regimes/uncertainty.
- [ ] **Who benefits:** student investment clubs, educators, retail learners.
- [ ] **Why it matters:** reduces catastrophic mistakes, builds risk literacy + decision discipline.
- [ ] **Concrete vignette:** one short story (“indicator looked fine → regime shift → losses; our system would have said STAND_DOWN + why”).
- [ ] **What success looks like:** better decisions on scenarios (not “outperformance”).

## 3) System story (what it does, and what it is not)
- [ ] **Inputs (simple):** price history → indicator proxies → agent votes.
- [ ] **AI logic (simple):** multiple specialist agents → aggregator → stance + rationale.
- [ ] **Outputs (concrete):**
  - [ ] desk stance label (`GREENLIGHT/WATCH/STAND_DOWN`)
  - [ ] regime label (`range/trend/transition`) + confidence/uncertainty
  - [ ] plain-language “why” + constraints (“do-not-trade” conditions)
- [ ] **Explicit “not” list:** no execution, no profit prediction, no personalized advice.

## 4) Evidence A — Baseline comparison on stress windows (must include)
Goal: show how a simple rules dashboard can give **false confidence** in unstable periods, and your system reduces that.

### Repro steps (screenshot-friendly)
- [ ] Run evaluation: `python3 BOTS/ATLAS_HYBRID/quant_team_eval.py`
- [ ] Confirm JSON written: `submission/evaluation_results.json`
- [ ] Run demo windows (for narrative screenshots):  
  - [ ] `python3 BOTS/ATLAS_HYBRID/quant_team_demo.py --window volatility-spike`  
  - [ ] `python3 BOTS/ATLAS_HYBRID/quant_team_demo.py --window regime-shift`

### What to include in the PDF (Page 6)
- [ ] A **1-row table per window** with:
  - [ ] baseline `GREENLIGHT-in-stress rate`
  - [ ] quant-team `GREENLIGHT-in-stress rate`
  - [ ] 1–2 sentence interpretation (“lower is better because false GREENLIGHT encourages unsafe risk-taking”)
- [ ] One screenshot per window (terminal output is fine) that shows **baseline vs quant-team** side-by-side.
- [ ] One short paragraph: “Baseline fails because it can’t represent uncertainty/regime transitions; multi-agent perspective flags disagreement + constraints.”

### Recommended headline metric (already supported)
- [ ] **Primary:** `GREENLIGHT-in-stress rate` (lower is better)
- [ ] **Optional secondary:** “STAND_DOWN recall in stress” (how often you correctly warn during stress steps)

## 5) Evidence B — Mini pilot study (5–10 people is enough)
Goal: demonstrate **learning and decision discipline** (pre/post), plus qualitative feedback.

### Minimum viable pilot (60–90 minutes total)
- [ ] Recruit `N=5–10` (club members/classmates).
- [ ] Pre-task (5–10 minutes): short scenario quiz (use `submission/pilot_quiz_and_task.md`).
- [ ] Intervention (10–15 minutes): show 2 stress windows + explain stance/constraints.
- [ ] Post-task (5–10 minutes): same or parallel scenario quiz.
- [ ] Debrief (5 minutes): 2–3 feedback questions.

### What to record (for Page 7)
- [ ] `N`, who they are (students/club), and session duration.
- [ ] Pre score → Post score (`% correct risk/regime decisions`).
- [ ] One chart or small table of results (even hand-made is fine).
- [ ] 2–3 short quotes (what helped, what confused them).
- [ ] One common misconception (e.g., “they wanted a buy/sell answer”) and how you addressed it.

### Data hygiene (recommended)
- [ ] Do **not** collect sensitive personal data; use anonymous IDs (`P1..P10`).
- [ ] Keep raw data local; summarize in the PDF.

## 6) Evidence C — Iteration documentation (required to look “real”)
Goal: show you learned from users and improved the system to reduce misuse/confusion.

- [ ] Identify one misuse/confusion from pilot (examples):
  - [ ] “Users interpreted GREENLIGHT as ‘buy now’”
  - [ ] “Users ignored uncertainty flags”
  - [ ] “Users didn’t understand ‘regime transition’”
- [ ] Make one concrete change (examples):
  - [ ] stronger disclaimer + “do-not-trade when…” block
  - [ ] add “uncertainty = signals disagree” line in output
  - [ ] add a simplified regime label + confidence
- [ ] Re-run the demo and include a before/after screenshot or quote in the PDF.
- [ ] Summarize the change in 3 bullets: problem → change → result.

## 7) Demo video (4 minutes, story-driven)
Goal: a reviewer should understand the value in **30 seconds**.

- [ ] **0:00–0:20:** hook (community + risk literacy problem).
- [ ] **0:20–0:40:** what it is / isn’t (no trading, no advice).
- [ ] **0:40–1:20:** baseline demo (static rules can fail).
- [ ] **1:20–2:40:** two windows with quant-team narrative:
  - [ ] show stance + regime + uncertainty + constraints
  - [ ] call out 1–2 “stand down” reasons
- [ ] **2:40–3:20:** one headline metric from `submission/evaluation_results.json`.
- [ ] **3:20–3:50:** mini pilot pre/post + 1 quote + 1 iteration.
- [ ] **3:50–4:00:** responsibility + limitations.

Tip: capture **one “baseline greenlights stress” moment** and **one “quant team stands down” moment** with a plain-language explanation.

## 8) Responsibility, limitations, and misuse prevention (Page 8)
- [ ] Clear disclaimer block: educational only, not financial advice.
- [ ] No execution, no broker integration, no personal data.
- [ ] “When not to use” section (thin data, illiquid assets, cherry-picked windows, etc.).
- [ ] Limitations: synthetic windows, indicator proxies, generalization, not predictive of profits.
- [ ] Uncertainty handling: how disagreement among agents triggers WATCH/STAND_DOWN.

## 9) Technical clarity (keep it simple for judges)
Goal: explain “AI-ness” without sounding like a trading bot.

- [ ] 1 architecture diagram (boxes/arrows): `data → indicators → agents → aggregator → stance+narrative`.
- [ ] 3–6 agent bullets (“regime,” “volatility,” “correlation,” “scenario risk,” “constraints”).
- [ ] Explain aggregation in 2–3 sentences (“multiple perspectives; traceable reasons; conservative when uncertain”).

## 10) Final packaging / QA (do this last)
- [ ] Run from a fresh terminal in repo root:
  - [ ] `python3 BOTS/ATLAS_HYBRID/quant_team_eval.py`
  - [ ] `python3 BOTS/ATLAS_HYBRID/quant_team_demo.py --window regime-shift`
- [ ] Confirm `submission/evaluation_results.json` exists and matches what you describe.
- [ ] Confirm PDF has: title, links, evidence, safety, reflection (500+ words), certification, citations.
- [ ] Confirm video has no secrets (API keys, accounts, balances).
- [ ] Confirm language never implies trading or profit prediction.

## Suggested “evidence table” layout (copy into PDF)
Columns: `Window | Baseline GREENLIGHT-in-stress | Quant-team GREENLIGHT-in-stress | What a student should do | Why (1 sentence)`

