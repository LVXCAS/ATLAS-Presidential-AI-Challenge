# ATLAS Demo Story (2-3 minutes)

Goal: present ATLAS as an educational, safety-first risk coach (not trading).

## 0:00-0:20 Hook (problem)
Say:
"Students and beginners see markets moving fast without understanding risk.
ATLAS is a safe, offline way to learn how uncertainty builds."

## 0:20-0:40 What ATLAS is (and is not)
Say:
"ATLAS is an agent-based AI reasoning system. It does not trade, does not
predict prices, and does not use live data. It explains risk so students know
when not to act."

## 0:40-1:20 Scenario walkthrough
On screen (synthetic stress window):
```bash
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --data-source synthetic --window volatility-spike
```
Say:
"Each agent scores a different risk lens: volatility, regime, correlation,
liquidity. The coordinator aggregates them into GREENLIGHT, WATCH, or
STAND_DOWN with a plain-language explanation."

## 1:20-1:50 Explainability moment
Say:
"Here the volatility proxy spikes, the regime becomes unstable, and correlation
risk rises. That combination triggers STAND_DOWN. Students see the exact
signals that drove the decision."

## 1:50-2:20 Evaluation snapshot
On screen:
```bash
python3 Agents/ATLAS_HYBRID/quant_team_eval.py --data-source synthetic
```
Say:
"Our headline metric is GREENLIGHT-in-stress. Lower is better because false
GREENLIGHTs teach the wrong lesson during risky conditions."

## 2:20-2:50 Safety and learning outcome
Say:
"ATLAS is deterministic and offline. No live data, no real money, no advice.
The goal is risk literacy: understanding uncertainty and practicing caution."

## 2:50-3:00 Close
Say:
"ATLAS helps students build safer habits by learning when not to act."

## Team roles
- AI / Logic: agent design, signal definitions, aggregation, evaluation metrics
- Frontend / UX: website, demo flow, visual explanations
- Safety & Documentation: ethics statement, disclaimers, explainability artifacts

Note: small teams can combine roles as needed.
