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

## AI agent definitions (interpretable)
Each agent uses simple, explainable signals. If data is insufficient, the agent returns NEUTRAL.

| Agent | Inputs | Reasoning method (proxy) | Output |
| --- | --- | --- | --- |
| TechnicalAgent | price, indicators | RSI/EMA/Bollinger/ATR volatility & momentum checks | Risk score + explanation |
| MarketRegimeAgent | price, ADX/EMA | Trend vs choppy regime proxy | Risk score + explanation |
| CorrelationAgent | pair, existing_positions | Currency overlap / concentration check | Risk score + explanation |
| GSQuantAgent | price history, ATR | VaR-style volatility proxy | Risk score + explanation |
| MonteCarloAgent | price history | Simulated return paths for tail risk | Risk score + explanation |
| RiskManagementAgent | account snapshot | Drawdown / loss streak safety rules | Risk score + explanation |
| NewsFilterAgent | calendar_events | Scheduled event risk flags | Risk score + explanation |
| SessionTimingAgent | timestamp | Liquidity timing proxy | Risk score + explanation |
| MultiTimeframeAgent | price history | Trend alignment across horizons | Risk score + explanation |
| VolumeLiquidityAgent | volume history | Liquidity and spread proxy | Risk score + explanation |
| SupportResistanceAgent | price history | Proximity to key levels | Risk score + explanation |
| DivergenceAgent | price history, RSI | RSI divergence vs price | Risk score + explanation |

## How to reproduce
From the repo root:
```bash
python3 Agents/ATLAS_HYBRID/quant_team_eval.py
```

Outputs:
- Human-readable summary to stdout (for screenshots in the PDF/video)
- `submission/evaluation_results.json` (machine-readable)

## Stress windows used
These are synthetic windows designed to be understandable and reproducible:
- `stable`: mostly normal conditions
- `volatility-spike`: whipsaw volatility, elevated risk
- `regime-shift`: calm -> transition into trend + higher volatility

Generator and indicator proxies live in:
- `Agents/ATLAS_HYBRID/quant_team_utils.py`

## Evaluation method (behavioral + educational)
ATLAS is evaluated on **behavioral correctness** and **pedagogical validity**, not profit.
The question is whether the risk posture aligns with known market conditions and whether
the explanations are consistent, transparent, and understandable for students.

Key checks:
- Same scenario => same posture (deterministic, reproducible)
- Different scenarios => different postures (signal sensitivity)
- Explanations reflect the signals (volatility, regime, correlation, liquidity)

## Scenario table (expected behavior)
| Scenario type | Data source | Key signals present | Expected posture | Educational outcome |
| --- | --- | --- | --- | --- |
| Calm market | Cached low-vol window | Low volatility, stable regime, low correlation | GREENLIGHT | Learn what “normal” looks like and why it is lower risk |
| Transition | Mixed window | Rising volatility, regime uncertainty | WATCH | Learn to slow down when signals conflict |
| Crisis | Crisis-like window | Volatility spike, correlation convergence, liquidity stress | STAND_DOWN | Learn systemic risk and when to pause |
| Synthetic stress | Designed shock | Forced extreme signals for stress testing | STAND_DOWN | Learn how stress testing changes posture |

## Consistency and explainability checks
- **Consistency:** Identical windows produce the same posture and flags.
- **Sensitivity:** When signals change (volatility or regime shift), posture changes accordingly.
- **Explainability:** Each posture includes plain-language reasons tied to the agent signals.

## Student comprehension angle (optional evaluation)
Before using ATLAS, students often describe market risk in vague terms. After using ATLAS,
students can reference concrete signals (volatility, regime, correlation) and explain why a
posture changed. This indicates learning, not just prediction.

## What to include in the PDF (recommended)
- A 1-row summary table per window with baseline vs quant-team GREENLIGHT-in-stress rates.
- 1-2 sentences explaining why false GREENLIGHT labels are dangerous for beginners.
