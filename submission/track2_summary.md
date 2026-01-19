# ATLAS Track II Summary (Educational, Simulation-Only)

## Problem Statement
Students and beginners are exposed to markets and AI tools without understanding
risk, uncertainty, or how AI decisions are formed. Existing tools often push
trading, hide reasoning, or involve real money, which can create harm rather
than learning. ATLAS is designed as a preventative, educational alternative.

## The Problem We Are Solving
Many young people want to start investing but do not know where to begin.
Professional advice is expensive, and most online information is not written
for beginners. That creates a financial literacy gap: students see markets
moving fast but lack a safe, explainable way to learn about risk.

ATLAS helps bridge that gap by explaining market conditions and how major
events can change risk in stocks and forex. It is an educational risk coach,
not a trading system. It does not place trades, does not use live data, and
does not claim predictive accuracy.

## Why AI Is Needed
Single-model forecasts are often opaque. Students need a system that makes
uncertainty visible and encourages caution when signals conflict. A multi-agent
approach mirrors how real teams work:

- Different agents watch different signals (volatility, regime, correlation)
- Explanations are short and student-friendly
- A safety-first veto rule prevents overconfidence

ATLAS uses lightweight indicators so outputs are understandable and
reproducible. The agents explain *why* risk is elevated, not *what* to buy.

## What ATLAS Does (and Does Not Do)
ATLAS outputs a desk-style risk posture label:

- GREENLIGHT (LOW): low risk/uncertainty
- WATCH (ELEVATED): moderate risk/uncertainty
- STAND_DOWN (HIGH): high risk/uncertainty

Each agent returns a short explanation plus a (vote, confidence, reasoning)
triple. If data is insufficient, the agent returns NEUTRAL and does not affect
the aggregate.

ATLAS is intentionally limited:

- No live trading
- No execution, brokers, or orders
- No guarantees or accuracy claims
- No API keys required to run the demo

## What ATLAS Is NOT
- Not a trading bot
- Not a prediction engine
- Not financial advice
- Not connected to live data or real money

## How the System Works
ATLAS runs fully offline using cached historical CSVs (public sources) or
synthetic data if cached files are missing. The pipeline is:

1. Load cached OHLCV data (or generate synthetic stress windows)
2. Compute simple indicators (EMA, RSI, ATR, Bollinger, MACD, ADX proxy)
3. Ask each agent to score risk and explain its reasoning
4. Aggregate scores with weights and a veto rule
5. Output a plain-English explanation and key risk flags

If a cached event calendar is provided, the news filter can flag scheduled
events that may raise volatility. All inputs are historical or delayed.
CSV files use lowercase headers: `date,open,high,low,close,volume`.

## Development Challenges and Learning
Two major challenges were data handling and explainability. We addressed these
by keeping the system offline, using deterministic indicators, and writing
student-friendly explanations for each agent. Building ATLAS helped our team
grow in teamwork, data science, and understanding the limits of AI.

## Future Improvements
With more resources, we would expand to more markets and timeframes, add
additional educational pages for global indices and forex, and refine the
learning materials. The system would remain simulation-only and focused on
risk literacy rather than predictions.

## How to Run (Reproducible)
From the repository root:

```bash
python3 Agents/ATLAS_HYBRID/quant_team_eval.py
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --window regime-shift
```

For cached data (optional):

```bash
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --data-source cached --asset-class fx --symbol EUR_USD
```

## Closing Note
ATLAS is not a trading bot. It is a learning tool that shows students how to
reason about uncertainty and when not to act.
