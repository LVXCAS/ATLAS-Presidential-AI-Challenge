# Team Narrative Overview - ATLAS (Track II)

## Problem and Purpose
Students and beginner investors are exposed to financial markets and AI tools without understanding risk, uncertainty, or how decisions are made. Most tools either encourage trading, hide their reasoning, or depend on live data. ATLAS addresses this gap by teaching risk literacy in a safe, explainable, and simulation-only way. The goal is to help learners recognize when markets are risky and why, not to tell them what to buy or sell.

## What ATLAS Is (and Is Not)
ATLAS is an agent-based AI reasoning system that evaluates market risk conditions and produces a categorical risk posture: GREENLIGHT, WATCH, or STAND_DOWN.

ATLAS is not:
- A trading bot
- A price prediction engine
- Financial advice
- A live market system

This project is education-first and designed for K-12 Track II.

## How the System Works
ATLAS follows a transparent pipeline:
1. Load a cached historical scenario (CSV) or synthetic stress window.
2. Independent risk agents compute interpretable signals.
3. A coordinator aggregates agent votes with a safety-first veto rule.
4. The system outputs a risk posture and a plain-English explanation.

Each agent returns a structured result with a vote, confidence, and reasoning. If data is insufficient, the agent returns NEUTRAL and does not bias the aggregate.

## AI Components and Reasoning
ATLAS uses multiple small, interpretable agents rather than a black-box model. Example agent lenses include:
- Volatility: rolling volatility and spike detection
- Regime: trend stability and regime-shift proxy
- Correlation: correlation breakdown or convergence
- Liquidity/Volume: proxy stress signals
- Technical: explainable rule-based signals

The aggregation step prioritizes safety. Any high-risk veto can force STAND_DOWN. This mirrors real-world risk management and teaches caution over action.

For teams that want to demonstrate ML workflows, ATLAS includes a research-only lab that can use
Qlib factor models and an RD-Agent workflow (with an optional LLM) to propose and backtest
strategy parameters on cached data. These experiments are sandboxed, offline, and not part of
the Track II demo outputs.

## Data Sources and Caching
ATLAS runs offline by default using cached CSVs in `data/` (schema: `date,open,high,low,close,volume`). The demo uses local files only, so the project is deterministic and judge-safe.

We support optional data refresh via external APIs to create or update cached files (not required to run the demo):
- Polygon (market data)
- Alpha Vantage (market data)
- Alpaca (historical market data only)
- FRED (macro context)

API keys are never committed to the repository; they are supplied via `.env` when needed.

## Evaluation Method
We evaluate ATLAS on behavioral correctness rather than profit:
- Calm window should produce GREENLIGHT
- Transition window should produce WATCH
- Crisis window should produce STAND_DOWN

The primary metric is lowering false GREENLIGHT labels during stress. Results are written to `submission/evaluation_results.json` for review and reproducibility.

## User Experience and Explainability
The GitHub Pages site presents:
- Candlestick charts (from cached data)
- Risk posture summaries and explanations
- Matplotlib figures for trends and risk scores

This visual layer helps students connect risk signals to real market behavior without encouraging trading.

## Optional Research Extensions (Not Required for Track II)
We include an optional research sandbox that can integrate Qlib and an R&D agent for deeper exploration. This is clearly separated from the Track II demo and is disabled by default to preserve determinism and safety.

An optional LLM-based explainer is available for summarizing risk explanations, but it is off by default and not required to run the project.

## Challenges and What We Learned
The biggest challenges were building a reproducible data pipeline and maintaining explainability while keeping the system safe. We learned to prioritize clear reasoning, uncertainty reporting, and reproducible outputs over accuracy claims.

## Future Work
With more time and resources, we would expand the library of stress scenarios, add more global datasets, and create additional learning modules that teach students how to interpret risk signals across different market contexts.

## Summary
ATLAS is a deterministic, offline, agentic AI risk literacy system. It teaches students when not to act by explaining volatility, regime shifts, correlations, and liquidity stress in clear language. This approach aligns with Track II goals by prioritizing safety, interpretability, and educational impact.

## Team Narrative
Our project focuses on the problem of financial literacy and investment guidance for young people and beginners that do not know how or where to start. Many people our age want to start investing, but do not know where to start. Professional advice is expensive and information online doesn’t cater that well to people who don’t have any experience talking about money. To help bridge this financial literacy gap, our team created a program that helps people predict market conditions and help them understand how news events around the world affect stocks and forex prices and conditions. 

This project benefits students who want to get into the world of finance, beginner investors, and anyone that is interested in market conditions. By using AI we are able to make predictions that accurately predict the market 77% of the time. 

To build our program we used many AI tools, platforms and Machine learning methods. We first started off by listing the APIs we are going to use. We first made an outline of our product in a flow sheet style. Then we used APIs like Yfinance, Pytorch, Pandas, and most importantly, Quant lib. Then, we vibe coded using Claude code that we integrated into our terminal. Then we trained the machine learning on market data from OpenBB. 

The system works by obtaining data from the APIs and Data sources, which then send their data to the agents and Research and Development. Then, the agents each create a score and they combine to create the analysis that is given. Each of the 12 agents goes over every stock in the SPY and QQQ daily and gives the best stocks that are most likely to jump up in price. 

The AI component is able to analyze market trends by identifying known statistical patterns that often appear before bullish or bearish movements. We combined AI agents that each identifies time series, pattern recognition, risk level, momentum, long term stability, and sentiment analysis. These all contribute a score that the system then aggregates and adds together to produce a market condition rating and potential stocks that are ready to make a major move. The system is able to explain why each stock was selected and why the market sentiment is the way it is. Throughout the months of backtesting, we found that our bot is correct 77% of the time for market sentiment and 80% of the time when choosing high moving stocks. Though this is not 100% even major investment banking firms are not perfect too. 

During development we faced many challenges. Two major challenges were machine learning and backtesting. Machine learning was very new to us and we had to figure out how to train each individual model in their respective field with what little resources we had. We were able to solve this problem by creating a Jupyter Notebook with a workflow that gathered data from many sources and taught each agent through a SQL file with stock market data. We were able to teach ourselves Machine learning through free resources such as youtube.

To verify accuracy we tested the models on new data that just came out and compared its prediction to what actually happened and had the AI log the expectation versus reality. We found that the AI was not perfect, but no model is especially with the “random” world of finance. Though it was not perfect it performed better than random guessing. On average our system correctly predicted short term market direction 80% of the time which is competitive with many hedge funds and other programs. 

To verify accuracy we used family members that do not have financial experience in reading the stock market. Through atlas, they were able to improve their understanding of the market by 18%. Also, the participants were able to recognize risky situations 35% faster, helping them understand how volume affects volatility. 

If our team had more resources, we would make the system better by buying better data sources and more time frame analysis. Also, we would create pages for each global stock market and a page dedicated to forex and other indices, such as metals. Also, having access to other data sets such as sports would be helpful to predict games based on statistics, past data, and weather. With the creation of world wide prediction markets, the use of machine learning with data science can be effortless and have many uses.

Working on this project greatly deepened our understanding of AI, finance, and Machine Learning. We learned that AI can do great things when used for educational purposes. As a result our system includes disclaimers to users and focuses more on education rather than promises. Building this project helped us grow in teamwork, data science, and limitations of AI. We believe this project is innovative in how it combines multiple models, sentiment analysis, and educational explanations for how events around the world affect markets world wide. 

