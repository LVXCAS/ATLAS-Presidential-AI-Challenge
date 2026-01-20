# Optional Research Sandbox (Not Required for Track II)

This folder is an optional, offline-only research area for backtest experiments.
It is **not** used by the Track II demo and should not be required for judges.

## Goals
- Explore Qlib-based backtests on cached data
- Prototype R&D factor discovery workflows
- Keep all outputs offline and reproducible

## Strategy lab (offline backtests)
The strategy lab runs simple, interpretable strategy templates against cached
CSV data. It is research-only and does not affect Track II outputs.

```bash
python3 research/strategy_lab.py --symbol SPY --asset-class equities
```

Optional flags (research only):
- `--use-qlib-agent`: compare signals from `QlibResearchAgent`
- `--use-autogen`: include AutoGen R&D parameter proposals
- `--use-rdagent`: run RD-Agent factor discovery (if installed)
- `--use-llm`: ask an OpenAI-compatible LLM for parameter ideas (requires `LLM_API_KEY`)

Outputs are written to `research/outputs/`.

## Install (optional)
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-research.txt
```

## Qlib setup (optional)
Qlib requires a local dataset. Follow the official Qlib instructions to prepare
data **offline**, then point your experiments at the local cache.

## R&D agent (optional)
The repo includes an experimental `RDAgentFactorDiscovery` agent in:
- `Agents/ATLAS_HYBRID/agents/rdagent_factor_discovery.py`

That integration expects Microsoft RDAgent components to be installed separately.
Follow the upstream installation instructions and keep it disabled for Track II.

## Safety note
All research should remain simulation-only and must not execute trades or provide
buy/sell recommendations.
