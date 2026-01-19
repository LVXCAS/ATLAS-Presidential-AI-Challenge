# Repository Guidelines

## Project Structure & Module Organization
- `Agents/ATLAS_HYBRID/`: Python “AI quant team” demo (simulation-only). Entry points: `quant_team_eval.py`, `quant_team_demo.py`; config in `Agents/ATLAS_HYBRID/config/`.
- `frontend/`: optional React + TypeScript UI (Create React App). Source in `frontend/src/`.
- `submission/`: Track II writeups, scripts, and generated artifacts (for example `submission/evaluation_results.json`).
- `.env.example`: optional; the Track II demo runs without a `.env` file (never commit secrets).
- The Track II demo code lives under `Agents/ATLAS_HYBRID/`.

## Build, Test, and Development Commands
Python (from repo root):
- `python3 Agents/ATLAS_HYBRID/quant_team_eval.py` - runs the Track II evaluation and writes `submission/evaluation_results.json`.
- `python3 Agents/ATLAS_HYBRID/quant_team_demo.py --window regime-shift` - prints a narrated demo for a selected stress window.
- Optional setup: `python3 -m venv .venv && source .venv/bin/activate && python -m pip install -U pip numpy pandas scipy scikit-learn pyyaml matplotlib`

Frontend (from `frontend/`):
- `npm ci` - install pinned dependencies from `package-lock.json`.
- `npm start` - run the dev server.
- `npm run build` - produce a production build.
- `npm test` - run the CRA/Jest test runner.
- `npm run type-check` - TypeScript checks without emitting output.
- `npm run lint` / `npm run lint:fix` - ESLint checks (and autofix where possible).

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints where practical, and keep modules import-safe (agents may have optional dependencies and should degrade gracefully).
- Frontend: `PascalCase` for React components/files (for example `BloombergTerminal.tsx`), `camelCase` for variables/functions; prefer functional components and typed props.

## Testing Guidelines
- No dedicated automated test suite for the Python demo; treat `quant_team_eval.py` and `quant_team_demo.py` as smoke tests.
- For UI changes, use `npm test`, `npm run lint`, and `npm run type-check`.

## Commit & Pull Request Guidelines
- Commit subjects in history are mixed; prefer Conventional Commits going forward (`feat:`, `fix:`, `docs:`, `chore:`) with an imperative summary.
- PRs should include: what changed, how to run/verify (commands + expected output), and screenshots for UI changes. Keep the project “simulation-only” unless explicitly scoped otherwise.

## Security & Configuration Tips
- Don’t commit API keys or tokens; use `.env` and update `.env.example` when adding new required variables.

## ATLAS Agent Philosophy (Track II)
- Agents are small, explainable "risk lenses" that score uncertainty, not price direction.
- Each agent provides a short, student-friendly explanation and returns NEUTRAL when data is insufficient.
- The system uses only delayed or synthetic inputs in this repository (no live feeds or execution).
- ATLAS uses cached historical market data from public sources. Live APIs are intentionally disabled for safety, reproducibility, and educational use.
- If cached data is missing, the demo falls back to synthetic data with a warning.
- Cached CSV loading uses pandas when available; the Track II demo runs without it.
- CSV schema: `date,open,high,low,close,volume` with lowercase headers, sorted by date.
- Sample CSVs in this repo are placeholders for format; replace with real historical data if desired.

## Why Multi-Agent AI
- Different agents specialize in different risk signals (volatility, regime shifts, correlation, liquidity).
- Combining perspectives reduces single-point failure and encourages uncertainty reporting.
- A veto rule keeps the system safety-first: any high-risk agent can force STAND_DOWN.

## Why This Is Safer Than Prediction Models
- Avoids black-box price forecasts; focuses on risk flags and data sufficiency.
- Emphasizes "when not to act" to teach caution rather than action.
- Deterministic and reproducible outputs; no API keys required to run the demo.
