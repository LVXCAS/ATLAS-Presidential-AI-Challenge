import React, { useMemo, useState } from 'react';
import cachedResults from './data/results_cached.json';
import syntheticResults from './data/results_synthetic.json';

type WindowResult = {
  name: string;
  description?: string;
  baseline_greenlight_in_stress_rate: number;
  quant_team_greenlight_in_stress_rate: number;
  avg_quant_team_score_in_stress?: number;
  stress_steps?: number;
  steps_total?: number;
};

type ResultsData = {
  data_source: 'cached' | 'synthetic' | string;
  primary_metric: string;
  summary: {
    baseline_greenlight_in_stress_rate: number;
    quant_team_greenlight_in_stress_rate: number;
    avg_quant_team_score_in_stress: number;
  };
  labels?: Record<string, string>;
  risk_posture_map?: Record<string, string>;
  windows: WindowResult[];
  note?: string;
};

type DemoResult = {
  label: string;
  riskScore: number;
  confidence: number;
  flags: string[];
  explanation: string;
};

type DemoSummary = {
  start: number;
  end: number;
  changePct: number;
  volatility: number;
};

const DATASETS: Record<'cached' | 'synthetic', ResultsData> = {
  cached: cachedResults as ResultsData,
  synthetic: syntheticResults as ResultsData,
};

const FEATURE_CARDS = [
  {
    title: 'Simulation-only risk coach',
    body: 'Explains when markets are risky and why. No prediction, no execution, no money.',
  },
  {
    title: 'Multi-agent explanations',
    body: 'Independent agents score volatility, regime shifts, correlation, and liquidity so students see multiple lenses.',
  },
  {
    title: 'Offline and reproducible',
    body: 'Runs fully offline using cached CSVs or synthetic stress windows so results can be reproduced.',
  },
];

const WORKFLOW_STEPS = [
  {
    title: 'Load the scenario',
    detail: 'Read cached CSVs (OHLCV) or generate deterministic synthetic stress windows.',
  },
  {
    title: 'Compute simple indicators',
    detail: 'EMA, RSI proxy, volatility and correlation proxies, and regime checks.',
  },
  {
    title: 'Agents score risk',
    detail: 'Each agent returns a score, confidence, and reasoning; insufficient data yields NEUTRAL.',
  },
  {
    title: 'Aggregate with a veto',
    detail: 'A safety-first coordinator combines scores and allows veto agents to force STAND_DOWN.',
  },
  {
    title: 'Explain the posture',
    detail: 'Output GREENLIGHT, WATCH, or STAND_DOWN with human-readable risk flags.',
  },
];

const AGENT_LENSES = [
  { name: 'TechnicalAgent', focus: 'RSI, EMA, Bollinger, and ATR volatility checks.' },
  { name: 'MarketRegimeAgent', focus: 'Trend vs choppy regime detection using ADX proxy.' },
  { name: 'CorrelationAgent', focus: 'Concentration and overlap risk across positions.' },
  { name: 'GSQuantAgent', focus: 'VaR-style risk proxy using historical volatility.' },
  { name: 'MonteCarloAgent', focus: 'Simulated return paths to gauge tail risk.' },
  { name: 'RiskManagementAgent', focus: 'Account-level safety checks and limits.' },
  { name: 'NewsFilterAgent', focus: 'Scheduled event risk from a cached calendar.' },
  { name: 'SessionTimingAgent', focus: 'Time-of-day liquidity and session effects.' },
  { name: 'MultiTimeframeAgent', focus: 'Short vs long trend agreement checks.' },
  { name: 'VolumeLiquidityAgent', focus: 'Liquidity and spread proxy from volume.' },
  { name: 'SupportResistanceAgent', focus: 'Key level proximity checks.' },
  { name: 'DivergenceAgent', focus: 'Momentum divergence signals (RSI vs price).' },
];

const DATA_CATALOG = [
  { label: 'Equities', items: ['AAPL', 'MSFT', 'TSLA', 'SPY'] },
  { label: 'FX', items: ['EURUSD', 'GBPUSD', 'USDJPY'] },
];

const TOOL_STACK = [
  {
    label: 'Core offline stack',
    items: ['Python 3', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'PyYAML', 'matplotlib'],
  },
  {
    label: 'Frontend & delivery',
    items: ['React + TypeScript', 'CRA build pipeline', 'Docker + Nginx'],
  },
  {
    label: 'Optional research tools (not required for the demo)',
    items: ['OpenBB (local exploration)', 'yfinance (local exploration)'],
  },
];

const COMMAND_SNIPPET = `# Track II evaluation (offline)
python3 Agents/ATLAS_HYBRID/quant_team_eval.py

# Narrated demo (offline)
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --window regime-shift

# Cached data (optional)
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --data-source cached --asset-class fx --symbol EURUSD`;

const OUTPUT_SNIPPET = `{
  "risk_posture": "ELEVATED",
  "risk_flags": ["REGIME_SHIFT", "HIGH_VOLATILITY"],
  "agent_votes": {
    "TechnicalAgent": "NEUTRAL",
    "MarketRegimeAgent": "RISK_OFF",
    "MonteCarloAgent": "RISK_OFF"
  },
  "explanation": "Caution: risk and uncertainty are elevated. Signals: choppy regime and volatility spike."
}`;

const DOCKER_SNIPPET = `# Build and run the site in Docker
cd frontend

docker build -t atlas-site .
docker run --rm -p 8080:80 atlas-site`;

const DEMO_SERIES = {
  stable: {
    title: 'Stable range',
    description: 'Low volatility, steady drift.',
    prices: [100, 100.2, 100.1, 100.3, 100.35, 100.4, 100.42, 100.38, 100.45, 100.5],
  },
  stress: {
    title: 'Volatility spike',
    description: 'Whipsaw moves and uncertainty.',
    prices: [100, 102, 98, 103, 97, 104, 96, 101, 95, 99],
  },
  regime: {
    title: 'Regime shift',
    description: 'Calm then abrupt trend change.',
    prices: [100, 100.3, 100.4, 100.6, 100.5, 100.3, 99.8, 99.1, 98.4, 97.9],
  },
};

function toPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function computeMiniDemo(prices: number[]): DemoResult {
  if (prices.length < 3) {
    return {
      label: 'NEUTRAL',
      riskScore: 0.5,
      confidence: 0,
      flags: ['INSUFFICIENT_DATA'],
      explanation: 'Not enough data for a stable risk estimate.',
    };
  }

  const returns: number[] = [];
  for (let i = 1; i < prices.length; i += 1) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const variance = returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / returns.length;
  const volatility = Math.sqrt(variance);

  const mid = Math.floor(prices.length / 2);
  const trendEarly = (prices[mid] - prices[0]) / prices[0];
  const trendLate = (prices[prices.length - 1] - prices[mid]) / prices[mid];
  const regimeShift = trendEarly * trendLate < 0 || Math.abs(trendLate - trendEarly) > 0.01;

  const volatilityScore = clamp(volatility * 35, 0, 0.7);
  const regimeScore = regimeShift ? 0.2 : 0.0;
  const driftPenalty = Math.abs(trendLate) < 0.002 ? 0.1 : 0.0;

  const riskScore = clamp(volatilityScore + regimeScore + driftPenalty, 0, 1);
  const confidence = clamp(1 - riskScore, 0, 1);

  let label = 'GREENLIGHT';
  if (riskScore >= 0.36) {
    label = 'STAND_DOWN';
  } else if (riskScore >= 0.25) {
    label = 'WATCH';
  }

  const flags = [] as string[];
  if (volatility > 0.012) flags.push('HIGH_VOLATILITY');
  if (regimeShift) flags.push('REGIME_SHIFT');
  if (!flags.length) flags.push('LOW_SIGNAL');

  const explanation = `Volatility ${volatility > 0.012 ? 'spiked' : 'stayed modest'}, ` +
    `${regimeShift ? 'regime behavior shifted' : 'trends are consistent'}.`;

  return { label, riskScore, confidence, flags, explanation };
}

function summarizeSeries(prices: number[]): DemoSummary {
  if (prices.length < 2) {
    return { start: 0, end: 0, changePct: 0, volatility: 0 };
  }

  const start = prices[0];
  const end = prices[prices.length - 1];
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i += 1) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const variance = returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / returns.length;
  const volatility = Math.sqrt(variance);

  return { start, end, changePct: (end - start) / start, volatility };
}

function App() {
  const [mode, setMode] = useState<'cached' | 'synthetic'>('cached');
  const [scenario, setScenario] = useState<'stable' | 'stress' | 'regime'>('stable');
  const [demoResult, setDemoResult] = useState<DemoResult | null>(null);

  const dataset = DATASETS[mode];
  const demoSummary = useMemo(() => summarizeSeries(DEMO_SERIES[scenario].prices), [scenario]);

  const summaryDelta = useMemo(() => {
    return dataset.summary.baseline_greenlight_in_stress_rate - dataset.summary.quant_team_greenlight_in_stress_rate;
  }, [dataset]);

  const showcaseMetrics = useMemo(() => ([
    { label: 'Risk literacy quiz lift', value: '+18%', note: 'Prototype metric (replace with study data)' },
    { label: 'Confidence calibration', value: '+0.21', note: 'Prototype metric' },
    { label: 'Time-to-caution', value: '-35%', note: 'Prototype metric' },
    { label: 'Agent coverage', value: `${AGENT_LENSES.length} agents`, note: 'Documented coverage' },
  ]), []);

  const handleRunDemo = () => {
    const series = DEMO_SERIES[scenario];
    setDemoResult(computeMiniDemo(series.prices));
  };

  return (
    <div className="app">
      <header className="nav">
        <div className="nav__brand">
          <span className="logo">ATLAS Field Notes</span>
          <span className="tag">Risk Literacy Notebook</span>
        </div>
        <nav className="nav__links">
          <a href="#overview">Overview</a>
          <a href="#workflow">Method</a>
          <a href="#results">Findings</a>
          <a href="#demo">Demo</a>
          <a href="#docs">Run</a>
          <a href="#links">Links</a>
        </nav>
      </header>

      <main>
        <section className="hero" id="overview">
          <div className="hero__content">
            <p className="eyebrow">Field notebook - Track II</p>
            <h1>
              ATLAS field notes on market risk, written for students.
            </h1>
            <p className="lead">
              We built ATLAS as a notebook for how markets feel under stress. It does not predict prices or
              trade. It just says when risk looks calm, uncertain, or too hot and explains why.
            </p>
            <div className="hero__actions">
              <a className="btn btn--primary" href="#demo">Run the mini demo</a>
              <a className="btn btn--ghost" href="#docs">Open quick start</a>
            </div>
            <div className="hero__stats">
              <div className="stat">
                <span>Simulation only</span>
                <strong>Offline, no trading</strong>
              </div>
              <div className="stat">
                <span>Offline first</span>
                <strong>Cached CSVs + synthetic fallback</strong>
              </div>
              <div className="stat">
                <span>Risk posture</span>
                <strong>GREENLIGHT / WATCH / STAND_DOWN</strong>
              </div>
            </div>
            <div className="margin-note">
              Margin note: this is a classroom-style simulator. No live data. No advice.
            </div>
          </div>
          <div className="hero__panel">
            <div className="panel__card">
              <p className="panel__title">Field notes</p>
              <p>
                Students often see markets moving fast without understanding the risk beneath the headlines. ATLAS
                is a safe, explainable alternative to prediction-heavy tools.
              </p>
              <div className="panel__grid">
                {FEATURE_CARDS.map((card, index) => (
                  <div className="panel__tile reveal" style={{ animationDelay: `${index * 0.1}s` }} key={card.title}>
                    <h3>{card.title}</h3>
                    <p>{card.body}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="section" id="workflow">
          <div className="section__header">
            <h2>How it works</h2>
            <p>Offline, deterministic workflow written like a field notebook.</p>
          </div>
          <div className="workflow">
            {WORKFLOW_STEPS.map((step, index) => (
              <div className="workflow__step reveal" style={{ animationDelay: `${0.1 + index * 0.08}s` }} key={step.title}>
                <span className="step__index">0{index + 1}</span>
                <div>
                  <h3>{step.title}</h3>
                  <p>{step.detail}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="agent-grid">
            {AGENT_LENSES.map((agent, index) => (
              <div className="agent-card reveal" style={{ animationDelay: `${0.15 + index * 0.05}s` }} key={agent.name}>
                <h3>{agent.name}</h3>
                <p>{agent.focus}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="section" id="results">
          <div className="section__header">
            <h2>Findings (draft)</h2>
            <p>Switch between cached data and synthetic stress windows to compare risk posture behavior.</p>
          </div>

          <div className="results__toolbar">
            <div className="toggle">
              <button
                className={mode === 'cached' ? 'toggle__btn is-active' : 'toggle__btn'}
                onClick={() => setMode('cached')}
                aria-pressed={mode === 'cached'}
              >
                Cached data
              </button>
              <button
                className={mode === 'synthetic' ? 'toggle__btn is-active' : 'toggle__btn'}
                onClick={() => setMode('synthetic')}
                aria-pressed={mode === 'synthetic'}
              >
                Synthetic windows
              </button>
            </div>
            <div className="results__note">
              <span className="pill">Data source: {dataset.data_source}</span>
              {dataset.note && <span>{dataset.note}</span>}
            </div>
          </div>

          <div className="results__summary">
            <div className="metric-card">
              <h3>Baseline GREENLIGHT-in-stress</h3>
              <strong>{toPercent(dataset.summary.baseline_greenlight_in_stress_rate)}</strong>
              <p>Lower is safer for beginners.</p>
            </div>
            <div className="metric-card">
              <h3>Quant-team GREENLIGHT-in-stress</h3>
              <strong>{toPercent(dataset.summary.quant_team_greenlight_in_stress_rate)}</strong>
              <p>Multi-agent scoring reduces false confidence.</p>
            </div>
            <div className="metric-card">
              <h3>Improvement</h3>
              <strong>{toPercent(Math.abs(summaryDelta))}</strong>
              <p>{summaryDelta >= 0 ? 'Reduction vs baseline.' : 'Increase vs baseline.'}</p>
            </div>
            <div className="metric-card">
              <h3>Avg risk score (stress)</h3>
              <strong>{dataset.summary.avg_quant_team_score_in_stress.toFixed(2)}</strong>
              <p>0 = calm, 1 = high uncertainty.</p>
            </div>
          </div>

          <div className="showcase-note">
            Margin note: prototype metrics for layout only (replace with pilot data when available).
          </div>
          <div className="showcase-grid">
            {showcaseMetrics.map((metric) => (
              <div className="showcase-card" key={metric.label}>
                <h3>{metric.label}</h3>
                <strong>{metric.value}</strong>
                <p>{metric.note}</p>
              </div>
            ))}
          </div>

          <div className="margin-note">
            Field check: posture should shift when volatility or regime signals change.
          </div>

          <div className="results__grid">
            {dataset.windows.map((window) => {
              return (
                <div className="result-card" key={window.name}>
                  <div className="result-card__header">
                    <h3>{window.name}</h3>
                    <span>{window.description}</span>
                  </div>
                  <div className="result-card__meta">
                    <div>
                      <strong>{toPercent(window.baseline_greenlight_in_stress_rate)}</strong>
                      <small>Baseline GREENLIGHT-in-stress</small>
                    </div>
                    <div>
                      <strong>{toPercent(window.quant_team_greenlight_in_stress_rate)}</strong>
                      <small>Quant-team GREENLIGHT-in-stress</small>
                    </div>
                    <div>
                      <strong>{window.avg_quant_team_score_in_stress?.toFixed(2) ?? 'N/A'}</strong>
                      <small>Avg risk score (stress)</small>
                    </div>
                    <div>
                      <strong>{window.stress_steps ?? 0}</strong>
                      <small>Stress steps</small>
                    </div>
                    <div>
                      <strong>{window.steps_total ?? 0}</strong>
                      <small>Total steps</small>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="section" id="demo">
          <div className="section__header">
            <h2>Mini simulation demo</h2>
            <p>Run a tiny, offline stress check to see how risk labels respond to volatility.</p>
          </div>
          <div className="demo">
            <div className="demo__controls">
              <div>
                <h3>Scenario</h3>
                <div className="toggle">
                  {(['stable', 'stress', 'regime'] as const).map((key) => (
                    <button
                      key={key}
                      className={scenario === key ? 'toggle__btn is-active' : 'toggle__btn'}
                      onClick={() => setScenario(key)}
                      aria-pressed={scenario === key}
                    >
                      {DEMO_SERIES[key].title}
                    </button>
                  ))}
                </div>
                <p className="demo__hint">{DEMO_SERIES[scenario].description}</p>
              </div>
              <button className="btn btn--primary" onClick={handleRunDemo}>Run demo</button>
            </div>
            <div className="demo__panel">
              <div className="demo__output">
                <h3>Result</h3>
                {demoResult ? (
                  <>
                    <div className="demo__badge">{demoResult.label}</div>
                    <p className="demo__score">Risk score: {demoResult.riskScore.toFixed(2)} | Confidence {demoResult.confidence.toFixed(2)}</p>
                    <p>{demoResult.explanation}</p>
                    <div className="demo__flags">
                      {demoResult.flags.map((flag) => (
                        <span key={flag}>{flag}</span>
                      ))}
                    </div>
                  </>
                ) : (
                  <p>Choose a scenario and run the demo to see a risk posture.</p>
                )}
              </div>
              <div className="demo__inputs">
                <h3>Scenario inputs</h3>
                <ul>
                  <li>Start price: {demoSummary.start.toFixed(2)}</li>
                  <li>End price: {demoSummary.end.toFixed(2)}</li>
                  <li>Change: {toPercent(demoSummary.changePct)}</li>
                  <li>Volatility proxy: {demoSummary.volatility.toFixed(3)}</li>
                </ul>
                <small>Deterministic offline values for a safe demo.</small>
              </div>
            </div>
          </div>
        </section>

        <section className="section" id="docs">
          <div className="section__header">
            <h2>Documentation & quick start</h2>
            <p>Everything you need to run the Track II demo locally or in a container.</p>
          </div>
          <div className="docs">
            <div className="docs__block">
              <h3>Run the demo</h3>
              <pre><code>{COMMAND_SNIPPET}</code></pre>
            </div>
            <div className="docs__block">
              <h3>Docker (production)</h3>
              <pre><code>{DOCKER_SNIPPET}</code></pre>
            </div>
            <div className="docs__block">
              <h3>Example output</h3>
              <pre><code>{OUTPUT_SNIPPET}</code></pre>
            </div>
            <div className="docs__block">
              <h3>Cached data catalog</h3>
              {DATA_CATALOG.map((group) => (
                <div key={group.label} className="catalog">
                  <strong>{group.label}</strong>
                  <div className="chips">
                    {group.items.map((item) => (
                      <span key={item}>{item}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="section" id="links">
          <div className="section__header">
            <h2>Links & tooling</h2>
            <p>Core dependencies, references, and places to learn more.</p>
          </div>
          <div className="links">
            <div className="links__card">
              <h3>Project links</h3>
              <ul>
                <li><a href="https://github.com/" target="_blank" rel="noreferrer">GitHub repo (replace with your URL)</a></li>
                <li><a href="https://openbb.co" target="_blank" rel="noreferrer">OpenBB</a></li>
                <li><a href="https://numpy.org" target="_blank" rel="noreferrer">numpy</a></li>
                <li><a href="https://pandas.pydata.org" target="_blank" rel="noreferrer">pandas</a></li>
              </ul>
            </div>
            <div className="links__card">
              <h3>What we used</h3>
              {TOOL_STACK.map((stack) => (
                <div key={stack.label} className="stack">
                  <strong>{stack.label}</strong>
                  <p>{stack.items.join(' / ')}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>ATLAS is a simulation-only learning tool. No live data, no execution, no financial advice.</p>
        <p>Built for the Presidential AI Challenge (K-12 Track II).</p>
      </footer>
    </div>
  );
}

export default App;
