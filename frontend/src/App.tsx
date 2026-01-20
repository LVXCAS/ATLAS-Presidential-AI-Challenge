import React, { useMemo, useState } from 'react';
import cachedResults from './data/results_cached.json';
import syntheticResults from './data/results_synthetic.json';
import candlesSpy from './data/candles_spy.json';

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

type Candle = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
};

type CandleSeries = {
  symbol: string;
  source: string;
  candles: Candle[];
};

const DATASETS: Record<'cached' | 'synthetic', ResultsData> = {
  cached: cachedResults as ResultsData,
  synthetic: syntheticResults as ResultsData,
};

const CANDLE_SERIES = candlesSpy as CandleSeries;

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

const DATA_SOURCES = [
  { label: 'Market data (cached)', items: ['Polygon.io', 'Alpha Vantage', 'FRED', 'Alpaca Data API'] },
  { label: 'Research + AI (optional)', items: ['Qlib (MSR)', 'RDAgent (Microsoft)', 'OpenAI (LLM)', 'DeepSeek (LLM)'] },
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

const FIGURES = [
  {
    title: 'SPY close (cached)',
    src: '/figures/spy_close.png',
    caption: 'Matplotlib candlesticks generated from cached OHLCV data.',
  },
  {
    title: 'Risk score timeline',
    src: '/figures/risk_scores.png',
    caption: 'Aggregated risk score with GREENLIGHT/WATCH/STAND_DOWN thresholds.',
  },
  {
    title: 'Agent risk lenses',
    src: '/figures/agent_scores.png',
    caption: 'Top agent score traces showing which lenses drive uncertainty over time.',
  },
];

function toPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatPrice(value: number): string {
  return value.toFixed(2);
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

function CandlestickChart({ candles }: { candles: Candle[] }) {
  if (!candles.length) {
    return <div className="candles__empty">No candles loaded.</div>;
  }

  const width = 760;
  const height = 280;
  const padding = 28;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  const highs = candles.map((c) => c.high);
  const lows = candles.map((c) => c.low);
  const maxHigh = Math.max(...highs);
  const minLow = Math.min(...lows);
  const range = maxHigh - minLow || 1;
  const slot = chartWidth / candles.length;
  const bodyWidth = Math.min(12, slot * 0.6);

  const yFor = (value: number) => padding + ((maxHigh - value) / range) * chartHeight;
  const xFor = (index: number) => padding + slot * index + slot / 2;

  const tickCount = 4;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => minLow + (range * i) / tickCount);

  return (
    <div className="candles__chart">
      <svg viewBox={`0 0 ${width} ${height}`} className="candles__svg" role="img" aria-label="Candlestick chart">
        <rect className="candles__bg" x="0" y="0" width={width} height={height} rx="14" />
        {ticks.map((value) => {
          const y = yFor(value);
          return (
            <g className="candles__grid" key={`tick-${value}`}>
              <line x1={padding} x2={width - padding} y1={y} y2={y} />
              <text className="candles__axis" x={padding - 6} y={y + 4} textAnchor="end">
                {formatPrice(value)}
              </text>
            </g>
          );
        })}
        {candles.map((candle, index) => {
          const x = xFor(index);
          const yOpen = yFor(candle.open);
          const yClose = yFor(candle.close);
          const yHigh = yFor(candle.high);
          const yLow = yFor(candle.low);
          const isUp = candle.close >= candle.open;
          const bodyHeight = Math.max(2, Math.abs(yClose - yOpen));
          const bodyY = Math.min(yOpen, yClose);

          return (
            <g className={isUp ? 'candle candle--up' : 'candle candle--down'} key={`${candle.date}-${index}`}>
              <line className="candle__wick" x1={x} x2={x} y1={yHigh} y2={yLow} />
              <rect
                className="candle__body"
                x={x - bodyWidth / 2}
                y={bodyY}
                width={bodyWidth}
                height={bodyHeight}
                rx={2}
              />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function App() {
  const [mode, setMode] = useState<'cached' | 'synthetic'>('cached');
  const [scenario, setScenario] = useState<'stable' | 'stress' | 'regime'>('stable');
  const [demoResult, setDemoResult] = useState<DemoResult | null>(null);

  const dataset = DATASETS[mode];
  const demoSummary = useMemo(() => summarizeSeries(DEMO_SERIES[scenario].prices), [scenario]);
  const candleStats = useMemo(() => {
    if (!CANDLE_SERIES.candles.length) {
      return null;
    }
    const first = CANDLE_SERIES.candles[0];
    const last = CANDLE_SERIES.candles[CANDLE_SERIES.candles.length - 1];
    const highs = CANDLE_SERIES.candles.map((c) => c.high);
    const lows = CANDLE_SERIES.candles.map((c) => c.low);
    return {
      start: first.date,
      end: last.date,
      high: Math.max(...highs),
      low: Math.min(...lows),
    };
  }, []);

  const summaryDelta = useMemo(() => {
    return dataset.summary.baseline_greenlight_in_stress_rate - dataset.summary.quant_team_greenlight_in_stress_rate;
  }, [dataset]);

  const showcaseMetrics = useMemo(() => {
    const primaryWindow = dataset.windows[0];
    const stressCoverage = primaryWindow && primaryWindow.steps_total && primaryWindow.stress_steps
      ? primaryWindow.stress_steps / primaryWindow.steps_total
      : 0;
    return [
      {
        label: 'Cached window size',
        value: `${primaryWindow?.steps_total ?? 0} steps`,
        note: 'Daily bars used in the evaluation window.',
      },
      {
        label: 'Stress coverage',
        value: toPercent(stressCoverage),
        note: 'Share of steps marked as stress.',
      },
      {
        label: 'Avg risk score (stress)',
        value: dataset.summary.avg_quant_team_score_in_stress.toFixed(2),
        note: '0 = calm, 1 = high uncertainty.',
      },
      {
        label: 'Candles displayed',
        value: `${CANDLE_SERIES.candles.length} days`,
        note: `Latest cached ${CANDLE_SERIES.symbol} candles.`,
      },
    ];
  }, [dataset]);

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
          <a href="#architecture">Architecture</a>
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

          <div className="margin-note">
            Research note: an optional ML lab can explore Qlib factors, RD-Agent discovery, and LLM parameter ideas
            offline. These experiments are sandboxed and never used in the Track II demo.
          </div>
        </section>

        <section className="section" id="architecture">
          <div className="section__header">
            <h2>System Architecture</h2>
            <p>Visual overview of our multi-agent AI system and how data flows through the risk analysis pipeline.</p>
          </div>
          <div className="diagrams-grid">
            <div className="diagram-card">
              <h3>Multi-Agent Architecture</h3>
              <p>13 independent AI agents analyze market conditions from different perspectives.</p>
              <img src="/ATLAS-Presidential-AI-Challenge/diagrams/architecture.svg" alt="ATLAS Multi-Agent Architecture Diagram" loading="lazy" />
            </div>
            <div className="diagram-card">
              <h3>Data Flow Pipeline</h3>
              <p>How data moves from input through agents to explainable risk posture output.</p>
              <img src="/ATLAS-Presidential-AI-Challenge/diagrams/dataflow.svg" alt="ATLAS Data Flow Diagram" loading="lazy" />
            </div>
            <div className="diagram-card">
              <h3>Agent Roster</h3>
              <p>Complete breakdown of all 13 AI risk agents and their specialized functions.</p>
              <img src="/ATLAS-Presidential-AI-Challenge/diagrams/agents.svg" alt="ATLAS Agent Roster Diagram" loading="lazy" />
            </div>
            <div className="diagram-card">
              <h3>Evaluation Metrics</h3>
              <p>How we measure the educational impact and risk literacy improvements.</p>
              <img src="/ATLAS-Presidential-AI-Challenge/diagrams/evaluation.svg" alt="ATLAS Evaluation Metrics" loading="lazy" />
            </div>
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

          <div className="candles">
            <div className="candles__header">
              <div>
                <h3>Candlestick snapshot</h3>
                <p>Latest cached daily candles (offline) for {CANDLE_SERIES.symbol}.</p>
              </div>
              <div className="candles__meta">
                <span className="pill">Symbol: {CANDLE_SERIES.symbol}</span>
                <span className="pill">Source: {CANDLE_SERIES.source}</span>
              </div>
            </div>
            <CandlestickChart candles={CANDLE_SERIES.candles} />
            {candleStats && (
              <div className="candles__footer">
                <span>Range: {candleStats.start} → {candleStats.end}</span>
                <span>High: {formatPrice(candleStats.high)} | Low: {formatPrice(candleStats.low)}</span>
              </div>
            )}
          </div>

          <div className="figure-grid">
            {FIGURES.map((figure) => (
              <div className="figure-card" key={figure.title}>
                <h4>{figure.title}</h4>
                <img src={figure.src} alt={figure.title} loading="lazy" />
                <p>{figure.caption}</p>
              </div>
            ))}
          </div>

          <div className="showcase-note">
            Margin note: metrics computed from the latest cached evaluation run.
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
                <li><a href="https://github.com/lvxcas/ATLAS-Presidential-AI-Challenge" target="_blank" rel="noreferrer">GitHub repo</a></li>
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
            <div className="links__card">
              <h3>Data sources</h3>
              {DATA_SOURCES.map((group) => (
                <div key={group.label} className="stack">
                  <strong>{group.label}</strong>
                  <p>{group.items.join(' / ')}</p>
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
