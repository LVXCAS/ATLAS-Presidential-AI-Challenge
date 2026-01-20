#!/usr/bin/env python3
"""
Research-only Strategy Lab (offline)

Backtests simple, interpretable strategy templates on cached OHLCV data.
Optional hooks:
- QlibResearchAgent (simplified if Qlib not installed)
- AutoGenRDAgent (parameter proposals)
- RDAgentFactorDiscovery (factor discovery report)
- LLM parameter proposals (OpenAI-compatible; optional)

This script is NOT part of Track II evaluation and never executes trades.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Tuple


try:
    import numpy as np
except ImportError:  # pragma: no cover - research-only convenience
    np = None


@dataclass
class Strategy:
    name: str
    params: Dict[str, Any]
    signal_fn: Callable[[int, Dict[str, List[float]]], int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline strategy lab (research-only).")
    parser.add_argument("--symbol", default="SPY", help="Symbol to backtest (default: SPY)")
    parser.add_argument("--asset-class", default="equities", choices=["equities", "fx"])
    parser.add_argument("--data-path", default=None, help="Optional CSV path override")
    parser.add_argument("--output-dir", default="research/outputs")
    parser.add_argument("--use-qlib-agent", action="store_true", help="Use QlibResearchAgent signals")
    parser.add_argument("--use-autogen", action="store_true", help="Include AutoGenRDAgent proposals")
    parser.add_argument("--use-rdagent", action="store_true", help="Run RDAgent factor discovery (optional)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM to suggest strategy params")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def load_ohlcv_csv(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"date", "open", "high", "low", "close", "volume"}
        headers = {h.lower() for h in (reader.fieldnames or [])}
        if not required.issubset(headers):
            raise ValueError(f"CSV missing required columns: {sorted(required - headers)}")

        dates: List[str] = []
        opens: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        volumes: List[float] = []

        for row in reader:
            dates.append(row["date"])
            opens.append(float(row["open"]))
            highs.append(float(row["high"]))
            lows.append(float(row["low"]))
            closes.append(float(row["close"]))
            volumes.append(float(row.get("volume", 0.0) or 0.0))

    return {
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


def ema_series(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    alpha = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append((alpha * v) + ((1 - alpha) * out[-1]))
    return out


def sma_series(values: List[float], period: int) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        window = values[max(0, i - period + 1) : i + 1]
        out.append(mean(window))
    return out


def rsi_series(values: List[float], period: int = 14) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        if i < period:
            out.append(50.0)
            continue
        gains = 0.0
        losses = 0.0
        for j in range(i - period + 1, i + 1):
            delta = values[j] - values[j - 1]
            if delta >= 0:
                gains += delta
            else:
                losses -= delta
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            out.append(100.0)
        else:
            rs = avg_gain / avg_loss
            out.append(100.0 - (100.0 / (1.0 + rs)))
    return out


def atr_series(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    tr_values: List[float] = []
    for i in range(len(closes)):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        tr_values.append(tr)

    out: List[float] = []
    for i in range(len(tr_values)):
        window = tr_values[max(0, i - period + 1) : i + 1]
        out.append(mean(window))
    return out


def macd_series(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
    fast_ema = ema_series(values, fast)
    slow_ema = ema_series(values, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = ema_series(macd_line, signal)
    hist = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist


def adx_proxy_series(values: List[float], window: int = 30) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        segment = values[max(0, i - window + 1) : i + 1]
        if len(segment) < 2:
            out.append(20.0)
            continue
        x = list(range(len(segment)))
        x_mean = mean(x)
        y_mean = mean(segment)
        denom = sum((xi - x_mean) ** 2 for xi in x) or 1.0
        slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, segment)) / denom
        avg_price = y_mean or 1.0
        strength = min(1.0, abs(slope) / (avg_price * 0.002))
        out.append(10.0 + (50.0 * strength))
    return out


def build_context(data: Dict[str, List[float]]) -> Dict[str, List[float]]:
    closes = data["close"]
    ctx = {
        "close": closes,
        "ema_fast": ema_series(closes, 12),
        "ema_slow": ema_series(closes, 26),
        "sma_20": sma_series(closes, 20),
        "rsi": rsi_series(closes, 14),
        "atr": atr_series(data["high"], data["low"], closes, 14),
        "adx": adx_proxy_series(closes, 30),
    }
    macd_line, macd_signal, macd_hist = macd_series(closes)
    ctx["macd"] = macd_line
    ctx["macd_signal"] = macd_signal
    ctx["macd_hist"] = macd_hist
    return ctx


def backtest(signals: List[int], closes: List[float]) -> Dict[str, float]:
    returns = [(closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes))]
    aligned_signals = signals[:-1] if len(signals) > 1 else []
    strat_returns = [r * s for r, s in zip(returns, aligned_signals)]

    equity = [1.0]
    for r in strat_returns:
        equity.append(equity[-1] * (1.0 + r))
    total_return = equity[-1] - 1.0

    if strat_returns:
        avg_daily = mean(strat_returns)
        std_daily = pstdev(strat_returns) if len(strat_returns) > 1 else 0.0
    else:
        avg_daily = 0.0
        std_daily = 0.0

    sharpe = (avg_daily / std_daily) * math.sqrt(252) if std_daily > 0 else 0.0
    ann_return = (equity[-1] ** (252 / max(1, len(strat_returns)))) - 1.0 if strat_returns else 0.0

    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        drawdown = (peak - v) / peak if peak else 0.0
        max_dd = max(max_dd, drawdown)

    non_zero = [r for r, s in zip(strat_returns, aligned_signals) if s != 0]
    win_rate = (sum(1 for r in non_zero if r > 0) / len(non_zero)) if non_zero else 0.0

    trades = 0
    for i in range(1, len(signals)):
        if signals[i] != signals[i - 1]:
            trades += 1

    exposure = sum(1 for s in signals if s != 0) / len(signals) if signals else 0.0

    return {
        "total_return": round(total_return, 4),
        "annualized_return": round(ann_return, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 3),
        "trades": trades,
        "exposure": round(exposure, 3),
    }


def rsi_mean_reversion_strategy(oversold: float, overbought: float, adx_min: float = 0.0, adx_max: float = 100.0) -> Strategy:
    def signal(i: int, ctx: Dict[str, List[float]]) -> int:
        rsi_val = ctx["rsi"][i]
        adx_val = ctx["adx"][i]
        if adx_val < adx_min or adx_val > adx_max:
            return 0
        if rsi_val <= oversold:
            return 1
        if rsi_val >= overbought:
            return -1
        return 0

    return Strategy(
        name="RSI Mean Reversion",
        params={"oversold": oversold, "overbought": overbought, "adx_min": adx_min, "adx_max": adx_max},
        signal_fn=signal,
    )


def ema_crossover_strategy() -> Strategy:
    def signal(i: int, ctx: Dict[str, List[float]]) -> int:
        return 1 if ctx["ema_fast"][i] >= ctx["ema_slow"][i] else -1

    return Strategy(
        name="EMA Crossover",
        params={"fast": 12, "slow": 26},
        signal_fn=signal,
    )


def volatility_breakout_strategy(k: float = 1.5) -> Strategy:
    def signal(i: int, ctx: Dict[str, List[float]]) -> int:
        mid = ctx["sma_20"][i]
        atr = ctx["atr"][i]
        price = ctx["close"][i]
        if price > (mid + k * atr):
            return 1
        if price < (mid - k * atr):
            return -1
        return 0

    return Strategy(
        name="Volatility Breakout",
        params={"sma": 20, "atr": 14, "k": k},
        signal_fn=signal,
    )


def macd_confirmed_rsi_strategy(oversold: float, overbought: float, macd_threshold: float, adx_min: float) -> Strategy:
    def signal(i: int, ctx: Dict[str, List[float]]) -> int:
        rsi_val = ctx["rsi"][i]
        macd_hist = ctx["macd_hist"][i]
        adx_val = ctx["adx"][i]
        if adx_val < adx_min:
            return 0
        if rsi_val <= oversold and macd_hist >= macd_threshold:
            return 1
        if rsi_val >= overbought and macd_hist <= -macd_threshold:
            return -1
        return 0

    return Strategy(
        name="RSI + MACD Confirmation",
        params={
            "oversold": oversold,
            "overbought": overbought,
            "macd_threshold": macd_threshold,
            "adx_min": adx_min,
        },
        signal_fn=signal,
    )


def llm_suggest_params() -> List[Dict[str, Any]]:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    base_url = os.getenv("LLM_API_BASE", "https://api.openai.com").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    prompt = (
        "Suggest 3 parameter sets for RSI-based strategies on daily equity data. "
        "Return JSON array with fields: name, oversold, overbought, adx_min, macd_threshold."
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500,
    }

    import urllib.request

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


def qlib_agent_signals(ctx: Dict[str, List[float]]) -> Optional[List[int]]:
    try:
        from Agents.ATLAS_HYBRID.agents.qlib_research_agent import QlibResearchAgent
    except Exception:
        return None

    agent = QlibResearchAgent()
    signals: List[int] = []
    for i in range(len(ctx["close"])):
        market_data = {
            "pair": "SYMBOL",
            "price": ctx["close"][i],
            "price_history": ctx["close"][: i + 1],
            "volume_history": [],
            "indicators": {
                "rsi": ctx["rsi"][i],
                "macd_hist": ctx["macd_hist"][i],
                "adx": ctx["adx"][i],
            },
        }
        vote, _, _ = agent.analyze(market_data)
        signals.append(1 if vote == "BUY" else -1 if vote == "SELL" else 0)
    return signals


async def run_rdagent(data: Dict[str, List[float]], baseline_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    try:
        from Agents.ATLAS_HYBRID.agents.rdagent_factor_discovery import RDAgentFactorDiscovery
    except Exception:
        return []

    try:
        import pandas as pd
    except ImportError:
        return []

    df = pd.DataFrame(
        {
            "date": data["date"],
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
        }
    )

    agent = RDAgentFactorDiscovery()
    return await agent.run_factor_discovery_cycle(df, baseline_metrics)


def main() -> None:
    args = parse_args()

    if np is None:
        raise RuntimeError("numpy is required for research/strategy_lab.py")

    if args.seed is not None:
        np.random.seed(args.seed)

    symbol = args.symbol.upper()
    data_path = Path(args.data_path) if args.data_path else Path("data") / args.asset_class / f"{symbol.lower()}.csv"
    data = load_ohlcv_csv(data_path)
    ctx = build_context(data)

    strategies: List[Strategy] = [
        rsi_mean_reversion_strategy(30, 70),
        ema_crossover_strategy(),
        volatility_breakout_strategy(1.5),
    ]

    if args.use_autogen:
        try:
            from Agents.ATLAS_HYBRID.agents.autogen_rd_agent import AutoGenRDAgent
            agent = AutoGenRDAgent()
            proposals = agent._simplified_strategy_discovery({}, {})
            for proposal in proposals:
                params = proposal.get("parameters", {})
                strategies.append(
                    macd_confirmed_rsi_strategy(
                        oversold=float(params.get("rsi_oversold", 40)),
                        overbought=float(params.get("rsi_overbought", 60)),
                        macd_threshold=float(params.get("macd_threshold", 0.0)),
                        adx_min=float(params.get("adx_min", 20)),
                    )
                )
        except Exception:
            pass

    if args.use_llm:
        try:
            suggestions = llm_suggest_params()
            for item in suggestions:
                strategies.append(
                    macd_confirmed_rsi_strategy(
                        oversold=float(item.get("oversold", 35)),
                        overbought=float(item.get("overbought", 65)),
                        macd_threshold=float(item.get("macd_threshold", 0.0)),
                        adx_min=float(item.get("adx_min", 20)),
                    )
                )
        except Exception:
            pass

    results: List[Dict[str, Any]] = []
    for strat in strategies:
        signals = [strat.signal_fn(i, ctx) for i in range(len(ctx["close"]))]
        metrics = backtest(signals, ctx["close"])
        results.append({"name": strat.name, "params": strat.params, "metrics": metrics})

    qlib_metrics: Optional[Dict[str, Any]] = None
    if args.use_qlib_agent:
        signals = qlib_agent_signals(ctx)
        if signals:
            qlib_metrics = {
                "name": "QlibResearchAgent (signal-based)",
                "metrics": backtest(signals, ctx["close"]),
            }

    baseline = results[0]["metrics"] if results else {}
    rdagent_factors: List[Dict[str, Any]] = []
    if args.use_rdagent:
        try:
            rdagent_factors = asyncio.run(run_rdagent(data, baseline))
        except Exception:
            rdagent_factors = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"strategy_lab_{symbol.lower()}.json"
    output_md = output_dir / f"strategy_lab_{symbol.lower()}.md"

    payload = {
        "symbol": symbol,
        "rows": len(ctx["close"]),
        "strategies": results,
        "qlib_agent": qlib_metrics,
        "rdagent_factors": rdagent_factors,
        "notes": "Research-only results. Not used for Track II evaluation.",
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# ATLAS Strategy Lab (Research Only)",
        "",
        f"**Symbol:** {symbol}",
        f"**Rows:** {len(ctx['close'])}",
        "",
        "## Strategy Results",
    ]
    for item in results:
        metrics = item["metrics"]
        md.append(f"### {item['name']}")
        md.append(f"- params: {item['params']}")
        md.append(f"- metrics: {metrics}")
        md.append("")

    if qlib_metrics:
        md.append("## QlibResearchAgent Signals")
        md.append(f"- metrics: {qlib_metrics['metrics']}")
        md.append("")

    if rdagent_factors:
        md.append("## RD-Agent Factor Discovery (Optional)")
        md.append(f"- factors discovered: {len(rdagent_factors)}")
        md.append("")

    md.append("## Safety Note")
    md.append("Research-only output. No trading or investment advice.")
    output_md.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
