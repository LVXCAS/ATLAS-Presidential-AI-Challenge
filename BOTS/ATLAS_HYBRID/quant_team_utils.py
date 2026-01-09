from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class StressWindow:
    name: str
    pair: str
    prices: List[float]
    is_stress_step: List[bool]
    description: str


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def ema(values: List[float], period: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    alpha = 2 / (period + 1)
    current = values[0]
    for v in values[1:]:
        current = (alpha * v) + ((1 - alpha) * current)
    return current


def rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        delta = prices[i] - prices[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr_like(prices: List[float], period: int = 14) -> float:
    if len(prices) < 2:
        return 0.0
    window = prices[-(period + 1) :]
    diffs = [abs(window[i] - window[i - 1]) for i in range(1, len(window))]
    return mean(diffs) if diffs else 0.0


def bollinger(prices: List[float], period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
    if len(prices) < 2:
        p = prices[-1] if prices else 0.0
        return (p, p, p)
    window = prices[-period:] if len(prices) >= period else prices
    mid = mean(window)
    sd = pstdev(window) if len(window) >= 2 else 0.0
    upper = mid + (num_std * sd)
    lower = mid - (num_std * sd)
    return (lower, mid, upper)


def macd(prices: List[float]) -> Tuple[float, float, float]:
    if len(prices) < 2:
        return (0.0, 0.0, 0.0)
    fast = ema(prices[-60:], 12)
    slow = ema(prices[-120:], 26)
    macd_line = fast - slow
    # crude signal approximation using EMA on recent macd line history
    macd_history = []
    for i in range(max(2, len(prices) - 80), len(prices) + 1):
        segment = prices[:i]
        fast_i = ema(segment[-60:], 12)
        slow_i = ema(segment[-120:], 26)
        macd_history.append(fast_i - slow_i)
    signal = ema(macd_history, 9) if macd_history else 0.0
    hist = macd_line - signal
    return (macd_line, signal, hist)


def adx_proxy(prices: List[float]) -> float:
    """
    Lightweight trend-strength proxy in the 10..60 range (not a true ADX).
    Higher means a stronger directional regime; lower means choppy/uncertain.
    """
    if len(prices) < 10:
        return 20.0
    window = prices[-30:] if len(prices) >= 30 else prices
    x = list(range(len(window)))
    x_mean = mean(x)
    y_mean = mean(window)
    denom = sum((xi - x_mean) ** 2 for xi in x) or 1.0
    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, window)) / denom
    avg_price = y_mean or 1.0
    strength = min(1.0, abs(slope) / (avg_price * 0.002))
    return 10.0 + (50.0 * strength)


def make_market_data(pair: str, prices: List[float], step: int) -> Dict[str, Any]:
    history = prices[: step + 1]
    price = history[-1]

    ema50 = ema(history[-200:], 50)
    ema200 = ema(history[-400:], 200)
    rsi14 = rsi(history, 14)
    atr14 = atr_like(history, 14) or (price * 0.0008)
    bb_lower, bb_mid, bb_upper = bollinger(history, 20, 2.0)
    macd_line, macd_signal, macd_hist = macd(history)
    adx = adx_proxy(history)

    return {
        "pair": pair,
        "price": price,
        "time": datetime.now() + timedelta(minutes=step),
        "session": "london",
        "direction": "long",
        "persist_state": False,
        "price_history": history[-200:],
        "volume_history": [],
        "account_balance": 100000,
        "indicators": {
            "rsi": rsi14,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "ema50": ema50,
            "ema200": ema200,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_mid,
            "adx": adx,
            "atr": atr14,
        },
    }


def baseline_risk(indicators: Dict[str, float]) -> Tuple[str, Dict[str, Any]]:
    """
    Simple rule baseline (intentionally limited).
    Returns GREENLIGHT/WATCH/STAND_DOWN + rationale.
    """
    rsi_val = float(indicators.get("rsi", 50.0))
    adx_val = float(indicators.get("adx", 20.0))
    atr = float(indicators.get("atr", 0.0))
    price = float(indicators.get("_price", 1.0)) or 1.0
    atr_pips = (atr / price) * 10000.0

    reasons: List[str] = []

    if atr_pips >= 25:
        return ("STAND_DOWN", {"atr_pips": round(atr_pips, 1), "reason": "Volatility spike (ATR high)"})

    if atr_pips >= 15:
        reasons.append("Volatility elevated (ATR medium)")

    if rsi_val >= 72 or rsi_val <= 28:
        reasons.append("Momentum extreme (RSI)")

    if adx_val <= 18:
        reasons.append("Choppy/uncertain regime (low ADX)")

    if reasons:
        return ("WATCH", {"atr_pips": round(atr_pips, 1), "rsi": round(rsi_val, 1), "adx": round(adx_val, 1), "reasons": reasons})

    return ("GREENLIGHT", {"atr_pips": round(atr_pips, 1), "rsi": round(rsi_val, 1), "adx": round(adx_val, 1), "reason": "No baseline risk flags"})


def quant_team_recommendation(agent_votes: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Convert multi-agent outputs into a desk-style recommendation.
    Returns GREENLIGHT/WATCH/STAND_DOWN + short rationale.
    """
    blocks = []
    cautions = []
    strong_directional = []

    for name, vd in agent_votes.items():
        vote = vd.get("vote")
        conf = float(vd.get("confidence", 0.0))
        if vote == "BLOCK":
            blocks.append(name)
        elif vote == "CAUTION":
            cautions.append(name)
        elif vote in {"BUY", "SELL"} and conf >= 0.70:
            strong_directional.append(name)

    if blocks:
        return ("STAND_DOWN", {"reason": "One or more agents flagged high risk", "flagged_by": blocks})

    if cautions or strong_directional:
        return ("WATCH", {"reason": "Mixed signals / elevated uncertainty", "flagged_by": sorted(set(cautions + strong_directional))})

    return ("GREENLIGHT", {"reason": "No agent flagged elevated risk"})


def generate_stress_windows(seed: int = 7) -> List[StressWindow]:
    """
    Generate 3 synthetic 'stress windows' used to demonstrate how a limited baseline can
    give false confidence and how a multi-agent quant team flags uncertainty/regime risk.
    """
    rng = random.Random(seed)

    def series(start: float, steps: int, drift: float, vol: float) -> List[float]:
        p = start
        out = [p]
        for _ in range(steps - 1):
            shock = rng.gauss(drift, vol)
            p = max(0.5, p * (1.0 + shock))
            out.append(p)
        return out

    # 1) Stable range: low vol, slight drift
    prices_a = series(1.10, 120, drift=0.0, vol=0.0003)
    stress_a = [False] * len(prices_a)

    # 2) Volatility spike: high vol whipsaw
    prices_b = series(1.10, 120, drift=0.0, vol=0.0022)
    stress_b = [True] * len(prices_b)

    # 3) Regime shift: calm -> trend + higher vol
    calm = series(1.10, 60, drift=0.00005, vol=0.00035)
    shift = series(calm[-1], 60, drift=-0.00025, vol=0.0016)
    prices_c = calm + shift[1:]
    stress_c = [False] * 60 + [True] * (len(prices_c) - 60)

    return [
        StressWindow(
            name="stable-range",
            pair="EUR_USD",
            prices=prices_a,
            is_stress_step=stress_a,
            description="Low-volatility range conditions (mostly normal risk).",
        ),
        StressWindow(
            name="volatility-spike",
            pair="EUR_USD",
            prices=prices_b,
            is_stress_step=stress_b,
            description="High-volatility whipsaw conditions (elevated risk throughout).",
        ),
        StressWindow(
            name="regime-shift",
            pair="EUR_USD",
            prices=prices_c,
            is_stress_step=stress_c,
            description="Regime shift: calm period followed by trend + elevated volatility.",
        ),
    ]


def initialize_coordinator(config: Dict[str, Any]):
    """
    Build an ATLASCoordinator from a config dict, skipping agents whose optional
    dependencies are missing.
    """
    from core.coordinator import ATLASCoordinator

    coordinator = ATLASCoordinator(config)

    # Import agent classes on demand (some trigger optional heavy deps).
    def _agent_class(agent_name: str):
        if agent_name == "TechnicalAgent":
            from agents.technical_agent import TechnicalAgent
            return TechnicalAgent
        if agent_name == "PatternRecognitionAgent":
            from agents.pattern_recognition_agent import PatternRecognitionAgent
            return PatternRecognitionAgent
        if agent_name == "NewsFilterAgent":
            from agents.news_filter_agent import NewsFilterAgent
            return NewsFilterAgent
        if agent_name == "MeanReversionAgent":
            from agents.mean_reversion_agent import MeanReversionAgent
            return MeanReversionAgent
        if agent_name == "XGBoostMLAgent":
            from agents.xgboost_ml_agent import XGBoostMLAgent
            return XGBoostMLAgent
        if agent_name == "SentimentAgent":
            from agents.sentiment_agent import SentimentAgent
            return SentimentAgent
        if agent_name == "QlibResearchAgent":
            from agents.qlib_research_agent import QlibResearchAgent
            return QlibResearchAgent
        if agent_name == "GSQuantAgent":
            from agents.gs_quant_agent import GSQuantAgent
            return GSQuantAgent
        if agent_name == "AutoGenRDAgent":
            from agents.autogen_rd_agent import AutoGenRDAgent
            return AutoGenRDAgent
        if agent_name == "MonteCarloAgent":
            from agents.monte_carlo_agent import MonteCarloAgent
            return MonteCarloAgent
        if agent_name == "MarketRegimeAgent":
            from agents.market_regime_agent import MarketRegimeAgent
            return MarketRegimeAgent
        if agent_name == "RiskManagementAgent":
            from agents.risk_management_agent import RiskManagementAgent
            return RiskManagementAgent
        if agent_name == "SessionTimingAgent":
            from agents.session_timing_agent import SessionTimingAgent
            return SessionTimingAgent
        if agent_name == "CorrelationAgent":
            from agents.correlation_agent import CorrelationAgent
            return CorrelationAgent
        if agent_name == "MultiTimeframeAgent":
            from agents.multi_timeframe_agent import MultiTimeframeAgent
            return MultiTimeframeAgent
        if agent_name == "VolumeLiquidityAgent":
            from agents.volume_liquidity_agent import VolumeLiquidityAgent
            return VolumeLiquidityAgent
        if agent_name == "SupportResistanceAgent":
            from agents.support_resistance_agent import SupportResistanceAgent
            return SupportResistanceAgent
        if agent_name == "DivergenceAgent":
            from agents.divergence_agent import DivergenceAgent
            return DivergenceAgent
        return None

    for agent_name, agent_cfg in (config.get("agents") or {}).items():
        if not agent_cfg.get("enabled", True):
            continue
        agent_class = _agent_class(agent_name)
        if agent_class is None:
            continue
        initial_weight = float(agent_cfg.get("initial_weight", 1.0))

        try:
            if agent_name == "PatternRecognitionAgent":
                agent = agent_class(
                    initial_weight=initial_weight,
                    min_pattern_samples=int(agent_cfg.get("min_pattern_samples", 10)),
                )
            elif agent_name == "MonteCarloAgent":
                agent = agent_class(
                    initial_weight=initial_weight,
                    is_veto=bool(agent_cfg.get("is_veto", False)),
                )
                if "num_simulations" in agent_cfg:
                    agent.num_simulations = int(agent_cfg["num_simulations"])
            else:
                agent = agent_class(initial_weight=initial_weight)
        except Exception:
            # Agent couldn't initialize (missing optional deps or runtime issues).
            continue

        is_veto = bool(agent_cfg.get("is_veto", False))
        coordinator.add_agent(agent, is_veto=is_veto)

    return coordinator
