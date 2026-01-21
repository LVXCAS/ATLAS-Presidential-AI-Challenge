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
    timestamps: Optional[List[datetime]] = None
    volumes: Optional[List[float]] = None
    data_source: str = "synthetic_historical"


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


def make_market_data(
    pair: str,
    prices: List[float],
    step: int,
    timestamps: Optional[List[datetime]] = None,
    volume_history: Optional[List[float]] = None,
    data_source: str = "synthetic_historical",
) -> Dict[str, Any]:
    history = prices[: step + 1]
    price = history[-1]

    # This project is simulation-only: treat the snapshot as historical/delayed data.
    # Use a fixed reference timestamp for reproducible evaluation artifacts unless
    # cached timestamps are provided.
    if timestamps and step < len(timestamps) and isinstance(timestamps[step], datetime):
        as_of = timestamps[step]
    else:
        as_of = datetime(2025, 1, 15, 9, 0) + timedelta(minutes=step)

    ema50 = ema(history[-200:], 50)
    ema200 = ema(history[-400:], 200)
    rsi14 = rsi(history, 14)
    atr14 = atr_like(history, 14) or (price * 0.0008)
    bb_lower, bb_mid, bb_upper = bollinger(history, 20, 2.0)
    macd_line, macd_signal, macd_hist = macd(history)
    adx = adx_proxy(history)

    volume_hist = volume_history[: step + 1] if volume_history else []

    return {
        "step": step,
        "pair": pair,
        "price": price,
        "time": as_of,
        "data_source": data_source,
        "data_is_delayed": True,
        "data_delay_minutes": 15,
        "session": "london",
        "direction": "long",
        "persist_state": False,
        "price_history": history[-200:],
        "volume_history": volume_hist[-200:],
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


def baseline_assessment(indicators: Dict[str, float]) -> Dict[str, Any]:
    """
    Educational baseline assessment used in Track II comparisons.

    Returns a normalized risk score (0..1) plus a short explanation.
    """
    label, meta = baseline_risk(indicators)

    # Map baseline labels to a normalized risk score.
    score_map = {"GREENLIGHT": 0.20, "WATCH": 0.55, "STAND_DOWN": 0.90}
    score = float(score_map.get(label, 0.50))

    explanation = ""
    if isinstance(meta, dict):
        if isinstance(meta.get("reason"), str):
            explanation = meta["reason"]
        elif isinstance(meta.get("reasons"), list) and meta["reasons"]:
            explanation = "; ".join(str(r) for r in meta["reasons"][:2])
    if not explanation:
        explanation = "Baseline rules applied to indicators."

    return {"label": label, "score": score, "explanation": explanation, "meta": meta}


def derive_vote_confidence(
    score: float,
    explanation: str,
    details: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Convert a risk score into a (vote, confidence, reasoning) triple.

    Vote values are education-focused: RISK_ON / NEUTRAL / RISK_OFF.
    If data is insufficient, return NEUTRAL.
    """
    meta = details or {}
    data_sufficiency = meta.get("data_sufficiency")
    if data_sufficiency == "insufficient":
        return (
            "NEUTRAL",
            0.0,
            {
                "explanation": explanation or "Insufficient data for a risk signal.",
                "data_sufficiency": "insufficient",
            },
        )

    risk = max(0.0, min(1.0, float(score)))
    if risk >= 0.60:
        vote = "RISK_OFF"
    elif risk <= 0.30:
        vote = "RISK_ON"
    else:
        vote = "NEUTRAL"

    # Confidence indicates signal clarity: high when risk is clearly low or high.
    confidence = max(0.0, min(1.0, abs(risk - 0.5) * 2.0))
    reasoning = {"explanation": explanation, "risk_score": round(risk, 3)}
    return (vote, round(confidence, 3), reasoning)


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


def quant_team_assessment(agent_assessments: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Combine agent assessments into a desk-style stance label.

    Each agent is expected to provide:
      - score: 0..1 (0 = low risk, 1 = high risk)
      - explanation: short string
      - weight: optional (defaults to 1.0)
      - is_veto: optional (if true and score is high -> immediate STAND_DOWN)
    """
    agents: List[Dict[str, Any]] = []
    insufficient_agents: List[str] = []
    for name, data in (agent_assessments or {}).items():
        try:
            score = float(data.get("score", 0.5))
        except Exception:
            score = 0.5
        try:
            weight = float(data.get("weight", 1.0))
        except Exception:
            weight = 1.0
        details = data.get("details") if isinstance(data.get("details"), dict) else {}
        data_sufficiency = details.get("data_sufficiency")
        data_ok = data_sufficiency != "insufficient"
        if not data_ok:
            insufficient_agents.append(str(name))
            weight = 0.0
        agents.append(
            {
                "name": str(name),
                "score": max(0.0, min(1.0, score)),
                "weight": max(0.0, weight),
                "explanation": str(data.get("explanation", "")).strip(),
                "is_veto": bool(data.get("is_veto", False)),
                "details": details,
                "data_ok": data_ok,
            }
        )

    def _collect_risk_flag_codes(items: List[Dict[str, Any]]) -> List[str]:
        codes: List[str] = []

        def _add(code: str) -> None:
            if code not in codes:
                codes.append(code)

        for a in items:
            if not a.get("data_ok", True):
                continue
            name = a.get("name", "")
            score = float(a.get("score", 0.0) or 0.0)
            details = a.get("details", {}) if isinstance(a.get("details"), dict) else {}

            if name in {"TechnicalAgent", "GSQuantAgent", "MonteCarloAgent", "VolumeLiquidityAgent"} and score >= 0.65:
                _add("HIGH_VOLATILITY")
            if name == "CorrelationAgent" and score >= 0.60:
                _add("CORRELATION_BREAKDOWN")
            if name in {"MarketRegimeAgent", "MultiTimeframeAgent"} and score >= 0.60:
                _add("REGIME_SHIFT")

            # Extra signals if explicit stats are available.
            atr_pips = details.get("atr_pips")
            if isinstance(atr_pips, (int, float)) and atr_pips >= 20:
                _add("HIGH_VOLATILITY")
            regime = details.get("regime")
            if isinstance(regime, str) and regime in {"choppy", "transition"}:
                _add("REGIME_SHIFT")

        return codes

    risk_flag_codes = _collect_risk_flag_codes(agents)

    if not agents:
        risk_score = 0.5
        return (
            "WATCH",
            {
                "risk_score": risk_score,
                "aggregated_score": risk_score,
                "market_condition": "UNCERTAIN",
                "risk_posture": "ELEVATED",
                "confidence": round(1.0 - risk_score, 3),
                "explanation": "No agents available; defaulting to WATCH.",
                "method": "none",
                "drivers": [],
                "risk_flags": [],
                "risk_flag_details": [],
                "agent_count": 0,
            },
        )

    # Veto: any veto agent can force STAND_DOWN if it flags high risk.
    veto_hits = [a for a in agents if a["is_veto"] and a["data_ok"] and a["score"] >= 0.80]
    if veto_hits:
        top_veto = sorted(veto_hits, key=lambda a: a["score"], reverse=True)[0]
        drivers = [
            {"agent": a["name"], "score": round(a["score"], 2), "weight": round(a["weight"], 2), "explanation": a["explanation"]}
            for a in sorted(veto_hits, key=lambda a: a["score"], reverse=True)[:3]
        ]
        explanation = top_veto["explanation"] or f"{top_veto['name']} flagged high risk."
        return (
            "STAND_DOWN",
            {
                "risk_score": 1.0,
                "aggregated_score": 1.0,
                "market_condition": "STRESS",
                "risk_posture": "HIGH",
                "confidence": 0.0,
                "explanation": explanation,
                "method": "veto",
                "drivers": drivers,
                "risk_flags": risk_flag_codes,
                "risk_flag_details": drivers,
                "agent_count": len(agents),
                "insufficient_agents": insufficient_agents,
            },
        )

    total_weight = sum(a["weight"] for a in agents)
    if total_weight <= 0:
        risk_score = 0.5
        return (
            "WATCH",
            {
                "risk_score": risk_score,
                "aggregated_score": risk_score,
                "market_condition": "UNCERTAIN",
                "risk_posture": "ELEVATED",
                "confidence": 0.5,
                "explanation": "Too few usable agent signals; defaulting to WATCH.",
                "method": "insufficient_data",
                "drivers": [],
                "risk_flags": [],
                "risk_flag_details": [],
                "agent_count": len(agents),
                "insufficient_agents": insufficient_agents,
                "total_weight": 0.0,
            },
        )

    risk_score = sum(a["score"] * a["weight"] for a in agents) / total_weight

    # Choose stance label based on overall risk.
    # These thresholds are tuned for an educational demo that prioritizes
    # avoiding false GREENLIGHT signals during stress windows.
    if risk_score >= 0.36:
        label = "STAND_DOWN"
    elif risk_score >= 0.25:
        label = "WATCH"
    else:
        label = "GREENLIGHT"

    market_condition = "CALM" if label == "GREENLIGHT" else ("ELEVATED" if label == "WATCH" else "STRESS")
    posture_map = {"GREENLIGHT": "LOW", "WATCH": "ELEVATED", "STAND_DOWN": "HIGH"}
    risk_posture = posture_map.get(label, "ELEVATED")

    top_drivers = sorted(agents, key=lambda a: a["score"] * a["weight"], reverse=True)[:3]
    drivers = [
        {"agent": a["name"], "score": round(a["score"], 2), "weight": round(a["weight"], 2), "explanation": a["explanation"]}
        for a in top_drivers
    ]

    flagged = [a for a in top_drivers if a["data_ok"] and a["score"] >= 0.60]
    flag_explanations = [a["explanation"] for a in flagged if a.get("explanation")]

    if label == "GREENLIGHT":
        explanation = f"Low overall risk (score {risk_score:.2f})."
    elif label == "WATCH":
        explanation = f"Caution: risk and uncertainty are elevated (score {risk_score:.2f})."
    else:
        explanation = f"Stand down: multiple risk flags are active (score {risk_score:.2f})."

    if flag_explanations:
        explanation = f"{explanation} Signals: " + "; ".join(flag_explanations[:2])

    # Add one driver hint if available (kept short for students/judges).
    if drivers and drivers[0].get("explanation"):
        explanation = f"{explanation} Top driver: {drivers[0]['agent']} - {drivers[0]['explanation']}"

    return (
        label,
        {
            "risk_score": round(risk_score, 3),
            "aggregated_score": round(risk_score, 3),
            "market_condition": market_condition,
            "risk_posture": risk_posture,
            "confidence": round(1.0 - risk_score, 3),
            "explanation": explanation,
            "method": "weighted_average",
            "drivers": drivers,
            "risk_flags": risk_flag_codes,
            "risk_flag_details": [
                {"agent": a["name"], "score": round(a["score"], 2), "explanation": a["explanation"]}
                for a in flagged
            ],
            "agent_count": len(agents),
            "total_weight": round(total_weight, 3),
            "insufficient_agents": insufficient_agents,
        },
    )


def _normalize_symbol(symbol: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in (symbol or "")).strip("_")


def infer_stress_flags(prices: List[float], threshold_pips: float = 8.0) -> List[bool]:
    """
    Tag stress steps using a simple ATR-like volatility proxy.
    """
    flags: List[bool] = []
    for i in range(len(prices)):
        if i < 2:
            flags.append(False)
            continue
        window = prices[: i + 1]
        atr_val = atr_like(window, 14)
        price = window[-1] or 1.0
        atr_pips = (atr_val / price) * 10000.0
        flags.append(atr_pips >= threshold_pips)
    return flags


def load_cached_window(
    asset_class: str,
    symbol: str,
    data_dir: Optional[str | Path] = None,
    max_rows: Optional[int] = None,
) -> Optional[StressWindow]:
    """
    Load a cached CSV and convert it into a StressWindow.

    Returns None if cached data is unavailable or pandas is missing.
    """
    try:
        from data_loader import DataLoaderError, load_cached_csv
    except Exception:
        print("[ATLAS] WARNING: Cached data loader unavailable; falling back to synthetic data.")
        return None

    try:
        df = load_cached_csv(symbol=symbol, asset_class=asset_class, data_dir=data_dir, max_rows=max_rows)
    except DataLoaderError as exc:
        print(f"[ATLAS] WARNING: {exc}. Falling back to synthetic data.")
        return None

    if df.empty:
        print("[ATLAS] WARNING: Cached CSV has no usable rows; falling back to synthetic data.")
        return None

    prices = [float(v) for v in df["close"].tolist()]
    volumes = [float(v) for v in df["volume"].tolist()] if "volume" in df.columns else []
    timestamps = [
        (ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts) for ts in df["date"].tolist()
    ]

    is_stress = infer_stress_flags(prices)
    name = f"cached-{asset_class}-{_normalize_symbol(symbol) or 'series'}"
    description = f"Cached historical data from CSV ({asset_class}/{symbol})."
    return StressWindow(
        name=name,
        pair=symbol,
        prices=prices,
        is_stress_step=is_stress,
        description=description,
        timestamps=timestamps,
        volumes=volumes,
        data_source="cached_csv",
    )


def get_stress_windows(
    data_source: str = "synthetic",
    asset_class: str = "fx",
    symbol: str = "EUR_USD",
    data_dir: Optional[str | Path] = None,
    max_rows: Optional[int] = None,
) -> List[StressWindow]:
    source = (data_source or "synthetic").lower()
    if source == "cached":
        cached = load_cached_window(
            asset_class=asset_class,
            symbol=symbol,
            data_dir=data_dir,
            max_rows=max_rows,
        )
        if cached:
            return [cached]
        print("[ATLAS] WARNING: Cached data missing; using synthetic stress windows.")
    return generate_stress_windows()


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
            name="stable",
            pair="EUR_USD",
            prices=prices_a,
            is_stress_step=stress_a,
            description="Stable range conditions with low volatility (mostly normal risk).",
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
        if agent_name == "OfflineMLRiskAgent":
            from agents.offline_ml_risk_agent import OfflineMLRiskAgent
            return OfflineMLRiskAgent
        if agent_name == "SentimentAgent":
            from agents.sentiment_agent import SentimentAgent
            return SentimentAgent
        if agent_name == "QlibResearchAgent":
            from agents.qlib_research_agent import QlibResearchAgent
            return QlibResearchAgent
        if agent_name == "GSQuantAgent":
            from agents.gs_quant_agent import GSQuantAgent
            return GSQuantAgent
        if agent_name == "E8ComplianceAgent":
            from agents.e8_compliance_agent import E8ComplianceAgent
            return E8ComplianceAgent
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
        if agent_name == "VolumeAgent":
            from agents.volume_agent import VolumeAgent
            return VolumeAgent
        if agent_name == "VolumeLiquidityAgent":
            from agents.volume_liquidity_agent import VolumeLiquidityAgent
            return VolumeLiquidityAgent
        if agent_name == "SupportResistanceAgent":
            from agents.support_resistance_agent import SupportResistanceAgent
            return SupportResistanceAgent
        if agent_name == "DivergenceAgent":
            from agents.divergence_agent import DivergenceAgent
            return DivergenceAgent
        if agent_name == "LLMTechnicalAgent":
            from agents.llm_technical_agent import LLMTechnicalAgent
            return LLMTechnicalAgent
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
