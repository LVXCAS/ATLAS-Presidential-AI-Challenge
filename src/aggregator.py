from __future__ import annotations

from typing import Dict, List

from agents import AgentSignal


def aggregate(signals: List[AgentSignal]) -> Dict[str, object]:
    if not signals:
        return {
            "posture": "WATCH",
            "risk_score": 0.5,
            "risk_flags": ["INSUFFICIENT_DATA"],
            "agent_signals": {},
        }

    risk_score = sum(s.score for s in signals) / len(signals)

    veto = any(s.score >= 0.9 for s in signals)
    if veto:
        posture = "STAND_DOWN"
    elif risk_score >= 0.36:
        posture = "STAND_DOWN"
    elif risk_score >= 0.25:
        posture = "WATCH"
    else:
        posture = "GREENLIGHT"

    flags: List[str] = []
    for signal in signals:
        if signal.name == "VolatilityAgent" and signal.score >= 0.6:
            flags.append("HIGH_VOLATILITY")
        if signal.name == "RegimeAgent" and signal.score >= 0.6:
            flags.append("REGIME_SHIFT")
        if signal.name == "CorrelationAgent" and signal.score >= 0.6:
            flags.append("CORRELATION_BREAKDOWN")
        if signal.name == "LiquidityAgent" and signal.score >= 0.6:
            flags.append("LIQUIDITY_STRESS")

    agent_signals = {
        s.name: {"score": round(s.score, 3), "reasoning": s.reasoning} for s in signals
    }

    return {
        "posture": posture,
        "risk_score": round(risk_score, 3),
        "risk_flags": flags,
        "agent_signals": agent_signals,
    }
