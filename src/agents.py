from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AgentSignal:
    name: str
    score: float
    reasoning: str


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _compute_returns(prices: List[float]) -> List[float]:
    returns: List[float] = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        if prev == 0:
            returns.append(0.0)
        else:
            returns.append((prices[i] - prev) / prev)
    return returns


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var ** 0.5


def volatility_agent(prices: List[float]) -> AgentSignal:
    returns = _compute_returns(prices)
    if len(returns) < 2:
        return AgentSignal("VolatilityAgent", 0.5, "Insufficient data for volatility proxy.")
    std = _std(returns)
    score = _clamp(std * 80.0)
    reasoning = f"Volatility proxy std={std:.4f}."
    return AgentSignal("VolatilityAgent", score, reasoning)


def regime_agent(prices: List[float]) -> AgentSignal:
    if len(prices) < 6:
        return AgentSignal("RegimeAgent", 0.5, "Insufficient data for regime proxy.")
    mid = len(prices) // 2
    early = (prices[mid - 1] - prices[0]) / (prices[0] or 1.0)
    late = (prices[-1] - prices[mid]) / (prices[mid] or 1.0)
    shift = abs(late - early)
    score = _clamp(shift * 10.0)
    regime = "stable" if score < 0.25 else ("transition" if score < 0.6 else "shift")
    reasoning = f"Regime {regime}: early {early:.2%}, late {late:.2%}."
    return AgentSignal("RegimeAgent", score, reasoning)


def correlation_agent(prices: List[float]) -> AgentSignal:
    returns = _compute_returns(prices)
    if len(returns) < 6:
        return AgentSignal("CorrelationAgent", 0.5, "Insufficient data for correlation proxy.")
    half = len(returns) // 2
    a = returns[:half]
    b = returns[-half:]
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((v - mean_a) ** 2 for v in a)
    var_b = sum((v - mean_b) ** 2 for v in b)
    if var_a == 0 or var_b == 0:
        corr = 0.0
    else:
        cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        corr = cov / ((var_a ** 0.5) * (var_b ** 0.5))
        corr = _clamp(corr, -1.0, 1.0)
    score = _clamp((1.0 - corr) / 2.0)
    reasoning = f"Return correlation across halves {corr:.2f}."
    return AgentSignal("CorrelationAgent", score, reasoning)


def liquidity_agent(volumes: List[float]) -> AgentSignal:
    if not volumes:
        return AgentSignal("LiquidityAgent", 0.5, "No volume data; liquidity uncertainty.")
    mean = sum(volumes) / len(volumes)
    std = _std(volumes)
    cv = (std / mean) if mean else 0.0
    low_volume = mean < 100000
    score = _clamp((cv * 1.2) + (0.3 if low_volume else 0.0))
    reasoning = f"Volume cv={cv:.2f}, avg={mean:.0f}."
    return AgentSignal("LiquidityAgent", score, reasoning)


def assess_agents(prices: List[float], volumes: List[float]) -> List[AgentSignal]:
    return [
        volatility_agent(prices),
        regime_agent(prices),
        correlation_agent(prices),
        liquidity_agent(volumes),
    ]
