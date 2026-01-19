"""
Monte Carlo Agent (Educational, Simulation-Only)

Monte Carlo simulation = "roll the dice many times" using historical returns to
estimate the probability of large moves.

This implementation:
- uses only `price_history` + optional `indicators["atr"]`
- outputs a normalized risk score (0..1) and a short explanation string
- does NOT place trades and does NOT use live data
"""

from __future__ import annotations

import random
from statistics import pstdev
from typing import Any, Dict, List, Tuple

from .base_agent import AgentAssessment, BaseAgent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _returns(prices: List[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        cur = prices[i]
        if prev:
            out.append((cur - prev) / prev)
    return out


class MonteCarloAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.2, is_veto: bool = False):
        super().__init__(name="MonteCarloAgent", initial_weight=initial_weight)

        self.is_veto = bool(is_veto)
        self.num_simulations = 250
        self.horizon_steps = 10

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        # Use a deterministic seed so evaluation artifacts are reproducible for judges.
        pair = str(market_data.get("pair", "") or "").upper()
        step = int(market_data.get("step", 0) or 0)
        pair_seed = sum((i + 1) * ord(c) for i, c in enumerate(pair))
        rng = random.Random((pair_seed * 1000003 + step * 9176 + 12345) & 0xFFFFFFFF)

        history: List[float] = list(market_data.get("price_history", []) or [])
        if len(history) < 40:
            return AgentAssessment(
                score=0.50,
                explanation="Not enough history for Monte Carlo risk estimation.",
                details={"history_len": len(history), "needed": 40, "data_sufficiency": "insufficient"},
            )

        price_now = float(market_data.get("price", history[-1]) or history[-1])
        if price_now <= 0:
            return AgentAssessment(
                score=0.50,
                explanation="Missing price; Monte Carlo skipped.",
                details={"data_sufficiency": "insufficient"},
            )

        indicators = market_data.get("indicators", {}) or {}
        atr = float(indicators.get("atr", 0.0) or 0.0)
        atr_pct = (atr / price_now) if price_now else 0.0

        rets = _returns(history[-200:])
        if len(rets) < 30:
            return AgentAssessment(
                score=0.50,
                explanation="Not enough return history for Monte Carlo sampling.",
                details={"returns_len": len(rets), "data_sufficiency": "insufficient"},
            )

        # Threshold: interpret as a "large move" over the horizon.
        # Use ATR if available, otherwise use return volatility.
        ret_vol = pstdev(rets) if len(rets) >= 2 else 0.0
        threshold_pct = 2.0 * atr_pct if atr_pct > 0 else 2.0 * ret_vol * (self.horizon_steps**0.5)
        threshold_pct = max(0.002, min(0.01, threshold_pct))  # clamp to 0.2%..1.0% for readability

        # Run bootstrap simulations.
        drop_hits = 0
        big_move_hits = 0
        worst_drawdown = 0.0

        for _ in range(max(1, int(self.num_simulations))):
            sim_price = price_now
            min_price = price_now
            max_price = price_now
            for _step in range(self.horizon_steps):
                r = rng.choice(rets)
                sim_price *= 1.0 + r
                min_price = min(min_price, sim_price)
                max_price = max(max_price, sim_price)

            drawdown = (min_price - price_now) / price_now  # negative number
            max_move = max(abs((max_price - price_now) / price_now), abs(drawdown))

            if drawdown <= -threshold_pct:
                drop_hits += 1
            if max_move >= threshold_pct:
                big_move_hits += 1
            worst_drawdown = min(worst_drawdown, drawdown)

        prob_drop = drop_hits / self.num_simulations
        prob_big = big_move_hits / self.num_simulations

        # Convert probability into a 0..1 risk score.
        # We treat frequent large moves as higher uncertainty/risk.
        score = _clamp01(prob_big / 0.60)  # ~60% chance of big move => max risk

        explanation = (
            f"Monte Carlo: ~{prob_big:.0%} chance of a move ≥ {threshold_pct*100:.1f}% "
            f"over the next {self.horizon_steps} steps."
        )

        # If veto-enabled, we still return a score; the coordinator can treat high scores as veto.
        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={
                "seed": (pair_seed * 1000003 + step * 9176 + 12345) & 0xFFFFFFFF,
                "num_simulations": self.num_simulations,
                "horizon_steps": self.horizon_steps,
                "threshold_pct": round(threshold_pct * 100, 2),
                "prob_drop": round(prob_drop, 3),
                "prob_big_move": round(prob_big, 3),
                "worst_drawdown_pct": round(worst_drawdown * 100, 2),
                "veto_capable": self.is_veto,
            },
        )
