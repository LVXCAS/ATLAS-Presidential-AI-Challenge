"""
Volume Agent (Educational, Simulation-Only)

Some configurations reference `VolumeAgent` separately from `VolumeLiquidityAgent`.
This lightweight agent focuses only on volume patterns, using data already present
in `market_data["volume_history"]` (historical/delayed).
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

from .base_agent import AgentAssessment, BaseAgent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class VolumeAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.0):
        super().__init__(name="VolumeAgent", initial_weight=initial_weight)
        self.lookback = 20

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        volume_history: List[float] = list(market_data.get("volume_history", []) or [])
        if len(volume_history) < 10:
            return AgentAssessment(
                score=0.40,
                explanation="No volume history provided; volume-based risk is uncertain.",
                details={"volume_len": len(volume_history)},
            )

        window = volume_history[-self.lookback :] if len(volume_history) >= self.lookback else volume_history
        avg = mean(window[:-1]) if len(window) > 1 else mean(window)
        current = float(window[-1])
        ratio = (current / avg) if avg > 0 else 1.0

        # Interpret extremes as higher uncertainty.
        low_risk = _clamp01((0.7 - ratio) / 0.7) if ratio < 0.7 else 0.0
        spike_risk = _clamp01((ratio - 2.0) / 2.0) if ratio > 2.0 else 0.0
        score = _clamp01(0.25 + 0.5 * max(low_risk, spike_risk))

        if ratio < 0.7:
            explanation = "Volume is below average; markets can be thinner and more jumpy."
        elif ratio > 2.0:
            explanation = "Volume spike detected; large moves can happen more often."
        else:
            explanation = "Volume looks normal; no strong volume-related warning."

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={"volume_ratio": round(ratio, 2), "lookback": len(window)},
        )

