"""
Volume/Liquidity Agent (Educational, Simulation-Only)

Liquidity is "how easy it is to trade without moving the price too much".
Low liquidity often shows up as:
- wide bid/ask spreads
- inconsistent volume (if volume data is available)

This agent never fetches real-time data. It only uses what is already present
in `market_data` (historical/delayed snapshot).
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

from .base_agent import AgentAssessment, BaseAgent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class VolumeLiquidityAgent(BaseAgent):
    def __init__(self, initial_weight: float = 0.9):
        super().__init__(name="VolumeLiquidityAgent", initial_weight=initial_weight)

        # Simple spread baselines (pips) for common FX pairs.
        self.typical_spread_pips = {"EUR_USD": 1.0, "GBP_USD": 1.5, "USD_JPY": 1.0}

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        pair = str(market_data.get("pair", "") or "")
        price = float(market_data.get("price", 0.0) or 0.0)

        bid = market_data.get("bid")
        ask = market_data.get("ask")
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask > bid and price > 0:
            spread = float(ask) - float(bid)
            pip_scale = 100.0 if pair.endswith("JPY") else 10000.0
            spread_pips = spread * pip_scale
            typical = float(self.typical_spread_pips.get(pair, 1.5))

            ratio = spread_pips / typical if typical > 0 else 1.0
            score = _clamp01((ratio - 1.0) / 3.0)  # 1x => 0, 4x => 1

            if ratio >= 3.0:
                explanation = f"Spread is very wide ({spread_pips:.1f} pips) — liquidity may be poor."
            elif ratio >= 2.0:
                explanation = f"Spread is wider than usual ({spread_pips:.1f} pips) — caution."
            else:
                explanation = f"Spread looks normal ({spread_pips:.1f} pips) — liquidity risk is lower."

            return AgentAssessment(
                score=score,
                explanation=explanation,
                details={"spread_pips": round(spread_pips, 2), "typical_pips": typical, "ratio": round(ratio, 2)},
            )

        # If spread is unavailable, try a basic volume signal (if present).
        volume_history: List[float] = list(market_data.get("volume_history", []) or [])
        if len(volume_history) >= 10:
            recent = volume_history[-10:]
            avg = mean(recent[:-1]) if len(recent) > 1 else mean(recent)
            current = recent[-1]
            ratio = (current / avg) if avg > 0 else 1.0

            # Very low volume can imply thinner markets; very high spikes can imply instability.
            low_vol_risk = _clamp01((0.6 - ratio) / 0.6) if ratio < 0.6 else 0.0
            spike_risk = _clamp01((ratio - 2.0) / 2.0) if ratio > 2.0 else 0.0
            score = _clamp01(0.3 + (0.4 * low_vol_risk) + (0.4 * spike_risk))

            if ratio < 0.6:
                explanation = "Recent volume is below average; liquidity may be thinner."
            elif ratio > 2.0:
                explanation = "Large volume spike; conditions may be unstable."
            else:
                explanation = "Volume looks typical; no strong liquidity warning."

            return AgentAssessment(
                score=score,
                explanation=explanation,
                details={"volume_ratio": round(ratio, 2), "volume_len": len(volume_history)},
            )

        return AgentAssessment(
            score=0.40,
            explanation="No bid/ask or volume data provided; liquidity risk is uncertain.",
            details={"pair": pair, "price": price, "data_sufficiency": "insufficient"},
        )
