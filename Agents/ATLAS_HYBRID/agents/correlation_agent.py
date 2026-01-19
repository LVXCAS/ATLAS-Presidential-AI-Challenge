"""
Correlation Agent (Educational)

This agent avoids "stacking the same bet twice" by checking whether the current
symbol shares major currencies with existing positions.

It uses only the provided (historical/delayed) snapshot in `market_data`.
"""

from typing import Dict, List, Tuple

from .base_agent import AgentAssessment, BaseAgent


def _split_pair(pair: str) -> Tuple[str, str]:
    p = (pair or "").replace("/", "_").upper()
    if "_" in p:
        base, quote = p.split("_", 1)
        return base, quote
    if len(p) == 6:
        return p[:3], p[3:]
    return (p, "")


class CorrelationAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.0):
        super().__init__(name="CorrelationAgent", initial_weight=initial_weight)

    def analyze(self, market_data: Dict) -> AgentAssessment:
        pair = str(market_data.get("pair", "") or "")
        base, quote = _split_pair(pair)

        existing_positions: List[Dict] = list(market_data.get("existing_positions", []) or [])
        if not existing_positions:
            return AgentAssessment(
                score=0.25,
                explanation="No existing positions provided; correlation risk assumed low.",
                details={"pair": pair, "existing_positions": 0},
            )

        shared = 0
        compared = 0
        for pos in existing_positions:
            pos_pair = str(pos.get("pair") or pos.get("symbol") or "")
            if not pos_pair:
                continue
            b2, q2 = _split_pair(pos_pair)
            compared += 1
            if base and (base == b2 or base == q2):
                shared += 1
            if quote and (quote == b2 or quote == q2):
                shared += 1

        if compared == 0:
            return AgentAssessment(
                score=0.35,
                explanation="Positions were provided, but symbols were missing; correlation risk unknown.",
                details={"pair": pair, "existing_positions": len(existing_positions), "data_sufficiency": "insufficient"},
            )

        # More shared currencies => more concentration risk.
        ratio = shared / (2 * compared)  # 0..1
        score = 0.25 + (0.55 * ratio)

        if ratio >= 0.60:
            explanation = "High overlap with existing positions — concentration risk is elevated."
        elif ratio >= 0.30:
            explanation = "Some overlap with existing positions — keep an eye on concentration."
        else:
            explanation = "Low overlap with existing positions — correlation risk appears limited."

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={"pair": pair, "positions_compared": compared, "shared_currency_ratio": round(ratio, 2)},
        )
