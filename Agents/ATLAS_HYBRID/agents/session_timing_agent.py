"""
Session Timing Agent (Educational)

Why it exists: markets can behave differently at different times of day
(liquidity, volatility, news releases). This agent uses the *provided* timestamp
in `market_data["time"]` (historical/delayed) — it never looks up real-time data.
"""

from datetime import datetime
from typing import Dict

from .base_agent import AgentAssessment, BaseAgent


class SessionTimingAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.2):
        super().__init__(name="SessionTimingAgent", initial_weight=initial_weight)

    def analyze(self, market_data: Dict) -> AgentAssessment:
        ts = market_data.get("time")
        if not isinstance(ts, datetime):
            return AgentAssessment(
                score=0.45,
                explanation="No timestamp provided; session risk is uncertain.",
                details={"time": str(ts), "data_sufficiency": "insufficient"},
            )

        hour = ts.hour  # interpret as local/naive time used by the simulator

        # Simple FX-style heuristic (UTC-like): London + NY overlap is often most liquid.
        if 8 <= hour < 17:
            session = "high_liquidity"
            score = 0.25
            explanation = f"Active session hour ({hour:02d}:00) — liquidity is usually better."
        else:
            session = "low_liquidity"
            score = 0.55
            explanation = f"Off-hours ({hour:02d}:00) — liquidity can be thinner and moves noisier."

        return AgentAssessment(score=score, explanation=explanation, details={"session": session, "hour": hour})
