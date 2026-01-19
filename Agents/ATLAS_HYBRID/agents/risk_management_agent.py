"""
Risk Management Agent (Educational, Simulation-Only)

This agent represents simple "safety rules" that help avoid taking action when:
- the day is already going poorly (drawdown),
- there have been multiple recent losses (losing streak),
- the system is over-trading.

It does not execute trades and uses only the provided historical/delayed snapshot
in `market_data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Dict

from .base_agent import AgentAssessment, BaseAgent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class RiskState:
    day: date
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    actions_today: int = 0


class RiskManagementAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.0):
        super().__init__(name="RiskManagementAgent", initial_weight=initial_weight)

        # These are educational defaults; tune for demos as needed.
        self.max_daily_loss_pct = 0.03  # 3%
        self.max_consecutive_losses = 3
        self.max_actions_per_day = 10

        self.state = RiskState(day=date.today())

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        ts = market_data.get("time")
        now_day = ts.date() if isinstance(ts, datetime) else date.today()

        # Reset state when the (simulated) day changes.
        if now_day != self.state.day:
            self.state = RiskState(day=now_day)

        balance = float(market_data.get("account_balance", 0.0) or 0.0)
        if balance <= 0:
            return AgentAssessment(
                score=0.50,
                explanation="No account balance provided; risk rules cannot be applied.",
                details={"account_balance": balance, "data_sufficiency": "insufficient"},
            )

        # Allow demos to pass in these fields directly (historical/delayed).
        self.state.daily_pnl = float(market_data.get("daily_pnl", self.state.daily_pnl) or 0.0)
        self.state.consecutive_losses = int(market_data.get("consecutive_losses", self.state.consecutive_losses) or 0)
        self.state.actions_today = int(market_data.get("actions_today", self.state.actions_today) or 0)

        daily_loss_limit = balance * self.max_daily_loss_pct

        if self.state.daily_pnl <= -daily_loss_limit:
            return AgentAssessment(
                score=1.0,
                explanation="Daily loss limit reached — best practice is to pause and review.",
                details={"daily_pnl": self.state.daily_pnl, "daily_loss_limit": daily_loss_limit},
            )

        if self.state.consecutive_losses >= self.max_consecutive_losses:
            score = 0.85
            explanation = "Several losses in a row — uncertainty is elevated; consider a cooldown."
        elif self.state.actions_today >= self.max_actions_per_day:
            score = 0.75
            explanation = "Many actions today — over-trading can increase mistakes; slow down."
        else:
            # Mild risk proportional to how close we are to the daily loss limit.
            proximity = abs(min(self.state.daily_pnl, 0.0)) / daily_loss_limit if daily_loss_limit > 0 else 0.0
            score = 0.25 + (0.35 * _clamp01(proximity))
            explanation = "Risk limits look OK based on the provided account snapshot."

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={
                "daily_pnl": round(self.state.daily_pnl, 2),
                "consecutive_losses": self.state.consecutive_losses,
                "actions_today": self.state.actions_today,
                "daily_loss_limit": round(daily_loss_limit, 2),
            },
        )

    # Optional helper for simulations that want to update state.
    def record_outcome(self, outcome: str, pnl: float) -> None:
        if str(outcome).upper() == "WIN":
            self.state.consecutive_losses = 0
        elif str(outcome).upper() == "LOSS":
            self.state.consecutive_losses += 1
        self.state.daily_pnl += float(pnl or 0.0)
