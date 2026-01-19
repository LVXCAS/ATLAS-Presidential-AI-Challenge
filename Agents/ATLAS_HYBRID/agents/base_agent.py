"""
Base Agent Class

All ATLAS agents inherit from this base class.
Provides a standard, educational interface for scoring risk and explaining why.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from datetime import datetime
import json


@dataclass(frozen=True)
class AgentAssessment:
    """
    Standardized agent output for the educational (simulation-only) system.

    `score` is normalized to the range 0..1 and is interpreted as a *risk/uncertainty*
    score: 0 = low risk, 1 = high risk.

    `explanation` is a short, student-friendly sentence describing the main driver.
    """

    score: float
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp score to [0, 1] and normalize explanation to a single line.
        clamped = max(0.0, min(1.0, float(self.score)))
        object.__setattr__(self, "score", clamped)
        object.__setattr__(self, "explanation", " ".join(str(self.explanation).split()))

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "explanation": self.explanation, "details": self.details}


LegacyAnalyzeOutput = Tuple[str, float, Dict[str, Any]]
AnalyzeOutput = Union[AgentAssessment, LegacyAnalyzeOutput]


class BaseAgent(ABC):
    """
    Abstract base class for all ATLAS agents.

    Each agent:
    - Analyzes market data
    - Emits a risk/uncertainty signal (ALLOW/CAUTION/BLOCK/NEUTRAL) or a score
    - Reports confidence in that signal (0.0 - 1.0)
    - Can track performance in simulation-only exercises (legacy, optional)
    """

    def __init__(self, name: str, initial_weight: float = 1.0):
        """
        Initialize agent.

        Args:
            name: Agent name (e.g., "TechnicalAgent")
            initial_weight: Starting weight for vote scoring (default 1.0)
        """
        self.name = name
        self.weight = initial_weight

        # Performance tracking
        self.total_votes = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.total_r = 0.0

        # Learning metrics
        self.performance_history = []
        self.discovered_patterns = []

    @abstractmethod
    def analyze(self, market_data: Dict) -> AnalyzeOutput:
        """
        Analyze market data and return an assessment.

        Args:
            market_data: Dictionary containing:
                - pair: Currency pair (e.g., "EUR_USD")
                - price: Current price
                - candles: Recent OHLCV data
                - indicators: Pre-calculated indicators (RSI, MACD, etc.)
                - time: Data timestamp (historical/delayed)
                - session: Trading session (Asian/London/NY)

        Returns:
            Preferred (new): `AgentAssessment(score=..., explanation=..., details=...)`

            Backward-compatible (legacy): `(vote, confidence, reasoning)` where:
              - vote: "ALLOW" | "CAUTION" | "BLOCK" | "NEUTRAL" | "BUY" | "SELL" | "HOLD"
              - confidence: 0..1 (signal clarity)
              - reasoning: dict with explanation fields (e.g., "reason" or "message")
        """
        pass

    def assess(self, market_data: Dict) -> AgentAssessment:
        """
        Return a standardized (score + explanation) assessment.

        This wraps legacy agents that still return (vote, confidence, reasoning).
        New/updated agents can directly return AgentAssessment from `analyze()`.
        """
        raw = self.analyze(market_data)
        if isinstance(raw, AgentAssessment):
            return raw

        try:
            vote, confidence, reasoning = raw
        except Exception:
            return AgentAssessment(
                score=0.5,
                explanation="Insufficient information to assess risk.",
                details={"error": "invalid_agent_output", "raw": repr(raw)},
            )

        score = self._legacy_to_score(vote=vote, confidence=confidence)
        explanation = self._legacy_to_explanation(vote=vote, confidence=confidence, reasoning=reasoning)
        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={"legacy_vote": vote, "legacy_confidence": float(confidence), "legacy_reasoning": reasoning},
        )

    @staticmethod
    def _legacy_to_score(vote: str, confidence: float) -> float:
        """
        Map legacy vote/confidence outputs into a 0..1 risk score.

        This is a fallback for older agents; updated agents should output risk
        directly via `AgentAssessment`.
        """
        v = str(vote or "").upper()
        c = max(0.0, min(1.0, float(confidence)))

        if v == "BLOCK":
            return 1.0
        if v in {"HOLD", "STOP"}:
            return 0.85
        if v == "CAUTION":
            return 0.70
        if v == "ALLOW":
            return 0.25
        if v in {"BUY", "SELL"}:
            return max(0.10, 1.0 - c)  # high confidence -> lower perceived risk
        # NEUTRAL/unknown: moderate uncertainty
        return 0.50

    @staticmethod
    def _legacy_to_explanation(vote: str, confidence: float, reasoning: Dict[str, Any]) -> str:
        """
        Extract a short explanation from a legacy reasoning dict.
        """
        if isinstance(reasoning, dict):
            for key in ("explanation", "reason", "message", "veto_reason", "signal"):
                val = reasoning.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        v = str(vote or "NEUTRAL").upper()
        c = max(0.0, min(1.0, float(confidence)))
        return f"{v} (confidence {c:.0%})"

    def record_vote(self, vote: str, confidence: float, trade_executed: bool):
        """
        Record that agent voted (legacy simulation hook).

        Args:
            vote: The vote cast
            confidence: Confidence level
            trade_executed: Whether a hypothetical action was executed in a simulation
        """
        if vote in ["BUY", "SELL"] and trade_executed:
            self.total_votes += 1

    def record_outcome(self, trade_result: Dict):
        """
        Record outcome and update agent performance (legacy simulation hook).

        Args:
            trade_result: Dictionary containing:
                - outcome: "WIN" or "LOSS"
                - pnl: Profit/loss in dollars
                - r_multiple: Risk-reward multiple (e.g., 2.3 means 2.3R)
                - entry_time: When the simulated action began
                - exit_time: When the simulated action ended
                - agent_voted: Whether this agent voted in the simulation
        """
        if not trade_result.get("agent_voted"):
            return  # Agent didn't participate in this trade

        outcome = trade_result["outcome"]
        pnl = trade_result["pnl"]
        r_multiple = trade_result.get("r_multiple", 0)

        if outcome == "WIN":
            self.wins += 1
        else:
            self.losses += 1

        self.total_pnl += pnl
        self.total_r += r_multiple

        # Store in history
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome,
            "pnl": pnl,
            "r_multiple": r_multiple,
        })

    def get_performance_metrics(self) -> Dict:
        """
        Calculate current performance metrics.

        Returns:
            Dictionary with:
                - win_rate: Percentage of winning trades
                - avg_r_multiple: Average R-multiple on wins
                - total_pnl: Total profit/loss
                - total_votes: Number of trades participated in
                - current_weight: Current vote weight
        """
        total_trades = self.wins + self.losses

        return {
            "agent_name": self.name,
            "win_rate": (self.wins / total_trades * 100) if total_trades > 0 else 0,
            "wins": self.wins,
            "losses": self.losses,
            "total_trades": total_trades,
            "avg_r_multiple": (self.total_r / self.wins) if self.wins > 0 else 0,
            "total_pnl": self.total_pnl,
            "current_weight": self.weight,
            "total_votes": self.total_votes,
        }

    def adjust_weight(self, learning_rate: float = 0.15):
        """
        Adjust agent weight based on performance.

        Called by learning engine after every N trades.

        Args:
            learning_rate: How aggressively to adjust (0.05 = conservative, 0.30 = aggressive)
        """
        metrics = self.get_performance_metrics()

        if metrics["total_trades"] < 10:
            return  # Need more data before adjusting

        win_rate = metrics["win_rate"] / 100  # Convert to 0.0-1.0
        avg_r = metrics["avg_r_multiple"]

        # Performance score (0.0 - 1.0+)
        # 60% weight on win rate, 40% weight on R-multiple
        performance = (win_rate * 0.6) + (min(avg_r / 3.0, 1.0) * 0.4)

        # Adjust weight based on performance
        if performance > 0.75:
            # Excellent performance - boost weight
            adjustment = 1.0 + learning_rate
        elif performance > 0.60:
            # Good performance - small boost
            adjustment = 1.0 + (learning_rate * 0.5)
        elif performance > 0.45:
            # Neutral performance - no change
            adjustment = 1.0
        elif performance > 0.30:
            # Poor performance - small reduction
            adjustment = 1.0 - (learning_rate * 0.5)
        else:
            # Very poor performance - reduce weight
            adjustment = 1.0 - learning_rate

        self.weight *= adjustment

        # Clip weight to reasonable range
        self.weight = max(0.3, min(2.5, self.weight))

    def learn_from_pattern(self, pattern: Dict):
        """
        Learn from discovered pattern.

        Called by PatternRecognitionAgent when it discovers
        a high-probability setup that this agent can incorporate.

        Args:
            pattern: Dictionary describing the pattern
        """
        # Base implementation - can be overridden by specific agents
        self.discovered_patterns.append(pattern)

    def save_state(self, filepath: str):
        """Save agent state to JSON file."""
        state = {
            "name": self.name,
            "weight": self.weight,
            "total_votes": self.total_votes,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": self.total_pnl,
            "total_r": self.total_r,
            "performance_history": self.performance_history[-100:],  # Last 100 trades
            "discovered_patterns": self.discovered_patterns,
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load agent state from JSON file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.weight = state.get("weight", 1.0)
            self.total_votes = state.get("total_votes", 0)
            self.wins = state.get("wins", 0)
            self.losses = state.get("losses", 0)
            self.total_pnl = state.get("total_pnl", 0.0)
            self.total_r = state.get("total_r", 0.0)
            self.performance_history = state.get("performance_history", [])
            self.discovered_patterns = state.get("discovered_patterns", [])

            print(f"[{self.name}] Loaded state: {self.wins}W-{self.losses}L, weight={self.weight:.2f}")
        except FileNotFoundError:
            print(f"[{self.name}] No saved state found, starting fresh")

    def __repr__(self):
        metrics = self.get_performance_metrics()
        return (f"<{self.name} "
                f"weight={self.weight:.2f} "
                f"wr={metrics['win_rate']:.1f}% "
                f"trades={metrics['total_trades']}>")
