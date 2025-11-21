"""
Base Agent Class

All ATLAS agents inherit from this base class.
Provides standard interface for voting, learning, and performance tracking.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from datetime import datetime
import json


class BaseAgent(ABC):
    """
    Abstract base class for all ATLAS agents.

    Each agent:
    - Analyzes market data
    - Votes on trade decisions (BUY/SELL/NEUTRAL/ALLOW/BLOCK)
    - Has a confidence level (0.0 - 1.0)
    - Tracks its own performance
    - Learns from trade outcomes
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
    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Analyze market data and return vote.

        Args:
            market_data: Dictionary containing:
                - pair: Currency pair (e.g., "EUR_USD")
                - price: Current price
                - candles: Recent OHLCV data
                - indicators: Pre-calculated indicators (RSI, MACD, etc.)
                - time: Current timestamp
                - session: Trading session (Asian/London/NY)

        Returns:
            Tuple of (vote, confidence, reasoning):
                - vote: "BUY" | "SELL" | "NEUTRAL" | "ALLOW" | "BLOCK" | "BOOST"
                - confidence: 0.0 - 1.0 (how confident agent is in its vote)
                - reasoning: Dict explaining why agent voted this way

        Example:
            return ("BUY", 0.85, {
                "rsi": 42,
                "trend": "bullish",
                "session": "london_open",
                "reason": "RSI pullback in uptrend during high-volume session"
            })
        """
        pass

    def record_vote(self, vote: str, confidence: float, trade_executed: bool):
        """
        Record that agent voted (for tracking participation).

        Args:
            vote: The vote cast
            confidence: Confidence level
            trade_executed: Whether trade was actually executed
        """
        if vote in ["BUY", "SELL"] and trade_executed:
            self.total_votes += 1

    def record_outcome(self, trade_result: Dict):
        """
        Record trade outcome and update agent performance.

        Args:
            trade_result: Dictionary containing:
                - outcome: "WIN" or "LOSS"
                - pnl: Profit/loss in dollars
                - r_multiple: Risk-reward multiple (e.g., 2.3 means 2.3R)
                - entry_time: When trade opened
                - exit_time: When trade closed
                - agent_voted: Whether this agent voted for the trade
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
