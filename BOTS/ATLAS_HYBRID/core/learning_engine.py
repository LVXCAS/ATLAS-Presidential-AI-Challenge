"""
Learning Engine

Analyzes trade outcomes and improves ATLAS performance over time.

Responsibilities:
1. Track all trade outcomes
2. Discover high-probability patterns
3. Adjust agent weights based on performance
4. Identify agent combinations that work well together
5. Tune score thresholds dynamically

This is where ATLAS gets smarter.
"""

from typing import Dict, List
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path


class LearningEngine:
    """
    Central learning system for ATLAS.

    Learns from every trade to improve future decisions.
    """

    def __init__(self, coordinator, pattern_recognition_agent):
        """
        Initialize learning engine.

        Args:
            coordinator: ATLAS coordinator instance
            pattern_recognition_agent: Pattern recognition agent for discovering setups
        """
        self.coordinator = coordinator
        self.pattern_agent = pattern_recognition_agent

        # Trade database
        self.trade_history = []

        # Learning metrics
        self.total_trades_analyzed = 0
        self.patterns_discovered = 0
        self.weight_adjustments_made = 0

        # Performance tracking by combination
        self.agent_combo_performance = defaultdict(lambda: {
            'wins': 0,
            'losses': 0,
            'total_r': 0
        })

    def process_trade(self, trade_result: Dict):
        """
        Process completed trade and extract learning.

        Args:
            trade_result: Dictionary containing:
                - pair: Currency pair
                - entry_time: Entry timestamp
                - exit_time: Exit timestamp
                - entry_price: Entry price
                - exit_price: Exit price
                - pnl: Profit/loss in dollars
                - r_multiple: R-multiple achieved
                - outcome: "WIN" or "LOSS"
                - agent_votes: All agent votes at entry
                - entry_conditions: Market conditions at entry
        """
        self.total_trades_analyzed += 1

        # Add to trade history
        self.trade_history.append(trade_result)

        # 1. Learn patterns (feed to PatternRecognitionAgent)
        self._learn_pattern(trade_result)

        # 2. Analyze agent combinations
        self._analyze_agent_combos(trade_result)

        # 3. Check if threshold needs adjustment
        if self.total_trades_analyzed % 100 == 0:
            self._consider_threshold_adjustment()

        # 4. Identify underperforming agents
        if self.total_trades_analyzed % 50 == 0:
            self._identify_problem_agents()

    def _learn_pattern(self, trade_result: Dict):
        """
        Extract and learn from trading pattern.

        Feeds data to PatternRecognitionAgent.
        """
        entry_conditions = trade_result.get("entry_conditions", {})

        if not entry_conditions:
            # Can't learn without entry conditions
            return

        # Feed to pattern recognition agent
        self.pattern_agent.learn_from_trade({
            "outcome": trade_result["outcome"],
            "r_multiple": trade_result.get("r_multiple", 0),
            "entry_conditions": entry_conditions
        })

    def _analyze_agent_combos(self, trade_result: Dict):
        """
        Analyze which agent combinations lead to wins vs losses.

        Example insight:
        "When TechnicalAgent + PatternAgent both vote BUY = 82% win rate"
        """
        agent_votes = trade_result.get("agent_votes", {})
        outcome = trade_result["outcome"]
        r_multiple = trade_result.get("r_multiple", 0)

        # Find agents that voted BUY
        agents_voted_buy = [
            name for name, vote_data in agent_votes.items()
            if vote_data.get("vote") == "BUY"
        ]

        if len(agents_voted_buy) < 2:
            return  # Need at least 2 agents for combination

        # Create combo key
        combo_key = "_".join(sorted(agents_voted_buy))

        # Update combo performance
        combo = self.agent_combo_performance[combo_key]

        if outcome == "WIN":
            combo['wins'] += 1
            combo['total_r'] += r_multiple
        else:
            combo['losses'] += 1

    def _consider_threshold_adjustment(self):
        """
        Consider adjusting score threshold based on recent performance.

        If win rate too low → raise threshold (be more selective)
        If trade frequency too low → lower threshold (take more trades)
        """
        # Get last 100 trades
        recent_trades = self.trade_history[-100:]

        if len(recent_trades) < 50:
            return  # Need more data

        wins = sum(1 for t in recent_trades if t["outcome"] == "WIN")
        win_rate = wins / len(recent_trades)

        current_threshold = self.coordinator.score_threshold

        print(f"\n[LEARNING] Recent win rate: {win_rate*100:.1f}% (last {len(recent_trades)} trades)")

        # Adjustment logic
        if win_rate < 0.50:
            # Win rate too low - raise threshold
            new_threshold = current_threshold + 0.2
            print(f"[LEARNING] Win rate below 50% - raising threshold: {current_threshold} → {new_threshold}")
            self.coordinator.score_threshold = min(new_threshold, 7.0)  # Cap at 7.0

        elif win_rate > 0.70 and len(recent_trades) < 30:
            # Win rate high but not enough trades - lower threshold slightly
            new_threshold = current_threshold - 0.1
            print(f"[LEARNING] Win rate excellent but trade frequency low - lowering threshold: {current_threshold} → {new_threshold}")
            self.coordinator.score_threshold = max(new_threshold, 3.0)  # Floor at 3.0

    def _identify_problem_agents(self):
        """
        Identify agents with poor performance.

        Alerts if agent consistently underperforms.
        """
        print(f"\n[LEARNING] Agent Performance Analysis ({self.total_trades_analyzed} trades)")

        for agent in self.coordinator.agents:
            metrics = agent.get_performance_metrics()

            if metrics['total_trades'] < 10:
                continue  # Not enough data

            win_rate = metrics['win_rate']
            weight = metrics['current_weight']

            # Flag underperformers
            if win_rate < 45 and weight > 0.8:
                print(f"  ⚠️ [{agent.name}] Underperforming: {win_rate:.1f}% WR (weight: {weight:.2f})")

            # Highlight top performers
            elif win_rate > 65 and weight < 1.8:
                print(f"  ⭐ [{agent.name}] High performer: {win_rate:.1f}% WR (weight: {weight:.2f})")

    def get_top_agent_combos(self, n: int = 5) -> List[Dict]:
        """
        Get top performing agent combinations.

        Args:
            n: Number of combos to return

        Returns:
            List of top combinations with metrics
        """
        combos = []

        for combo_key, perf in self.agent_combo_performance.items():
            total = perf['wins'] + perf['losses']

            if total < 10:
                continue  # Need more samples

            win_rate = perf['wins'] / total
            avg_r = perf['total_r'] / perf['wins'] if perf['wins'] > 0 else 0

            combos.append({
                'agents': combo_key.split('_'),
                'win_rate': win_rate,
                'avg_r': avg_r,
                'sample_size': total,
                'wins': perf['wins'],
                'losses': perf['losses']
            })

        # Sort by win rate
        sorted_combos = sorted(combos, key=lambda c: c['win_rate'], reverse=True)

        return sorted_combos[:n]

    def generate_learning_report(self) -> Dict:
        """
        Generate comprehensive learning report.

        Returns:
            Dictionary with all learning insights
        """
        # Recent performance
        recent_trades = self.trade_history[-100:]
        recent_wins = sum(1 for t in recent_trades if t["outcome"] == "WIN")
        recent_win_rate = (recent_wins / len(recent_trades) * 100) if recent_trades else 0

        # Agent leaderboard
        agent_leaderboard = self.coordinator.get_agent_leaderboard()

        # Top patterns
        top_patterns = self.pattern_agent.get_top_patterns(n=5)

        # Top agent combos
        top_combos = self.get_top_agent_combos(n=5)

        return {
            "total_trades_analyzed": self.total_trades_analyzed,
            "recent_win_rate": round(recent_win_rate, 1),
            "recent_trade_count": len(recent_trades),
            "current_threshold": self.coordinator.score_threshold,
            "agent_leaderboard": agent_leaderboard[:5],  # Top 5 agents
            "top_patterns": top_patterns,
            "top_agent_combos": top_combos,
            "patterns_discovered": len(self.pattern_agent.patterns),
        }

    def save_learning_data(self, filepath: str):
        """
        Save all learning data to file.

        Args:
            filepath: Path to save learning data
        """
        data = {
            "trade_history": self.trade_history[-500:],  # Last 500 trades
            "total_trades_analyzed": self.total_trades_analyzed,
            "patterns_discovered": self.patterns_discovered,
            "weight_adjustments_made": self.weight_adjustments_made,
            "agent_combo_performance": dict(self.agent_combo_performance),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[LEARNING] Saved learning data to {filepath}")

    def load_learning_data(self, filepath: str):
        """
        Load learning data from file.

        Args:
            filepath: Path to learning data file
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.trade_history = data.get("trade_history", [])
            self.total_trades_analyzed = data.get("total_trades_analyzed", 0)
            self.patterns_discovered = data.get("patterns_discovered", 0)
            self.weight_adjustments_made = data.get("weight_adjustments_made", 0)

            combo_data = data.get("agent_combo_performance", {})
            self.agent_combo_performance = defaultdict(
                lambda: {'wins': 0, 'losses': 0, 'total_r': 0},
                combo_data
            )

            print(f"[LEARNING] Loaded learning data: {self.total_trades_analyzed} trades analyzed")

        except FileNotFoundError:
            print(f"[LEARNING] No saved learning data found at {filepath}")
