"""
ATLAS Coordinator

The "brain" that:
1. Collects votes from all agents
2. Applies learned weights
3. Makes final BUY/SELL/HOLD decision
4. Routes to paper vs live trading
5. Logs all decisions for learning

This is the orchestrator that brings all agents together.
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path

# Will import agents when we have them all
# from ..agents.technical_agent import TechnicalAgent
# from ..agents.pattern_recognition_agent import PatternRecognitionAgent
# etc...


class ATLASCoordinator:
    """
    Central coordinator for ATLAS multi-agent system.

    Workflow:
    1. Receive market data
    2. Ask all agents for votes
    3. Check VETO agents (NewsFilter, E8Compliance)
    4. Calculate weighted score
    5. Make final decision
    6. Execute trade (if score >= threshold)
    7. Log decision for learning
    """

    def __init__(self, config: Dict):
        """
        Initialize coordinator with configuration.

        Args:
            config: Dictionary containing:
                - mode: "paper" or "live"
                - strategy: "hybrid_optimized" or "ultra_aggressive"
                - score_threshold: Minimum score to execute trade
                - max_position_size_lots: Maximum lot size
                - agents: List of agent configurations
        """
        self.config = config
        self.mode = config.get("mode", "paper")
        self.strategy = config.get("strategy", "hybrid_optimized")
        self.score_threshold = config.get("score_threshold", 4.5)

        # Initialize agents
        self.agents = []
        self.veto_agents = []  # NewsFilter, E8Compliance

        print(f"[ATLAS] Initializing coordinator in {self.mode} mode...")
        print(f"[ATLAS] Strategy: {self.strategy}")
        print(f"[ATLAS] Score threshold: {self.score_threshold}")

        # Decision log
        self.decision_log = []

        # Performance tracking
        self.total_decisions = 0
        self.trades_executed = 0
        self.trades_blocked = 0

    def add_agent(self, agent, is_veto: bool = False):
        """
        Add an agent to the system.

        Args:
            agent: Agent instance
            is_veto: Whether agent has VETO power
        """
        self.agents.append(agent)

        if is_veto:
            self.veto_agents.append(agent)

        print(f"[ATLAS] Added agent: {agent.name} (veto={is_veto}, weight={agent.weight})")

    def analyze_opportunity(self, market_data: Dict) -> Dict:
        """
        Analyze trading opportunity by consulting all agents.

        Args:
            market_data: Current market state

        Returns:
            Dictionary with:
                - decision: "BUY" | "SELL" | "HOLD"
                - score: Final weighted score
                - confidence: Overall confidence (0-1)
                - agent_votes: All individual agent votes
                - reasoning: Why decision was made
        """
        self.total_decisions += 1

        pair = market_data.get("pair", "UNKNOWN")
        price = market_data.get("price", 0)
        timestamp = market_data.get("time", datetime.now())

        print(f"\n[ATLAS] Analyzing {pair} at {price} ({timestamp})")

        # Collect votes from all agents
        agent_votes = {}
        veto_triggered = False
        veto_reason = None

        for agent in self.agents:
            vote, confidence, reasoning = agent.analyze(market_data)

            agent_votes[agent.name] = {
                "vote": vote,
                "confidence": confidence,
                "reasoning": reasoning,
                "weight": agent.weight
            }

            print(f"  [{agent.name}] {vote} (confidence: {confidence:.2f}, weight: {agent.weight:.2f})")

            # Check for VETO
            if agent in self.veto_agents and vote == "BLOCK":
                veto_triggered = True
                veto_reason = reasoning.get("reason", "Veto triggered")
                print(f"    ⚠️ VETO: {veto_reason}")

        # If VETO triggered, block trade immediately
        if veto_triggered:
            self.trades_blocked += 1

            decision = {
                "decision": "HOLD",
                "score": 0,
                "confidence": 0,
                "agent_votes": agent_votes,
                "reasoning": {
                    "type": "VETO",
                    "reason": veto_reason,
                    "blocked_by": [a.name for a in self.veto_agents if agent_votes[a.name]["vote"] == "BLOCK"]
                },
                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                "pair": pair,
                "price": price,
            }

            self.decision_log.append(decision)
            return decision

        # Calculate weighted score
        total_score = 0
        total_confidence = 0
        vote_count = 0

        for agent_name, vote_data in agent_votes.items():
            vote = vote_data["vote"]
            confidence = vote_data["confidence"]
            weight = vote_data["weight"]

            if vote == "BUY":
                total_score += confidence * weight
                total_confidence += confidence
                vote_count += 1
            elif vote == "SELL":
                total_score -= confidence * weight
                total_confidence += confidence
                vote_count += 1
            elif vote == "BOOST":
                # Session timing or other boosters
                total_score *= (1.0 + confidence * 0.2)  # Up to 20% boost
            # "NEUTRAL" and "ALLOW" don't affect score

        # Average confidence
        avg_confidence = total_confidence / vote_count if vote_count > 0 else 0

        # Make final decision
        if total_score >= self.score_threshold:
            final_decision = "BUY"
            self.trades_executed += 1
        elif total_score <= -self.score_threshold:
            final_decision = "SELL"
            self.trades_executed += 1
        else:
            final_decision = "HOLD"

        print(f"\n[ATLAS] Final Decision: {final_decision} (score: {total_score:.2f}, threshold: {self.score_threshold})")

        decision = {
            "decision": final_decision,
            "score": round(total_score, 2),
            "confidence": round(avg_confidence, 2),
            "agent_votes": agent_votes,
            "reasoning": {
                "type": "CONSENSUS",
                "vote_count": vote_count,
                "threshold": self.score_threshold,
            },
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            "pair": pair,
            "price": price,
        }

        self.decision_log.append(decision)
        return decision

    def record_trade_outcome(self, trade_result: Dict):
        """
        Record trade outcome and distribute to agents for learning.

        Args:
            trade_result: Trade result with outcome, P/L, R-multiple, etc.
        """
        print(f"\n[ATLAS] Recording trade outcome: {trade_result.get('outcome')} (P/L: ${trade_result.get('pnl', 0):+,.2f})")

        # Send to all agents that voted for this trade
        for agent in self.agents:
            # Check if agent voted for this trade
            agent_voted = trade_result.get("agent_votes", {}).get(agent.name, {}).get("vote") in ["BUY", "SELL"]

            if agent_voted:
                trade_result_with_vote_flag = {
                    **trade_result,
                    "agent_voted": True
                }
                agent.record_outcome(trade_result_with_vote_flag)

        # Update agent weights (every 50 trades)
        if self.trades_executed % 50 == 0:
            self.adjust_agent_weights()

    def adjust_agent_weights(self):
        """
        Adjust all agent weights based on performance.

        Called after every 50 trades.
        """
        print("\n[ATLAS] Adjusting agent weights based on performance...")

        # Determine learning rate based on training phase
        if self.mode == "paper":
            # More aggressive learning in paper trading
            learning_rate = 0.20
        else:
            # Conservative learning in live trading
            learning_rate = 0.10

        for agent in self.agents:
            old_weight = agent.weight

            # Skip VETO agents (they maintain fixed weight)
            if agent in self.veto_agents:
                continue

            agent.adjust_weight(learning_rate)

            metrics = agent.get_performance_metrics()

            print(f"  [{agent.name}] "
                  f"weight: {old_weight:.2f} → {agent.weight:.2f} | "
                  f"WR: {metrics['win_rate']:.1f}% | "
                  f"Trades: {metrics['total_trades']}")

    def get_agent_leaderboard(self) -> List[Dict]:
        """
        Get agent performance leaderboard.

        Returns:
            List of agents sorted by performance
        """
        agent_metrics = []

        for agent in self.agents:
            metrics = agent.get_performance_metrics()

            # Calculate performance score
            win_rate = metrics['win_rate'] / 100
            avg_r = metrics.get('avg_r_multiple', 0)
            performance_score = (win_rate * 0.6) + (min(avg_r / 3.0, 1.0) * 0.4)

            agent_metrics.append({
                **metrics,
                'performance_score': performance_score
            })

        # Sort by performance score
        sorted_agents = sorted(agent_metrics, key=lambda a: a['performance_score'], reverse=True)

        return sorted_agents

    def get_statistics(self) -> Dict:
        """
        Get overall ATLAS statistics.

        Returns:
            Dictionary with system-wide metrics
        """
        return {
            "mode": self.mode,
            "strategy": self.strategy,
            "total_decisions": self.total_decisions,
            "trades_executed": self.trades_executed,
            "trades_blocked": self.trades_blocked,
            "execution_rate": (self.trades_executed / self.total_decisions * 100) if self.total_decisions > 0 else 0,
            "num_agents": len(self.agents),
            "num_veto_agents": len(self.veto_agents),
        }

    def save_state(self, directory: str):
        """
        Save entire ATLAS state (all agents + coordinator).

        Args:
            directory: Directory to save state files
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save coordinator state
        coordinator_state = {
            "config": self.config,
            "total_decisions": self.total_decisions,
            "trades_executed": self.trades_executed,
            "trades_blocked": self.trades_blocked,
            "decision_log": self.decision_log[-100:],  # Last 100 decisions
        }

        with open(save_dir / "coordinator_state.json", 'w') as f:
            json.dump(coordinator_state, f, indent=2)

        # Save each agent
        for agent in self.agents:
            agent_file = save_dir / f"{agent.name.lower()}_state.json"
            agent.save_state(str(agent_file))

        print(f"[ATLAS] State saved to {directory}")

    def load_state(self, directory: str):
        """
        Load entire ATLAS state.

        Args:
            directory: Directory containing state files
        """
        load_dir = Path(directory)

        if not load_dir.exists():
            print(f"[ATLAS] No saved state found at {directory}")
            return

        # Load coordinator state
        try:
            with open(load_dir / "coordinator_state.json", 'r') as f:
                coordinator_state = json.load(f)

            self.total_decisions = coordinator_state.get("total_decisions", 0)
            self.trades_executed = coordinator_state.get("trades_executed", 0)
            self.trades_blocked = coordinator_state.get("trades_blocked", 0)
            self.decision_log = coordinator_state.get("decision_log", [])

            print(f"[ATLAS] Loaded coordinator state: {self.trades_executed} trades executed")
        except FileNotFoundError:
            print(f"[ATLAS] No coordinator state found")

        # Load each agent
        for agent in self.agents:
            agent_file = load_dir / f"{agent.name.lower()}_state.json"
            agent.load_state(str(agent_file))

    def __repr__(self):
        stats = self.get_statistics()
        return (f"<ATLASCoordinator "
                f"mode={self.mode} "
                f"agents={stats['num_agents']} "
                f"trades={stats['trades_executed']}>")
