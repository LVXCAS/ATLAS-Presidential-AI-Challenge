"""
ATLAS Coordinator (Educational, Simulation-Only)

Collects agent assessments, applies weights, and outputs a risk posture label.
No trade execution or brokerage integration is performed in this repository.
"""

from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path

# Educational scoring output (new). Legacy agents still return (vote, confidence, reasoning).
from agents.base_agent import AgentAssessment
from quant_team_utils import derive_vote_confidence, quant_team_assessment
from simulation_config import SIMULATION_ONLY

# Will import agents when we have them all
# from ..agents.technical_agent import TechnicalAgent
# from ..agents.pattern_recognition_agent import PatternRecognitionAgent
# etc...


class ATLASCoordinator:
    """
    Central coordinator for ATLAS multi-agent system.

    Workflow:
    1. Receive market data (historical/delayed snapshot)
    2. Ask all agents for risk assessments
    3. Apply veto rule for safety-first posture
    4. Calculate weighted risk score
    5. Output a desk-style risk label (GREENLIGHT/WATCH/STAND_DOWN)
    6. Log the decision for educational review
    """

    def __init__(self, config: Dict):
        """
        Initialize coordinator with configuration.

        Args:
            config: Dictionary containing:
                - mode: "education" (default for this repo)
                - strategy: descriptive label for the demo
                - score_threshold: legacy field (not used for trading here)
                - max_position_size_lots: Maximum lot size
                - agents: List of agent configurations
        """
        self.config = config
        self.mode = config.get("mode", "education")
        self.strategy = config.get("strategy", "hybrid_optimized")
        self.score_threshold = config.get("score_threshold", 4.5)

        self.simulation_only = SIMULATION_ONLY
        if not self.simulation_only:
            print("[ATLAS] WARNING: Simulation-only flag disabled; forcing simulation-only mode.")
            self.simulation_only = True

        if self.mode != "education":
            print("[ATLAS] WARNING: This repository is simulation-only; forcing education mode.")
            self.mode = "education"

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

    def _assess_risk_posture(self, market_data: Dict) -> Dict:
        """
        Education-mode assessment: return a risk posture label and explanation.
        """
        self.total_decisions += 1

        pair = market_data.get("pair", "UNKNOWN")
        price = market_data.get("price", 0)
        timestamp = market_data.get("time", datetime.now())

        agent_votes: Dict[str, Dict[str, Any]] = {}
        for agent in self.agents:
            assessment = agent.assess(market_data)
            vote, confidence, reasoning = derive_vote_confidence(
                assessment.score,
                assessment.explanation,
                assessment.details,
            )
            agent_votes[agent.name] = {
                "score": float(assessment.score),
                "explanation": assessment.explanation,
                "details": assessment.details,
                "vote": vote,
                "confidence": confidence,
                "reasoning": reasoning,
                "weight": float(getattr(agent, "weight", 1.0)),
                "is_veto": bool(agent in getattr(self, "veto_agents", [])),
            }

        label, meta = quant_team_assessment(agent_votes)
        score = float(meta.get("aggregated_score", meta.get("risk_score", 0.5)) or 0.5)

        decision = {
            "decision": label,
            "score": round(score, 3),
            "confidence": meta.get("confidence", None),
            "agent_votes": agent_votes,
            "reasoning": {
                "type": "RISK_POSTURE",
                "explanation": meta.get("explanation", ""),
                "drivers": meta.get("drivers", []),
                "risk_flags": meta.get("risk_flags", []),
                "insufficient_agents": meta.get("insufficient_agents", []),
                "method": meta.get("method", ""),
            },
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            "pair": pair,
            "price": price,
        }

        self.decision_log.append(decision)
        return decision

    def analyze_opportunity(self, market_data: Dict) -> Dict:
        """
        Analyze risk posture by consulting all agents.

        Args:
            market_data: Current market state

        Returns:
            Dictionary with:
                - decision: "GREENLIGHT" | "WATCH" | "STAND_DOWN"
                - score: Final weighted risk score
                - confidence: Overall confidence (0-1)
                - agent_votes: All individual agent signals
                - reasoning: Why decision was made
        """
        if self.mode == "education":
            return self._assess_risk_posture(market_data)
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
            raw = agent.analyze(market_data)

            # New-style agents return a normalized 0..1 risk score + explanation.
            if isinstance(raw, AgentAssessment):
                assessment = raw
                vote = "BLOCK" if assessment.score >= 0.80 else ("CAUTION" if assessment.score >= 0.50 else "ALLOW")
                confidence = 1.0 - assessment.score
                reasoning = {"explanation": assessment.explanation, "risk_score": assessment.score, **(assessment.details or {})}
            else:
                # Legacy agent output: (vote, confidence, reasoning).
                vote, confidence, reasoning = raw  # type: ignore[misc]
                assessment = AgentAssessment(
                    score=agent._legacy_to_score(vote=vote, confidence=confidence),  # type: ignore[attr-defined]
                    explanation=agent._legacy_to_explanation(vote=vote, confidence=confidence, reasoning=reasoning),  # type: ignore[attr-defined]
                    details={"legacy_reasoning": reasoning},
                )

            agent_votes[agent.name] = {
                "vote": vote,
                "confidence": confidence,
                "reasoning": reasoning,
                "weight": agent.weight
            }
            agent_votes[agent.name]["risk_score"] = float(assessment.score)
            agent_votes[agent.name]["explanation"] = assessment.explanation

            print(f"  [{agent.name}] {vote} (confidence: {confidence:.2f}, weight: {agent.weight:.2f})")

            # Check for VETO (from veto agents OR high-confidence blocks from any agent)
            if vote == "BLOCK":
                # Always honor BLOCK votes from veto agents
                if agent in self.veto_agents:
                    veto_triggered = True
                    veto_reason = reasoning.get("reason", "Veto triggered")
                    print(f"    ⚠️ VETO: {veto_reason}")
                # Also honor high-confidence BLOCK votes from other agents (e.g., TechnicalAgent trend blocks)
                elif confidence >= 0.85:
                    veto_triggered = True
                    veto_reason = reasoning.get("reason", reasoning.get("message", "High-confidence block"))
                    print(f"    ⚠️ BLOCK: {veto_reason} (from {agent.name})")

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
                    "blocked_by": [name for name, vote_data in agent_votes.items() if vote_data["vote"] == "BLOCK"]
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
