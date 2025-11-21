"""
ATLAS Paper Training Runner

Starts ATLAS in paper trading mode for 60-day training period.

Usage:
    python run_paper_training.py --phase exploration
    python run_paper_training.py --phase refinement
    python run_paper_training.py --phase validation
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import core components
from core.coordinator import ATLASCoordinator
from core.learning_engine import LearningEngine

# Import agents
from agents.technical_agent import TechnicalAgent
from agents.pattern_recognition_agent import PatternRecognitionAgent
from agents.news_filter_agent import NewsFilterAgent
from agents.e8_compliance_agent import E8ComplianceAgent

# Import OANDA client (if available)
try:
    from BOTS.HYBRID_OANDA_TRADELOCKER import HybridAdapter
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    print("[WARNING] OANDA client not available - using simulation mode")


def load_config(phase: str = "validation") -> dict:
    """
    Load configuration for specified training phase.

    Args:
        phase: "exploration", "refinement", or "validation"

    Returns:
        Configuration dictionary
    """
    config_file = Path(__file__).parent / "config" / "hybrid_optimized.json"

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Override with phase-specific settings
    if phase in ["exploration", "refinement", "validation"]:
        phase_config = config["paper_training"]["phases"][phase]

        config["score_threshold"] = phase_config["score_threshold"]
        config["training_phase"] = phase
        config["learning_rate"] = phase_config["learning_rate"]

        print(f"\n{'='*80}")
        print(f"ATLAS PAPER TRAINING - {phase.upper()} PHASE")
        print(f"{'='*80}")
        print(f"Duration: {phase_config['days']} days")
        print(f"Score Threshold: {phase_config['score_threshold']}")
        print(f"Learning Rate: {phase_config['learning_rate']}")
        print(f"{'='*80}\n")

    return config


def initialize_atlas(config: dict) -> tuple:
    """
    Initialize ATLAS system with all agents.

    Args:
        config: Configuration dictionary

    Returns:
        (coordinator, learning_engine)
    """
    print("[ATLAS] Initializing system...")

    # Create coordinator
    coordinator = ATLASCoordinator(config)

    # Initialize agents
    agents_config = config.get("agents", {})

    # 1. TechnicalAgent
    if agents_config.get("TechnicalAgent", {}).get("enabled", True):
        tech_agent = TechnicalAgent(
            initial_weight=agents_config["TechnicalAgent"]["initial_weight"]
        )
        coordinator.add_agent(tech_agent)

    # 2. PatternRecognitionAgent
    if agents_config.get("PatternRecognitionAgent", {}).get("enabled", True):
        pattern_agent = PatternRecognitionAgent(
            initial_weight=agents_config["PatternRecognitionAgent"]["initial_weight"],
            min_pattern_samples=agents_config["PatternRecognitionAgent"]["min_pattern_samples"]
        )
        coordinator.add_agent(pattern_agent)
    else:
        pattern_agent = None

    # 3. NewsFilterAgent (VETO)
    if agents_config.get("NewsFilterAgent", {}).get("enabled", True):
        news_agent = NewsFilterAgent(
            initial_weight=agents_config["NewsFilterAgent"]["initial_weight"]
        )
        coordinator.add_agent(news_agent, is_veto=True)

    # 4. E8ComplianceAgent (VETO)
    if agents_config.get("E8ComplianceAgent", {}).get("enabled", True):
        e8_agent = E8ComplianceAgent(
            starting_balance=config["e8_challenge"]["starting_balance"],
            initial_weight=agents_config["E8ComplianceAgent"]["initial_weight"]
        )
        coordinator.add_agent(e8_agent, is_veto=True)

    # TODO: Add remaining agents (Volume, MarketRegime, Risk, SessionTiming, Correlation)
    # For now, we have the core agents to demonstrate the system

    # Create learning engine
    learning_engine = LearningEngine(coordinator, pattern_agent)

    print(f"[ATLAS] Initialized with {len(coordinator.agents)} agents")

    return coordinator, learning_engine


def run_simulation_mode(coordinator: ATLASCoordinator, learning_engine: LearningEngine, days: int = 7):
    """
    Run in simulation mode (for demo/testing without live data).

    Args:
        coordinator: ATLAS coordinator
        learning_engine: Learning engine
        days: Number of days to simulate
    """
    print(f"\n[SIMULATION] Running {days}-day simulation...")

    import random

    pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]

    for day in range(days):
        print(f"\n{'='*80}")
        print(f"DAY {day + 1}/{days}")
        print(f"{'='*80}")

        # Simulate 10-20 market opportunities per day
        num_opportunities = random.randint(10, 20)

        for opp in range(num_opportunities):
            # Generate fake market data
            pair = random.choice(pairs)

            market_data = {
                "pair": pair,
                "price": random.uniform(1.0, 1.5),
                "time": datetime.now(),
                "session": random.choice(["asian", "london", "ny"]),
                "indicators": {
                    "rsi": random.uniform(30, 70),
                    "macd": random.uniform(-0.001, 0.001),
                    "macd_signal": random.uniform(-0.001, 0.001),
                    "macd_hist": random.uniform(-0.0005, 0.0005),
                    "ema50": random.uniform(1.0, 1.5),
                    "ema200": random.uniform(1.0, 1.5),
                    "bb_upper": random.uniform(1.0, 1.6),
                    "bb_lower": random.uniform(0.9, 1.4),
                    "bb_middle": random.uniform(1.0, 1.5),
                    "adx": random.uniform(15, 40),
                    "atr": random.uniform(0.0005, 0.002),
                },
                "account_balance": 200000,  # Simulated
                "date": datetime.now().date(),
            }

            # Get decision
            decision = coordinator.analyze_opportunity(market_data)

            # Simulate trade execution if decision was BUY
            if decision["decision"] == "BUY":
                # Simulate random outcome
                outcome = "WIN" if random.random() > 0.45 else "LOSS"  # 55% win rate
                pnl = random.uniform(800, 1500) if outcome == "WIN" else random.uniform(-400, -800)
                r_multiple = random.uniform(1.5, 3.0) if outcome == "WIN" else random.uniform(-0.5, -1.0)

                trade_result = {
                    "outcome": outcome,
                    "pnl": pnl,
                    "r_multiple": r_multiple,
                    "agent_votes": decision["agent_votes"],
                    "entry_conditions": market_data.get("indicators", {}),
                    "pair": pair,
                }

                # Record outcome
                coordinator.record_trade_outcome(trade_result)
                learning_engine.process_trade(trade_result)

        # Daily summary
        stats = coordinator.get_statistics()
        print(f"\n[DAY {day + 1} SUMMARY]")
        print(f"  Total Decisions: {stats['total_decisions']}")
        print(f"  Trades Executed: {stats['trades_executed']}")
        print(f"  Execution Rate: {stats['execution_rate']:.1f}%")

    # Final report
    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")

    report = learning_engine.generate_learning_report()

    print(f"\nLEARNING REPORT:")
    print(f"  Total Trades: {report['total_trades_analyzed']}")
    print(f"  Recent Win Rate: {report['recent_win_rate']:.1f}%")
    print(f"  Patterns Discovered: {report['patterns_discovered']}")
    print(f"  Current Threshold: {report['current_threshold']}")

    print(f"\nAGENT LEADERBOARD:")
    for i, agent_metrics in enumerate(report['agent_leaderboard'], 1):
        print(f"  {i}. {agent_metrics['agent_name']}: {agent_metrics['win_rate']:.1f}% WR, "
              f"{agent_metrics['total_trades']} trades, weight={agent_metrics['current_weight']:.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ATLAS Paper Training")
    parser.add_argument("--phase", choices=["exploration", "refinement", "validation"],
                        default="validation", help="Training phase")
    parser.add_argument("--days", type=int, default=7, help="Number of days to run (simulation mode)")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.phase)

    # Initialize ATLAS
    coordinator, learning_engine = initialize_atlas(config)

    # Load previous state if exists
    state_dir = Path(__file__).parent / "learning" / "state"
    coordinator.load_state(str(state_dir))

    if args.simulation or not OANDA_AVAILABLE:
        # Run simulation
        run_simulation_mode(coordinator, learning_engine, days=args.days)
    else:
        print("[ERROR] Live paper trading mode not yet implemented")
        print("[INFO] Run with --simulation flag for demo")
        return

    # Save state
    coordinator.save_state(str(state_dir))
    learning_engine.save_learning_data(str(state_dir / "learning_data.json"))


if __name__ == "__main__":
    main()
