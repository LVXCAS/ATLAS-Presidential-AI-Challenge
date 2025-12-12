"""
ATLAS Paper Training Runner

Starts ATLAS in paper trading mode for 60-day training period.

Usage:
    python run_paper_training.py --phase exploration
    python run_paper_training.py --phase refinement
    python run_paper_training.py --phase validation
"""

print("[STARTUP] Script execution started")

import sys
print("[STARTUP] sys imported")
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
from agents.mean_reversion_agent import MeanReversionAgent
from agents.xgboost_ml_agent import XGBoostMLAgent
from agents.sentiment_agent import SentimentAgent
from agents.qlib_research_agent import QlibResearchAgent
from agents.gs_quant_agent import GSQuantAgent
from agents.autogen_rd_agent import AutoGenRDAgent
from agents.monte_carlo_agent import MonteCarloAgent
from agents.market_regime_agent import MarketRegimeAgent
from agents.risk_management_agent import RiskManagementAgent
from agents.session_timing_agent import SessionTimingAgent
from agents.correlation_agent import CorrelationAgent
from agents.multi_timeframe_agent import MultiTimeframeAgent
from agents.volume_liquidity_agent import VolumeLiquidityAgent
from agents.support_resistance_agent import SupportResistanceAgent
from agents.divergence_agent import DivergenceAgent

# Import OANDA client
try:
    from adapters.oanda_adapter import OandaAdapter
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    print("[WARNING] OANDA adapter not available - using simulation mode")


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
    print("[DEBUG] Creating coordinator...")

    # Create coordinator
    coordinator = ATLASCoordinator(config)
    print("[DEBUG] Coordinator created")

    # Initialize agents
    agents_config = config.get("agents", {})
    print(f"[DEBUG] Loading {len(agents_config)} agent types...")

    # 1. TechnicalAgent
    print("[DEBUG] Loading TechnicalAgent...")
    if agents_config.get("TechnicalAgent", {}).get("enabled", True):
        tech_config = agents_config["TechnicalAgent"]
        tech_agent = TechnicalAgent(
            initial_weight=tech_config["initial_weight"]
        )
        coordinator.add_agent(tech_agent, is_veto=tech_config.get("is_veto", False))
    print("[DEBUG] TechnicalAgent loaded")

    # 2. PatternRecognitionAgent
    print("[DEBUG] Loading PatternRecognitionAgent...")
    if agents_config.get("PatternRecognitionAgent", {}).get("enabled", True):
        pattern_agent = PatternRecognitionAgent(
            initial_weight=agents_config["PatternRecognitionAgent"]["initial_weight"],
            min_pattern_samples=agents_config["PatternRecognitionAgent"]["min_pattern_samples"]
        )
        coordinator.add_agent(pattern_agent)
    else:
        pattern_agent = None
    print("[DEBUG] PatternRecognitionAgent loaded")

    # 3. NewsFilterAgent (VETO)
    print("[DEBUG] Loading NewsFilterAgent...")
    if agents_config.get("NewsFilterAgent", {}).get("enabled", True):
        news_agent = NewsFilterAgent(
            initial_weight=agents_config["NewsFilterAgent"]["initial_weight"]
        )
        coordinator.add_agent(news_agent, is_veto=True)
    print("[DEBUG] NewsFilterAgent loaded")

    # 3.5 MeanReversionAgent (for range-bound markets)
    print("[DEBUG] Loading MeanReversionAgent...")
    if agents_config.get("MeanReversionAgent", {}).get("enabled", True):
        mean_reversion_agent = MeanReversionAgent(
            initial_weight=agents_config.get("MeanReversionAgent", {}).get("initial_weight", 1.5)
        )
        coordinator.add_agent(mean_reversion_agent)
    print("[DEBUG] MeanReversionAgent loaded")

    # 4. XGBoostMLAgent (Machine Learning predictions)
    print("[DEBUG] Loading XGBoostMLAgent...")
    if agents_config.get("XGBoostMLAgent", {}).get("enabled", True):
        xgboost_agent = XGBoostMLAgent(
            initial_weight=agents_config.get("XGBoostMLAgent", {}).get("initial_weight", 2.5)
        )
        coordinator.add_agent(xgboost_agent)
    print("[DEBUG] XGBoostMLAgent loaded")

    # 5. SentimentAgent (FinBERT news sentiment)
    print("[DEBUG] Loading SentimentAgent...")
    if agents_config.get("SentimentAgent", {}).get("enabled", True):
        sentiment_agent = SentimentAgent(
            initial_weight=agents_config.get("SentimentAgent", {}).get("initial_weight", 1.5)
        )
        coordinator.add_agent(sentiment_agent)
    print("[DEBUG] SentimentAgent loaded")

    # 6. QlibResearchAgent (Microsoft Qlib - 1000+ factors)
    print("[DEBUG] Loading QlibResearchAgent...")
    if agents_config.get("QlibResearchAgent", {}).get("enabled", True):
        qlib_agent = QlibResearchAgent(
            initial_weight=agents_config["QlibResearchAgent"]["initial_weight"]
        )
        coordinator.add_agent(qlib_agent)
    print("[DEBUG] QlibResearchAgent loaded")

    # 7. GSQuantAgent (Goldman Sachs risk models)
    print("[DEBUG] Loading GSQuantAgent...")
    if agents_config.get("GSQuantAgent", {}).get("enabled", True):
        gs_agent = GSQuantAgent(
            initial_weight=agents_config["GSQuantAgent"]["initial_weight"]
        )
        coordinator.add_agent(gs_agent)
    print("[DEBUG] GSQuantAgent loaded")

    # 8. AutoGenRDAgent (Microsoft AutoGen - strategy discovery)
    # Note: R&D agent runs in background, doesn't vote on trades
    print("[DEBUG] Loading AutoGenRDAgent...")
    if agents_config.get("AutoGenRDAgent", {}).get("enabled", True):
        rd_agent = AutoGenRDAgent(
            initial_weight=agents_config["AutoGenRDAgent"]["initial_weight"]
        )
        coordinator.add_agent(rd_agent)
    print("[DEBUG] AutoGenRDAgent loaded")

    # 9. MarketRegimeAgent (bull/bear/range detection)
    print("[DEBUG] Loading MarketRegimeAgent...")
    if agents_config.get("MarketRegimeAgent", {}).get("enabled", True):
        regime_agent = MarketRegimeAgent(
            initial_weight=agents_config.get("MarketRegimeAgent", {}).get("initial_weight", 1.2)
        )
        coordinator.add_agent(regime_agent)
    print("[DEBUG] MarketRegimeAgent loaded")

    # 10. RiskManagementAgent (position sizing, VaR)
    print("[DEBUG] Loading RiskManagementAgent...")
    if agents_config.get("RiskManagementAgent", {}).get("enabled", True):
        risk_agent = RiskManagementAgent(
            initial_weight=agents_config.get("RiskManagementAgent", {}).get("initial_weight", 1.5)
        )
        coordinator.add_agent(risk_agent)
    print("[DEBUG] RiskManagementAgent loaded")

    # 11. SessionTimingAgent (London/NY/Asian sessions)
    print("[DEBUG] Loading SessionTimingAgent...")
    if agents_config.get("SessionTimingAgent", {}).get("enabled", True):
        session_agent = SessionTimingAgent(
            initial_weight=agents_config.get("SessionTimingAgent", {}).get("initial_weight", 1.2)
        )
        coordinator.add_agent(session_agent)
    print("[DEBUG] SessionTimingAgent loaded")

    # 12. CorrelationAgent (prevent over-exposure)
    print("[DEBUG] Loading CorrelationAgent...")
    if agents_config.get("CorrelationAgent", {}).get("enabled", True):
        correlation_agent = CorrelationAgent(
            initial_weight=agents_config.get("CorrelationAgent", {}).get("initial_weight", 1.0)
        )
        coordinator.add_agent(correlation_agent)
    print("[DEBUG] CorrelationAgent loaded")

    # Skip E8ComplianceAgent and MonteCarloAgent (disabled per user request)

    # 8. MonteCarloAgent (Real-time probabilistic risk simulation)
    print("[DEBUG] Loading MonteCarloAgent...")
    if agents_config.get("MonteCarloAgent", {}).get("enabled", True):
        mc_config = agents_config.get("MonteCarloAgent", {})
        mc_agent = MonteCarloAgent(
            initial_weight=mc_config.get("initial_weight", 2.0),
            is_veto=mc_config.get("is_veto", False)
        )
        # Set custom parameters if specified
        if "num_simulations" in mc_config:
            mc_agent.num_simulations = mc_config["num_simulations"]
        if "min_win_probability" in mc_config:
            mc_agent.min_win_probability = mc_config["min_win_probability"]
        if "max_dd_risk" in mc_config:
            mc_agent.max_acceptable_dd_risk = mc_config["max_dd_risk"]

        coordinator.add_agent(mc_agent, is_veto=mc_config.get("is_veto", False))
    print("[DEBUG] MonteCarloAgent loaded")

    # 13. MultiTimeframeAgent (M5/M15/H1/H4/D1 trend confirmation)
    print("[DEBUG] Loading MultiTimeframeAgent...")
    if agents_config.get("MultiTimeframeAgent", {}).get("enabled", True):
        mtf_agent = MultiTimeframeAgent(
            initial_weight=agents_config.get("MultiTimeframeAgent", {}).get("initial_weight", 2.0)
        )
        coordinator.add_agent(mtf_agent)
    print("[DEBUG] MultiTimeframeAgent loaded")

    # 14. VolumeLiquidityAgent (spread detection, institutional flows)
    print("[DEBUG] Loading VolumeLiquidityAgent...")
    if agents_config.get("VolumeLiquidityAgent", {}).get("enabled", True):
        vol_agent = VolumeLiquidityAgent(
            initial_weight=agents_config.get("VolumeLiquidityAgent", {}).get("initial_weight", 1.8)
        )
        coordinator.add_agent(vol_agent)
    print("[DEBUG] VolumeLiquidityAgent loaded")

    # 15. SupportResistanceAgent (key price level trading)
    print("[DEBUG] Loading SupportResistanceAgent...")
    if agents_config.get("SupportResistanceAgent", {}).get("enabled", True):
        sr_agent = SupportResistanceAgent(
            initial_weight=agents_config.get("SupportResistanceAgent", {}).get("initial_weight", 1.7)
        )
        coordinator.add_agent(sr_agent)
    print("[DEBUG] SupportResistanceAgent loaded")

    # 16. DivergenceAgent (RSI/MACD divergence detection)
    print("[DEBUG] Loading DivergenceAgent...")
    if agents_config.get("DivergenceAgent", {}).get("enabled", True):
        div_agent = DivergenceAgent(
            initial_weight=agents_config.get("DivergenceAgent", {}).get("initial_weight", 1.6)
        )
        coordinator.add_agent(div_agent)
    print("[DEBUG] DivergenceAgent loaded")

    # Create learning engine
    print("[DEBUG] Creating learning engine...")
    learning_engine = LearningEngine(coordinator, pattern_agent)
    print("[DEBUG] Learning engine created")

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
    parser.add_argument("--fast-scan", action="store_true", help="Use 1-minute scans for testing (default: 5 min)")

    args = parser.parse_args()

    print("[DEBUG] Step 1: Parsing args complete")

    # Load configuration
    config = load_config(args.phase)
    print("[DEBUG] Step 2: Config loaded")

    # Initialize ATLAS
    coordinator, learning_engine = initialize_atlas(config)
    print("[DEBUG] Step 3: ATLAS initialized")

    # Load previous state if exists
    state_dir = Path(__file__).parent / "learning" / "state"
    print(f"[DEBUG] Step 4: Loading state from {state_dir}")
    coordinator.load_state(str(state_dir))
    print("[DEBUG] Step 5: State loaded successfully")

    if args.simulation or not OANDA_AVAILABLE:
        # Run simulation
        run_simulation_mode(coordinator, learning_engine, days=args.days)
    else:
        # Run live paper trading with OANDA
        from live_trader import run_live_trading
        run_live_trading(coordinator, learning_engine, days=args.days, fast_scan=args.fast_scan)

    # Save state
    coordinator.save_state(str(state_dir))
    learning_engine.save_learning_data(str(state_dir / "learning_data.json"))


if __name__ == "__main__":
    main()
