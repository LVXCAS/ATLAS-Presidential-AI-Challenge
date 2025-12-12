"""
RD-Agent Discovery Scheduler

Runs weekly factor discovery cycles to autonomously improve ATLAS.
Can be scheduled via Windows Task Scheduler or run manually.
"""
import asyncio
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from agents.rdagent_factor_discovery import RDAgentFactorDiscovery
from adapters.oanda_adapter import OandaAdapter


async def main():
    """
    Run one complete RD-Agent discovery cycle.

    Steps:
    1. Load historical market data from OANDA (6 months)
    2. Load current ATLAS performance metrics
    3. Run RD-Agent factor discovery
    4. Generate recommendations
    5. Optionally auto-deploy top factors
    """
    print("\n" + "="*80)
    print("RD-AGENT AUTONOMOUS FACTOR DISCOVERY")
    print("Microsoft RD-Agent for ATLAS Trading System")
    print("="*80 + "\n")

    # Initialize RD-Agent - Choose your LLM:

    # Option 1: DeepSeek V3 (RECOMMENDED - 14× cheaper than GPT-4o-mini, same quality)
    # Cost: ~$2-5/month for weekly discovery cycles
    # Add to .env: DEEPSEEK_API_KEY=sk-...
    rd_agent = RDAgentFactorDiscovery(
        llm_model="deepseek-chat"  # DeepSeek V3
    )

    # Option 2: GPT-4o from OpenAI (best quality, higher cost)
    # Cost: ~$10-20/month for weekly discovery
    # Add to .env: OPENAI_API_KEY=sk-...
    # rd_agent = RDAgentFactorDiscovery(
    #     llm_model="gpt-4o"  # or "gpt-4o-mini" for cheaper
    # )

    # Option 3: Local Qwen 2.5 Coder 120B or GPT OSS 120B (FREE, unlimited)
    # Cost: $0/month (just electricity)
    # Requires: Qwen/GPT server running on http://localhost:8000
    # rd_agent = RDAgentFactorDiscovery(
    #     local_llm_url="http://localhost:8000"
    # )

    # Initialize OANDA adapter
    print("[SETUP] Connecting to OANDA...")
    oanda = OandaAdapter()

    # Step 1: Fetch historical data (6 months, H1 timeframe)
    print("[DATA] Fetching 6 months of historical data...")
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    all_data = []

    for pair in pairs:
        print(f"[DATA] Fetching {pair}...")
        candles = oanda.get_candles(pair, 'H1', count=4320)  # 6 months ≈ 4320 hours

        if candles and len(candles) > 0:
            df = pd.DataFrame(candles)
            df['pair'] = pair
            all_data.append(df)
            print(f"[DATA] ✓ {pair}: {len(candles)} candles")
        else:
            print(f"[DATA] ✗ {pair}: No data available")

    if not all_data:
        print("[ERROR] No historical data available - cannot run discovery")
        return

    historical_data = pd.concat(all_data, ignore_index=True)
    print(f"[DATA] Total: {len(historical_data)} data points across {len(pairs)} pairs\n")

    # Step 2: Load current ATLAS performance
    print("[METRICS] Loading current ATLAS performance...")
    performance_metrics = load_performance_metrics()

    print(f"[METRICS] Current Statistics:")
    print(f"  - Total Trades: {performance_metrics.get('total_trades', 0)}")
    print(f"  - Win Rate: {performance_metrics.get('win_rate', 0)*100:.1f}%")
    print(f"  - Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  - Max Drawdown: {performance_metrics.get('max_drawdown', 0)*100:.1f}%\n")

    # Step 3: Run factor discovery
    print("[DISCOVERY] Starting RD-Agent factor discovery cycle...\n")

    discovered_factors = await rd_agent.run_factor_discovery_cycle(
        historical_data=historical_data,
        performance_metrics=performance_metrics
    )

    # Step 4: Generate recommendations
    print(f"\n[RECOMMENDATIONS] Generating deployment recommendations...")
    recommendations = rd_agent.get_deployment_recommendations()

    print(f"\n{'='*80}")
    print("DISCOVERY COMPLETE")
    print(f"{'='*80}\n")

    print(f"Discovered Factors: {len(discovered_factors)}")
    print(f"\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    # Step 5: Show top factors
    if discovered_factors:
        print(f"\nTop 3 Factors:\n")
        for i, factor in enumerate(discovered_factors[:3], 1):
            results = factor['backtest_results']
            print(f"{i}. {factor['name']}")
            print(f"   Sharpe: {results['sharpe_ratio']:.2f} | WR: {results['win_rate']*100:.1f}% | DD: {results['max_drawdown']*100:.1f}%")
            print(f"   Description: {factor['description']}\n")

    print("\n" + "="*80)
    print("Next Steps:")
    print("1. Review discovery report in rdagent_workspace/")
    print("2. Manually test top factors in paper trading")
    print("3. Deploy validated factors to production")
    print("4. Schedule next discovery cycle for next week")
    print("="*80 + "\n")


def load_performance_metrics() -> dict:
    """
    Load current ATLAS performance from coordinator_state.json.

    Returns:
        Dictionary with performance metrics
    """
    state_file = Path(__file__).parent / "learning" / "state" / "coordinator_state.json"

    try:
        with open(state_file, 'r') as f:
            state = json.load(f)

        total_trades = state.get('total_trades_executed', 0)
        wins = state.get('total_wins', 0)
        losses = state.get('total_losses', 0)

        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Calculate Sharpe (simplified - would need full trade history)
        # For now, estimate based on win rate
        sharpe = (win_rate - 0.5) * 4 + 1.0  # Rough estimate

        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'sharpe_ratio': max(0.5, sharpe),
            'max_drawdown': 0.08,  # Would calculate from trade history
            'total_pnl': state.get('total_pnl', 0),
            'avg_trade_pnl': state.get('total_pnl', 0) / total_trades if total_trades > 0 else 0
        }

    except FileNotFoundError:
        print("[WARNING] No coordinator state found - using default metrics")
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 1.0,
            'max_drawdown': 0.0,
            'total_pnl': 0,
            'avg_trade_pnl': 0
        }
    except Exception as e:
        print(f"[WARNING] Error loading performance metrics: {e}")
        return {'total_trades': 0, 'win_rate': 0.0, 'sharpe_ratio': 1.0, 'max_drawdown': 0.0}


if __name__ == "__main__":
    # Run discovery cycle
    asyncio.run(main())
