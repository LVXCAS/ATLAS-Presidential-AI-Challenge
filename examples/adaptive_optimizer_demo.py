
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.adaptive_optimizer_agent import AdaptiveOptimizerAgent

class MockAgent:
    """A mock agent for demonstration purposes."""
    def __init__(self, name, params):
        self.name = name
        self.params = params
        print(f"Initialized MockAgent: {self.name} with params: {self.params}")

    def get_params(self):
        """Gets the agent's parameters."""
        return self.params

    def set_params(self, params):
        """Sets the agent's parameters."""
        self.params = params
        print(f"Updated params for {self.name} to: {self.params}")

class MockBacktestingEngine:
    """A mock backtesting engine for demonstration purposes."""
    def run_backtest(self, agent):
        """Simulates running a backtest."""
        print(f"Running backtest for {agent.name} with params {agent.params}...")
        # In a real engine, this would return actual performance data.
        # We'll return a low Sharpe ratio to trigger the optimizer.
        return {"sharpe_ratio": 0.8, "max_drawdown": 0.15, "pnl": 5000}

def main():
    """Main function to run the demo."""
    print("--- Adaptive Optimizer Agent Demo ---")

    # 1. Create mock agents
    momentum_agent = MockAgent("MomentumAgent", {"lookback_period": 50, "strategy_active": True})
    mean_reversion_agent = MockAgent("MeanReversionAgent", {"reversion_threshold": 0.02, "strategy_active": True})

    # 2. Create a mock backtesting engine
    backtester = MockBacktestingEngine()

    # 3. Initialize the AdaptiveOptimizerAgent with the agents and backtester
    optimizer = AdaptiveOptimizerAgent(
        agents_to_manage=[momentum_agent, mean_reversion_agent],
        backtesting_engine=backtester
    )

    # 4. Run an optimization cycle
    print("\nRunning optimization cycle...")
    optimizer.run_optimization_cycle()

    # 5. Verify that the parameters have been updated
    print("\n--- Verification ---")
    print(f"Momentum agent's new params: {momentum_agent.get_params()}")
    print(f"Mean Reversion agent's new params: {mean_reversion_agent.get_params()}")
    print("\nDemo finished.")

if __name__ == "__main__":
    main()
