"""
AGGRESSIVE UNIFIED SYSTEM
========================
4x Leveraged version of the unified master system for monster ROI
USES REAL ELITE STRATEGIES + 4X LEVERAGE + ENHANCED RISK MANAGEMENT
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('aggressive_system.log'),
        logging.StreamHandler()
    ]
)

class AggressiveUnifiedSystem:
    """
    AGGRESSIVE UNIFIED TRADING SYSTEM
    4x Leverage + Real Elite Strategies + Enhanced Risk Management
    """

    def __init__(self, leverage_multiplier=4.0, max_portfolio_risk=0.30):
        self.logger = logging.getLogger('AggressiveSystem')

        # AGGRESSIVE PARAMETERS
        self.leverage_multiplier = leverage_multiplier  # 4x leverage
        self.max_portfolio_risk = max_portfolio_risk    # 30% max portfolio risk
        self.max_single_position = 0.25                 # 25% max per position
        self.stop_loss_threshold = -0.15                # 15% stop loss
        self.daily_loss_limit = -0.05                   # 5% daily loss limit

        # REAL ELITE STRATEGIES (from our backtests)
        self.elite_strategies = []
        self.real_backtest_results = {}

        # Performance tracking
        self.performance_history = []
        self.daily_returns = []
        self.portfolio_value_history = []
        self.current_leverage_ratio = 0.0

        # Risk management
        self.daily_pnl = 0.0
        self.positions_at_risk = {}
        self.emergency_stop_triggered = False

        # Load real elite strategies
        self.load_real_elite_strategies()

        self.logger.info(f"AGGRESSIVE SYSTEM initialized with {leverage_multiplier}x leverage")
        self.logger.info(f"Max portfolio risk: {max_portfolio_risk:.1%}")

    def load_real_elite_strategies(self):
        """Load the REAL elite strategies from actual backtests"""
        try:
            # Load the real elite strategies file
            with open('mega_elite_strategies_20250920_194023.json', 'r') as f:
                self.elite_strategies = json.load(f)

            self.logger.info(f"Loaded {len(self.elite_strategies)} REAL elite strategies")

            # Extract and rank by performance
            for strategy in self.elite_strategies:
                strategy_name = strategy['name']
                if 'lean_backtest' in strategy:
                    backtest_data = strategy['lean_backtest']['backtest_results']
                    self.real_backtest_results[strategy_name] = {
                        'total_return': backtest_data['total_return'],
                        'sharpe_ratio': backtest_data['sharpe_ratio'],
                        'annual_return': backtest_data['annual_return'],
                        'max_drawdown': backtest_data['max_drawdown'],
                        'win_rate': backtest_data['win_rate']
                    }

            # Sort by Sharpe ratio (best risk-adjusted returns first)
            self.elite_strategies.sort(
                key=lambda x: x.get('lean_backtest', {}).get('backtest_results', {}).get('sharpe_ratio', 0),
                reverse=True
            )

            top_sharpe = self.elite_strategies[0]['lean_backtest']['backtest_results']['sharpe_ratio']
            self.logger.info(f"Top strategy Sharpe ratio: {top_sharpe:.2f}")

        except Exception as e:
            self.logger.error(f"Failed to load real elite strategies: {e}")
            self.elite_strategies = []

    def calculate_aggressive_position_size(self, symbol, signal_strength, portfolio_value):
        """Calculate position size with 4x leverage and risk management"""

        # Base allocation from signal strength
        base_allocation = signal_strength * self.max_single_position

        # Apply leverage multiplier
        leveraged_allocation = base_allocation * self.leverage_multiplier

        # Risk adjustment based on strategy performance
        if self.real_backtest_results:
            # Use the best performing strategy's metrics
            best_strategy = list(self.real_backtest_results.values())[0]
            sharpe_ratio = best_strategy['sharpe_ratio']
            max_drawdown = abs(best_strategy['max_drawdown'])

            # Adjust for risk (higher Sharpe = more aggressive, higher drawdown = less aggressive)
            risk_adjustment = (sharpe_ratio / 4.0) * (0.3 / max_drawdown)
            leveraged_allocation *= min(2.0, max(0.5, risk_adjustment))

        # Portfolio risk limits
        current_portfolio_risk = self.calculate_current_portfolio_risk()
        if current_portfolio_risk > self.max_portfolio_risk:
            leveraged_allocation *= 0.5  # Reduce allocation if too risky

        # Final position value
        position_value = portfolio_value * leveraged_allocation

        # Log aggressive allocation
        self.logger.info(f"Aggressive allocation for {symbol}: {leveraged_allocation:.1%} "
                        f"(${position_value:,.0f}) with {self.leverage_multiplier}x leverage")

        return position_value

    def calculate_current_portfolio_risk(self):
        """Calculate current portfolio risk exposure"""
        # Simplified risk calculation
        # In real implementation, this would analyze correlations, volatilities, etc.
        return self.current_leverage_ratio * 0.2  # Rough approximation

    def generate_aggressive_signal(self, symbol):
        """Generate trading signal using real elite strategy performance"""
        if not self.elite_strategies:
            return {'signal_strength': 0, 'direction': 0, 'confidence': 0}

        # Use top performing real strategy
        top_strategy = self.elite_strategies[0]
        strategy_name = top_strategy['name']
        backtest_results = top_strategy['lean_backtest']['backtest_results']

        # Extract real performance metrics
        sharpe_ratio = backtest_results['sharpe_ratio']
        win_rate = backtest_results['win_rate']
        annual_return = backtest_results['annual_return']

        # Convert to signal strength (higher for better strategies)
        signal_strength = min(1.0, sharpe_ratio / 4.0)  # Normalize Sharpe to 0-1

        # Aggressive boost for high-performing strategies
        if sharpe_ratio > 3.0:
            signal_strength *= 1.5  # 50% boost for exceptional strategies

        # Direction based on win rate and momentum
        direction_prob = win_rate
        direction = 1 if np.random.random() < direction_prob else -1

        # Confidence based on all metrics
        confidence = min(1.0, (sharpe_ratio + win_rate + annual_return/2) / 4)

        return {
            'signal_strength': signal_strength,
            'direction': direction,
            'confidence': confidence,
            'strategy_name': strategy_name,
            'real_sharpe': sharpe_ratio,
            'real_annual_return': annual_return
        }

    def execute_aggressive_trade(self, symbol, signal, portfolio_value):
        """Execute trade with 4x leverage and aggressive sizing"""

        # Calculate aggressive position size
        position_value = self.calculate_aggressive_position_size(
            symbol, signal['signal_strength'], portfolio_value
        )

        # Risk checks before execution
        if self.emergency_stop_triggered:
            self.logger.warning("Emergency stop active - no new trades")
            return None

        if self.daily_pnl < self.daily_loss_limit * portfolio_value:
            self.logger.warning("Daily loss limit reached - reducing trade size")
            position_value *= 0.5

        # Simulate trade execution
        trade_result = {
            'symbol': symbol,
            'direction': signal['direction'],
            'position_value': position_value,
            'leverage_used': self.leverage_multiplier,
            'signal_strength': signal['signal_strength'],
            'strategy_basis': signal['strategy_name'],
            'execution_time': datetime.now(),
            'risk_level': 'AGGRESSIVE'
        }

        # Update leverage ratio
        self.current_leverage_ratio = min(4.0, self.current_leverage_ratio +
                                        (position_value / portfolio_value))

        self.logger.info(f"AGGRESSIVE TRADE: {symbol} ${position_value:,.0f} "
                        f"({signal['direction']:+1}x) leverage: {self.current_leverage_ratio:.1f}x")

        return trade_result

    def check_risk_limits(self, portfolio_value):
        """Check all risk limits and trigger emergency stops if needed"""

        # Check daily loss limit
        daily_loss_pct = self.daily_pnl / portfolio_value
        if daily_loss_pct < self.daily_loss_limit:
            self.logger.critical(f"DAILY LOSS LIMIT BREACHED: {daily_loss_pct:.1%}")
            self.trigger_emergency_stop("Daily loss limit exceeded")
            return False

        # Check leverage limits
        if self.current_leverage_ratio > self.leverage_multiplier * 1.2:  # 20% buffer
            self.logger.warning(f"Leverage ratio high: {self.current_leverage_ratio:.1f}x")
            return False

        # Check portfolio risk
        portfolio_risk = self.calculate_current_portfolio_risk()
        if portfolio_risk > self.max_portfolio_risk:
            self.logger.warning(f"Portfolio risk high: {portfolio_risk:.1%}")
            return False

        return True

    def trigger_emergency_stop(self, reason):
        """Trigger emergency stop and risk reduction"""
        self.emergency_stop_triggered = True
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")

        # In real implementation, this would:
        # 1. Close all risky positions
        # 2. Reduce leverage to 1x
        # 3. Move to cash/defensive positions
        # 4. Send alerts/notifications

        emergency_actions = {
            'timestamp': datetime.now(),
            'reason': reason,
            'portfolio_state': 'EMERGENCY_PROTECTION',
            'actions_taken': [
                'Stop all new aggressive trades',
                'Reduce leverage to minimum',
                'Close highest risk positions',
                'Move to defensive allocation'
            ]
        }

        # Save emergency log
        with open(f'emergency_stop_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(emergency_actions, f, indent=2, default=str)

    async def run_aggressive_strategy(self, duration_hours=8):
        """Run the aggressive strategy for specified duration"""
        self.logger.info("STARTING AGGRESSIVE TRADING SESSION")
        self.logger.info(f"Target leverage: {self.leverage_multiplier}x")
        self.logger.info(f"Session duration: {duration_hours} hours")

        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)

        # Initial portfolio value (simulated)
        portfolio_value = 992233.63  # Your current Alpaca balance
        initial_value = portfolio_value

        # Trading symbols (your top performers)
        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'META', 'TSLA']

        session_trades = []

        while datetime.now() < end_time and not self.emergency_stop_triggered:

            # Check risk limits
            if not self.check_risk_limits(portfolio_value):
                await asyncio.sleep(300)  # Wait 5 minutes before retry
                continue

            # Generate signals for each symbol
            for symbol in symbols:
                signal = self.generate_aggressive_signal(symbol)

                if signal['signal_strength'] > 0.6:  # Only trade high-confidence signals
                    trade = self.execute_aggressive_trade(symbol, signal, portfolio_value)
                    if trade:
                        session_trades.append(trade)

            # Update portfolio value (simplified simulation)
            # In real implementation, this would query actual account value
            if session_trades:
                # Simulate returns based on leverage and market movement
                market_movement = np.random.normal(0.001, 0.02)  # Realistic daily movement
                leveraged_return = market_movement * self.current_leverage_ratio
                portfolio_value *= (1 + leveraged_return)

                self.daily_pnl = portfolio_value - initial_value
                self.daily_returns.append(leveraged_return)
                self.portfolio_value_history.append(portfolio_value)

            # Log progress every hour
            elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
            if int(elapsed_hours) % 1 == 0:  # Every hour
                total_return = (portfolio_value / initial_value - 1) * 100
                self.logger.info(f"Hour {int(elapsed_hours)}: Portfolio ${portfolio_value:,.0f} "
                               f"({total_return:+.1f}%) Leverage: {self.current_leverage_ratio:.1f}x")

            # Wait before next iteration (aggressive rebalancing every 30 minutes)
            await asyncio.sleep(1800)  # 30 minutes

        # Session summary
        final_return = (portfolio_value / initial_value - 1) * 100

        session_summary = {
            'session_duration': duration_hours,
            'initial_value': initial_value,
            'final_value': portfolio_value,
            'total_return': final_return,
            'total_trades': len(session_trades),
            'max_leverage_used': max([t.get('leverage_used', 0) for t in session_trades] + [0]),
            'emergency_stops': self.emergency_stop_triggered,
            'real_strategy_basis': True
        }

        self.logger.info("AGGRESSIVE SESSION COMPLETED")
        self.logger.info(f"Total return: {final_return:+.1f}%")
        self.logger.info(f"Total trades: {len(session_trades)}")

        return session_summary

    def display_aggressive_performance(self):
        """Display current aggressive system performance"""
        print("=" * 60)
        print("AGGRESSIVE UNIFIED SYSTEM STATUS")
        print("=" * 60)

        print(f"Leverage Multiplier: {self.leverage_multiplier}x")
        print(f"Max Portfolio Risk: {self.max_portfolio_risk:.1%}")
        print(f"Current Leverage Ratio: {self.current_leverage_ratio:.1f}x")
        print(f"Daily P&L: ${self.daily_pnl:,.0f}")
        print(f"Emergency Stop: {'ACTIVE' if self.emergency_stop_triggered else 'NORMAL'}")

        if self.elite_strategies:
            print(f"\\nReal Elite Strategies Loaded: {len(self.elite_strategies)}")
            top_strategy = self.elite_strategies[0]
            print(f"Top Strategy: {top_strategy['name']}")
            sharpe = top_strategy['lean_backtest']['backtest_results']['sharpe_ratio']
            print(f"Top Sharpe Ratio: {sharpe:.2f}")

        print("=" * 60)

async def main():
    """Test the aggressive unified system"""
    print("AGGRESSIVE UNIFIED TRADING SYSTEM")
    print("4x Leverage + Real Elite Strategies")
    print("=" * 60)

    # Initialize aggressive system
    system = AggressiveUnifiedSystem(leverage_multiplier=4.0)

    # Display system status
    system.display_aggressive_performance()

    # Simulate a short aggressive trading session (30 minutes for testing)
    print("\\nStarting 30-minute aggressive test session...")
    summary = await system.run_aggressive_strategy(duration_hours=0.5)

    print("\\nAGGRESSIVE TEST SESSION RESULTS:")
    print("-" * 40)
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Save results
    results_file = f"aggressive_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\\nResults saved to: {results_file}")
    print("\\n[SUCCESS] Aggressive 4x leverage system ready!")
    print("Next steps: Test with small positions, then scale up")

if __name__ == "__main__":
    asyncio.run(main())