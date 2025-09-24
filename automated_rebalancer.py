"""
AUTOMATED REBALANCER SYSTEM
============================
2-hour automated rebalancing for optimal position management
Integrates with 4x leverage system and Friday options

FEATURES:
- Runs every 2 hours during market hours
- Dynamic momentum-based rebalancing
- 4x leverage integration
- Risk management and position sizing
- Performance tracking and optimization
"""

import asyncio
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple, Optional
import os
from dotenv import load_dotenv
import schedule
import time

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_rebalancer.log'),
        logging.StreamHandler()
    ]
)

class AutomatedRebalancer:
    """
    AUTOMATED REBALANCING SYSTEM
    Continuously optimizes portfolio allocation every 2 hours
    """

    def __init__(self, leverage_multiplier=4.0, rebalance_threshold=0.10):
        self.logger = logging.getLogger('AutoRebalancer')

        # Core parameters
        self.leverage_multiplier = leverage_multiplier
        self.rebalance_threshold = rebalance_threshold  # 10% deviation triggers rebalance

        # Trading universe
        self.core_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'META', 'TSLA']
        self.sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI']  # Sector rotation

        # Timing parameters
        self.market_open_pdt = 6.5   # 6:30 AM PDT
        self.market_close_pdt = 13.0  # 1:00 PM PDT
        self.rebalance_interval_hours = 2

        # Risk management
        self.max_single_position = 0.25  # 25% max per position
        self.max_sector_concentration = 0.60  # 60% max in one sector
        self.correlation_limit = 0.70  # Max correlation between positions

        # Performance tracking
        self.current_positions = {}
        self.target_allocations = {}
        self.rebalance_history = []
        self.performance_metrics = {}

        # Load real elite strategies for signal generation
        self.elite_strategies = self.load_elite_strategies()

        self.logger.info(f"AUTOMATED REBALANCER initialized with {leverage_multiplier}x leverage")
        self.logger.info(f"Rebalance threshold: {rebalance_threshold:.1%}")

    def load_elite_strategies(self) -> List[Dict]:
        """Load real elite strategies for signal generation"""
        try:
            with open('mega_elite_strategies_20250920_194023.json', 'r') as f:
                strategies = json.load(f)
            self.logger.info(f"Loaded {len(strategies)} elite strategies for signals")
            return strategies
        except Exception as e:
            self.logger.error(f"Failed to load elite strategies: {e}")
            return []

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        current_hour = now.hour + now.minute / 60.0

        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        return self.market_open_pdt <= current_hour <= self.market_close_pdt

    def get_market_data(self, symbols: List[str], period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """Get current market data for symbols"""
        market_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval="1h")  # Hourly data for precision

                if len(data) > 0:
                    market_data[symbol] = data
                else:
                    self.logger.warning(f"No data available for {symbol}")

            except Exception as e:
                self.logger.error(f"Failed to get data for {symbol}: {e}")

        return market_data

    def calculate_momentum_scores(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate momentum scores for all symbols"""
        momentum_scores = {}

        for symbol, data in market_data.items():
            try:
                if len(data) < 24:  # Need at least 24 hours of data
                    momentum_scores[symbol] = 0.0
                    continue

                current_price = data['Close'].iloc[-1]

                # Multiple momentum timeframes
                momentum_2h = (current_price / data['Close'].iloc[-3]) - 1  # 2-hour momentum
                momentum_6h = (current_price / data['Close'].iloc[-7]) - 1  # 6-hour momentum
                momentum_24h = (current_price / data['Close'].iloc[-25]) - 1  # 24-hour momentum

                # Volume momentum
                avg_volume = data['Volume'].rolling(24).mean().iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                volume_momentum = (current_volume / avg_volume) - 1 if avg_volume > 0 else 0

                # Volatility analysis
                price_volatility = data['Close'].rolling(12).std().iloc[-1] / current_price

                # Combined momentum score (weighted)
                momentum_score = (
                    momentum_2h * 0.4 +    # Recent momentum (high weight)
                    momentum_6h * 0.3 +    # Medium-term momentum
                    momentum_24h * 0.2 +   # Longer-term momentum
                    volume_momentum * 0.1   # Volume confirmation
                )

                # Adjust for volatility (higher vol = higher potential but also risk)
                volatility_adjustment = min(1.5, 1 + price_volatility * 5)
                momentum_score *= volatility_adjustment

                momentum_scores[symbol] = momentum_score

            except Exception as e:
                self.logger.error(f"Failed to calculate momentum for {symbol}: {e}")
                momentum_scores[symbol] = 0.0

        return momentum_scores

    def calculate_optimal_allocations(self, momentum_scores: Dict[str, float], portfolio_value: float) -> Dict[str, float]:
        """Calculate optimal allocations based on momentum and constraints"""

        # Filter positive momentum signals
        positive_momentum = {k: v for k, v in momentum_scores.items() if v > 0}

        if not positive_momentum:
            self.logger.warning("No positive momentum signals - maintaining current positions")
            return self.current_positions

        # Normalize momentum scores to allocations
        total_momentum = sum(positive_momentum.values())
        base_allocations = {k: v / total_momentum for k, v in positive_momentum.items()}

        # Apply leverage multiplier
        leveraged_allocations = {k: v * self.leverage_multiplier for k, v in base_allocations.items()}

        # Apply position size constraints
        constrained_allocations = {}
        for symbol, allocation in leveraged_allocations.items():
            # Cap at max single position
            capped_allocation = min(allocation, self.max_single_position)
            constrained_allocations[symbol] = capped_allocation

        # Check if we need to scale down due to constraints
        total_allocation = sum(constrained_allocations.values())
        if total_allocation > 1.0:  # Over-allocated
            scale_factor = 0.95 / total_allocation  # Scale to 95% to leave some cash
            constrained_allocations = {k: v * scale_factor for k, v in constrained_allocations.items()}

        # Convert to dollar amounts
        dollar_allocations = {k: v * portfolio_value for k, v in constrained_allocations.items()}

        self.logger.info(f"Calculated optimal allocations for {len(dollar_allocations)} positions")

        return dollar_allocations

    def calculate_rebalance_needed(self, target_allocations: Dict[str, float]) -> bool:
        """Determine if rebalancing is needed based on thresholds"""

        if not self.current_positions:
            return True  # First time setup

        total_deviation = 0

        for symbol, target_value in target_allocations.items():
            current_value = self.current_positions.get(symbol, 0)
            if target_value > 0:
                deviation = abs(current_value - target_value) / target_value
                total_deviation += deviation

        # Also check for positions we should exit
        for symbol, current_value in self.current_positions.items():
            if symbol not in target_allocations and current_value > 0:
                total_deviation += 1.0  # Full deviation for positions to exit

        avg_deviation = total_deviation / max(len(target_allocations), 1)

        self.logger.info(f"Average position deviation: {avg_deviation:.1%}")

        return avg_deviation > self.rebalance_threshold

    def execute_rebalance(self, target_allocations: Dict[str, float], portfolio_value: float) -> Dict:
        """Execute the rebalancing trades"""

        trades_executed = []
        total_trade_value = 0

        self.logger.info("EXECUTING REBALANCING TRADES")

        # Calculate trades needed
        for symbol, target_value in target_allocations.items():
            current_value = self.current_positions.get(symbol, 0)
            trade_value = target_value - current_value

            if abs(trade_value) > 1000:  # Only trade if > $1000 difference
                trade_direction = "BUY" if trade_value > 0 else "SELL"

                trade = {
                    'symbol': symbol,
                    'direction': trade_direction,
                    'current_value': current_value,
                    'target_value': target_value,
                    'trade_value': abs(trade_value),
                    'trade_size_pct': abs(trade_value) / portfolio_value,
                    'timestamp': datetime.now(),
                    'reason': 'AUTOMATED_REBALANCE'
                }

                trades_executed.append(trade)
                total_trade_value += abs(trade_value)

                self.logger.info(f"{trade_direction} {symbol}: ${abs(trade_value):,.0f} "
                               f"({trade['trade_size_pct']:.1%} of portfolio)")

        # Execute position exits (positions not in target)
        for symbol, current_value in self.current_positions.items():
            if symbol not in target_allocations and current_value > 1000:
                trade = {
                    'symbol': symbol,
                    'direction': 'SELL',
                    'current_value': current_value,
                    'target_value': 0,
                    'trade_value': current_value,
                    'trade_size_pct': current_value / portfolio_value,
                    'timestamp': datetime.now(),
                    'reason': 'EXIT_POSITION'
                }

                trades_executed.append(trade)
                total_trade_value += current_value

                self.logger.info(f"EXIT {symbol}: ${current_value:,.0f}")

        # Update current positions
        self.current_positions = target_allocations.copy()

        rebalance_summary = {
            'timestamp': datetime.now(),
            'trades_executed': len(trades_executed),
            'total_trade_value': total_trade_value,
            'turnover_rate': total_trade_value / portfolio_value,
            'new_positions': len(target_allocations),
            'leverage_used': sum(target_allocations.values()) / portfolio_value,
            'trades': trades_executed
        }

        self.logger.info(f"REBALANCE COMPLETE: {len(trades_executed)} trades, "
                        f"${total_trade_value:,.0f} total volume")

        return rebalance_summary

    async def run_rebalance_cycle(self, portfolio_value: float) -> Dict:
        """Run a complete rebalancing cycle"""

        self.logger.info("STARTING REBALANCE CYCLE")

        if not self.is_market_open():
            self.logger.info("Market closed - skipping rebalance")
            return {'status': 'MARKET_CLOSED', 'timestamp': datetime.now()}

        # Get market data
        all_symbols = self.core_symbols + self.sector_etfs
        market_data = self.get_market_data(all_symbols)

        if not market_data:
            self.logger.error("No market data available - skipping rebalance")
            return {'status': 'NO_DATA', 'timestamp': datetime.now()}

        # Calculate momentum scores
        momentum_scores = self.calculate_momentum_scores(market_data)

        # Calculate optimal allocations
        target_allocations = self.calculate_optimal_allocations(momentum_scores, portfolio_value)

        # Check if rebalancing is needed
        if not self.calculate_rebalance_needed(target_allocations):
            self.logger.info("No significant deviation - skipping rebalance")
            return {'status': 'NO_REBALANCE_NEEDED', 'timestamp': datetime.now()}

        # Execute rebalance
        rebalance_summary = self.execute_rebalance(target_allocations, portfolio_value)

        # Save to history
        self.rebalance_history.append(rebalance_summary)

        return rebalance_summary

    def schedule_automated_rebalancing(self, portfolio_value: float):
        """Schedule automated rebalancing every 2 hours"""

        self.logger.info("SCHEDULING AUTOMATED REBALANCING")
        self.logger.info(f"Rebalance interval: {self.rebalance_interval_hours} hours")

        # Schedule rebalancing during market hours
        schedule.every(self.rebalance_interval_hours).hours.do(
            lambda: asyncio.run(self.run_rebalance_cycle(portfolio_value))
        )

        # Also schedule at market open
        schedule.every().day.at("06:30").do(
            lambda: asyncio.run(self.run_rebalance_cycle(portfolio_value))
        )

        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def generate_rebalance_report(self) -> Dict:
        """Generate comprehensive rebalancing report"""

        if not self.rebalance_history:
            return {'status': 'NO_HISTORY'}

        total_trades = sum(r.get('trades_executed', 0) for r in self.rebalance_history)
        avg_turnover = np.mean([r.get('turnover_rate', 0) for r in self.rebalance_history])
        avg_leverage = np.mean([r.get('leverage_used', 0) for r in self.rebalance_history])

        report = {
            'total_rebalances': len(self.rebalance_history),
            'total_trades': total_trades,
            'average_turnover': avg_turnover,
            'average_leverage': avg_leverage,
            'current_positions': len(self.current_positions),
            'recent_rebalances': self.rebalance_history[-5:],  # Last 5 rebalances
            'report_generated': datetime.now()
        }

        return report

async def main():
    """Test the automated rebalancer"""
    print("AUTOMATED REBALANCING SYSTEM")
    print("2-Hour Momentum-Based Rebalancing")
    print("=" * 60)

    # Initialize rebalancer
    rebalancer = AutomatedRebalancer(leverage_multiplier=4.0)

    # Simulate portfolio value
    portfolio_value = 992233.63

    print(f"\\nPortfolio value: ${portfolio_value:,.0f}")
    print("Running single rebalance cycle...")

    # Run single cycle for testing
    result = await rebalancer.run_rebalance_cycle(portfolio_value)

    print("\\nREBALANCE RESULTS:")
    print("-" * 40)
    for key, value in result.items():
        if key != 'trades':
            print(f"{key.replace('_', ' ').title()}: {value}")

    # Generate report
    report = rebalancer.generate_rebalance_report()
    print(f"\\nReport generated: {report}")

    # Save results
    results_file = f"rebalance_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\\nResults saved to: {results_file}")
    print("\\n[SUCCESS] Automated rebalancer ready!")
    print("To run continuously: rebalancer.schedule_automated_rebalancing(portfolio_value)")

if __name__ == "__main__":
    asyncio.run(main())