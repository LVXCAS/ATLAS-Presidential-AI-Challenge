"""
Comprehensive backtesting demo showcasing the new backtesting engine capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add backend to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backtesting import (
    BacktestEngine, BacktestResults, DataLoader, StrategyTester, 
    ParameterRange, create_data_feed
)
from agents.base_agent import TradingSignal, SignalType
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_moving_average_crossover_strategy(short_window: int = 10, long_window: int = 30):
    """
    Create a moving average crossover strategy.
    
    Args:
        short_window: Period for short-term moving average
        long_window: Period for long-term moving average
    """
    
    # Store price history for moving average calculation
    price_history = {}
    
    def moving_average_strategy(timestamp: datetime, context: Dict[str, Any]) -> List[TradingSignal]:
        """Moving average crossover strategy implementation."""
        signals = []
        portfolio = context['portfolio']
        current_prices = context['current_prices']
        
        for symbol in current_prices:
            current_price = current_prices[symbol]
            
            # Initialize price history if needed
            if symbol not in price_history:
                price_history[symbol] = []
            
            # Add current price to history
            price_history[symbol].append(current_price)
            
            # Keep only required history length
            max_length = max(short_window, long_window) + 1
            if len(price_history[symbol]) > max_length:
                price_history[symbol] = price_history[symbol][-max_length:]
            
            # Calculate moving averages if we have enough data
            if len(price_history[symbol]) >= long_window:
                prices = price_history[symbol]
                
                short_ma = sum(prices[-short_window:]) / short_window
                long_ma = sum(prices[-long_window:]) / long_window
                
                # Previous MAs for crossover detection
                if len(prices) > long_window:
                    prev_short_ma = sum(prices[-short_window-1:-1]) / short_window
                    prev_long_ma = sum(prices[-long_window-1:-1]) / long_window
                    
                    current_position = portfolio.get_position(symbol)
                    position_quantity = current_position.quantity if current_position else 0
                    
                    # Generate signals based on crossover
                    bullish_crossover = prev_short_ma <= prev_long_ma and short_ma > long_ma
                    bearish_crossover = prev_short_ma >= prev_long_ma and short_ma < long_ma
                    
                    if bullish_crossover and position_quantity <= 0:
                        # Buy signal
                        signal = TradingSignal(
                            id=str(uuid.uuid4()),
                            agent_name="MovingAverageCrossover",
                            symbol=symbol,
                            timestamp=timestamp,
                            signal_type=SignalType.BUY,
                            confidence=0.75,
                            strength=0.8,
                            reasoning={
                                "strategy": "ma_crossover",
                                "short_ma": short_ma,
                                "long_ma": long_ma,
                                "crossover_type": "bullish"
                            },
                            features_used={
                                "short_window": short_window,
                                "long_window": long_window
                            },
                            prediction_horizon=10,
                            target_price=current_price * 1.05,
                            stop_loss=current_price * 0.97,
                            risk_score=0.4,
                            expected_return=0.05
                        )
                        signals.append(signal)
                        
                    elif bearish_crossover and position_quantity > 0:
                        # Sell signal
                        signal = TradingSignal(
                            id=str(uuid.uuid4()),
                            agent_name="MovingAverageCrossover",
                            symbol=symbol,
                            timestamp=timestamp,
                            signal_type=SignalType.SELL,
                            confidence=0.75,
                            strength=0.8,
                            reasoning={
                                "strategy": "ma_crossover",
                                "short_ma": short_ma,
                                "long_ma": long_ma,
                                "crossover_type": "bearish"
                            },
                            features_used={
                                "short_window": short_window,
                                "long_window": long_window
                            },
                            prediction_horizon=10,
                            target_price=current_price * 0.95,
                            stop_loss=current_price * 1.03,
                            risk_score=0.4,
                            expected_return=0.05
                        )
                        signals.append(signal)
        
        return signals
    
    return moving_average_strategy


async def run_simple_backtest_demo():
    """Demonstrate basic backtesting functionality."""
    logger.info("=== Running Simple Backtest Demo ===")
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    initial_capital = 1000000.0
    
    # Create strategy
    strategy = create_moving_average_crossover_strategy(short_window=10, long_window=30)
    
    # Create backtest engine
    config = {
        'market_simulator': {
            'base_slippage_bps': 3,
            'commission_per_share': 0.01
        }
    }
    engine = BacktestEngine(initial_capital, config)
    
    # Create data feed (will use mock data)
    data_feed = await create_data_feed(symbols, start_date, end_date, '1d')
    engine.add_data_feed(data_feed)
    
    # Run backtest
    results = await engine.run_backtest(
        start_date, end_date, strategy, symbols
    )
    
    # Print results
    logger.info(f"Backtest Results:")
    logger.info(f"Initial Capital: ${results.initial_capital:,.2f}")
    logger.info(f"Final Capital: ${results.final_capital:,.2f}")
    logger.info(f"Total Return: {results.total_return:.2%}")
    logger.info(f"Annualized Return: {results.annualized_return:.2%}")
    logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
    logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
    logger.info(f"Win Rate: {results.win_rate:.2%}")
    logger.info(f"Total Trades: {results.total_trades}")
    logger.info(f"Total Commission: ${results.total_commission:.2f}")
    
    return results


async def run_optimization_demo():
    """Demonstrate parameter optimization."""
    logger.info("=== Running Parameter Optimization Demo ===")
    
    # Configuration
    symbols = ['AAPL', 'MSFT']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 6, 30)  # Shorter period for optimization
    initial_capital = 1000000.0
    
    # Define parameter ranges to optimize
    parameter_ranges = [
        ParameterRange(
            name='short_window',
            min_value=5,
            max_value=20,
            step=5,
            param_type='int'
        ),
        ParameterRange(
            name='long_window',
            min_value=20,
            max_value=50,
            step=10,
            param_type='int'
        )
    ]
    
    # Create parameterized strategy function
    def strategy_function(timestamp: datetime, context: Dict[str, Any]) -> List[TradingSignal]:
        params = context.get('parameters', {})
        short_window = params.get('short_window', 10)
        long_window = params.get('long_window', 30)
        
        strategy = create_moving_average_crossover_strategy(short_window, long_window)
        return strategy(timestamp, context)
    
    # Create strategy tester
    config = {
        'optimization_metric': 'sharpe_ratio',
        'backtest_engine': {
            'market_simulator': {
                'base_slippage_bps': 2
            }
        }
    }
    tester = StrategyTester(config)
    
    # Run optimization
    optimization_results = await tester.optimize_parameters(
        strategy_function,
        parameter_ranges,
        symbols,
        start_date,
        end_date,
        initial_capital,
        max_combinations=50
    )
    
    # Print optimization results
    logger.info(f"Optimization Results (Top 5):")
    for i, result in enumerate(optimization_results[:5]):
        logger.info(f"Rank {result.rank}: Parameters: {result.parameters}")
        logger.info(f"  Sharpe Ratio: {result.performance_metric:.3f}")
        logger.info(f"  Total Return: {result.backtest_results.total_return:.2%}")
        logger.info(f"  Max Drawdown: {result.backtest_results.max_drawdown:.2%}")
        logger.info("")
    
    return optimization_results


async def main():
    """Run backtesting demo."""
    logger.info("Starting Comprehensive Backtesting Demo")
    logger.info("=" * 60)
    
    try:
        # Run simple backtest
        await run_simple_backtest_demo()
        print("\n" + "=" * 60 + "\n")
        
        # Run parameter optimization
        await run_optimization_demo()
        
        logger.info("All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())