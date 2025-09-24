"""
Simple Momentum Strategy Example

This example demonstrates how to:
1. Set up the quantitative research platform
2. Load market data from multiple sources
3. Implement a simple momentum strategy
4. Run comprehensive backtests
5. Analyze performance results

Strategy Logic:
- Buy stocks with strong recent performance (top momentum quartile)
- Hold for a fixed period (e.g., 1 month)
- Rebalance monthly
- Include risk management rules
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Quantitative Research Platform imports
from quant_research.config import get_config
from quant_research.data.sources import DataSourceManager, YahooDataSource, AlpacaDataSource
from quant_research.backtest import BacktestEngine, BacktestConfig
from quant_research.utils import get_logger, calculate_returns, calculate_sharpe_ratio

# Set up logging
logger = get_logger(__name__)


class SimpleMomentumStrategy:
    """Simple momentum strategy implementation."""
    
    def __init__(
        self,
        lookback_period: int = 252,  # 1 year lookback
        holding_period: int = 21,    # 1 month holding
        top_n: int = 10,             # Top 10 momentum stocks
        rebalance_frequency: str = "monthly"
    ):
        """Initialize strategy parameters.
        
        Args:
            lookback_period: Days to calculate momentum
            holding_period: Days to hold positions
            top_n: Number of top momentum stocks to hold
            rebalance_frequency: How often to rebalance
        """
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.top_n = top_n
        self.rebalance_frequency = rebalance_frequency
        
        # State tracking
        self.last_rebalance = None
        self.current_positions = set()
        
        logger.info(f"Initialized momentum strategy: lookback={lookback_period}, top_n={top_n}")
    
    def calculate_momentum_scores(
        self, 
        data: Dict[str, pd.DataFrame], 
        timestamp: datetime
    ) -> Dict[str, float]:
        """Calculate momentum scores for all symbols.
        
        Args:
            data: Historical price data by symbol
            timestamp: Current timestamp
            
        Returns:
            Dictionary mapping symbols to momentum scores
        """
        momentum_scores = {}
        
        for symbol, symbol_data in data.items():
            try:
                if len(symbol_data) < self.lookback_period:
                    continue
                
                # Calculate momentum as cumulative return over lookback period
                current_price = symbol_data['close'].iloc[-1]
                lookback_price = symbol_data['close'].iloc[-self.lookback_period]
                
                momentum = (current_price - lookback_price) / lookback_price
                momentum_scores[symbol] = momentum
                
            except Exception as e:
                logger.warning(f"Error calculating momentum for {symbol}: {e}")
                continue
        
        return momentum_scores
    
    def should_rebalance(self, timestamp: datetime) -> bool:
        """Determine if portfolio should be rebalanced.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if should rebalance
        """
        if self.last_rebalance is None:
            return True
        
        if self.rebalance_frequency == "daily":
            return True
        elif self.rebalance_frequency == "weekly":
            return (timestamp - self.last_rebalance).days >= 7
        elif self.rebalance_frequency == "monthly":
            return (timestamp - self.last_rebalance).days >= 30
        
        return False
    
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio,
        timestamp: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """Generate trading signals based on momentum.
        
        Args:
            data: Current market data
            portfolio: Current portfolio state
            timestamp: Current timestamp
            
        Returns:
            Dictionary of trading signals
        """
        signals = {}
        
        try:
            # Check if we should rebalance
            if not self.should_rebalance(timestamp):
                return signals
            
            # Calculate momentum scores
            momentum_scores = self.calculate_momentum_scores(data, timestamp)
            
            if not momentum_scores:
                logger.warning("No momentum scores calculated")
                return signals
            
            # Sort by momentum (descending)
            sorted_symbols = sorted(
                momentum_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Select top momentum stocks
            top_momentum_symbols = [symbol for symbol, score in sorted_symbols[:self.top_n]]
            
            # Close positions not in top momentum
            current_positions = set(portfolio.positions.keys())
            positions_to_close = current_positions - set(top_momentum_symbols)
            
            for symbol in positions_to_close:
                if portfolio.has_position(symbol):
                    signals[symbol] = {
                        "action": "sell",
                        "quantity": abs(portfolio.get_position_quantity(symbol)),
                        "reason": "not_in_top_momentum"
                    }
            
            # Open new positions
            target_weight = 1.0 / self.top_n  # Equal weight allocation
            
            for symbol in top_momentum_symbols:
                if symbol not in data:
                    continue
                
                symbol_data = data[symbol]
                if symbol_data.empty:
                    continue
                
                current_price = symbol_data['close'].iloc[-1]
                current_position = portfolio.get_position_quantity(symbol)
                
                # Calculate target position size
                target_value = portfolio.total_value * target_weight
                target_quantity = target_value / current_price
                
                # Generate signal if position needs adjustment
                quantity_diff = target_quantity - current_position
                
                if abs(quantity_diff) > 1:  # Minimum trade size
                    signals[symbol] = {
                        "action": "buy" if quantity_diff > 0 else "sell",
                        "quantity": abs(quantity_diff),
                        "target_weight": target_weight,
                        "momentum_score": momentum_scores[symbol],
                        "reason": "momentum_rebalance"
                    }
            
            # Update tracking
            self.last_rebalance = timestamp
            self.current_positions = set(top_momentum_symbols)
            
            logger.info(f"Generated {len(signals)} signals at {timestamp}")
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals


async def run_momentum_backtest():
    """Run the momentum strategy backtest."""
    
    logger.info("Starting momentum strategy backtest")
    
    # Initialize configuration
    config = get_config()
    
    # Set up data sources
    data_manager = DataSourceManager()
    
    # Add Yahoo Finance as primary data source
    yahoo_source = YahooDataSource()
    data_manager.add_source("yahoo", yahoo_source, is_primary=True)
    
    # Optionally add Alpaca as backup (requires API keys)
    if config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY:
        alpaca_source = AlpacaDataSource(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            paper=True  # Use paper trading
        )
        data_manager.add_source("alpaca", alpaca_source)
    
    # Define universe of stocks (large cap tech and financial stocks)
    universe = [
        # Technology
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA",
        # Financial
        "JPM", "BAC", "WFC", "GS", "MS", "C",
        # Healthcare
        "JNJ", "PFE", "UNH", "ABBV",
        # Consumer
        "PG", "KO", "WMT", "HD",
        # ETFs for diversification
        "SPY", "QQQ", "IWM"
    ]
    
    # Set backtest period
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    # Initialize backtest configuration
    backtest_config = BacktestConfig(
        start_date=start_date.date(),
        end_date=end_date.date(),
        initial_capital=100000.0,
        commission=0.001,  # 0.1% commission
        benchmark="SPY"
    )
    
    # Initialize backtest engine
    engine = BacktestEngine(
        config=backtest_config,
        data_manager=data_manager,
        initial_capital=100000.0
    )
    
    # Initialize strategy
    strategy = SimpleMomentumStrategy(
        lookback_period=252,  # 1 year momentum
        holding_period=21,    # Hold for 1 month
        top_n=10,            # Top 10 momentum stocks
        rebalance_frequency="monthly"
    )
    
    try:
        # Load historical data
        logger.info("Loading historical data...")
        await engine.load_data(
            symbols=universe,
            start_date=start_date,
            end_date=end_date,
            frequency="1D"
        )
        
        # Run backtest
        logger.info("Running backtest...")
        results = await engine.run(strategy.generate_signals)
        
        # Print results
        print("\n" + "="*60)
        print("MOMENTUM STRATEGY BACKTEST RESULTS")
        print("="*60)
        
        print(f"Backtest Period: {start_date.date()} to {end_date.date()}")
        print(f"Initial Capital: ${backtest_config.initial_capital:,.2f}")
        print(f"Universe Size: {len(universe)} stocks")
        print()
        
        print("PERFORMANCE METRICS:")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Annualized Return: {results.annual_return:.2%}")
        print(f"Volatility: {results.volatility:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {results.sortino_ratio:.2f}")
        print(f"Maximum Drawdown: {results.max_drawdown:.2%}")
        print(f"Calmar Ratio: {results.calmar_ratio:.2f}")
        print()
        
        print("TRADE STATISTICS:")
        print(f"Total Trades: {results.total_trades}")
        print(f"Winning Trades: {results.winning_trades}")
        print(f"Losing Trades: {results.losing_trades}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Average Win: {results.avg_win:.2%}")
        print(f"Average Loss: {results.avg_loss:.2%}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print()
        
        if hasattr(results, 'benchmark_return'):
            print("BENCHMARK COMPARISON:")
            print(f"Benchmark Return: {results.benchmark_return:.2%}")
            print(f"Alpha: {results.alpha:.2%}")
            print(f"Beta: {results.beta:.2f}")
        
        print("="*60)
        
        # Save detailed results
        if not results.portfolio_history.empty:
            results.portfolio_history.to_csv("momentum_strategy_portfolio.csv")
            logger.info("Portfolio history saved to momentum_strategy_portfolio.csv")
        
        if not results.trades_history.empty:
            results.trades_history.to_csv("momentum_strategy_trades.csv")
            logger.info("Trades history saved to momentum_strategy_trades.csv")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


async def run_parameter_optimization():
    """Run parameter optimization for the momentum strategy."""
    
    logger.info("Starting parameter optimization...")
    
    # Parameter grid to test
    param_grid = {
        "lookback_period": [126, 189, 252, 315],  # 6, 9, 12, 15 months
        "top_n": [5, 8, 10, 12, 15],
        "rebalance_frequency": ["monthly", "weekly"]
    }
    
    results = []
    
    # Test all combinations (simplified - in practice use proper optimization)
    for lookback in param_grid["lookback_period"]:
        for top_n in param_grid["top_n"]:
            for rebalance_freq in param_grid["rebalance_frequency"]:
                
                logger.info(f"Testing: lookback={lookback}, top_n={top_n}, freq={rebalance_freq}")
                
                # Initialize strategy with parameters
                strategy = SimpleMomentumStrategy(
                    lookback_period=lookback,
                    top_n=top_n,
                    rebalance_frequency=rebalance_freq
                )
                
                # Run backtest (simplified version)
                # In practice, you would run full backtest here
                # For now, we'll just record the parameters
                
                result_entry = {
                    "lookback_period": lookback,
                    "top_n": top_n,
                    "rebalance_frequency": rebalance_freq,
                    "sharpe_ratio": np.random.normal(1.0, 0.3),  # Placeholder
                    "total_return": np.random.normal(0.15, 0.1),  # Placeholder
                    "max_drawdown": np.random.normal(0.12, 0.05)  # Placeholder
                }
                
                results.append(result_entry)
    
    # Convert to DataFrame and analyze
    results_df = pd.DataFrame(results)
    
    # Find best parameters
    best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    
    print("\nPARAMETER OPTIMIZATION RESULTS:")
    print("="*50)
    print("\nBest Parameters (by Sharpe Ratio):")
    print(f"Lookback Period: {best_sharpe['lookback_period']} days")
    print(f"Top N: {best_sharpe['top_n']} stocks")
    print(f"Rebalance Frequency: {best_sharpe['rebalance_frequency']}")
    print(f"Expected Sharpe Ratio: {best_sharpe['sharpe_ratio']:.2f}")
    
    # Save optimization results
    results_df.to_csv("momentum_optimization_results.csv", index=False)
    logger.info("Optimization results saved to momentum_optimization_results.csv")
    
    return results_df


async def main():
    """Main execution function."""
    
    try:
        # Run basic backtest
        results = await run_momentum_backtest()
        
        # Optionally run parameter optimization
        # await run_parameter_optimization()
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    # Run the analysis
    asyncio.run(main())