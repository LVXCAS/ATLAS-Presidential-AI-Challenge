"""
Main entry point for the Quantitative Research Platform

This module provides the primary interface for running quantitative analysis,
backtests, and model training workflows.
"""

import asyncio
import click
from typing import Optional, List
from datetime import datetime, timedelta

from quant_research.config import get_config
from quant_research.utils import setup_logging, get_logger
from quant_research.data.sources import DataSourceManager, YahooDataSource
try:
    from quant_research.data.sources import AlpacaDataSource
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    AlpacaDataSource = None
from quant_research.config import BacktestConfig

# Initialize configuration and logging
config = get_config()
setup_logging(config.LOG_LEVEL)
logger = get_logger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Quantitative Research Platform - Professional trading research tools."""
    pass


@cli.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to analyze')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--frequency', default='1D', help='Data frequency (1min, 1H, 1D)')
@click.option('--output', '-o', help='Output file path')
def fetch_data(symbols, start_date, end_date, frequency, output):
    """Fetch market data from configured sources."""
    
    async def _fetch_data():
        """Async data fetching function."""
        
        # Default symbols if none provided
        if not symbols:
            symbols_list = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
        else:
            symbols_list = list(symbols)
        
        # Default date range if not provided
        if not start_date:
            start_date_parsed = datetime.now() - timedelta(days=365)
        else:
            start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d')
        
        if not end_date:
            end_date_parsed = datetime.now()
        else:
            end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Set up data manager
        data_manager = DataSourceManager()
        yahoo_source = YahooDataSource()
        data_manager.add_source("yahoo", yahoo_source, is_primary=True)
        
        # Add Alpaca if available
        if ALPACA_AVAILABLE and config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY:
            alpaca_source = AlpacaDataSource(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                paper=True
            )
            data_manager.add_source("alpaca", alpaca_source)
        
        logger.info(f"Fetching data for {len(symbols_list)} symbols")
        
        from quant_research.data.sources.base import DataRequest
        
        # Create data request
        request = DataRequest(
            symbols=symbols_list,
            start_date=start_date_parsed.date(),
            end_date=end_date_parsed.date(),
            frequency=frequency
        )
        
        # Fetch data
        async with data_manager.connect_all():
            response = await data_manager.get_bars(request)
        
        if not response.is_empty:
            df = response.to_pandas()
            
            if output:
                df.to_csv(output)
                logger.info(f"Data saved to {output}")
            else:
                print(df.head())
                print(f"\nTotal records: {len(df)}")
                print(f"Date range: {df.index.min()} to {df.index.max()}")
                print(f"Symbols: {df['symbol'].unique().tolist()}")
        else:
            logger.warning("No data retrieved")
    
    asyncio.run(_fetch_data())


@cli.command()
@click.option('--config-file', help='Strategy configuration file')
@click.option('--symbols', '-s', multiple=True, help='Symbols to backtest')
@click.option('--start-date', help='Backtest start date (YYYY-MM-DD)')
@click.option('--end-date', help='Backtest end date (YYYY-MM-DD)')
@click.option('--initial-capital', type=float, default=100000, help='Initial capital')
@click.option('--strategy', default='momentum', help='Strategy type (momentum, mean_reversion)')
@click.option('--output-dir', help='Output directory for results')
def backtest(config_file, symbols, start_date, end_date, initial_capital, strategy, output_dir):
    """Run strategy backtests."""
    
    async def _run_backtest():
        """Async backtest execution."""
        
        # Default parameters
        if not symbols:
            symbols_list = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
        else:
            symbols_list = list(symbols)
        
        if not start_date:
            start_date_parsed = datetime(2023, 1, 1)
        else:
            start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d')
        
        if not end_date:
            end_date_parsed = datetime(2024, 1, 1)
        else:
            end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Set up data manager
        data_manager = DataSourceManager()
        yahoo_source = YahooDataSource()
        data_manager.add_source("yahoo", yahoo_source, is_primary=True)
        
        # Configure backtest
        backtest_config = BacktestConfig(
            start_date=start_date_parsed.date(),
            end_date=end_date_parsed.date(),
            initial_capital=initial_capital,
            commission=0.001,
            benchmark="SPY"
        )
        
        # Initialize backtest engine
        engine = BacktestEngine(
            config=backtest_config,
            data_manager=data_manager,
            initial_capital=initial_capital
        )
        
        # Simple momentum strategy
        def momentum_strategy(data, portfolio, timestamp):
            """Simple momentum strategy."""
            signals = {}
            
            for symbol, symbol_data in data.items():
                if len(symbol_data) > 20:
                    # Simple momentum: price above 20-day SMA
                    sma_20 = symbol_data['close'].rolling(20).mean().iloc[-1]
                    current_price = symbol_data['close'].iloc[-1]
                    
                    if current_price > sma_20 * 1.02:  # 2% above SMA
                        signals[symbol] = {"action": "buy", "quantity": 100}
                    elif current_price < sma_20 * 0.98:  # 2% below SMA
                        if portfolio.has_position(symbol):
                            signals[symbol] = {
                                "action": "sell", 
                                "quantity": abs(portfolio.get_position_quantity(symbol))
                            }
            
            return signals
        
        logger.info(f"Running {strategy} backtest")
        
        # Load data
        await engine.load_data(
            symbols=symbols_list,
            start_date=start_date_parsed,
            end_date=end_date_parsed,
            frequency="1D"
        )
        
        # Run backtest
        results = await engine.run(momentum_strategy)
        
        # Display results
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS - {strategy.upper()} STRATEGY")
        print(f"{'='*60}")
        print(f"Period: {start_date_parsed.date()} to {end_date_parsed.date()}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Universe: {len(symbols_list)} symbols")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Total Return: {results.total_return:.2%}")
        print(f"  Annual Return: {results.annual_return:.2%}")
        print(f"  Volatility: {results.volatility:.2%}")
        print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {results.max_drawdown:.2%}")
        print()
        print("TRADE STATISTICS:")
        print(f"  Total Trades: {results.total_trades}")
        print(f"  Win Rate: {results.win_rate:.2%}")
        print(f"  Profit Factor: {results.profit_factor:.2f}")
        
        # Save results if output directory specified
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            if not results.portfolio_history.empty:
                portfolio_file = os.path.join(output_dir, "portfolio_history.csv")
                results.portfolio_history.to_csv(portfolio_file)
                print(f"\nPortfolio history saved to: {portfolio_file}")
            
            if not results.trades_history.empty:
                trades_file = os.path.join(output_dir, "trades_history.csv")
                results.trades_history.to_csv(trades_file, index=False)
                print(f"Trades history saved to: {trades_file}")
    
    asyncio.run(_run_backtest())


@cli.command()
@click.option('--model-type', default='xgboost', help='Model type (xgboost, sklearn, pytorch)')
@click.option('--symbols', '-s', multiple=True, help='Symbols to train on')
@click.option('--features', help='Feature configuration file')
@click.option('--target', default='returns_1d', help='Target variable')
@click.option('--output', help='Model output directory')
def train_model(model_type, symbols, features, target, output):
    """Train machine learning models."""
    
    logger.info(f"Training {model_type} model")
    
    # Default symbols
    if not symbols:
        symbols_list = ['SPY', 'QQQ', 'AAPL', 'MSFT']
    else:
        symbols_list = list(symbols)
    
    print(f"Training {model_type} model on {len(symbols_list)} symbols")
    print(f"Target variable: {target}")
    
    # Placeholder for model training
    # In a real implementation, this would:
    # 1. Load and prepare data
    # 2. Engineer features
    # 3. Split data into train/test
    # 4. Train the model
    # 5. Evaluate performance
    # 6. Save the model
    
    print("Model training completed (placeholder)")


@cli.command()
def health_check():
    """Check system health and data source availability."""
    
    async def _health_check():
        """Async health check."""
        
        print("Quantitative Research Platform Health Check")
        print("=" * 50)
        
        # Check configuration
        print(f"Environment: {config.ENVIRONMENT}")
        print(f"Debug Mode: {config.DEBUG}")
        print(f"Log Level: {config.LOG_LEVEL}")
        print()
        
        # Check data sources
        data_manager = DataSourceManager()
        
        # Add available sources
        yahoo_source = YahooDataSource()
        data_manager.add_source("yahoo", yahoo_source)
        
        if ALPACA_AVAILABLE and config.ALPACA_API_KEY:
            alpaca_source = AlpacaDataSource(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY or "test",
                paper=True
            )
            data_manager.add_source("alpaca", alpaca_source)
        
        # Run health check
        health_status = await data_manager.health_check()
        
        print("DATA SOURCES:")
        for source_name, status in health_status.items():
            status_indicator = "[OK]" if status["status"] == "healthy" else "[ERROR]"
            print(f"  {status_indicator} {source_name}: {status['status']}")
            
            if status["status"] == "healthy":
                print(f"    Response time: {status.get('response_time_seconds', 0):.2f}s")
            else:
                print(f"    Error: {status.get('error', 'Unknown')}")
        
        print()
        print("CAPABILITIES:")
        capabilities = data_manager.get_source_capabilities()
        for source_name, caps in capabilities.items():
            print(f"  {source_name}:")
            for capability, available in caps.items():
                indicator = "[YES]" if available else "[NO]"
                print(f"    {indicator} {capability}")
    
    asyncio.run(_health_check())


@cli.command()
@click.option('--symbol', default='SPY', help='Symbol to analyze')
@click.option('--days', type=int, default=30, help='Days of data to analyze')
def analyze(symbol, days):
    """Perform quick market analysis."""
    
    async def _analyze():
        """Async analysis function."""
        
        # Set up data manager
        data_manager = DataSourceManager()
        yahoo_source = YahooDataSource()
        data_manager.add_source("yahoo", yahoo_source)
        
        from quant_research.data.sources.base import DataRequest
        
        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        request = DataRequest(
            symbols=[symbol],
            start_date=start_date.date(),
            end_date=end_date.date(),
            frequency="1D"
        )
        
        async with data_manager.connect_all():
            response = await data_manager.get_bars(request)
        
        if response.is_empty:
            print(f"No data available for {symbol}")
            return
        
        df = response.to_pandas()
        symbol_data = df[df['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            print(f"No data for {symbol}")
            return
        
        # Basic analysis
        current_price = symbol_data['close'].iloc[-1]
        prev_price = symbol_data['close'].iloc[-2] if len(symbol_data) > 1 else current_price
        daily_return = (current_price - prev_price) / prev_price
        
        # Calculate simple metrics
        returns = symbol_data['close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
        
        # Moving averages
        sma_20 = symbol_data['close'].rolling(20).mean().iloc[-1] if len(symbol_data) >= 20 else None
        sma_50 = symbol_data['close'].rolling(50).mean().iloc[-1] if len(symbol_data) >= 50 else None
        
        print(f"\n{symbol} ANALYSIS ({days} days)")
        print("=" * 40)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Daily Return: {daily_return:.2%}")
        print(f"Volatility (Ann.): {volatility:.2%}")
        
        if sma_20:
            print(f"20-day SMA: ${sma_20:.2f} ({(current_price/sma_20-1)*100:+.1f}%)")
        
        if sma_50:
            print(f"50-day SMA: ${sma_50:.2f} ({(current_price/sma_50-1)*100:+.1f}%)")
        
        print(f"\nHighest: ${symbol_data['high'].max():.2f}")
        print(f"Lowest: ${symbol_data['low'].min():.2f}")
        print(f"Average Volume: {symbol_data['volume'].mean():,.0f}")
    
    asyncio.run(_analyze())


if __name__ == '__main__':
    cli()