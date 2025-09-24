#!/usr/bin/env python3
"""
LEAN MEGA INTEGRATION - Advanced LEAN Engine Integration
Real-time strategy deployment with mega factory output
"""

import json
import logging
import asyncio
import subprocess
import os
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lean_mega_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LEANMegaIntegration:
    """Advanced LEAN integration for mega factory strategies"""

    def __init__(self):
        self.lean_cli_path = self._find_lean_cli()
        self.project_dir = Path.cwd()
        self.algorithms_dir = self.project_dir / "lean_algorithms"
        self.results_dir = self.project_dir / "lean_results"
        self.config_dir = self.project_dir / "lean_configs"

        # Create directories
        self.algorithms_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)

        logger.info("LEAN Mega Integration initialized")
        logger.info(f"LEAN CLI: {self.lean_cli_path}")
        logger.info(f"Algorithms Directory: {self.algorithms_dir}")

    def _find_lean_cli(self):
        """Find LEAN CLI executable"""
        possible_paths = [
            "lean",
            "C:\\Program Files\\QuantConnect\\Lean\\Launcher\\bin\\Debug\\QuantConnect.Lean.Launcher.exe",
            "C:\\Users\\lucas\\.dotnet\\tools\\lean",
            "/usr/local/bin/lean",
            "./lean"
        ]

        for path in possible_paths:
            if shutil.which(path):
                return path

        # Default to lean command
        return "lean"

    async def process_mega_strategies(self, strategies_file: str):
        """Process all strategies from mega factory output"""
        logger.info(f"Processing mega strategies from: {strategies_file}")

        # Load strategies
        with open(strategies_file, 'r') as f:
            strategies = json.load(f)

        logger.info(f"Found {len(strategies)} elite strategies")

        results = []
        for i, strategy in enumerate(strategies, 1):
            logger.info(f"Processing strategy {i}/{len(strategies)}: {strategy['name']}")

            try:
                # Create LEAN algorithm
                algorithm_path = await self._create_lean_algorithm(strategy)

                # Run backtest
                backtest_result = await self._run_lean_backtest(algorithm_path, strategy['name'])

                # Process results
                processed_result = self._process_backtest_result(strategy, backtest_result)
                results.append(processed_result)

            except Exception as e:
                logger.error(f"Error processing strategy {strategy['name']}: {e}")
                continue

        # Save comprehensive results
        results_file = f"lean_mega_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved {len(results)} strategy results to {results_file}")
        return results

    async def _create_lean_algorithm(self, strategy):
        """Create advanced LEAN algorithm from strategy data"""
        algorithm_name = f"{strategy['name']}_LEAN"
        algorithm_file = self.algorithms_dir / f"{algorithm_name}.py"

        # Enhanced algorithm template
        algorithm_code = f'''
from AlgorithmImports import *
import numpy as np
import pandas as pd

class {algorithm_name.replace("-", "_")}(QCAlgorithm):
    def Initialize(self):
        """Initialize the algorithm with enhanced settings"""
        # Time range for backtesting
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)  # $1M starting capital

        # Strategy metadata
        self.strategy_name = "{strategy['name']}"
        self.strategy_type = "{strategy['type']}"
        self.options_type = "{strategy.get('options_type', 'N/A')}"
        self.source = "{strategy['source']}"
        self.expected_sharpe = {strategy['expected_sharpe']}
        self.leverage_multiplier = {strategy.get('leverage_multiplier', 1.0)}

        # Add universe
        self.symbols = []
        universe_symbols = ["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD", "VIX"]

        for symbol in universe_symbols:
            if symbol == "VIX":
                # Add VIX as custom data
                continue
            equity = self.AddEquity(symbol, Resolution.Minute)
            equity.SetFeeModel(EquityFeeModel())
            equity.SetSlippageModel(VolumeShareSlippageModel())
            equity.SetFillModel(ImmediateFillModel())
            self.symbols.append(equity.Symbol)

        # Options universe for options strategies
        if self.strategy_type == "options":
            self.spy_option = self.AddOption("SPY", Resolution.Minute)
            self.spy_option.SetFilter(-20, 20, 0, 60)  # Strike range and expiry

        # Performance tracking
        self.previous_portfolio_value = self.Portfolio.TotalPortfolioValue
        self.daily_returns = []
        self.trade_count = 0
        self.winning_trades = 0
        self.max_drawdown = 0
        self.peak_value = self.Portfolio.TotalPortfolioValue

        # Risk management
        self.max_position_size = 0.15  # 15% max per position
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit

        # Rebalancing schedule
        if "{strategy.get('rebalancing_frequency', 'weekly')}" == "daily":
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.AfterMarketOpen("SPY", 30),
                self.Rebalance
            )
        else:
            self.Schedule.On(
                self.DateRules.WeekStart(),
                self.TimeRules.AfterMarketOpen("SPY", 30),
                self.Rebalance
            )

        # Daily performance tracking
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.BeforeMarketClose("SPY", 0),
            self.TrackPerformance
        )

        self.Debug(f"Initialized {{self.strategy_name}} - Expected Sharpe: {{self.expected_sharpe:.2f}}")

    def Rebalance(self):
        """Execute strategy-specific rebalancing"""
        try:
            if self.strategy_type == "options":
                self._rebalance_options_strategy()
            elif "momentum" in self.strategy_name.lower():
                self._rebalance_momentum_strategy()
            elif "mean_reversion" in self.strategy_name.lower():
                self._rebalance_mean_reversion_strategy()
            elif "volatility" in self.strategy_name.lower():
                self._rebalance_volatility_strategy()
            else:
                self._rebalance_factor_strategy()

        except Exception as e:
            self.Debug(f"Rebalancing error: {{e}}")

    def _rebalance_options_strategy(self):
        """Advanced options strategy rebalancing"""
        if hasattr(self, 'spy_option'):
            # Get option chain
            chain = self.spy_option.GetCurrentChain()
            if chain is None or len(chain) == 0:
                return

            # Options strategy based on type
            if "{strategy.get('options_type')}" == "long_call":
                self._execute_long_call_strategy(chain)
            elif "{strategy.get('options_type')}" == "iron_condor":
                self._execute_iron_condor_strategy(chain)
            elif "{strategy.get('options_type')}" == "straddle":
                self._execute_straddle_strategy(chain)
            else:
                # Default covered call
                self._execute_covered_call_strategy(chain)

    def _execute_long_call_strategy(self, chain):
        """Execute long call options strategy"""
        # Find ATM calls with 30-45 days to expiry
        underlying_price = self.Securities["SPY"].Price

        calls = [contract for contract in chain if contract.Right == OptionRight.Call
                and 30 <= (contract.Expiry.date() - self.Time.date()).days <= 45
                and abs(contract.Strike - underlying_price) <= underlying_price * 0.05]

        if calls:
            # Select closest to ATM
            best_call = min(calls, key=lambda x: abs(x.Strike - underlying_price))

            # Calculate position size
            option_price = self.Securities[best_call].Price
            if option_price > 0:
                position_value = self.Portfolio.TotalPortfolioValue * 0.10  # 10% allocation
                contracts = int(position_value / (option_price * 100))

                if contracts > 0:
                    self.MarketOrder(best_call, contracts)
                    self.Debug(f"Long call: {{contracts}} contracts of {{best_call.Strike}} strike")

    def _execute_iron_condor_strategy(self, chain):
        """Execute iron condor options strategy"""
        underlying_price = self.Securities["SPY"].Price

        # Find options with 30-45 days to expiry
        valid_contracts = [contract for contract in chain
                         if 30 <= (contract.Expiry.date() - self.Time.date()).days <= 45]

        if len(valid_contracts) >= 4:
            # Iron condor construction
            otm_put_strike = underlying_price * 0.95
            itm_put_strike = underlying_price * 0.97
            itm_call_strike = underlying_price * 1.03
            otm_call_strike = underlying_price * 1.05

            # Execute iron condor (simplified)
            position_size = int(self.Portfolio.TotalPortfolioValue * 0.05 / 10000)  # Conservative
            if position_size > 0:
                self.Debug(f"Iron condor: {{position_size}} spreads")

    def _execute_straddle_strategy(self, chain):
        """Execute straddle options strategy"""
        underlying_price = self.Securities["SPY"].Price

        # Find ATM options
        calls = [c for c in chain if c.Right == OptionRight.Call
                and abs(c.Strike - underlying_price) <= underlying_price * 0.02]
        puts = [p for p in chain if p.Right == OptionRight.Put
               and abs(p.Strike - underlying_price) <= underlying_price * 0.02]

        if calls and puts:
            best_call = min(calls, key=lambda x: abs(x.Strike - underlying_price))
            best_put = min(puts, key=lambda x: abs(x.Strike - underlying_price))

            contracts = int(self.Portfolio.TotalPortfolioValue * 0.08 / 20000)  # Conservative
            if contracts > 0:
                self.MarketOrder(best_call, contracts)
                self.MarketOrder(best_put, contracts)
                self.Debug(f"Straddle: {{contracts}} contracts each")

    def _execute_covered_call_strategy(self, chain):
        """Execute covered call strategy"""
        # Buy underlying and sell calls
        spy_weight = 0.50
        self.SetHoldings("SPY", spy_weight)

        # Sell OTM calls
        underlying_price = self.Securities["SPY"].Price
        calls = [c for c in chain if c.Right == OptionRight.Call
                and c.Strike > underlying_price * 1.02
                and 30 <= (c.Expiry.date() - self.Time.date()).days <= 45]

        if calls:
            best_call = min(calls, key=lambda x: abs(x.Strike - underlying_price * 1.05))
            shares_held = self.Portfolio["SPY"].Quantity
            contracts_to_sell = int(shares_held / 100)

            if contracts_to_sell > 0:
                self.MarketOrder(best_call, -contracts_to_sell)
                self.Debug(f"Covered call: sold {{contracts_to_sell}} contracts")

    def _rebalance_momentum_strategy(self):
        """Momentum strategy implementation"""
        for symbol in self.symbols:
            history = self.History(symbol, 252, Resolution.Daily)
            if not history.empty:
                # Calculate 3-month and 12-month momentum
                prices = history['close']
                momentum_3m = (prices.iloc[-1] / prices.iloc[-63]) - 1
                momentum_12m = (prices.iloc[-1] / prices.iloc[-252]) - 1

                # Combined momentum score
                momentum_score = 0.3 * momentum_3m + 0.7 * momentum_12m

                # Position sizing based on momentum
                if momentum_score > 0.05:  # Strong momentum
                    weight = min(self.max_position_size, momentum_score * 0.5)
                elif momentum_score < -0.05:  # Negative momentum
                    weight = 0  # No position
                else:
                    weight = momentum_score * 0.1  # Small position

                self.SetHoldings(symbol, weight * self.leverage_multiplier)

    def _rebalance_mean_reversion_strategy(self):
        """Mean reversion strategy implementation"""
        for symbol in self.symbols:
            history = self.History(symbol, 60, Resolution.Daily)
            if not history.empty:
                prices = history['close']
                mean_price = prices.rolling(20).mean().iloc[-1]
                current_price = prices.iloc[-1]
                std_dev = prices.rolling(20).std().iloc[-1]

                # Z-score calculation
                z_score = (current_price - mean_price) / std_dev

                # Mean reversion positioning
                if z_score > 2:  # Overbought
                    weight = -min(self.max_position_size, abs(z_score) * 0.05)
                elif z_score < -2:  # Oversold
                    weight = min(self.max_position_size, abs(z_score) * 0.05)
                else:
                    weight = -z_score * 0.02  # Fade the move

                self.SetHoldings(symbol, weight * self.leverage_multiplier)

    def _rebalance_volatility_strategy(self):
        """Volatility-based strategy implementation"""
        for symbol in self.symbols:
            history = self.History(symbol, 30, Resolution.Daily)
            if not history.empty:
                returns = history['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)

                # Inverse volatility weighting
                if volatility > 0:
                    weight = min(self.max_position_size, (0.20 / volatility))
                    self.SetHoldings(symbol, weight * self.leverage_multiplier)

    def _rebalance_factor_strategy(self):
        """Multi-factor strategy implementation"""
        # Equal weight with momentum and quality tilt
        num_symbols = len(self.symbols)
        if num_symbols > 0:
            base_weight = 1.0 / num_symbols

            for symbol in self.symbols:
                # Apply basic momentum tilt
                history = self.History(symbol, 126, Resolution.Daily)
                if not history.empty:
                    momentum = (history['close'].iloc[-1] / history['close'].iloc[-63]) - 1
                    momentum_tilt = 1 + np.clip(momentum, -0.5, 0.5)

                    final_weight = base_weight * momentum_tilt * self.leverage_multiplier
                    final_weight = min(final_weight, self.max_position_size)

                    self.SetHoldings(symbol, final_weight)

    def TrackPerformance(self):
        """Enhanced performance tracking"""
        current_value = self.Portfolio.TotalPortfolioValue
        daily_return = (current_value / self.previous_portfolio_value) - 1
        self.daily_returns.append(daily_return)
        self.previous_portfolio_value = current_value

        # Update peak and drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value

        current_drawdown = (current_value / self.peak_value) - 1
        if current_drawdown < self.max_drawdown:
            self.max_drawdown = current_drawdown

        # Log metrics monthly
        if len(self.daily_returns) % 21 == 0:  # Roughly monthly
            returns_series = pd.Series(self.daily_returns[-63:])  # Last 3 months
            if len(returns_series) > 10:
                sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
                total_return = (current_value / 1000000) - 1

                self.Debug(f"Performance Update:")
                self.Debug(f"  3M Sharpe: {{sharpe:.2f}}")
                self.Debug(f"  Total Return: {{total_return:.2%}}")
                self.Debug(f"  Max Drawdown: {{self.max_drawdown:.2%}}")
                self.Debug(f"  Portfolio Value: ${{current_value:,.0f}}")

    def OnOrderEvent(self, orderEvent):
        """Track trade statistics"""
        if orderEvent.Status == OrderStatus.Filled:
            self.trade_count += 1

            # Simple P&L tracking for completed trades
            if orderEvent.FillQuantity != 0:
                self.Debug(f"Trade executed: {{orderEvent.Symbol}} {{orderEvent.FillQuantity}} @ {{orderEvent.FillPrice}}")

    def OnEndOfAlgorithm(self):
        """Final performance summary"""
        if len(self.daily_returns) > 0:
            returns_series = pd.Series(self.daily_returns)

            # Calculate metrics
            total_return = (self.Portfolio.TotalPortfolioValue / 1000000) - 1
            annual_return = (1 + total_return) ** (252 / len(self.daily_returns)) - 1
            volatility = returns_series.std() * np.sqrt(252)
            sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0

            self.Debug("="*50)
            self.Debug(f"FINAL PERFORMANCE - {{self.strategy_name}}")
            self.Debug("="*50)
            self.Debug(f"Total Return: {{total_return:.2%}}")
            self.Debug(f"Annual Return: {{annual_return:.2%}}")
            self.Debug(f"Sharpe Ratio: {{sharpe_ratio:.2f}}")
            self.Debug(f"Volatility: {{volatility:.2%}}")
            self.Debug(f"Max Drawdown: {{self.max_drawdown:.2%}}")
            self.Debug(f"Total Trades: {{self.trade_count}}")
            self.Debug(f"Expected Sharpe: {{self.expected_sharpe:.2f}}")
            self.Debug(f"Sharpe vs Expected: {{sharpe_ratio - self.expected_sharpe:.2f}}")
            self.Debug("="*50)
'''

        # Write algorithm file
        with open(algorithm_file, 'w') as f:
            f.write(algorithm_code)

        logger.info(f"Created LEAN algorithm: {algorithm_file}")
        return algorithm_file

    async def _run_lean_backtest(self, algorithm_path, strategy_name):
        """Run LEAN backtest with enhanced configuration"""
        try:
            # Create custom config for this strategy
            config = self._create_lean_config(strategy_name)
            config_file = self.config_dir / f"config_{strategy_name}.json"

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            # Run LEAN backtest
            cmd = [
                str(self.lean_cli_path),
                "backtest",
                str(algorithm_path),
                "--config", str(config_file),
                "--output", str(self.results_dir / f"results_{strategy_name}")
            ]

            logger.info(f"Running LEAN backtest: {' '.join(cmd)}")

            # Run with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_dir)
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=3600  # 1 hour timeout
            )

            if process.returncode == 0:
                logger.info(f"LEAN backtest completed successfully for {strategy_name}")
                return self._parse_lean_results(stdout.decode(), strategy_name)
            else:
                logger.error(f"LEAN backtest failed for {strategy_name}: {stderr.decode()}")
                return None

        except asyncio.TimeoutError:
            logger.error(f"LEAN backtest timed out for {strategy_name}")
            return None
        except Exception as e:
            logger.error(f"Error running LEAN backtest for {strategy_name}: {e}")
            return None

    def _create_lean_config(self, strategy_name):
        """Create LEAN configuration for strategy"""
        return {
            "environment": "backtesting",
            "algorithm-type-name": f"{strategy_name}_LEAN",
            "algorithm-language": "Python",
            "data-folder": "../Data",
            "debugging": False,
            "debugging-method": "LocalCmdline",
            "log-handler": "ConsoleLogHandler",
            "messaging-handler": "MessagingHandler",
            "job-queue-handler": "JobQueue",
            "api-handler": "LocalDiskApiHandler",
            "map-file-provider": "LocalDiskMapFileProvider",
            "factor-file-provider": "LocalDiskFactorFileProvider",
            "data-provider": "DefaultDataProvider",
            "alpha-handler": "DefaultAlphaHandler",
            "object-store": "LocalObjectStore",
            "data-channel-provider": "DataChannelProvider"
        }

    def _parse_lean_results(self, output, strategy_name):
        """Parse LEAN backtest output"""
        try:
            # Extract key metrics from output
            lines = output.split('\n')
            metrics = {}

            for line in lines:
                if 'Total Return' in line:
                    metrics['total_return'] = float(line.split(':')[-1].strip().replace('%', '')) / 100
                elif 'Sharpe Ratio' in line:
                    metrics['sharpe_ratio'] = float(line.split(':')[-1].strip())
                elif 'Max Drawdown' in line:
                    metrics['max_drawdown'] = float(line.split(':')[-1].strip().replace('%', '')) / 100
                elif 'Volatility' in line:
                    metrics['volatility'] = float(line.split(':')[-1].strip().replace('%', '')) / 100
                elif 'Total Trades' in line:
                    metrics['total_trades'] = int(line.split(':')[-1].strip())

            return metrics

        except Exception as e:
            logger.error(f"Error parsing LEAN results for {strategy_name}: {e}")
            return {}

    def _process_backtest_result(self, strategy, lean_result):
        """Process and combine strategy with LEAN results"""
        if lean_result is None:
            lean_result = {}

        return {
            'strategy_name': strategy['name'],
            'strategy_type': strategy['type'],
            'source': strategy['source'],
            'expected_sharpe': strategy['expected_sharpe'],
            'mega_factory_score': strategy.get('comprehensive_score', 0),
            'lean_backtest': lean_result,
            'performance_comparison': {
                'expected_vs_actual_sharpe': lean_result.get('sharpe_ratio', 0) - strategy['expected_sharpe'],
                'lean_total_return': lean_result.get('total_return', 0),
                'lean_max_drawdown': lean_result.get('max_drawdown', 0),
                'lean_volatility': lean_result.get('volatility', 0),
                'lean_total_trades': lean_result.get('total_trades', 0)
            },
            'validation_status': 'LEAN_VALIDATED' if lean_result else 'LEAN_FAILED',
            'timestamp': datetime.now().isoformat()
        }

async def main():
    """Main execution function"""
    logger.info("Starting LEAN Mega Integration")

    # Initialize LEAN integration
    lean_integration = LEANMegaIntegration()

    # Find latest mega strategies file
    import glob
    strategy_files = glob.glob("mega_elite_strategies_*.json")
    if not strategy_files:
        logger.error("No mega strategies file found")
        return

    latest_file = max(strategy_files)
    logger.info(f"Processing strategies from: {latest_file}")

    # Process all strategies
    results = await lean_integration.process_mega_strategies(latest_file)

    # Summary statistics
    if results:
        validated_strategies = [r for r in results if r['validation_status'] == 'LEAN_VALIDATED']

        logger.info("="*60)
        logger.info("LEAN MEGA INTEGRATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Strategies Processed: {len(results)}")
        logger.info(f"Successfully Validated: {len(validated_strategies)}")
        logger.info(f"Validation Rate: {len(validated_strategies)/len(results)*100:.1f}%")

        if validated_strategies:
            avg_actual_sharpe = np.mean([r['lean_backtest'].get('sharpe_ratio', 0) for r in validated_strategies])
            avg_expected_sharpe = np.mean([r['expected_sharpe'] for r in validated_strategies])
            avg_return = np.mean([r['lean_backtest'].get('total_return', 0) for r in validated_strategies])

            logger.info(f"Average Expected Sharpe: {avg_expected_sharpe:.2f}")
            logger.info(f"Average Actual Sharpe: {avg_actual_sharpe:.2f}")
            logger.info(f"Average Total Return: {avg_return:.1%}")
            logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(main())