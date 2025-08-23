"""
Strategy testing framework with parameter optimization and walk-forward analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import json

from .backtest_engine import BacktestEngine, BacktestResults
from .data_loader import DataLoader, create_data_feed
from agents.base_agent import TradingSignal

logger = logging.getLogger(__name__)


@dataclass
class ParameterRange:
    """Parameter range for optimization."""
    name: str
    min_value: float
    max_value: float
    step: float
    param_type: str = 'float'  # 'float', 'int', 'bool'


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    parameters: Dict[str, Any]
    performance_metric: float
    backtest_results: BacktestResults
    rank: int


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis."""
    in_sample_results: BacktestResults
    out_sample_results: BacktestResults
    optimal_parameters: Dict[str, Any]
    period_start: datetime
    period_end: datetime


class StrategyTester:
    """
    Comprehensive strategy testing framework with:
    - Parameter optimization
    - Walk-forward analysis
    - Monte Carlo simulation
    - Statistical testing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data_loader = DataLoader(config=self.config.get('data_loader', {}))
        
        # Optimization settings
        self.optimization_metric = self.config.get('optimization_metric', 'sharpe_ratio')
        self.max_parallel_processes = self.config.get('max_parallel_processes', 4)
        
        logger.info(f"StrategyTester initialized with metric: {self.optimization_metric}")
    
    async def optimize_parameters(
        self,
        strategy_function: Callable,
        parameter_ranges: List[ParameterRange],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 1000000.0,
        max_combinations: int = 1000
    ) -> List[OptimizationResult]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy_function: Strategy function to optimize
            parameter_ranges: List of parameter ranges to test
            symbols: List of symbols to trade
            start_date: Optimization start date
            end_date: Optimization end date
            initial_capital: Initial capital for backtests
            max_combinations: Maximum parameter combinations to test
        
        Returns:
            List of optimization results sorted by performance
        """
        logger.info(f"Starting parameter optimization for {len(parameter_ranges)} parameters")
        
        # Generate parameter combinations
        combinations = self._generate_parameter_combinations(parameter_ranges, max_combinations)
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        # Create data feed
        data_feed = await create_data_feed(symbols, start_date, end_date)
        
        # Run optimization
        results = []
        total_combinations = len(combinations)
        
        for i, params in enumerate(combinations):
            try:
                # Create strategy function with parameters
                parameterized_strategy = self._create_parameterized_strategy(strategy_function, params)
                
                # Run backtest
                engine = BacktestEngine(initial_capital, self.config.get('backtest_engine', {}))
                engine.add_data_feed(data_feed)
                
                backtest_result = await engine.run_backtest(
                    start_date, end_date, parameterized_strategy, symbols
                )
                
                # Extract performance metric
                performance = self._extract_performance_metric(backtest_result)
                
                results.append(OptimizationResult(
                    parameters=params.copy(),
                    performance_metric=performance,
                    backtest_results=backtest_result,
                    rank=0  # Will be set later
                ))
                
                # Log progress
                if (i + 1) % max(1, total_combinations // 10) == 0:
                    progress = ((i + 1) / total_combinations) * 100
                    logger.info(f"Optimization progress: {progress:.1f}% - Best: {max([r.performance_metric for r in results]):.4f}")
                
            except Exception as e:
                logger.error(f"Error in parameter combination {i}: {e}")
                continue
        
        # Sort and rank results
        results.sort(key=lambda x: x.performance_metric, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        
        logger.info(f"Optimization complete. Best {self.optimization_metric}: {results[0].performance_metric:.4f}")
        
        return results
    
    async def walk_forward_analysis(
        self,
        strategy_function: Callable,
        parameter_ranges: List[ParameterRange],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        in_sample_months: int = 12,
        out_sample_months: int = 3,
        step_months: int = 1,
        initial_capital: float = 1000000.0
    ) -> List[WalkForwardResult]:
        """
        Perform walk-forward analysis with rolling parameter optimization.
        
        Args:
            strategy_function: Strategy function to test
            parameter_ranges: Parameter ranges for optimization
            symbols: List of symbols to trade
            start_date: Analysis start date
            end_date: Analysis end date
            in_sample_months: Months of in-sample data for optimization
            out_sample_months: Months of out-of-sample testing
            step_months: Months to step forward each iteration
            initial_capital: Initial capital for backtests
        
        Returns:
            List of walk-forward results
        """
        logger.info(f"Starting walk-forward analysis: {in_sample_months}M in-sample, {out_sample_months}M out-sample")
        
        results = []
        current_date = start_date
        
        while current_date + timedelta(days=in_sample_months*30 + out_sample_months*30) <= end_date:
            # Define periods
            in_sample_start = current_date
            in_sample_end = current_date + timedelta(days=in_sample_months*30)
            out_sample_start = in_sample_end
            out_sample_end = out_sample_start + timedelta(days=out_sample_months*30)
            
            logger.info(f"Walk-forward period: {in_sample_start.date()} to {out_sample_end.date()}")
            
            try:
                # Optimize parameters on in-sample data
                optimization_results = await self.optimize_parameters(
                    strategy_function,
                    parameter_ranges,
                    symbols,
                    in_sample_start,
                    in_sample_end,
                    initial_capital,
                    max_combinations=100  # Reduced for walk-forward
                )
                
                if not optimization_results:
                    logger.warning(f"No optimization results for period {in_sample_start.date()}")
                    current_date += timedelta(days=step_months*30)
                    continue
                
                # Get best parameters
                best_params = optimization_results[0].parameters
                in_sample_results = optimization_results[0].backtest_results
                
                # Test on out-of-sample data
                parameterized_strategy = self._create_parameterized_strategy(strategy_function, best_params)
                
                out_sample_data_feed = await create_data_feed(symbols, out_sample_start, out_sample_end)
                out_sample_engine = BacktestEngine(initial_capital, self.config.get('backtest_engine', {}))
                out_sample_engine.add_data_feed(out_sample_data_feed)
                
                out_sample_results = await out_sample_engine.run_backtest(
                    out_sample_start, out_sample_end, parameterized_strategy, symbols
                )
                
                results.append(WalkForwardResult(
                    in_sample_results=in_sample_results,
                    out_sample_results=out_sample_results,
                    optimal_parameters=best_params,
                    period_start=in_sample_start,
                    period_end=out_sample_end
                ))
                
                logger.info(f"Period complete - In-sample {self.optimization_metric}: {self._extract_performance_metric(in_sample_results):.4f}, "
                           f"Out-sample: {self._extract_performance_metric(out_sample_results):.4f}")
                
            except Exception as e:
                logger.error(f"Error in walk-forward period {current_date.date()}: {e}")
            
            # Step forward
            current_date += timedelta(days=step_months*30)
        
        logger.info(f"Walk-forward analysis complete. {len(results)} periods tested.")
        
        return results
    
    async def monte_carlo_analysis(
        self,
        strategy_function: Callable,
        parameters: Dict[str, Any],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        num_simulations: int = 1000,
        initial_capital: float = 1000000.0
    ) -> Dict[str, Any]:
        """
        Perform Monte Carlo analysis by randomizing trade order and timing.
        
        Args:
            strategy_function: Strategy function to test
            parameters: Fixed strategy parameters
            symbols: List of symbols to trade
            start_date: Analysis start date
            end_date: Analysis end date
            num_simulations: Number of Monte Carlo simulations
            initial_capital: Initial capital for backtests
        
        Returns:
            Dictionary with Monte Carlo results and statistics
        """
        logger.info(f"Starting Monte Carlo analysis with {num_simulations} simulations")
        
        # Run base backtest to get original trades
        parameterized_strategy = self._create_parameterized_strategy(strategy_function, parameters)
        data_feed = await create_data_feed(symbols, start_date, end_date)
        
        base_engine = BacktestEngine(initial_capital, self.config.get('backtest_engine', {}))
        base_engine.add_data_feed(data_feed)
        
        base_results = await base_engine.run_backtest(
            start_date, end_date, parameterized_strategy, symbols
        )
        
        # Extract base performance
        base_performance = self._extract_performance_metric(base_results)
        
        # Run Monte Carlo simulations
        simulation_results = []
        
        for sim in range(num_simulations):
            try:
                # Create randomized strategy
                randomized_strategy = self._create_randomized_strategy(
                    strategy_function, parameters, randomization_factor=0.1
                )
                
                # Run simulation
                sim_engine = BacktestEngine(initial_capital, self.config.get('backtest_engine', {}))
                sim_engine.add_data_feed(data_feed)
                
                sim_results = await sim_engine.run_backtest(
                    start_date, end_date, randomized_strategy, symbols
                )
                
                performance = self._extract_performance_metric(sim_results)
                simulation_results.append(performance)
                
                if (sim + 1) % max(1, num_simulations // 10) == 0:
                    progress = ((sim + 1) / num_simulations) * 100
                    logger.info(f"Monte Carlo progress: {progress:.1f}%")
                
            except Exception as e:
                logger.error(f"Error in simulation {sim}: {e}")
                continue
        
        # Calculate statistics
        simulation_results = np.array(simulation_results)
        
        results = {
            'base_performance': base_performance,
            'num_simulations': len(simulation_results),
            'mean_performance': np.mean(simulation_results),
            'std_performance': np.std(simulation_results),
            'min_performance': np.min(simulation_results),
            'max_performance': np.max(simulation_results),
            'percentiles': {
                '5th': np.percentile(simulation_results, 5),
                '25th': np.percentile(simulation_results, 25),
                '50th': np.percentile(simulation_results, 50),
                '75th': np.percentile(simulation_results, 75),
                '95th': np.percentile(simulation_results, 95)
            },
            'probability_positive': np.mean(simulation_results > 0),
            'probability_better_than_base': np.mean(simulation_results > base_performance),
            'simulation_results': simulation_results.tolist()
        }
        
        logger.info(f"Monte Carlo complete - Mean: {results['mean_performance']:.4f}, "
                   f"Std: {results['std_performance']:.4f}")
        
        return results
    
    def _generate_parameter_combinations(
        self,
        parameter_ranges: List[ParameterRange],
        max_combinations: int
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations within limits."""
        param_lists = []
        
        for param_range in parameter_ranges:
            if param_range.param_type == 'float':
                values = np.arange(
                    param_range.min_value,
                    param_range.max_value + param_range.step,
                    param_range.step
                ).tolist()
            elif param_range.param_type == 'int':
                values = list(range(
                    int(param_range.min_value),
                    int(param_range.max_value) + 1,
                    int(param_range.step)
                ))
            elif param_range.param_type == 'bool':
                values = [True, False]
            else:
                values = [param_range.min_value]
            
            param_lists.append((param_range.name, values))
        
        # Generate all combinations
        names, value_lists = zip(*param_lists)
        all_combinations = list(itertools.product(*value_lists))
        
        # Limit combinations if necessary
        if len(all_combinations) > max_combinations:
            logger.warning(f"Limiting combinations from {len(all_combinations)} to {max_combinations}")
            # Use random sampling to select combinations
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        
        # Convert to list of dictionaries
        combinations = []
        for combo in all_combinations:
            combinations.append(dict(zip(names, combo)))
        
        return combinations
    
    def _create_parameterized_strategy(
        self,
        strategy_function: Callable,
        parameters: Dict[str, Any]
    ) -> Callable:
        """Create a strategy function with fixed parameters."""
        def parameterized_strategy(timestamp: datetime, context: Dict[str, Any]) -> List[TradingSignal]:
            # Add parameters to context
            context['parameters'] = parameters
            return strategy_function(timestamp, context)
        
        return parameterized_strategy
    
    def _create_randomized_strategy(
        self,
        strategy_function: Callable,
        parameters: Dict[str, Any],
        randomization_factor: float = 0.1
    ) -> Callable:
        """Create a strategy with randomized parameters for Monte Carlo."""
        randomized_params = {}
        
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                # Add random noise to numeric parameters
                noise = np.random.normal(0, abs(value) * randomization_factor)
                randomized_params[key] = value + noise
            else:
                randomized_params[key] = value
        
        return self._create_parameterized_strategy(strategy_function, randomized_params)
    
    def _extract_performance_metric(self, results: BacktestResults) -> float:
        """Extract the optimization metric from backtest results."""
        if self.optimization_metric == 'sharpe_ratio':
            return results.sharpe_ratio
        elif self.optimization_metric == 'total_return':
            return results.total_return
        elif self.optimization_metric == 'annualized_return':
            return results.annualized_return
        elif self.optimization_metric == 'profit_factor':
            return results.profit_factor
        elif self.optimization_metric == 'win_rate':
            return results.win_rate
        elif self.optimization_metric == 'max_drawdown':
            return -results.max_drawdown  # Negative because we want to minimize drawdown
        else:
            logger.warning(f"Unknown optimization metric: {self.optimization_metric}, using Sharpe ratio")
            return results.sharpe_ratio
    
    async def generate_optimization_report(
        self,
        optimization_results: List[OptimizationResult],
        output_path: str = "optimization_report.json"
    ):
        """Generate comprehensive optimization report."""
        
        if not optimization_results:
            logger.warning("No optimization results to report")
            return
        
        # Extract top results
        top_10 = optimization_results[:10]
        
        # Calculate parameter stability (how often parameters appear in top results)
        param_frequency = {}
        for result in top_10:
            for param, value in result.parameters.items():
                if param not in param_frequency:
                    param_frequency[param] = {}
                if value not in param_frequency[param]:
                    param_frequency[param][value] = 0
                param_frequency[param][value] += 1
        
        # Performance distribution
        performances = [r.performance_metric for r in optimization_results]
        
        report = {
            'optimization_summary': {
                'metric': self.optimization_metric,
                'total_combinations': len(optimization_results),
                'best_performance': optimization_results[0].performance_metric,
                'worst_performance': optimization_results[-1].performance_metric,
                'mean_performance': np.mean(performances),
                'std_performance': np.std(performances),
                'performance_distribution': {
                    '25th_percentile': np.percentile(performances, 25),
                    '50th_percentile': np.percentile(performances, 50),
                    '75th_percentile': np.percentile(performances, 75),
                    '90th_percentile': np.percentile(performances, 90)
                }
            },
            'top_10_results': [
                {
                    'rank': result.rank,
                    'parameters': result.parameters,
                    'performance': result.performance_metric,
                    'total_return': result.backtest_results.total_return,
                    'sharpe_ratio': result.backtest_results.sharpe_ratio,
                    'max_drawdown': result.backtest_results.max_drawdown,
                    'win_rate': result.backtest_results.win_rate
                }
                for result in top_10
            ],
            'parameter_stability': param_frequency,
            'best_parameters': optimization_results[0].parameters
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Optimization report saved to {output_path}")
        
        return report
    
    async def generate_walkforward_report(
        self,
        walkforward_results: List[WalkForwardResult],
        output_path: str = "walkforward_report.json"
    ):
        """Generate comprehensive walk-forward analysis report."""
        
        if not walkforward_results:
            logger.warning("No walk-forward results to report")
            return
        
        # Extract performance metrics
        in_sample_performances = [self._extract_performance_metric(r.in_sample_results) for r in walkforward_results]
        out_sample_performances = [self._extract_performance_metric(r.out_sample_results) for r in walkforward_results]
        
        # Calculate correlation between in-sample and out-of-sample
        correlation = np.corrcoef(in_sample_performances, out_sample_performances)[0, 1] if len(in_sample_performances) > 1 else 0
        
        report = {
            'walkforward_summary': {
                'total_periods': len(walkforward_results),
                'correlation_in_out_sample': correlation,
                'avg_in_sample_performance': np.mean(in_sample_performances),
                'avg_out_sample_performance': np.mean(out_sample_performances),
                'std_in_sample_performance': np.std(in_sample_performances),
                'std_out_sample_performance': np.std(out_sample_performances),
                'percentage_profitable_periods': np.mean([p > 0 for p in out_sample_performances]) * 100
            },
            'period_results': [
                {
                    'period': i + 1,
                    'start_date': result.period_start.isoformat(),
                    'end_date': result.period_end.isoformat(),
                    'optimal_parameters': result.optimal_parameters,
                    'in_sample_performance': self._extract_performance_metric(result.in_sample_results),
                    'out_sample_performance': self._extract_performance_metric(result.out_sample_results),
                    'in_sample_total_return': result.in_sample_results.total_return,
                    'out_sample_total_return': result.out_sample_results.total_return
                }
                for i, result in enumerate(walkforward_results)
            ],
            'performance_stability': {
                'in_sample_sharpe': np.mean([r.in_sample_results.sharpe_ratio for r in walkforward_results]),
                'out_sample_sharpe': np.mean([r.out_sample_results.sharpe_ratio for r in walkforward_results]),
                'avg_degradation': np.mean(in_sample_performances) - np.mean(out_sample_performances)
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Walk-forward report saved to {output_path}")
        
        return report