"""
Parameter Optimization Framework for Technical Indicators

This module provides optimization capabilities for technical indicator parameters
using various optimization algorithms including grid search, random search,
and Bayesian optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from dataclasses import dataclass
from itertools import product
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .technical_indicators import IndicatorLibrary, IndicatorResult


@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict]
    optimization_time: float
    total_evaluations: int
    method: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_evaluations': self.total_evaluations,
            'optimization_time': self.optimization_time,
            'method': self.method,
            'convergence_history': [r['score'] for r in self.all_results]
        }


@dataclass
class ParameterSpace:
    """Define parameter search space"""
    name: str
    min_value: Union[int, float]
    max_value: Union[int, float]
    step: Optional[Union[int, float]] = None
    values: Optional[List[Union[int, float]]] = None
    param_type: str = 'int'  # 'int', 'float', 'categorical'
    
    def get_values(self) -> List[Union[int, float]]:
        """Get all possible values for this parameter"""
        if self.values is not None:
            return self.values
        
        if self.step is None:
            if self.param_type == 'int':
                self.step = 1
            else:
                self.step = (self.max_value - self.min_value) / 10
        
        if self.param_type == 'int':
            return list(range(int(self.min_value), int(self.max_value) + 1, int(self.step)))
        else:
            values = []
            current = self.min_value
            while current <= self.max_value + 1e-10:  # Add small epsilon for floating point comparison
                values.append(round(current, 10))  # Round to avoid floating point precision issues
                current += self.step
            return values


class ObjectiveFunction:
    """Base class for optimization objective functions"""
    
    def __init__(self, name: str):
        self.name = name
    
    def evaluate(self, indicator_result: IndicatorResult, 
                price_data: np.ndarray, **kwargs) -> float:
        """Evaluate the objective function"""
        raise NotImplementedError


class SharpeRatioObjective(ObjectiveFunction):
    """Sharpe ratio objective function for indicator optimization"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        super().__init__("SharpeRatio")
        self.risk_free_rate = risk_free_rate
    
    def evaluate(self, indicator_result: IndicatorResult, 
                price_data: np.ndarray, **kwargs) -> float:
        """
        Calculate Sharpe ratio based on indicator signals
        
        Args:
            indicator_result: Result from technical indicator
            price_data: Price data for return calculation
            **kwargs: Additional parameters
        
        Returns:
            Sharpe ratio (higher is better)
        """
        try:
            # Generate simple trading signals based on indicator
            signals = self._generate_signals(indicator_result, **kwargs)
            
            # Calculate returns
            returns = self._calculate_strategy_returns(signals, price_data)
            
            if len(returns) == 0 or np.std(returns) == 0:
                return -np.inf
            
            # Calculate Sharpe ratio
            excess_returns = np.mean(returns) - self.risk_free_rate / 252  # Daily risk-free rate
            sharpe_ratio = excess_returns / np.std(returns) * np.sqrt(252)  # Annualized
            
            return sharpe_ratio
            
        except Exception as e:
            warnings.warn(f"Error calculating Sharpe ratio: {str(e)}")
            return -np.inf
    
    def _generate_signals(self, indicator_result: IndicatorResult, **kwargs) -> np.ndarray:
        """Generate trading signals from indicator values"""
        values = indicator_result.values
        
        if indicator_result.name == "RSI":
            # RSI signals: buy when < 30, sell when > 70
            oversold_threshold = kwargs.get('oversold_threshold', 30)
            overbought_threshold = kwargs.get('overbought_threshold', 70)
            
            signals = np.zeros_like(values)
            signals[values < oversold_threshold] = 1  # Buy signal
            signals[values > overbought_threshold] = -1  # Sell signal
            
        elif indicator_result.name == "MACD":
            # MACD signals: buy when MACD > signal, sell when MACD < signal
            macd_line = values[:, 0]
            signal_line = values[:, 1]
            
            signals = np.zeros_like(macd_line)
            signals[macd_line > signal_line] = 1  # Buy signal
            signals[macd_line < signal_line] = -1  # Sell signal
            
        elif indicator_result.name == "BollingerBands":
            # Bollinger Bands signals: buy at lower band, sell at upper band
            upper_band = values[:, 0]
            lower_band = values[:, 2]
            
            # Need price data for comparison
            price_data = kwargs.get('price_data', np.zeros_like(upper_band))
            
            signals = np.zeros_like(upper_band)
            signals[price_data <= lower_band] = 1  # Buy signal
            signals[price_data >= upper_band] = -1  # Sell signal
            
        elif indicator_result.name == "ZScore":
            # Z-Score signals: buy when < -2, sell when > 2
            buy_threshold = kwargs.get('buy_threshold', -2)
            sell_threshold = kwargs.get('sell_threshold', 2)
            
            signals = np.zeros_like(values)
            signals[values < buy_threshold] = 1  # Buy signal
            signals[values > sell_threshold] = -1  # Sell signal
            
        else:
            # Default: simple momentum signals
            signals = np.zeros_like(values)
            signals[1:] = np.where(np.diff(values) > 0, 1, -1)
        
        return signals
    
    def _calculate_strategy_returns(self, signals: np.ndarray, 
                                  price_data: np.ndarray) -> np.ndarray:
        """Calculate strategy returns based on signals"""
        if len(signals) != len(price_data):
            min_len = min(len(signals), len(price_data))
            signals = signals[:min_len]
            price_data = price_data[:min_len]
        
        # Calculate price returns
        price_returns = np.diff(price_data) / price_data[:-1]
        
        # Apply signals with 1-day lag (realistic trading assumption)
        strategy_returns = signals[:-1] * price_returns
        
        return strategy_returns[~np.isnan(strategy_returns)]


class ProfitFactorObjective(ObjectiveFunction):
    """Profit factor objective function"""
    
    def __init__(self):
        super().__init__("ProfitFactor")
    
    def evaluate(self, indicator_result: IndicatorResult, 
                price_data: np.ndarray, **kwargs) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        try:
            sharpe_obj = SharpeRatioObjective()
            signals = sharpe_obj._generate_signals(indicator_result, price_data=price_data, **kwargs)
            returns = sharpe_obj._calculate_strategy_returns(signals, price_data)
            
            if len(returns) == 0:
                return 0
            
            gross_profit = np.sum(returns[returns > 0])
            gross_loss = abs(np.sum(returns[returns < 0]))
            
            if gross_loss == 0:
                return np.inf if gross_profit > 0 else 0
            
            return gross_profit / gross_loss
            
        except Exception as e:
            warnings.warn(f"Error calculating profit factor: {str(e)}")
            return 0


class ParameterOptimizer:
    """Main parameter optimization class"""
    
    def __init__(self, indicator_library: Optional[IndicatorLibrary] = None):
        self.indicator_library = indicator_library or IndicatorLibrary()
        self.objective_functions = {
            'sharpe_ratio': SharpeRatioObjective(),
            'profit_factor': ProfitFactorObjective()
        }
    
    def grid_search(self, indicator_name: str, data: np.ndarray,
                   parameter_space: List[ParameterSpace],
                   objective: str = 'sharpe_ratio',
                   n_jobs: int = 1, **kwargs) -> OptimizationResult:
        """
        Perform grid search optimization
        
        Args:
            indicator_name: Name of the indicator to optimize
            data: Price data for optimization
            parameter_space: List of ParameterSpace objects defining search space
            objective: Objective function name
            n_jobs: Number of parallel jobs
            **kwargs: Additional parameters for objective function
        
        Returns:
            OptimizationResult with best parameters and performance
        """
        start_time = time.time()
        
        # Generate all parameter combinations
        param_names = [ps.name for ps in parameter_space]
        param_values = [ps.get_values() for ps in parameter_space]
        param_combinations = list(product(*param_values))
        
        print(f"Starting grid search with {len(param_combinations)} combinations...")
        
        # Evaluate all combinations
        all_results = []
        
        if n_jobs == 1:
            # Sequential execution
            for i, param_combo in enumerate(param_combinations):
                params = dict(zip(param_names, param_combo))
                score = self._evaluate_parameters(indicator_name, data, params, objective, **kwargs)
                
                result = {
                    'params': params,
                    'score': score,
                    'iteration': i
                }
                all_results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(param_combinations)} evaluations")
        
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_params = {
                    executor.submit(self._evaluate_parameters, indicator_name, data, 
                                  dict(zip(param_names, param_combo)), objective, **kwargs): 
                    (i, dict(zip(param_names, param_combo)))
                    for i, param_combo in enumerate(param_combinations)
                }
                
                for future in as_completed(future_to_params):
                    i, params = future_to_params[future]
                    try:
                        score = future.result()
                        result = {
                            'params': params,
                            'score': score,
                            'iteration': i
                        }
                        all_results.append(result)
                        
                        if len(all_results) % 10 == 0:
                            print(f"Completed {len(all_results)}/{len(param_combinations)} evaluations")
                            
                    except Exception as e:
                        warnings.warn(f"Error evaluating parameters {params}: {str(e)}")
        
        # Find best result
        if not all_results:
            raise ValueError("No valid results found during optimization")
        
        best_result = max(all_results, key=lambda x: x['score'])
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=all_results,
            optimization_time=optimization_time,
            total_evaluations=len(all_results),
            method='grid_search'
        )
    
    def random_search(self, indicator_name: str, data: np.ndarray,
                     parameter_space: List[ParameterSpace],
                     n_iterations: int = 100,
                     objective: str = 'sharpe_ratio',
                     random_seed: Optional[int] = None,
                     **kwargs) -> OptimizationResult:
        """
        Perform random search optimization
        
        Args:
            indicator_name: Name of the indicator to optimize
            data: Price data for optimization
            parameter_space: List of ParameterSpace objects defining search space
            n_iterations: Number of random evaluations
            objective: Objective function name
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters for objective function
        
        Returns:
            OptimizationResult with best parameters and performance
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        start_time = time.time()
        all_results = []
        
        print(f"Starting random search with {n_iterations} iterations...")
        
        for i in range(n_iterations):
            # Generate random parameters
            params = {}
            for ps in parameter_space:
                possible_values = ps.get_values()
                params[ps.name] = np.random.choice(possible_values)
            
            # Evaluate parameters
            score = self._evaluate_parameters(indicator_name, data, params, objective, **kwargs)
            
            result = {
                'params': params,
                'score': score,
                'iteration': i
            }
            all_results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{n_iterations} evaluations")
        
        # Find best result
        best_result = max(all_results, key=lambda x: x['score'])
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=all_results,
            optimization_time=optimization_time,
            total_evaluations=len(all_results),
            method='random_search'
        )
    
    def _evaluate_parameters(self, indicator_name: str, data: np.ndarray,
                           params: Dict[str, Any], objective: str, **kwargs) -> float:
        """Evaluate a specific parameter combination"""
        try:
            # Calculate indicator with given parameters
            indicator_result = self.indicator_library.calculate_indicator(
                indicator_name, data, **params
            )
            
            # Evaluate objective function
            objective_func = self.objective_functions[objective]
            score = objective_func.evaluate(indicator_result, data, **kwargs)
            
            return score
            
        except Exception as e:
            warnings.warn(f"Error evaluating parameters {params}: {str(e)}")
            return -np.inf
    
    def optimize_indicator(self, indicator_name: str, data: np.ndarray,
                          method: str = 'grid_search',
                          **kwargs) -> OptimizationResult:
        """
        Optimize indicator parameters using predefined parameter spaces
        
        Args:
            indicator_name: Name of the indicator to optimize
            data: Price data for optimization
            method: Optimization method ('grid_search' or 'random_search')
            **kwargs: Additional parameters for optimization method
        
        Returns:
            OptimizationResult with best parameters
        """
        # Define default parameter spaces for each indicator
        default_spaces = {
            'ema': [ParameterSpace('period', 5, 50, 5, param_type='int')],
            'rsi': [ParameterSpace('period', 5, 30, 5, param_type='int')],
            'macd': [
                ParameterSpace('fast_period', 5, 20, 5, param_type='int'),
                ParameterSpace('slow_period', 20, 50, 5, param_type='int'),
                ParameterSpace('signal_period', 5, 15, 2, param_type='int')
            ],
            'bollinger_bands': [
                ParameterSpace('period', 10, 30, 5, param_type='int'),
                ParameterSpace('std_dev', 1.5, 3.0, 0.5, param_type='float')
            ],
            'z_score': [ParameterSpace('period', 10, 50, 5, param_type='int')]
        }
        
        if indicator_name not in default_spaces:
            raise ValueError(f"No default parameter space for indicator: {indicator_name}")
        
        parameter_space = kwargs.pop('parameter_space', default_spaces[indicator_name])
        
        if method == 'grid_search':
            return self.grid_search(indicator_name, data, parameter_space, **kwargs)
        elif method == 'random_search':
            return self.random_search(indicator_name, data, parameter_space, **kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")


# Convenience functions
def optimize_rsi(data: np.ndarray, method: str = 'grid_search', **kwargs) -> OptimizationResult:
    """Optimize RSI parameters"""
    optimizer = ParameterOptimizer()
    return optimizer.optimize_indicator('rsi', data, method=method, **kwargs)


def optimize_macd(data: np.ndarray, method: str = 'grid_search', **kwargs) -> OptimizationResult:
    """Optimize MACD parameters"""
    optimizer = ParameterOptimizer()
    return optimizer.optimize_indicator('macd', data, method=method, **kwargs)


def optimize_bollinger_bands(data: np.ndarray, method: str = 'grid_search', **kwargs) -> OptimizationResult:
    """Optimize Bollinger Bands parameters"""
    optimizer = ParameterOptimizer()
    return optimizer.optimize_indicator('bollinger_bands', data, method=method, **kwargs)