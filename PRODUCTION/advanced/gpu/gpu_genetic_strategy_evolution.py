"""
GPU GENETIC ALGORITHM STRATEGY EVOLUTION
Evolve sophisticated trading strategies using GTX 1660 Super acceleration
Genetic algorithms discover optimal parameter combinations and strategy logic
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
import json
import copy
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@dataclass
class TradingGene:
    """Individual gene representing a trading parameter or rule"""
    parameter_name: str
    value: Union[float, int, str]
    min_value: Union[float, int] = 0
    max_value: Union[float, int] = 100
    mutation_rate: float = 0.1
    gene_type: str = 'float'  # 'float', 'int', 'categorical'
    choices: Optional[List] = None

@dataclass
class TradingStrategy:
    """Complete trading strategy genome"""
    strategy_id: str
    genes: List[TradingGene] = field(default_factory=list)
    fitness_score: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

class GPUStrategyEvaluator:
    """GPU-accelerated strategy backtesting and evaluation"""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('StrategyEvaluator')

        # Market data for backtesting
        self.price_data = None
        self.volume_data = None
        self.returns_data = None

        # Evaluation metrics
        self.evaluation_count = 0
        self.strategies_per_second = 0

        self.logger.info(f"Strategy evaluator initialized on {self.device}")

    def generate_synthetic_market_data(self, days: int = 252, symbols: int = 10) -> torch.Tensor:
        """
        Generate synthetic market data for strategy testing

        Args:
            days: Number of trading days
            symbols: Number of symbols

        Returns:
            Market data tensor (symbols, days, features)
        """
        # Generate realistic price movements
        np.random.seed(42)  # For reproducible results

        market_data = torch.zeros(symbols, days, 6, device=self.device)  # OHLCV + Returns

        for symbol in range(symbols):
            # Starting price
            initial_price = 50 + np.random.uniform(0, 200)

            # Generate price series with trend and volatility
            drift = np.random.uniform(-0.001, 0.002)  # Daily drift
            volatility = np.random.uniform(0.01, 0.03)  # Daily volatility

            prices = [initial_price]
            for day in range(days - 1):
                # Random walk with drift
                price_change = np.random.normal(drift, volatility)
                new_price = prices[-1] * (1 + price_change)
                prices.append(max(new_price, 0.01))  # Prevent negative prices

            prices = np.array(prices)

            # Generate OHLC from close prices
            opens = prices + np.random.normal(0, 0.001, len(prices))
            highs = np.maximum(prices, opens) + np.abs(np.random.normal(0, 0.002, len(prices)))
            lows = np.minimum(prices, opens) - np.abs(np.random.normal(0, 0.002, len(prices)))
            volumes = np.random.lognormal(10, 1, len(prices))

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])  # Add zero return for first day

            # Store in tensor
            market_data[symbol, :, 0] = torch.tensor(opens, device=self.device)
            market_data[symbol, :, 1] = torch.tensor(highs, device=self.device)
            market_data[symbol, :, 2] = torch.tensor(lows, device=self.device)
            market_data[symbol, :, 3] = torch.tensor(prices, device=self.device)
            market_data[symbol, :, 4] = torch.tensor(volumes, device=self.device)
            market_data[symbol, :, 5] = torch.tensor(returns, device=self.device)

        return market_data

    def calculate_technical_indicators(self, market_data: torch.Tensor) -> torch.Tensor:
        """
        Calculate technical indicators for strategy evaluation

        Args:
            market_data: Market data tensor (symbols, days, features)

        Returns:
            Technical indicators tensor
        """
        symbols, days, features = market_data.shape
        # Calculate multiple technical indicators
        indicators = torch.zeros(symbols, days, 10, device=self.device)

        # Extract price and volume data
        opens = market_data[:, :, 0]
        highs = market_data[:, :, 1]
        lows = market_data[:, :, 2]
        closes = market_data[:, :, 3]
        volumes = market_data[:, :, 4]
        returns = market_data[:, :, 5]

        # Moving averages
        sma_5 = self.moving_average(closes, 5)
        sma_20 = self.moving_average(closes, 20)
        sma_50 = self.moving_average(closes, 50)

        # RSI
        rsi = self.calculate_rsi(closes, 14)

        # MACD
        macd, macd_signal = self.calculate_macd(closes)

        # Bollinger Bands
        bb_upper, bb_lower = self.calculate_bollinger_bands(closes, 20, 2.0)

        # Volume indicators
        volume_sma = self.moving_average(volumes, 20)
        volume_ratio = volumes / torch.clamp(volume_sma, min=1e-8)

        # Store indicators
        indicators[:, :, 0] = sma_5
        indicators[:, :, 1] = sma_20
        indicators[:, :, 2] = sma_50
        indicators[:, :, 3] = rsi
        indicators[:, :, 4] = macd
        indicators[:, :, 5] = macd_signal
        indicators[:, :, 6] = bb_upper
        indicators[:, :, 7] = bb_lower
        indicators[:, :, 8] = volume_ratio
        indicators[:, :, 9] = returns

        return indicators

    def moving_average(self, data: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate moving average using GPU convolution"""
        kernel = torch.ones(window, device=self.device) / window
        padded_data = torch.nn.functional.pad(data, (window - 1, 0), mode='replicate')
        return torch.nn.functional.conv1d(
            padded_data.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0)
        ).squeeze(1)

    def calculate_rsi(self, prices: torch.Tensor, window: int = 14) -> torch.Tensor:
        """Calculate RSI indicator"""
        price_changes = torch.diff(prices, dim=1)
        gains = torch.clamp(price_changes, min=0)
        losses = torch.clamp(-price_changes, min=0)

        # Calculate average gains and losses
        avg_gains = self.moving_average(
            torch.cat([torch.zeros(prices.shape[0], 1, device=self.device), gains], dim=1),
            window
        )
        avg_losses = self.moving_average(
            torch.cat([torch.zeros(prices.shape[0], 1, device=self.device), losses], dim=1),
            window
        )

        # RSI calculation
        rs = avg_gains / torch.clamp(avg_losses, min=1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, prices: torch.Tensor, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate MACD indicator"""
        ema_fast = self.exponential_moving_average(prices, fast)
        ema_slow = self.exponential_moving_average(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = self.exponential_moving_average(macd, signal)
        return macd, macd_signal

    def exponential_moving_average(self, data: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate exponential moving average"""
        alpha = 2.0 / (window + 1)
        ema = torch.zeros_like(data)
        ema[:, 0] = data[:, 0]

        for i in range(1, data.shape[1]):
            ema[:, i] = alpha * data[:, i] + (1 - alpha) * ema[:, i - 1]

        return ema

    def calculate_bollinger_bands(self, prices: torch.Tensor, window: int = 20, std_mult: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate Bollinger Bands"""
        sma = self.moving_average(prices, window)

        # Calculate rolling standard deviation
        squared_diffs = (prices - sma) ** 2
        variance = self.moving_average(squared_diffs, window)
        std = torch.sqrt(variance)

        upper_band = sma + (std_mult * std)
        lower_band = sma - (std_mult * std)

        return upper_band, lower_band

    def batch_evaluate_strategies(self, strategies: List[TradingStrategy], market_data: torch.Tensor) -> List[TradingStrategy]:
        """
        Evaluate multiple trading strategies simultaneously using GPU

        Args:
            strategies: List of strategies to evaluate
            market_data: Market data for backtesting

        Returns:
            Evaluated strategies with fitness scores
        """
        start_time = time.time()

        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(market_data)

        # Evaluate each strategy
        evaluated_strategies = []

        for strategy in strategies:
            # Generate trading signals
            signals = self.generate_strategy_signals(strategy, indicators)

            # Calculate performance metrics
            performance = self.calculate_strategy_performance(signals, market_data)

            # Update strategy with results
            strategy.fitness_score = performance['fitness']
            strategy.sharpe_ratio = performance['sharpe_ratio']
            strategy.max_drawdown = performance['max_drawdown']
            strategy.total_return = performance['total_return']
            strategy.win_rate = performance['win_rate']

            evaluated_strategies.append(strategy)

        evaluation_time = time.time() - start_time
        self.evaluation_count += len(strategies)
        self.strategies_per_second = len(strategies) / evaluation_time if evaluation_time > 0 else 0

        self.logger.info(f"Evaluated {len(strategies)} strategies in {evaluation_time:.4f}s "
                        f"({self.strategies_per_second:.1f} strategies/second)")

        return evaluated_strategies

    def generate_strategy_signals(self, strategy: TradingStrategy, indicators: torch.Tensor) -> torch.Tensor:
        """
        Generate trading signals based on strategy parameters

        Args:
            strategy: Trading strategy
            indicators: Technical indicators

        Returns:
            Trading signals tensor (symbols, days)
        """
        symbols, days, num_indicators = indicators.shape
        signals = torch.zeros(symbols, days, device=self.device)

        # Extract strategy parameters
        params = {gene.parameter_name: gene.value for gene in strategy.genes}

        # Simple momentum strategy example
        sma_short_period = int(params.get('sma_short_period', 5))
        sma_long_period = int(params.get('sma_long_period', 20))
        rsi_overbought = params.get('rsi_overbought', 70)
        rsi_oversold = params.get('rsi_oversold', 30)
        volume_threshold = params.get('volume_threshold', 1.5)

        # Get indicators
        sma_short = indicators[:, :, 0] if sma_short_period == 5 else self.moving_average(indicators[:, :, 0], sma_short_period)
        sma_long = indicators[:, :, 1] if sma_long_period == 20 else self.moving_average(indicators[:, :, 1], sma_long_period)
        rsi = indicators[:, :, 3]
        volume_ratio = indicators[:, :, 8]

        # Generate signals
        # Buy signal: SMA crossover + RSI oversold + high volume
        buy_signals = (
            (sma_short > sma_long) &
            (rsi < rsi_oversold) &
            (volume_ratio > volume_threshold)
        ).float()

        # Sell signal: SMA crossover down + RSI overbought
        sell_signals = (
            (sma_short < sma_long) &
            (rsi > rsi_overbought)
        ).float()

        # Combine signals (1 = buy, -1 = sell, 0 = hold)
        signals = buy_signals - sell_signals

        return signals

    def calculate_strategy_performance(self, signals: torch.Tensor, market_data: torch.Tensor) -> Dict[str, float]:
        """
        Calculate strategy performance metrics

        Args:
            signals: Trading signals
            market_data: Market data

        Returns:
            Performance metrics dictionary
        """
        returns = market_data[:, :, 5]  # Daily returns
        symbols, days = signals.shape

        # Calculate strategy returns
        strategy_returns = signals[:, :-1] * returns[:, 1:]  # Shift returns for realistic trading

        # Portfolio returns (equal weight across symbols)
        portfolio_returns = torch.mean(strategy_returns, dim=0)

        # Calculate metrics
        total_return = torch.sum(portfolio_returns).item()
        returns_std = torch.std(portfolio_returns).item()
        sharpe_ratio = (torch.mean(portfolio_returns) / torch.clamp(torch.std(portfolio_returns), min=1e-8)).item() * np.sqrt(252)

        # Maximum drawdown
        cumulative_returns = torch.cumsum(portfolio_returns, dim=0)
        running_max = torch.cummax(cumulative_returns, dim=0)[0]
        drawdowns = cumulative_returns - running_max
        max_drawdown = torch.min(drawdowns).item()

        # Win rate
        positive_returns = (portfolio_returns > 0).float()
        win_rate = torch.mean(positive_returns).item()

        # Fitness score (combination of metrics)
        fitness = sharpe_ratio - abs(max_drawdown) + total_return * 0.1

        return {
            'fitness': fitness,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'win_rate': win_rate
        }

class GeneticAlgorithmEngine:
    """Genetic algorithm engine for strategy evolution"""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('GeneticEngine')

        # Evolution parameters
        self.population_size = 200
        self.elite_size = 20
        self.mutation_rate = 0.15
        self.crossover_rate = 0.8

        # Strategy evaluator
        self.evaluator = GPUStrategyEvaluator(device)

        # Evolution tracking
        self.generation = 0
        self.best_strategies = []
        self.evolution_history = []

        self.logger.info(f"Genetic algorithm engine initialized on {self.device}")

    def create_initial_population(self) -> List[TradingStrategy]:
        """Create initial population of random trading strategies"""
        population = []

        strategy_templates = [
            # Momentum strategies
            {
                'sma_short_period': (3, 15, 'int'),
                'sma_long_period': (15, 50, 'int'),
                'rsi_overbought': (60, 85, 'float'),
                'rsi_oversold': (15, 40, 'float'),
                'volume_threshold': (1.0, 3.0, 'float')
            },
            # Mean reversion strategies
            {
                'bollinger_period': (10, 30, 'int'),
                'bollinger_std': (1.5, 3.0, 'float'),
                'rsi_period': (10, 20, 'int'),
                'mean_revert_threshold': (0.02, 0.10, 'float'),
                'stop_loss': (0.02, 0.05, 'float')
            },
            # Breakout strategies
            {
                'breakout_period': (5, 25, 'int'),
                'breakout_threshold': (0.01, 0.05, 'float'),
                'volume_confirm': (1.2, 2.5, 'float'),
                'momentum_filter': (0.5, 2.0, 'float'),
                'profit_target': (0.02, 0.08, 'float')
            }
        ]

        for i in range(self.population_size):
            # Select random strategy template
            template = random.choice(strategy_templates)

            # Create genes
            genes = []
            for param_name, (min_val, max_val, param_type) in template.items():
                if param_type == 'int':
                    value = random.randint(min_val, max_val)
                else:
                    value = random.uniform(min_val, max_val)

                gene = TradingGene(
                    parameter_name=param_name,
                    value=value,
                    min_value=min_val,
                    max_value=max_val,
                    gene_type=param_type
                )
                genes.append(gene)

            # Create strategy
            strategy = TradingStrategy(
                strategy_id=f"gen0_strategy_{i}",
                genes=genes,
                generation=0
            )
            population.append(strategy)

        self.logger.info(f"Created initial population of {len(population)} strategies")
        return population

    def select_parents(self, population: List[TradingStrategy]) -> List[TradingStrategy]:
        """Select parents for reproduction using tournament selection"""
        tournament_size = 5
        parents = []

        for _ in range(self.population_size):
            # Tournament selection
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda s: s.fitness_score)
            parents.append(winner)

        return parents

    def crossover(self, parent1: TradingStrategy, parent2: TradingStrategy) -> Tuple[TradingStrategy, TradingStrategy]:
        """Create offspring through crossover"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        # Single-point crossover
        min_genes = min(len(parent1.genes), len(parent2.genes))
        if min_genes <= 1:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        crossover_point = random.randint(1, min_genes - 1)

        # Create offspring
        offspring1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        offspring2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]

        offspring1 = TradingStrategy(
            strategy_id=f"gen{self.generation + 1}_offspring_{random.randint(1000, 9999)}",
            genes=offspring1_genes,
            generation=self.generation + 1,
            parent_ids=[parent1.strategy_id, parent2.strategy_id]
        )

        offspring2 = TradingStrategy(
            strategy_id=f"gen{self.generation + 1}_offspring_{random.randint(1000, 9999)}",
            genes=offspring2_genes,
            generation=self.generation + 1,
            parent_ids=[parent1.strategy_id, parent2.strategy_id]
        )

        return offspring1, offspring2

    def mutate(self, strategy: TradingStrategy) -> TradingStrategy:
        """Apply mutation to strategy genes"""
        mutated_strategy = copy.deepcopy(strategy)

        for gene in mutated_strategy.genes:
            if random.random() < self.mutation_rate:
                if gene.gene_type == 'int':
                    gene.value = random.randint(gene.min_value, gene.max_value)
                elif gene.gene_type == 'float':
                    # Gaussian mutation
                    mutation_strength = (gene.max_value - gene.min_value) * 0.1
                    new_value = gene.value + random.gauss(0, mutation_strength)
                    gene.value = max(gene.min_value, min(gene.max_value, new_value))

        return mutated_strategy

    def evolve_generation(self, population: List[TradingStrategy], market_data: torch.Tensor) -> List[TradingStrategy]:
        """Evolve one generation of strategies"""
        self.logger.info(f"Evolving generation {self.generation}...")

        # Evaluate population
        evaluated_population = self.evaluator.batch_evaluate_strategies(population, market_data)

        # Sort by fitness
        evaluated_population.sort(key=lambda s: s.fitness_score, reverse=True)

        # Track best strategies
        generation_best = evaluated_population[:5]
        self.best_strategies.extend(generation_best)

        # Log generation statistics
        avg_fitness = np.mean([s.fitness_score for s in evaluated_population])
        best_fitness = evaluated_population[0].fitness_score

        self.logger.info(f"Generation {self.generation} - Best fitness: {best_fitness:.4f}, "
                        f"Average fitness: {avg_fitness:.4f}")

        # Selection and reproduction
        parents = self.select_parents(evaluated_population)
        new_population = []

        # Elite selection (keep best strategies)
        elite = evaluated_population[:self.elite_size]
        new_population.extend(elite)

        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            offspring1, offspring2 = self.crossover(parent1, parent2)

            # Mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)

            new_population.extend([offspring1, offspring2])

        # Trim to exact population size
        new_population = new_population[:self.population_size]

        # Update generation counter
        self.generation += 1

        return new_population

    def run_evolution(self, generations: int = 50) -> List[TradingStrategy]:
        """
        Run complete genetic algorithm evolution

        Args:
            generations: Number of generations to evolve

        Returns:
            Best evolved strategies
        """
        self.logger.info(f"Starting genetic algorithm evolution for {generations} generations...")

        # Generate market data for backtesting
        market_data = self.evaluator.generate_synthetic_market_data(days=252, symbols=5)

        # Create initial population
        population = self.create_initial_population()

        # Evolution loop
        for gen in range(generations):
            population = self.evolve_generation(population, market_data)

            # Save progress periodically
            if gen % 10 == 0:
                self.save_evolution_progress()

        # Final evaluation and selection
        final_population = self.evaluator.batch_evaluate_strategies(population, market_data)
        final_population.sort(key=lambda s: s.fitness_score, reverse=True)

        self.logger.info(f"Evolution completed. Best strategy fitness: {final_population[0].fitness_score:.4f}")

        return final_population[:10]  # Return top 10 strategies

    def save_evolution_progress(self):
        """Save evolution progress to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare data for saving
        progress_data = {
            'generation': self.generation,
            'best_strategies': [],
            'evolution_stats': {
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            }
        }

        # Add best strategies (limit to prevent huge files)
        for strategy in self.best_strategies[-20:]:  # Last 20 best strategies
            strategy_data = {
                'strategy_id': strategy.strategy_id,
                'fitness_score': strategy.fitness_score,
                'sharpe_ratio': strategy.sharpe_ratio,
                'total_return': strategy.total_return,
                'generation': strategy.generation,
                'parameters': {gene.parameter_name: gene.value for gene in strategy.genes}
            }
            progress_data['best_strategies'].append(strategy_data)

        # Save to file
        filename = f'genetic_evolution_progress_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(progress_data, f, indent=2)

        self.logger.info(f"Evolution progress saved to {filename}")

def demo_genetic_evolution():
    """Demonstration of genetic algorithm strategy evolution"""
    print("\n" + "="*80)
    print("GPU GENETIC ALGORITHM STRATEGY EVOLUTION DEMONSTRATION")
    print("="*80)

    # Initialize genetic engine
    genetic_engine = GeneticAlgorithmEngine()

    print(f"\n>> Genetic Algorithm Engine initialized on {genetic_engine.device}")
    print(f">> Population size: {genetic_engine.population_size}")
    print(f">> Elite size: {genetic_engine.elite_size}")
    print(f">> Mutation rate: {genetic_engine.mutation_rate}")

    # Run short evolution for demonstration
    print(f"\n>> Running evolution for 5 generations (demo)...")

    best_strategies = genetic_engine.run_evolution(generations=5)

    print(f"\n>> Evolution completed!")
    print(f">> Strategy evaluation rate: {genetic_engine.evaluator.strategies_per_second:.1f} strategies/second")

    # Display best strategies
    print(f"\n>> TOP EVOLVED STRATEGIES:")
    for i, strategy in enumerate(best_strategies[:3]):
        print(f"   {i+1}. Strategy {strategy.strategy_id}")
        print(f"      Fitness Score: {strategy.fitness_score:.4f}")
        print(f"      Sharpe Ratio: {strategy.sharpe_ratio:.3f}")
        print(f"      Total Return: {strategy.total_return:.3f}")
        print(f"      Max Drawdown: {strategy.max_drawdown:.3f}")
        print(f"      Win Rate: {strategy.win_rate:.3f}")

        # Show key parameters
        params = {gene.parameter_name: gene.value for gene in strategy.genes[:3]}
        print(f"      Key Parameters: {params}")

    print(f"\n" + "="*80)
    print("GENETIC STRATEGY EVOLUTION SYSTEM READY!")
    print("Use genetic_engine.run_evolution(50) for full evolution")
    print("="*80)

if __name__ == "__main__":
    demo_genetic_evolution()