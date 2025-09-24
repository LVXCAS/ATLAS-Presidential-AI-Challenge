import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import random
import json
import pickle
from pathlib import Path
import uuid
import copy
from collections import defaultdict, deque
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from event_bus import TradingEventBus, Event, Priority


class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    MOMENTUM_RSI = "momentum_rsi"
    BOLLINGER_BANDS = "bollinger_bands"
    MOVING_AVERAGE_CROSSOVER = "moving_average_crossover"
    VOLATILITY_TRADING = "volatility_trading"


class IndicatorType(Enum):
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER = "bollinger"
    STOCHASTIC = "stochastic"
    ATR = "atr"
    VOLUME = "volume"
    PRICE_VELOCITY = "price_velocity"
    MOMENTUM = "momentum"


@dataclass
class StrategyGene:
    """Individual gene representing a strategy parameter"""
    gene_type: str
    value: Any
    min_value: Any = None
    max_value: Any = None
    mutation_rate: float = 0.1
    is_discrete: bool = False
    allowed_values: List[Any] = field(default_factory=list)
    
    def mutate(self) -> 'StrategyGene':
        """Mutate this gene"""
        if random.random() > self.mutation_rate:
            return copy.deepcopy(self)
        
        new_gene = copy.deepcopy(self)
        
        if self.is_discrete and self.allowed_values:
            # Discrete mutation
            new_gene.value = random.choice(self.allowed_values)
        elif self.min_value is not None and self.max_value is not None:
            if isinstance(self.value, int):
                # Integer mutation
                mutation_range = max(1, int((self.max_value - self.min_value) * 0.1))
                new_gene.value = max(self.min_value, min(self.max_value, 
                    self.value + random.randint(-mutation_range, mutation_range)))
            elif isinstance(self.value, float):
                # Float mutation
                mutation_range = (self.max_value - self.min_value) * 0.1
                new_gene.value = max(self.min_value, min(self.max_value,
                    self.value + random.uniform(-mutation_range, mutation_range)))
        elif isinstance(self.value, bool):
            # Boolean mutation
            new_gene.value = not self.value
        
        return new_gene
    
    def crossover(self, other: 'StrategyGene') -> Tuple['StrategyGene', 'StrategyGene']:
        """Crossover with another gene"""
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other)
        
        if random.random() < 0.5:
            child1.value = other.value
            child2.value = self.value
        
        return child1, child2


@dataclass
class TradingStrategy:
    """Complete trading strategy genome"""
    strategy_id: str
    generation: int
    strategy_type: StrategyType
    genes: Dict[str, StrategyGene]
    
    # Performance metrics
    fitness_score: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    
    # Testing metrics
    backtest_results: Dict[str, Any] = field(default_factory=dict)
    live_performance: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_tested: Optional[datetime] = None
    
    # Evolution tracking
    parent_ids: List[str] = field(default_factory=list)
    mutations_count: int = 0
    crossover_count: int = 0
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get parameter value by name"""
        if name in self.genes:
            return self.genes[name].value
        return default
    
    def set_parameter(self, name: str, value: Any, gene_type: str = "custom"):
        """Set parameter value"""
        if name in self.genes:
            self.genes[name].value = value
        else:
            self.genes[name] = StrategyGene(gene_type=gene_type, value=value)
    
    def mutate(self, mutation_rate: float = 0.1) -> 'TradingStrategy':
        """Mutate this strategy"""
        new_strategy = copy.deepcopy(self)
        new_strategy.strategy_id = str(uuid.uuid4())
        new_strategy.mutations_count += 1
        new_strategy.parent_ids = [self.strategy_id]
        new_strategy.created_at = datetime.now()
        
        # Mutate genes
        for gene_name, gene in new_strategy.genes.items():
            if random.random() < mutation_rate:
                new_strategy.genes[gene_name] = gene.mutate()
        
        # Reset performance metrics
        new_strategy.fitness_score = 0.0
        new_strategy.backtest_results = {}
        new_strategy.live_performance = {}
        
        return new_strategy
    
    def crossover(self, other: 'TradingStrategy') -> Tuple['TradingStrategy', 'TradingStrategy']:
        """Crossover with another strategy"""
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other)
        
        # Generate new IDs
        child1.strategy_id = str(uuid.uuid4())
        child2.strategy_id = str(uuid.uuid4())
        child1.crossover_count += 1
        child2.crossover_count += 1
        child1.parent_ids = [self.strategy_id, other.strategy_id]
        child2.parent_ids = [self.strategy_id, other.strategy_id]
        child1.created_at = datetime.now()
        child2.created_at = datetime.now()
        
        # Crossover genes
        common_genes = set(self.genes.keys()) & set(other.genes.keys())
        
        for gene_name in common_genes:
            if random.random() < 0.5:  # 50% chance to crossover each gene
                gene1, gene2 = self.genes[gene_name].crossover(other.genes[gene_name])
                child1.genes[gene_name] = gene1
                child2.genes[gene_name] = gene2
        
        # Reset performance metrics
        for child in [child1, child2]:
            child.fitness_score = 0.0
            child.backtest_results = {}
            child.live_performance = {}
        
        return child1, child2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary"""
        return {
            'strategy_id': self.strategy_id,
            'generation': self.generation,
            'strategy_type': self.strategy_type.value,
            'genes': {name: asdict(gene) for name, gene in self.genes.items()},
            'fitness_score': self.fitness_score,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'backtest_results': self.backtest_results,
            'live_performance': self.live_performance,
            'created_at': self.created_at.isoformat(),
            'last_tested': self.last_tested.isoformat() if self.last_tested else None,
            'parent_ids': self.parent_ids,
            'mutations_count': self.mutations_count,
            'crossover_count': self.crossover_count
        }


class StrategyGenerator:
    """Generate random trading strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StrategyGenerator")
        
        # Strategy templates with parameter ranges
        self.strategy_templates = {
            StrategyType.MOMENTUM: {
                'lookback_period': StrategyGene('int', 20, 5, 100, 0.2, False),
                'momentum_threshold': StrategyGene('float', 0.02, 0.005, 0.1, 0.15, False),
                'exit_threshold': StrategyGene('float', 0.01, 0.005, 0.05, 0.15, False),
                'max_holding_period': StrategyGene('int', 10, 1, 50, 0.1, False),
                'volume_filter': StrategyGene('bool', True, mutation_rate=0.05),
                'min_volume': StrategyGene('int', 1000000, 100000, 10000000, 0.2, False)
            },
            StrategyType.MEAN_REVERSION: {
                'lookback_period': StrategyGene('int', 30, 10, 200, 0.2, False),
                'deviation_threshold': StrategyGene('float', 2.0, 1.0, 4.0, 0.15, False),
                'mean_reversion_speed': StrategyGene('float', 0.5, 0.1, 2.0, 0.15, False),
                'stop_loss_pct': StrategyGene('float', 0.05, 0.01, 0.2, 0.1, False),
                'take_profit_pct': StrategyGene('float', 0.03, 0.01, 0.1, 0.1, False),
                'rsi_oversold': StrategyGene('int', 30, 10, 40, 0.1, False),
                'rsi_overbought': StrategyGene('int', 70, 60, 90, 0.1, False)
            },
            StrategyType.BREAKOUT: {
                'consolidation_period': StrategyGene('int', 20, 5, 100, 0.2, False),
                'breakout_threshold': StrategyGene('float', 0.02, 0.005, 0.1, 0.15, False),
                'volume_confirmation': StrategyGene('bool', True, mutation_rate=0.05),
                'volume_multiplier': StrategyGene('float', 1.5, 1.0, 5.0, 0.2, False),
                'false_breakout_filter': StrategyGene('bool', True, mutation_rate=0.1),
                'minimum_range': StrategyGene('float', 0.01, 0.005, 0.05, 0.1, False)
            },
            StrategyType.TREND_FOLLOWING: {
                'short_ma_period': StrategyGene('int', 10, 5, 50, 0.15, False),
                'long_ma_period': StrategyGene('int', 50, 20, 200, 0.15, False),
                'ma_type': StrategyGene('str', 'SMA', allowed_values=['SMA', 'EMA', 'WMA'], 
                                     mutation_rate=0.1, is_discrete=True),
                'trend_strength_filter': StrategyGene('bool', True, mutation_rate=0.05),
                'min_trend_strength': StrategyGene('float', 0.3, 0.1, 0.8, 0.1, False),
                'trailing_stop_pct': StrategyGene('float', 0.02, 0.005, 0.1, 0.1, False)
            },
            StrategyType.BOLLINGER_BANDS: {
                'period': StrategyGene('int', 20, 10, 50, 0.15, False),
                'std_dev': StrategyGene('float', 2.0, 1.0, 3.0, 0.1, False),
                'squeeze_threshold': StrategyGene('float', 0.01, 0.005, 0.03, 0.1, False),
                'band_touch_strategy': StrategyGene('str', 'mean_reversion', 
                                                  allowed_values=['mean_reversion', 'breakout'], 
                                                  mutation_rate=0.1, is_discrete=True),
                'rsi_confirmation': StrategyGene('bool', True, mutation_rate=0.1),
                'position_sizing': StrategyGene('str', 'fixed', 
                                              allowed_values=['fixed', 'volatility_adjusted', 'kelly'], 
                                              mutation_rate=0.1, is_discrete=True)
            },
            StrategyType.MOVING_AVERAGE_CROSSOVER: {
                'fast_period': StrategyGene('int', 12, 5, 50, 0.15, False),
                'slow_period': StrategyGene('int', 26, 20, 200, 0.15, False),
                'signal_period': StrategyGene('int', 9, 3, 30, 0.15, False),
                'ma_type': StrategyGene('str', 'EMA', allowed_values=['SMA', 'EMA'], 
                                     mutation_rate=0.1, is_discrete=True),
                'divergence_filter': StrategyGene('bool', False, mutation_rate=0.1),
                'volume_confirmation': StrategyGene('bool', True, mutation_rate=0.1)
            }
        }
    
    def generate_random_strategy(self, generation: int = 0, strategy_type: Optional[StrategyType] = None) -> TradingStrategy:
        """Generate a random trading strategy"""
        
        # Select strategy type
        if strategy_type is None:
            strategy_type = random.choice(list(self.strategy_templates.keys()))
        
        # Get template genes
        template_genes = self.strategy_templates[strategy_type]
        
        # Create randomized genes
        genes = {}
        for gene_name, template_gene in template_genes.items():
            gene = copy.deepcopy(template_gene)
            
            # Randomize the value within constraints
            if gene.is_discrete and gene.allowed_values:
                gene.value = random.choice(gene.allowed_values)
            elif gene.min_value is not None and gene.max_value is not None:
                if isinstance(gene.min_value, int):
                    gene.value = random.randint(gene.min_value, gene.max_value)
                elif isinstance(gene.min_value, float):
                    gene.value = random.uniform(gene.min_value, gene.max_value)
            elif isinstance(gene.value, bool):
                gene.value = random.choice([True, False])
            
            genes[gene_name] = gene
        
        # Add common genes that apply to all strategies
        genes.update({
            'position_size_pct': StrategyGene('float', random.uniform(0.01, 0.1), 0.005, 0.2, 0.1, False),
            'max_positions': StrategyGene('int', random.randint(1, 10), 1, 20, 0.15, False),
            'risk_per_trade': StrategyGene('float', random.uniform(0.005, 0.02), 0.001, 0.05, 0.1, False),
            'commission_rate': StrategyGene('float', 0.001, 0.0005, 0.01, 0.05, False),
            'slippage_rate': StrategyGene('float', 0.001, 0.0001, 0.005, 0.05, False)
        })
        
        return TradingStrategy(
            strategy_id=str(uuid.uuid4()),
            generation=generation,
            strategy_type=strategy_type,
            genes=genes
        )
    
    def generate_population(self, size: int, generation: int = 0) -> List[TradingStrategy]:
        """Generate a population of random strategies"""
        population = []
        
        # Ensure diversity by including all strategy types
        strategy_types = list(self.strategy_templates.keys())
        strategies_per_type = size // len(strategy_types)
        
        for strategy_type in strategy_types:
            for _ in range(strategies_per_type):
                strategy = self.generate_random_strategy(generation, strategy_type)
                population.append(strategy)
        
        # Fill remaining slots with random types
        while len(population) < size:
            strategy = self.generate_random_strategy(generation)
            population.append(strategy)
        
        return population


class StrategyBacktester:
    """Backtest trading strategies"""
    
    def __init__(self, initial_capital: float = 100000):
        self.logger = logging.getLogger(f"{__name__}.StrategyBacktester")
        self.initial_capital = initial_capital
    
    async def backtest_strategy(self, strategy: TradingStrategy, 
                               market_data: pd.DataFrame,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Backtest a strategy against historical data"""
        
        try:
            # Filter data by date range
            if start_date and end_date:
                mask = (market_data.index >= start_date) & (market_data.index <= end_date)
                data = market_data.loc[mask].copy()
            else:
                data = market_data.copy()
            
            if data.empty:
                return {'error': 'No data available for backtesting'}
            
            # Initialize backtest state
            capital = self.initial_capital
            positions = {}
            trades = []
            equity_curve = []
            
            # Generate signals based on strategy
            signals = await self._generate_signals(strategy, data)
            
            # Process each signal
            for timestamp, signal_data in signals.items():
                if timestamp not in data.index:
                    continue
                
                current_price = data.loc[timestamp, 'close']
                
                for symbol, signal in signal_data.items():
                    if signal['action'] == 'buy' and symbol not in positions:
                        # Open long position
                        position_size = min(
                            capital * strategy.get_parameter('position_size_pct', 0.05),
                            capital * strategy.get_parameter('risk_per_trade', 0.02) * 50  # 2% risk = 50x leverage limit
                        )
                        
                        if position_size > 100:  # Minimum position size
                            shares = int(position_size / current_price)
                            cost = shares * current_price * (1 + strategy.get_parameter('commission_rate', 0.001))
                            
                            if cost <= capital:
                                positions[symbol] = {
                                    'shares': shares,
                                    'entry_price': current_price,
                                    'entry_time': timestamp,
                                    'stop_loss': signal.get('stop_loss'),
                                    'take_profit': signal.get('take_profit')
                                }
                                capital -= cost
                    
                    elif signal['action'] == 'sell' and symbol in positions:
                        # Close position
                        position = positions[symbol]
                        shares = position['shares']
                        proceeds = shares * current_price * (1 - strategy.get_parameter('commission_rate', 0.001))
                        
                        trade_return = (proceeds - shares * position['entry_price']) / (shares * position['entry_price'])
                        
                        trades.append({
                            'symbol': symbol,
                            'entry_time': position['entry_time'],
                            'exit_time': timestamp,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'shares': shares,
                            'return': trade_return,
                            'profit': proceeds - shares * position['entry_price']
                        })
                        
                        capital += proceeds
                        del positions[symbol]
                
                # Calculate current portfolio value
                portfolio_value = capital
                for symbol, position in positions.items():
                    if timestamp in data.index:
                        current_price = data.loc[timestamp, 'close']
                        portfolio_value += position['shares'] * current_price
                
                equity_curve.append({
                    'timestamp': timestamp,
                    'portfolio_value': portfolio_value,
                    'cash': capital,
                    'positions_value': portfolio_value - capital
                })
            
            # Close remaining positions at end
            final_timestamp = data.index[-1]
            final_price = data.loc[final_timestamp, 'close']
            
            for symbol, position in positions.items():
                shares = position['shares']
                proceeds = shares * final_price * (1 - strategy.get_parameter('commission_rate', 0.001))
                
                trade_return = (proceeds - shares * position['entry_price']) / (shares * position['entry_price'])
                
                trades.append({
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'exit_time': final_timestamp,
                    'entry_price': position['entry_price'],
                    'exit_price': final_price,
                    'shares': shares,
                    'return': trade_return,
                    'profit': proceeds - shares * position['entry_price']
                })
                
                capital += proceeds
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(trades, equity_curve, self.initial_capital)
            results['total_trades'] = len(trades)
            results['strategy_id'] = strategy.strategy_id
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error backtesting strategy {strategy.strategy_id}: {e}")
            return {'error': str(e)}
    
    async def _generate_signals(self, strategy: TradingStrategy, data: pd.DataFrame) -> Dict[datetime, Dict[str, Dict[str, Any]]]:
        """Generate trading signals based on strategy"""
        signals = {}
        
        # This is a simplified signal generation - in reality, you'd implement
        # the specific logic for each strategy type
        
        strategy_type = strategy.strategy_type
        
        if strategy_type == StrategyType.MOMENTUM:
            signals = await self._generate_momentum_signals(strategy, data)
        elif strategy_type == StrategyType.MEAN_REVERSION:
            signals = await self._generate_mean_reversion_signals(strategy, data)
        elif strategy_type == StrategyType.BREAKOUT:
            signals = await self._generate_breakout_signals(strategy, data)
        elif strategy_type == StrategyType.TREND_FOLLOWING:
            signals = await self._generate_trend_following_signals(strategy, data)
        elif strategy_type == StrategyType.BOLLINGER_BANDS:
            signals = await self._generate_bollinger_signals(strategy, data)
        else:
            # Default momentum-based signals
            signals = await self._generate_momentum_signals(strategy, data)
        
        return signals
    
    async def _generate_momentum_signals(self, strategy: TradingStrategy, data: pd.DataFrame) -> Dict[datetime, Dict[str, Dict[str, Any]]]:
        """Generate momentum-based trading signals"""
        signals = {}
        
        lookback = strategy.get_parameter('lookback_period', 20)
        threshold = strategy.get_parameter('momentum_threshold', 0.02)
        
        # Calculate momentum
        data['momentum'] = data['close'].pct_change(lookback)
        data['volume_ma'] = data['volume'].rolling(lookback).mean()
        
        for i in range(lookback, len(data)):
            timestamp = data.index[i]
            momentum = data.iloc[i]['momentum']
            volume_ratio = data.iloc[i]['volume'] / data.iloc[i]['volume_ma'] if data.iloc[i]['volume_ma'] > 0 else 1
            
            signal_data = {}
            
            # Buy signal: strong positive momentum with volume confirmation
            if momentum > threshold and (not strategy.get_parameter('volume_filter', True) or volume_ratio > 1.2):
                signal_data['SYMBOL'] = {
                    'action': 'buy',
                    'confidence': min(1.0, momentum / threshold),
                    'stop_loss': data.iloc[i]['close'] * (1 - strategy.get_parameter('exit_threshold', 0.01)),
                    'take_profit': data.iloc[i]['close'] * (1 + strategy.get_parameter('exit_threshold', 0.01) * 2)
                }
            
            # Sell signal: negative momentum
            elif momentum < -threshold/2:
                signal_data['SYMBOL'] = {
                    'action': 'sell',
                    'confidence': min(1.0, abs(momentum) / threshold),
                }
            
            if signal_data:
                signals[timestamp] = signal_data
        
        return signals
    
    async def _generate_mean_reversion_signals(self, strategy: TradingStrategy, data: pd.DataFrame) -> Dict[datetime, Dict[str, Dict[str, Any]]]:
        """Generate mean reversion signals"""
        signals = {}
        
        lookback = strategy.get_parameter('lookback_period', 30)
        deviation_threshold = strategy.get_parameter('deviation_threshold', 2.0)
        
        # Calculate Bollinger Bands
        data['sma'] = data['close'].rolling(lookback).mean()
        data['std'] = data['close'].rolling(lookback).std()
        data['upper_band'] = data['sma'] + (data['std'] * deviation_threshold)
        data['lower_band'] = data['sma'] - (data['std'] * deviation_threshold)
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        rsi_oversold = strategy.get_parameter('rsi_oversold', 30)
        rsi_overbought = strategy.get_parameter('rsi_overbought', 70)
        
        for i in range(lookback, len(data)):
            timestamp = data.index[i]
            close = data.iloc[i]['close']
            lower_band = data.iloc[i]['lower_band']
            upper_band = data.iloc[i]['upper_band']
            rsi = data.iloc[i]['rsi']
            
            signal_data = {}
            
            # Buy signal: price below lower band and RSI oversold
            if close < lower_band and rsi < rsi_oversold:
                signal_data['SYMBOL'] = {
                    'action': 'buy',
                    'confidence': min(1.0, (lower_band - close) / lower_band),
                    'stop_loss': close * (1 - strategy.get_parameter('stop_loss_pct', 0.05)),
                    'take_profit': close * (1 + strategy.get_parameter('take_profit_pct', 0.03))
                }
            
            # Sell signal: price above upper band and RSI overbought
            elif close > upper_band and rsi > rsi_overbought:
                signal_data['SYMBOL'] = {
                    'action': 'sell',
                    'confidence': min(1.0, (close - upper_band) / upper_band),
                }
            
            if signal_data:
                signals[timestamp] = signal_data
        
        return signals
    
    async def _generate_breakout_signals(self, strategy: TradingStrategy, data: pd.DataFrame) -> Dict[datetime, Dict[str, Dict[str, Any]]]:
        """Generate breakout signals"""
        signals = {}
        
        period = strategy.get_parameter('consolidation_period', 20)
        threshold = strategy.get_parameter('breakout_threshold', 0.02)
        
        # Calculate price ranges
        data['high_max'] = data['high'].rolling(period).max()
        data['low_min'] = data['low'].rolling(period).min()
        data['range'] = (data['high_max'] - data['low_min']) / data['close']
        data['volume_ma'] = data['volume'].rolling(period).mean()
        
        for i in range(period, len(data)):
            timestamp = data.index[i]
            close = data.iloc[i]['close']
            high_max = data.iloc[i]['high_max']
            low_min = data.iloc[i]['low_min']
            volume_ratio = data.iloc[i]['volume'] / data.iloc[i]['volume_ma'] if data.iloc[i]['volume_ma'] > 0 else 1
            
            signal_data = {}
            
            # Upward breakout
            if close > high_max * (1 + threshold):
                volume_confirmed = not strategy.get_parameter('volume_confirmation', True) or volume_ratio > strategy.get_parameter('volume_multiplier', 1.5)
                
                if volume_confirmed:
                    signal_data['SYMBOL'] = {
                        'action': 'buy',
                        'confidence': min(1.0, (close - high_max) / high_max / threshold),
                        'stop_loss': low_min,
                        'take_profit': close * (1 + threshold * 2)
                    }
            
            # Downward breakout
            elif close < low_min * (1 - threshold):
                volume_confirmed = not strategy.get_parameter('volume_confirmation', True) or volume_ratio > strategy.get_parameter('volume_multiplier', 1.5)
                
                if volume_confirmed:
                    signal_data['SYMBOL'] = {
                        'action': 'sell',
                        'confidence': min(1.0, (low_min - close) / low_min / threshold),
                    }
            
            if signal_data:
                signals[timestamp] = signal_data
        
        return signals
    
    async def _generate_trend_following_signals(self, strategy: TradingStrategy, data: pd.DataFrame) -> Dict[datetime, Dict[str, Dict[str, Any]]]:
        """Generate trend following signals"""
        signals = {}
        
        short_period = strategy.get_parameter('short_ma_period', 10)
        long_period = strategy.get_parameter('long_ma_period', 50)
        
        # Calculate moving averages
        if strategy.get_parameter('ma_type', 'SMA') == 'EMA':
            data['short_ma'] = data['close'].ewm(span=short_period).mean()
            data['long_ma'] = data['close'].ewm(span=long_period).mean()
        else:
            data['short_ma'] = data['close'].rolling(short_period).mean()
            data['long_ma'] = data['close'].rolling(long_period).mean()
        
        for i in range(long_period, len(data)):
            timestamp = data.index[i]
            short_ma = data.iloc[i]['short_ma']
            long_ma = data.iloc[i]['long_ma']
            prev_short_ma = data.iloc[i-1]['short_ma']
            prev_long_ma = data.iloc[i-1]['long_ma']
            
            signal_data = {}
            
            # Golden cross: short MA crosses above long MA
            if short_ma > long_ma and prev_short_ma <= prev_long_ma:
                signal_data['SYMBOL'] = {
                    'action': 'buy',
                    'confidence': min(1.0, (short_ma - long_ma) / long_ma * 10),
                    'stop_loss': data.iloc[i]['close'] * (1 - strategy.get_parameter('trailing_stop_pct', 0.02)),
                    'take_profit': None  # Trend following uses trailing stops
                }
            
            # Death cross: short MA crosses below long MA
            elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
                signal_data['SYMBOL'] = {
                    'action': 'sell',
                    'confidence': min(1.0, (long_ma - short_ma) / long_ma * 10),
                }
            
            if signal_data:
                signals[timestamp] = signal_data
        
        return signals
    
    async def _generate_bollinger_signals(self, strategy: TradingStrategy, data: pd.DataFrame) -> Dict[datetime, Dict[str, Dict[str, Any]]]:
        """Generate Bollinger Bands signals"""
        signals = {}
        
        period = strategy.get_parameter('period', 20)
        std_dev = strategy.get_parameter('std_dev', 2.0)
        band_strategy = strategy.get_parameter('band_touch_strategy', 'mean_reversion')
        
        # Calculate Bollinger Bands
        data['sma'] = data['close'].rolling(period).mean()
        data['std'] = data['close'].rolling(period).std()
        data['upper_band'] = data['sma'] + (data['std'] * std_dev)
        data['lower_band'] = data['sma'] - (data['std'] * std_dev)
        data['bb_width'] = (data['upper_band'] - data['lower_band']) / data['sma']
        
        squeeze_threshold = strategy.get_parameter('squeeze_threshold', 0.01)
        
        for i in range(period, len(data)):
            timestamp = data.index[i]
            close = data.iloc[i]['close']
            upper_band = data.iloc[i]['upper_band']
            lower_band = data.iloc[i]['lower_band']
            sma = data.iloc[i]['sma']
            bb_width = data.iloc[i]['bb_width']
            
            signal_data = {}
            
            if band_strategy == 'mean_reversion':
                # Mean reversion strategy
                if close <= lower_band:
                    signal_data['SYMBOL'] = {
                        'action': 'buy',
                        'confidence': min(1.0, (lower_band - close) / lower_band),
                        'stop_loss': close * 0.95,
                        'take_profit': sma
                    }
                elif close >= upper_band:
                    signal_data['SYMBOL'] = {
                        'action': 'sell',
                        'confidence': min(1.0, (close - upper_band) / upper_band),
                    }
            
            else:  # breakout strategy
                # Bollinger squeeze breakout
                if bb_width < squeeze_threshold:
                    if close > sma:
                        signal_data['SYMBOL'] = {
                            'action': 'buy',
                            'confidence': 0.7,  # Medium confidence for squeeze breakout
                            'stop_loss': lower_band,
                            'take_profit': upper_band * 1.1
                        }
            
            if signal_data:
                signals[timestamp] = signal_data
        
        return signals
    
    def _calculate_performance_metrics(self, trades: List[Dict], equity_curve: List[Dict], initial_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'total_trades': 0
            }
        
        # Basic metrics
        returns = [trade['return'] for trade in trades]
        profits = [trade['profit'] for trade in trades]
        
        total_return = sum(profits) / initial_capital
        avg_return = np.mean(returns)
        volatility = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        # Win rate
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]
        win_rate = len(winning_trades) / len(returns) if returns else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1e-10
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Maximum drawdown
        if equity_curve:
            portfolio_values = [point['portfolio_value'] for point in equity_curve]
            peak = portfolio_values[0]
            max_drawdown = 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_return,
            'volatility': volatility,
            'total_trades': len(trades),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }


class StrategyEvolver:
    """Main genetic algorithm engine for strategy evolution"""
    
    def __init__(self, 
                 event_bus: Optional[TradingEventBus] = None,
                 population_size: int = 100,
                 elite_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 max_generations: int = 100):
        
        self.logger = logging.getLogger(__name__)
        self.event_bus = event_bus
        
        # Evolution parameters
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        
        # Components
        self.generator = StrategyGenerator()
        self.backtester = StrategyBacktester()
        
        # Evolution state
        self.current_generation = 0
        self.population: List[TradingStrategy] = []
        self.best_strategies: List[TradingStrategy] = []
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
        # Database for persistence
        self.db_path = "strategy_evolution.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for strategy storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS strategies (
                        strategy_id TEXT PRIMARY KEY,
                        generation INTEGER NOT NULL,
                        strategy_type TEXT NOT NULL,
                        genes TEXT NOT NULL,
                        fitness_score REAL NOT NULL,
                        total_return REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        win_rate REAL NOT NULL,
                        profit_factor REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        parent_ids TEXT,
                        mutations_count INTEGER DEFAULT 0,
                        crossover_count INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        last_tested TEXT,
                        is_elite INTEGER DEFAULT 0
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS evolution_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        generation INTEGER NOT NULL,
                        best_fitness REAL NOT NULL,
                        avg_fitness REAL NOT NULL,
                        diversity_score REAL NOT NULL,
                        elite_count INTEGER NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    async def evolve(self, market_data: pd.DataFrame, generations: Optional[int] = None) -> List[TradingStrategy]:
        """Run the genetic algorithm evolution process"""
        
        generations = generations or self.max_generations
        self.logger.info(f"Starting evolution for {generations} generations with population size {self.population_size}")
        
        # Initialize population if empty
        if not self.population:
            self.population = self.generator.generate_population(self.population_size, self.current_generation)
            self.logger.info(f"Generated initial population of {len(self.population)} strategies")
        
        for generation in range(generations):
            self.current_generation = generation
            self.logger.info(f"Starting generation {generation + 1}/{generations}")
            
            # Evaluate fitness
            await self._evaluate_population(market_data)
            
            # Select elite strategies
            elite_strategies = self._select_elite()
            
            # Calculate generation statistics
            generation_stats = self._calculate_generation_stats()
            self.evolution_history.append(generation_stats)
            
            # Publish generation results
            if self.event_bus:
                await self.event_bus.publish(
                    "evolution_generation_complete",
                    {
                        'generation': generation + 1,
                        'best_fitness': generation_stats['best_fitness'],
                        'avg_fitness': generation_stats['avg_fitness'],
                        'diversity': generation_stats['diversity_score'],
                        'elite_strategies': [s.strategy_id for s in elite_strategies[:5]]
                    },
                    priority=Priority.NORMAL
                )
            
            # Store results in database
            await self._store_generation_results(generation_stats, elite_strategies)
            
            # Create next generation
            if generation < generations - 1:
                self.population = await self._create_next_generation(elite_strategies)
            
            self.logger.info(f"Generation {generation + 1} complete. Best fitness: {generation_stats['best_fitness']:.4f}")
        
        # Final elite selection
        final_elite = self._select_elite()
        self.best_strategies = final_elite
        
        self.logger.info(f"Evolution complete. Best strategy fitness: {final_elite[0].fitness_score:.4f}")
        
        return final_elite
    
    async def _evaluate_population(self, market_data: pd.DataFrame):
        """Evaluate fitness of all strategies in population"""
        
        # Use multiprocessing for parallel evaluation
        with ThreadPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
            futures = []
            
            for strategy in self.population:
                if strategy.fitness_score == 0.0:  # Only test untested strategies
                    future = executor.submit(self._evaluate_strategy_sync, strategy, market_data)
                    futures.append((strategy, future))
            
            # Collect results
            for strategy, future in futures:
                try:
                    results = future.result(timeout=30)  # 30 second timeout per strategy
                    if 'error' not in results:
                        strategy.fitness_score = self._calculate_fitness(results)
                        strategy.total_return = results.get('total_return', 0)
                        strategy.sharpe_ratio = results.get('sharpe_ratio', 0)
                        strategy.max_drawdown = results.get('max_drawdown', 0)
                        strategy.win_rate = results.get('win_rate', 0)
                        strategy.profit_factor = results.get('profit_factor', 0)
                        strategy.total_trades = results.get('total_trades', 0)
                        strategy.backtest_results = results
                        strategy.last_tested = datetime.now()
                    else:
                        self.logger.warning(f"Strategy {strategy.strategy_id} failed: {results['error']}")
                        strategy.fitness_score = 0.0
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating strategy {strategy.strategy_id}: {e}")
                    strategy.fitness_score = 0.0
    
    def _evaluate_strategy_sync(self, strategy: TradingStrategy, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Synchronous wrapper for strategy evaluation"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.backtester.backtest_strategy(strategy, market_data))
        finally:
            loop.close()
    
    def _calculate_fitness(self, results: Dict[str, Any]) -> float:
        """Calculate fitness score from backtest results"""
        
        # Multi-objective fitness function
        total_return = results.get('total_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 1)
        win_rate = results.get('win_rate', 0)
        profit_factor = results.get('profit_factor', 0)
        total_trades = results.get('total_trades', 0)
        
        # Normalize and weight components
        return_score = min(2.0, max(-2.0, total_return * 5))  # -2 to 2
        sharpe_score = min(3.0, max(-1.0, sharpe_ratio))      # -1 to 3
        drawdown_score = 1 - min(1.0, max_drawdown)           # 0 to 1
        win_rate_score = win_rate                              # 0 to 1
        pf_score = min(2.0, max(0.0, np.log(profit_factor))) if profit_factor > 0 else 0  # 0 to 2
        trade_count_score = min(1.0, total_trades / 100)      # 0 to 1 (prefer strategies that trade)
        
        # Weighted fitness
        fitness = (
            return_score * 0.25 +
            sharpe_score * 0.25 +
            drawdown_score * 0.20 +
            win_rate_score * 0.15 +
            pf_score * 0.10 +
            trade_count_score * 0.05
        )
        
        return max(0.0, fitness)
    
    def _select_elite(self) -> List[TradingStrategy]:
        """Select elite strategies for next generation"""
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Select top performers
        elite = self.population[:self.elite_size].copy()
        
        return elite
    
    def _calculate_generation_stats(self) -> Dict[str, Any]:
        """Calculate statistics for current generation"""
        
        fitness_scores = [s.fitness_score for s in self.population if s.fitness_score > 0]
        
        if not fitness_scores:
            return {
                'generation': self.current_generation,
                'best_fitness': 0.0,
                'avg_fitness': 0.0,
                'worst_fitness': 0.0,
                'diversity_score': 0.0,
                'strategies_tested': 0
            }
        
        # Diversity calculation (simplified)
        strategy_types = [s.strategy_type for s in self.population]
        unique_types = len(set(strategy_types))
        diversity_score = unique_types / len(StrategyType)
        
        return {
            'generation': self.current_generation,
            'best_fitness': max(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'diversity_score': diversity_score,
            'strategies_tested': len(fitness_scores)
        }
    
    async def _store_generation_results(self, stats: Dict[str, Any], elite_strategies: List[TradingStrategy]):
        """Store generation results in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Store generation stats
                cursor.execute('''
                    INSERT INTO evolution_history (
                        generation, best_fitness, avg_fitness, diversity_score, elite_count, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    stats['generation'], stats['best_fitness'], stats['avg_fitness'],
                    stats['diversity_score'], len(elite_strategies), datetime.now().isoformat()
                ))
                
                # Store/update strategies
                for strategy in self.population:
                    cursor.execute('''
                        INSERT OR REPLACE INTO strategies (
                            strategy_id, generation, strategy_type, genes, fitness_score,
                            total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor,
                            total_trades, parent_ids, mutations_count, crossover_count,
                            created_at, last_tested, is_elite
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        strategy.strategy_id, strategy.generation, strategy.strategy_type.value,
                        json.dumps({name: asdict(gene) for name, gene in strategy.genes.items()}),
                        strategy.fitness_score, strategy.total_return, strategy.sharpe_ratio,
                        strategy.max_drawdown, strategy.win_rate, strategy.profit_factor,
                        strategy.total_trades, json.dumps(strategy.parent_ids),
                        strategy.mutations_count, strategy.crossover_count,
                        strategy.created_at.isoformat(),
                        strategy.last_tested.isoformat() if strategy.last_tested else None,
                        1 if strategy in elite_strategies else 0
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing generation results: {e}")
    
    async def _create_next_generation(self, elite_strategies: List[TradingStrategy]) -> List[TradingStrategy]:
        """Create next generation through selection, crossover, and mutation"""
        
        next_generation = []
        
        # Keep elite strategies (elitism)
        for strategy in elite_strategies:
            strategy.generation = self.current_generation + 1
            next_generation.append(copy.deepcopy(strategy))
        
        # Fill remaining population through crossover and mutation
        while len(next_generation) < self.population_size:
            
            if random.random() < self.crossover_rate and len(elite_strategies) >= 2:
                # Crossover
                parent1, parent2 = self._tournament_selection(elite_strategies, 2)
                child1, child2 = parent1.crossover(parent2)
                
                child1.generation = self.current_generation + 1
                child2.generation = self.current_generation + 1
                
                next_generation.extend([child1, child2])
                
            else:
                # Mutation
                parent = self._tournament_selection(elite_strategies, 1)[0]
                child = parent.mutate(self.mutation_rate)
                child.generation = self.current_generation + 1
                
                next_generation.append(child)
        
        # Add some completely random strategies for diversity
        random_count = max(5, self.population_size // 20)
        for _ in range(random_count):
            if len(next_generation) >= self.population_size:
                break
            
            random_strategy = self.generator.generate_random_strategy(self.current_generation + 1)
            next_generation.append(random_strategy)
        
        # Trim to exact population size
        return next_generation[:self.population_size]
    
    def _tournament_selection(self, strategies: List[TradingStrategy], count: int) -> List[TradingStrategy]:
        """Select strategies using tournament selection"""
        
        selected = []
        tournament_size = 3
        
        for _ in range(count):
            tournament = random.sample(strategies, min(tournament_size, len(strategies)))
            winner = max(tournament, key=lambda x: x.fitness_score)
            selected.append(winner)
        
        return selected
    
    async def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics"""
        
        if not self.evolution_history:
            return {'error': 'No evolution history available'}
        
        latest_stats = self.evolution_history[-1]
        
        fitness_progression = [gen['best_fitness'] for gen in self.evolution_history]
        diversity_progression = [gen['diversity_score'] for gen in self.evolution_history]
        
        return {
            'current_generation': self.current_generation,
            'population_size': self.population_size,
            'latest_best_fitness': latest_stats['best_fitness'],
            'latest_avg_fitness': latest_stats['avg_fitness'],
            'fitness_improvement': fitness_progression[-1] - fitness_progression[0] if len(fitness_progression) > 1 else 0,
            'avg_diversity': np.mean(diversity_progression) if diversity_progression else 0,
            'total_strategies_tested': sum(gen['strategies_tested'] for gen in self.evolution_history),
            'best_strategies_count': len(self.best_strategies),
            'evolution_history': self.evolution_history
        }
    
    def get_best_strategies(self, count: int = 10) -> List[TradingStrategy]:
        """Get the best strategies discovered"""
        
        all_strategies = self.population + self.best_strategies
        unique_strategies = {s.strategy_id: s for s in all_strategies}.values()
        
        sorted_strategies = sorted(unique_strategies, key=lambda x: x.fitness_score, reverse=True)
        
        return sorted_strategies[:count]


# Example usage
async def main():
    """Example usage of the strategy evolution system"""
    logging.basicConfig(level=logging.INFO)
    
    # Create mock market data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate price data with some trends
    prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
    volumes = np.random.normal(1000000, 200000, len(dates))
    
    market_data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'volume': np.abs(volumes)
    }, index=dates)
    
    # Create event bus
    event_bus = TradingEventBus()
    await event_bus.start()
    
    try:
        # Create evolution engine
        evolver = StrategyEvolver(
            event_bus=event_bus,
            population_size=20,  # Small for demo
            elite_size=5,
            max_generations=5
        )
        
        # Run evolution
        best_strategies = await evolver.evolve(market_data, generations=5)
        
        # Get statistics
        stats = await evolver.get_evolution_statistics()
        
        print("Evolution Results:")
        print(f"Best fitness achieved: {stats['latest_best_fitness']:.4f}")
        print(f"Total strategies tested: {stats['total_strategies_tested']}")
        
        print(f"\nTop 3 strategies:")
        for i, strategy in enumerate(best_strategies[:3], 1):
            print(f"{i}. {strategy.strategy_type.value} - Fitness: {strategy.fitness_score:.4f}")
            print(f"   Return: {strategy.total_return:.2%}, Sharpe: {strategy.sharpe_ratio:.2f}")
        
    finally:
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(main())