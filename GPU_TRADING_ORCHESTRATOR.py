#!/usr/bin/env python3
"""
GPU TRADING ORCHESTRATOR
========================
Combines GPU AI Trading Agent + Genetic Strategy Evolution
for maximum computational power and strategy generation.

Key Features:
- Runs GPU AI agent on separate thread (2-3.5 Sharpe target)
- Runs genetic evolution on separate thread (strategy discovery)
- Shares signals between systems via queue
- Combines signals intelligently using ensemble voting
- Adapts to CUDA availability (fallback to CPU)
- Real-time performance tracking

Target: 2-4% monthly from GPU systems
"""

import asyncio
import threading
import queue
import time
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import yfinance as yf
import logging

# Import GPU systems
from PRODUCTION.advanced.gpu.gpu_ai_trading_agent import GPUAITradingAgent, TradingEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GPUSignal:
    """Signal from GPU systems"""
    source: str  # 'ai_agent' or 'genetic'
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    expected_return: float
    position_size: float  # 0-1 (portfolio percentage)
    reasoning: str
    timestamp: datetime
    metadata: Dict

class GeneticStrategyEvolution:
    """
    Simplified Genetic Algorithm for strategy evolution
    Evolves trading strategy parameters
    """

    def __init__(self, device='cuda', population_size=20):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.population_size = population_size
        self.generation = 0
        self.best_strategies = []

        # Strategy gene space
        self.gene_space = {
            'ema_fast': (5, 20),
            'ema_slow': (20, 50),
            'rsi_period': (10, 20),
            'adx_threshold': (20, 30),
            'position_size': (0.05, 0.20),
            'stop_loss_pct': (0.01, 0.03),
            'take_profit_pct': (0.02, 0.05)
        }

        # Initialize population
        self.population = self._initialize_population()

        logger.info(f"Genetic Evolution initialized on {self.device}")

    def _initialize_population(self) -> List[Dict]:
        """Initialize random population"""
        population = []

        for i in range(self.population_size):
            strategy = {}
            for gene, (min_val, max_val) in self.gene_space.items():
                if isinstance(min_val, float):
                    strategy[gene] = np.random.uniform(min_val, max_val)
                else:
                    strategy[gene] = np.random.randint(min_val, max_val + 1)

            strategy['fitness'] = 0.0
            strategy['strategy_id'] = f"gen0_strat{i}"
            population.append(strategy)

        return population

    def evaluate_strategy(self, strategy: Dict, market_data: pd.DataFrame) -> float:
        """
        Evaluate strategy fitness on market data

        Returns:
            Fitness score (higher is better)
        """
        try:
            # Simple backtest
            closes = market_data['Close'].values

            # Calculate indicators
            ema_fast = pd.Series(closes).ewm(span=int(strategy['ema_fast'])).mean().values
            ema_slow = pd.Series(closes).ewm(span=int(strategy['ema_slow'])).mean().values

            # Generate signals
            signals = []
            position = 0
            entry_price = 0
            total_return = 1.0

            for i in range(50, len(closes)):  # Skip warmup
                # Simple crossover strategy
                if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
                    if position == 0:
                        position = 1
                        entry_price = closes[i]

                elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
                    if position == 1:
                        exit_price = closes[i]
                        trade_return = (exit_price - entry_price) / entry_price
                        total_return *= (1 + trade_return * strategy['position_size'])
                        position = 0

            # Calculate fitness (Sharpe-like metric)
            if total_return > 1:
                fitness = (total_return - 1) * 100  # Percent return
            else:
                fitness = (total_return - 1) * 100  # Negative for losses

            return fitness

        except Exception as e:
            logger.error(f"Strategy evaluation error: {e}")
            return -999.0

    def evolve_generation(self, market_data: pd.DataFrame) -> List[Dict]:
        """
        Evolve one generation

        Returns:
            Best strategies from this generation
        """
        self.generation += 1

        # Evaluate all strategies
        for strategy in self.population:
            strategy['fitness'] = self.evaluate_strategy(strategy, market_data)

        # Sort by fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)

        # Keep top 20%
        elite_count = max(1, self.population_size // 5)
        elites = self.population[:elite_count]

        # Track best
        self.best_strategies.append(elites[0].copy())

        logger.info(f"Generation {self.generation}: Best fitness = {elites[0]['fitness']:.2f}")

        # Create new population
        new_population = []

        # Keep elites
        new_population.extend([s.copy() for s in elites])

        # Crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = np.random.choice(elites)
            parent2 = np.random.choice(elites)

            # Crossover
            child = {}
            for gene in self.gene_space.keys():
                child[gene] = parent1[gene] if np.random.random() < 0.5 else parent2[gene]

            # Mutation
            if np.random.random() < 0.2:
                gene_to_mutate = np.random.choice(list(self.gene_space.keys()))
                min_val, max_val = self.gene_space[gene_to_mutate]
                if isinstance(min_val, float):
                    child[gene_to_mutate] = np.random.uniform(min_val, max_val)
                else:
                    child[gene_to_mutate] = np.random.randint(min_val, max_val + 1)

            child['fitness'] = 0.0
            child['strategy_id'] = f"gen{self.generation}_strat{len(new_population)}"
            new_population.append(child)

        self.population = new_population

        return elites

    def get_best_strategy(self) -> Dict:
        """Get current best strategy"""
        if self.best_strategies:
            return self.best_strategies[-1]
        return self.population[0]

class GPUTradingOrchestrator:
    """
    Orchestrates GPU AI Agent + Genetic Evolution
    Combines their signals for superior trading decisions
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()

        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_available else 'cpu')

        if self.cuda_available:
            logger.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.info("GPU not available - using CPU")

        # Initialize systems
        self.ai_agent = None
        self.genetic_evolver = None

        # Signal queues
        self.ai_signal_queue = queue.Queue()
        self.genetic_signal_queue = queue.Queue()
        self.combined_signal_queue = queue.Queue()

        # Performance tracking
        self.performance = {
            'ai_agent': {'signals': 0, 'accepted': 0, 'roi': []},
            'genetic': {'signals': 0, 'accepted': 0, 'roi': []},
            'combined': {'signals': 0, 'executed': 0, 'roi': []}
        }

        # Control flags
        self.running = False
        self.threads = []

        logger.info("GPU Trading Orchestrator initialized")

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'ai_agent': {
                'state_size': 100,
                'action_size': 21,
                'training_episodes': 50,
                'learning_rate': 0.001
            },
            'genetic': {
                'population_size': 20,
                'generations_per_cycle': 5
            },
            'ensemble': {
                'min_confidence': 0.70,
                'min_agreement': 0.60,
                'voting_weights': {'ai_agent': 0.6, 'genetic': 0.4}
            },
            'execution': {
                'max_position_size': 0.10,
                'min_position_size': 0.02,
                'max_concurrent_trades': 5
            },
            'update_frequency': 3600  # 1 hour
        }

    def initialize_systems(self, market_data: pd.DataFrame) -> None:
        """Initialize AI agent and genetic evolver"""

        logger.info("Initializing GPU systems...")

        # Initialize AI Agent
        self.ai_agent = GPUAITradingAgent(
            state_size=self.config['ai_agent']['state_size'],
            action_size=self.config['ai_agent']['action_size'],
            learning_rate=self.config['ai_agent']['learning_rate']
        )

        # Train AI agent on initial data
        logger.info("Training AI agent...")
        training_results = self.ai_agent.train_multiple_episodes(
            market_data,
            num_episodes=self.config['ai_agent']['training_episodes']
        )

        logger.info(f"AI Agent trained: Sharpe = {training_results['final_metrics']['sharpe_ratio']:.2f}")

        # Initialize Genetic Evolver
        self.genetic_evolver = GeneticStrategyEvolution(
            device=str(self.device),
            population_size=self.config['genetic']['population_size']
        )

        # Evolve initial generations
        logger.info("Evolving genetic strategies...")
        for gen in range(5):
            self.genetic_evolver.evolve_generation(market_data)

        logger.info(f"Genetic evolver ready: Best fitness = {self.genetic_evolver.get_best_strategy()['fitness']:.2f}")

        logger.info("GPU systems initialized successfully")

    def ai_agent_thread(self):
        """Thread for AI agent signal generation"""

        logger.info("AI Agent thread started")

        while self.running:
            try:
                # Get market data
                market_data = self._fetch_market_data()

                if market_data is not None and len(market_data) > 100:
                    # Create trading environment
                    env = TradingEnvironment(market_data)
                    state = env.reset()

                    # Get AI decision
                    action = self.ai_agent.select_action(state, training=False)

                    # Convert action to signal
                    action_mapping = {
                        0: 'HOLD', 1: 'HOLD', 2: 'HOLD', 3: 'HOLD', 4: 'HOLD',
                        5: 'HOLD', 6: 'HOLD', 7: 'HOLD', 8: 'HOLD', 9: 'HOLD',
                        10: 'HOLD', 11: 'BUY', 12: 'BUY', 13: 'BUY', 14: 'BUY',
                        15: 'BUY', 16: 'SELL', 17: 'SELL', 18: 'SELL', 19: 'SELL',
                        20: 'SELL'
                    }

                    action_type = action_mapping.get(action, 'HOLD')

                    if action_type != 'HOLD':
                        # Calculate confidence
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            q_values = self.ai_agent.q_network(state_tensor)
                            confidence = torch.softmax(q_values, dim=1).max().item()

                        # Create signal
                        signal = GPUSignal(
                            source='ai_agent',
                            symbol='SPY',  # Would be dynamic in production
                            action=action_type,
                            confidence=confidence,
                            expected_return=0.02,  # Would be calculated
                            position_size=0.05,
                            reasoning=f"AI Agent DQN action {action}",
                            timestamp=datetime.now(),
                            metadata={'action_index': action}
                        )

                        self.ai_signal_queue.put(signal)
                        self.performance['ai_agent']['signals'] += 1

                        logger.info(f"AI Signal: {action_type} {signal.symbol} (confidence: {confidence:.2f})")

                # Sleep
                time.sleep(self.config['update_frequency'])

            except Exception as e:
                logger.error(f"AI agent thread error: {e}")
                time.sleep(60)

        logger.info("AI Agent thread stopped")

    def genetic_thread(self):
        """Thread for genetic evolution signal generation"""

        logger.info("Genetic evolution thread started")

        while self.running:
            try:
                # Get market data
                market_data = self._fetch_market_data()

                if market_data is not None and len(market_data) > 100:
                    # Evolve strategies
                    for _ in range(self.config['genetic']['generations_per_cycle']):
                        elite_strategies = self.genetic_evolver.evolve_generation(market_data)

                    # Get best strategy
                    best_strategy = self.genetic_evolver.get_best_strategy()

                    # Generate signal from best strategy
                    if best_strategy['fitness'] > 0:  # Profitable strategy
                        # Simple trend determination
                        closes = market_data['Close'].values
                        ema_fast = pd.Series(closes).ewm(span=int(best_strategy['ema_fast'])).mean().values
                        ema_slow = pd.Series(closes).ewm(span=int(best_strategy['ema_slow'])).mean().values

                        if ema_fast[-1] > ema_slow[-1]:
                            action = 'BUY'
                        elif ema_fast[-1] < ema_slow[-1]:
                            action = 'SELL'
                        else:
                            action = 'HOLD'

                        if action != 'HOLD':
                            signal = GPUSignal(
                                source='genetic',
                                symbol='SPY',
                                action=action,
                                confidence=min(0.95, best_strategy['fitness'] / 100 + 0.5),
                                expected_return=best_strategy['fitness'] / 100,
                                position_size=float(best_strategy['position_size']),
                                reasoning=f"Genetic strategy gen{self.genetic_evolver.generation}",
                                timestamp=datetime.now(),
                                metadata=best_strategy
                            )

                            self.genetic_signal_queue.put(signal)
                            self.performance['genetic']['signals'] += 1

                            logger.info(f"Genetic Signal: {action} {signal.symbol} (fitness: {best_strategy['fitness']:.2f})")

                # Sleep
                time.sleep(self.config['update_frequency'])

            except Exception as e:
                logger.error(f"Genetic thread error: {e}")
                time.sleep(60)

        logger.info("Genetic evolution thread stopped")

    def signal_combiner_thread(self):
        """Thread for combining signals from both systems"""

        logger.info("Signal combiner thread started")

        while self.running:
            try:
                # Collect signals from both queues
                ai_signals = []
                genetic_signals = []

                # Get AI signals
                while not self.ai_signal_queue.empty():
                    try:
                        signal = self.ai_signal_queue.get_nowait()
                        ai_signals.append(signal)
                    except queue.Empty:
                        break

                # Get genetic signals
                while not self.genetic_signal_queue.empty():
                    try:
                        signal = self.genetic_signal_queue.get_nowait()
                        genetic_signals.append(signal)
                    except queue.Empty:
                        break

                # Combine signals
                if ai_signals or genetic_signals:
                    combined_signal = self._combine_signals(ai_signals, genetic_signals)

                    if combined_signal:
                        self.combined_signal_queue.put(combined_signal)
                        self.performance['combined']['signals'] += 1

                        logger.info(f"Combined Signal: {combined_signal.action} {combined_signal.symbol} "
                                  f"(confidence: {combined_signal.confidence:.2f})")

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Signal combiner error: {e}")
                time.sleep(10)

        logger.info("Signal combiner thread stopped")

    def _combine_signals(self, ai_signals: List[GPUSignal], genetic_signals: List[GPUSignal]) -> Optional[GPUSignal]:
        """
        Combine signals using ensemble voting

        Returns:
            Combined signal or None
        """
        if not ai_signals and not genetic_signals:
            return None

        # Aggregate by action
        votes = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        weights = self.config['ensemble']['voting_weights']

        # AI agent votes
        for signal in ai_signals:
            votes[signal.action] += weights['ai_agent'] * signal.confidence

        # Genetic votes
        for signal in genetic_signals:
            votes[signal.action] += weights['genetic'] * signal.confidence

        # Determine consensus
        max_vote = max(votes.values())
        winning_action = max(votes, key=votes.get)

        # Calculate combined confidence
        total_votes = sum(votes.values())
        combined_confidence = max_vote / total_votes if total_votes > 0 else 0

        # Check minimum confidence
        if combined_confidence < self.config['ensemble']['min_confidence']:
            return None

        # Check agreement (both systems should somewhat agree)
        ai_action = ai_signals[0].action if ai_signals else winning_action
        genetic_action = genetic_signals[0].action if genetic_signals else winning_action

        agreement = 1.0 if ai_action == genetic_action else 0.5

        if agreement < self.config['ensemble']['min_agreement']:
            return None

        # Create combined signal
        combined_signal = GPUSignal(
            source='combined',
            symbol=ai_signals[0].symbol if ai_signals else genetic_signals[0].symbol,
            action=winning_action,
            confidence=combined_confidence,
            expected_return=(ai_signals[0].expected_return if ai_signals else 0 +
                           genetic_signals[0].expected_return if genetic_signals else 0) / 2,
            position_size=min(self.config['execution']['max_position_size'],
                            max(self.config['execution']['min_position_size'],
                                combined_confidence * 0.10)),
            reasoning=f"Ensemble: AI({len(ai_signals)}) + Genetic({len(genetic_signals)})",
            timestamp=datetime.now(),
            metadata={
                'ai_signals': len(ai_signals),
                'genetic_signals': len(genetic_signals),
                'votes': votes,
                'agreement': agreement
            }
        )

        return combined_signal

    def _fetch_market_data(self) -> Optional[pd.DataFrame]:
        """Fetch market data for analysis"""
        try:
            ticker = yf.Ticker('SPY')
            data = ticker.history(period='3mo', interval='1h')
            return data if not data.empty else None
        except Exception as e:
            logger.error(f"Market data fetch error: {e}")
            return None

    def start(self):
        """Start the orchestrator"""

        logger.info("Starting GPU Trading Orchestrator...")

        # Get initial market data
        market_data = self._fetch_market_data()
        if market_data is None:
            logger.error("Failed to fetch initial market data")
            return

        # Initialize systems
        self.initialize_systems(market_data)

        # Set running flag
        self.running = True

        # Start threads
        threads = [
            threading.Thread(target=self.ai_agent_thread, daemon=True, name='AIAgent'),
            threading.Thread(target=self.genetic_thread, daemon=True, name='Genetic'),
            threading.Thread(target=self.signal_combiner_thread, daemon=True, name='Combiner')
        ]

        for thread in threads:
            thread.start()
            self.threads.append(thread)

        logger.info("GPU Trading Orchestrator started successfully")
        logger.info(f"Threads running: {[t.name for t in self.threads]}")

    def stop(self):
        """Stop the orchestrator"""

        logger.info("Stopping GPU Trading Orchestrator...")

        self.running = False

        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=5)

        logger.info("GPU Trading Orchestrator stopped")

    def get_combined_signals(self, max_signals: int = 10) -> List[GPUSignal]:
        """
        Get combined signals from queue

        Returns:
            List of combined signals
        """
        signals = []

        while not self.combined_signal_queue.empty() and len(signals) < max_signals:
            try:
                signal = self.combined_signal_queue.get_nowait()
                signals.append(signal)
            except queue.Empty:
                break

        return signals

    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cuda_available': self.cuda_available,
            'device': str(self.device),
            'running': self.running,
            'threads_active': len([t for t in self.threads if t.is_alive()]),
            'performance': self.performance,
            'ai_agent_generation': self.ai_agent.performance_metrics.get('total_episodes', 0) if self.ai_agent else 0,
            'genetic_generation': self.genetic_evolver.generation if self.genetic_evolver else 0
        }

def main():
    """Demo the orchestrator"""

    print("\n" + "="*70)
    print("GPU TRADING ORCHESTRATOR")
    print("="*70)

    # Create orchestrator
    orchestrator = GPUTradingOrchestrator()

    try:
        # Start orchestrator
        orchestrator.start()

        print("\nOrchestrator running...")
        print("Generating signals from GPU AI + Genetic Evolution")
        print("Press Ctrl+C to stop\n")

        # Run for demo period
        for i in range(10):
            time.sleep(30)  # Wait 30 seconds between checks

            # Get signals
            signals = orchestrator.get_combined_signals()

            if signals:
                print(f"\n[SIGNALS] Received {len(signals)} combined signal(s)")
                for signal in signals:
                    print(f"  {signal.action} {signal.symbol} - Confidence: {signal.confidence:.2%}")
                    print(f"    Source: {signal.source} | Size: {signal.position_size:.1%}")

            # Performance summary
            if i % 3 == 0:
                perf = orchestrator.get_performance_summary()
                print(f"\n[PERFORMANCE]")
                print(f"  AI Signals: {perf['performance']['ai_agent']['signals']}")
                print(f"  Genetic Signals: {perf['performance']['genetic']['signals']}")
                print(f"  Combined Signals: {perf['performance']['combined']['signals']}")
                print(f"  Threads Active: {perf['threads_active']}/3")

    except KeyboardInterrupt:
        print("\n\nStopping orchestrator...")

    finally:
        orchestrator.stop()

        print("\n" + "="*70)
        print("GPU Trading Orchestrator Demo Complete")
        print("="*70)

if __name__ == "__main__":
    main()
