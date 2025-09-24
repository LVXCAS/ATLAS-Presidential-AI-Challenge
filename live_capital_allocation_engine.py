"""
LIVE CAPITAL ALLOCATION ENGINE
=============================
Intelligent capital allocation across trading strategies
Dynamic position sizing and portfolio optimization
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AllocationMethod(Enum):
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    KELLY_CRITERION = "kelly_criterion"
    SHARPE_OPTIMIZATION = "sharpe_optimization"
    DYNAMIC_MOMENTUM = "dynamic_momentum"

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_id: str
    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trade_count: int
    last_updated: datetime
    is_active: bool = True

@dataclass
class AllocationTarget:
    """Capital allocation target"""
    strategy_id: str
    target_weight: float
    current_weight: float
    target_capital: float
    current_capital: float
    rebalance_threshold: float
    last_rebalance: datetime

class LiveCapitalAllocationEngine:
    """
    LIVE CAPITAL ALLOCATION ENGINE
    Optimizes capital allocation across strategies in real-time
    """

    def __init__(self, total_capital: float = 100000.0):
        self.logger = logging.getLogger('CapitalAllocation')

        # Portfolio parameters
        self.total_capital = total_capital
        self.available_capital = total_capital
        self.reserved_capital = 0.0
        self.emergency_reserve_pct = 0.05  # 5% emergency reserve

        # Strategy tracking
        self.strategy_performances = {}
        self.allocation_targets = {}
        self.allocation_history = []

        # Allocation parameters
        self.allocation_method = AllocationMethod.SHARPE_OPTIMIZATION
        self.rebalance_frequency_minutes = 30
        self.min_allocation_pct = 0.02  # Minimum 2% allocation
        self.max_allocation_pct = 0.25  # Maximum 25% allocation per strategy
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance

        # Risk parameters
        self.max_portfolio_volatility = 0.20  # 20% max portfolio volatility
        self.correlation_penalty = 0.5  # Penalty for high correlation
        self.lookback_periods = 252  # Trading days for performance calculation

        # Performance tracking
        self.portfolio_returns = []
        self.portfolio_values = [total_capital]
        self.allocation_changes = []

        # Real-time monitoring
        self.allocation_active = False
        self.last_rebalance = datetime.now()

        self.logger.info(f"Live Capital Allocation Engine initialized")
        self.logger.info(f"Total capital: ${total_capital:,.2f}")
        self.logger.info(f"Allocation method: {self.allocation_method.value}")

    def register_strategy(self, strategy_id: str, initial_performance: Optional[Dict] = None):
        """Register a new trading strategy"""
        try:
            # Create initial performance metrics
            performance = StrategyPerformance(
                strategy_id=strategy_id,
                total_return=initial_performance.get('total_return', 0.0) if initial_performance else 0.0,
                volatility=initial_performance.get('volatility', 0.15) if initial_performance else 0.15,
                sharpe_ratio=initial_performance.get('sharpe_ratio', 0.0) if initial_performance else 0.0,
                max_drawdown=initial_performance.get('max_drawdown', 0.0) if initial_performance else 0.0,
                win_rate=initial_performance.get('win_rate', 0.5) if initial_performance else 0.5,
                profit_factor=initial_performance.get('profit_factor', 1.0) if initial_performance else 1.0,
                trade_count=0,
                last_updated=datetime.now(),
                is_active=True
            )

            self.strategy_performances[strategy_id] = performance

            # Create initial allocation target
            initial_weight = 1.0 / max(len(self.strategy_performances), 1)
            allocation = AllocationTarget(
                strategy_id=strategy_id,
                target_weight=initial_weight,
                current_weight=0.0,
                target_capital=self.total_capital * initial_weight,
                current_capital=0.0,
                rebalance_threshold=self.rebalance_threshold,
                last_rebalance=datetime.now()
            )

            self.allocation_targets[strategy_id] = allocation

            self.logger.info(f"Strategy registered: {strategy_id}")
            self.logger.info(f"  Initial allocation: {initial_weight:.1%}")

        except Exception as e:
            self.logger.error(f"Error registering strategy {strategy_id}: {e}")

    def update_strategy_performance(self, strategy_id: str, performance_data: Dict):
        """Update strategy performance metrics"""
        try:
            if strategy_id not in self.strategy_performances:
                self.logger.warning(f"Strategy {strategy_id} not registered")
                return

            performance = self.strategy_performances[strategy_id]

            # Update performance metrics
            performance.total_return = performance_data.get('total_return', performance.total_return)
            performance.volatility = performance_data.get('volatility', performance.volatility)
            performance.sharpe_ratio = performance_data.get('sharpe_ratio', performance.sharpe_ratio)
            performance.max_drawdown = performance_data.get('max_drawdown', performance.max_drawdown)
            performance.win_rate = performance_data.get('win_rate', performance.win_rate)
            performance.profit_factor = performance_data.get('profit_factor', performance.profit_factor)
            performance.trade_count = performance_data.get('trade_count', performance.trade_count)
            performance.last_updated = datetime.now()

            self.logger.info(f"Updated performance for {strategy_id}")
            self.logger.info(f"  Return: {performance.total_return:.2%} | Sharpe: {performance.sharpe_ratio:.2f}")

        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {e}")

    def calculate_optimal_allocation(self) -> Dict[str, float]:
        """Calculate optimal capital allocation using selected method"""
        try:
            if len(self.strategy_performances) == 0:
                return {}

            active_strategies = {sid: perf for sid, perf in self.strategy_performances.items() if perf.is_active}

            if len(active_strategies) == 0:
                return {}

            # Calculate allocation based on method
            if self.allocation_method == AllocationMethod.EQUAL_WEIGHT:
                return self._equal_weight_allocation(active_strategies)
            elif self.allocation_method == AllocationMethod.RISK_PARITY:
                return self._risk_parity_allocation(active_strategies)
            elif self.allocation_method == AllocationMethod.SHARPE_OPTIMIZATION:
                return self._sharpe_optimization_allocation(active_strategies)
            elif self.allocation_method == AllocationMethod.KELLY_CRITERION:
                return self._kelly_criterion_allocation(active_strategies)
            elif self.allocation_method == AllocationMethod.DYNAMIC_MOMENTUM:
                return self._dynamic_momentum_allocation(active_strategies)
            else:
                return self._equal_weight_allocation(active_strategies)

        except Exception as e:
            self.logger.error(f"Error calculating optimal allocation: {e}")
            return {}

    def _equal_weight_allocation(self, strategies: Dict) -> Dict[str, float]:
        """Equal weight allocation"""
        n_strategies = len(strategies)
        equal_weight = 1.0 / n_strategies
        return {sid: equal_weight for sid in strategies.keys()}

    def _risk_parity_allocation(self, strategies: Dict) -> Dict[str, float]:
        """Risk parity allocation - allocate inversely proportional to volatility"""
        try:
            # Calculate inverse volatility weights
            inv_vols = {}
            for sid, perf in strategies.items():
                inv_vols[sid] = 1.0 / max(perf.volatility, 0.01)  # Avoid division by zero

            # Normalize weights
            total_inv_vol = sum(inv_vols.values())
            weights = {sid: inv_vol / total_inv_vol for sid, inv_vol in inv_vols.items()}

            # Apply min/max constraints
            return self._apply_allocation_constraints(weights)

        except Exception as e:
            self.logger.error(f"Risk parity allocation error: {e}")
            return self._equal_weight_allocation(strategies)

    def _sharpe_optimization_allocation(self, strategies: Dict) -> Dict[str, float]:
        """Sharpe ratio optimization allocation"""
        try:
            # Calculate Sharpe ratio weights
            sharpe_ratios = {}
            for sid, perf in strategies.items():
                # Use positive Sharpe ratios, set minimum floor
                sharpe_ratios[sid] = max(perf.sharpe_ratio, 0.1)

            # Exponential weighting for Sharpe ratios
            exp_sharpes = {sid: np.exp(sharpe * 2) for sid, sharpe in sharpe_ratios.items()}

            # Normalize weights
            total_exp_sharpe = sum(exp_sharpes.values())
            weights = {sid: exp_sharpe / total_exp_sharpe for sid, exp_sharpe in exp_sharpes.items()}

            # Apply min/max constraints
            return self._apply_allocation_constraints(weights)

        except Exception as e:
            self.logger.error(f"Sharpe optimization error: {e}")
            return self._equal_weight_allocation(strategies)

    def _kelly_criterion_allocation(self, strategies: Dict) -> Dict[str, float]:
        """Kelly criterion allocation"""
        try:
            kelly_weights = {}

            for sid, perf in strategies.items():
                # Kelly formula: f = (bp - q) / b
                # where b = odds, p = win probability, q = loss probability
                win_rate = perf.win_rate
                loss_rate = 1 - win_rate

                # Estimate average win/loss ratio from profit factor
                if loss_rate > 0:
                    avg_win_loss_ratio = (perf.profit_factor * loss_rate) / win_rate if win_rate > 0 else 1.0
                else:
                    avg_win_loss_ratio = 1.0

                # Kelly fraction
                kelly_fraction = (win_rate * avg_win_loss_ratio - loss_rate) / avg_win_loss_ratio

                # Cap Kelly fraction to avoid excessive leverage
                kelly_weights[sid] = max(min(kelly_fraction, 0.25), 0.02)

            # Normalize weights
            total_kelly = sum(kelly_weights.values())
            if total_kelly > 0:
                weights = {sid: weight / total_kelly for sid, weight in kelly_weights.items()}
            else:
                weights = self._equal_weight_allocation(strategies)

            return self._apply_allocation_constraints(weights)

        except Exception as e:
            self.logger.error(f"Kelly criterion allocation error: {e}")
            return self._equal_weight_allocation(strategies)

    def _dynamic_momentum_allocation(self, strategies: Dict) -> Dict[str, float]:
        """Dynamic momentum-based allocation"""
        try:
            momentum_scores = {}

            for sid, perf in strategies.items():
                # Combine multiple momentum factors
                return_momentum = max(perf.total_return, 0)
                sharpe_momentum = max(perf.sharpe_ratio, 0.1)
                consistency_momentum = min(perf.win_rate * 2, 1.0)

                # Composite momentum score
                momentum_score = (return_momentum * 0.4 +
                                sharpe_momentum * 0.4 +
                                consistency_momentum * 0.2)

                momentum_scores[sid] = momentum_score

            # Exponential weighting
            exp_momentum = {sid: np.exp(score) for sid, score in momentum_scores.items()}

            # Normalize weights
            total_exp_momentum = sum(exp_momentum.values())
            weights = {sid: exp_momentum / total_exp_momentum for sid, exp_momentum in exp_momentum.items()}

            return self._apply_allocation_constraints(weights)

        except Exception as e:
            self.logger.error(f"Dynamic momentum allocation error: {e}")
            return self._equal_weight_allocation(strategies)

    def _apply_allocation_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max allocation constraints"""
        try:
            # Apply minimum allocation constraint
            constrained_weights = {}
            for sid, weight in weights.items():
                constrained_weights[sid] = max(weight, self.min_allocation_pct)

            # Apply maximum allocation constraint
            for sid, weight in constrained_weights.items():
                constrained_weights[sid] = min(weight, self.max_allocation_pct)

            # Renormalize to sum to 1
            total_weight = sum(constrained_weights.values())
            if total_weight > 0:
                constrained_weights = {sid: weight / total_weight for sid, weight in constrained_weights.items()}

            return constrained_weights

        except Exception as e:
            self.logger.error(f"Error applying allocation constraints: {e}")
            return weights

    def calculate_rebalance_requirements(self) -> Dict[str, Dict]:
        """Calculate which strategies need rebalancing"""
        try:
            rebalance_requirements = {}

            for strategy_id, allocation in self.allocation_targets.items():
                weight_deviation = abs(allocation.current_weight - allocation.target_weight)
                capital_deviation = abs(allocation.current_capital - allocation.target_capital)

                needs_rebalance = (
                    weight_deviation > allocation.rebalance_threshold or
                    capital_deviation > (self.total_capital * 0.02)  # 2% of total capital
                )

                if needs_rebalance:
                    rebalance_requirements[strategy_id] = {
                        'current_weight': allocation.current_weight,
                        'target_weight': allocation.target_weight,
                        'weight_deviation': weight_deviation,
                        'current_capital': allocation.current_capital,
                        'target_capital': allocation.target_capital,
                        'capital_adjustment': allocation.target_capital - allocation.current_capital
                    }

            return rebalance_requirements

        except Exception as e:
            self.logger.error(f"Error calculating rebalance requirements: {e}")
            return {}

    async def execute_rebalancing(self) -> bool:
        """Execute portfolio rebalancing"""
        try:
            # Calculate optimal allocation
            optimal_weights = self.calculate_optimal_allocation()

            if not optimal_weights:
                self.logger.warning("No optimal allocation calculated")
                return False

            # Update allocation targets
            available_capital = self.total_capital * (1 - self.emergency_reserve_pct)

            for strategy_id, target_weight in optimal_weights.items():
                if strategy_id in self.allocation_targets:
                    allocation = self.allocation_targets[strategy_id]
                    allocation.target_weight = target_weight
                    allocation.target_capital = available_capital * target_weight

            # Calculate rebalance requirements
            rebalance_requirements = self.calculate_rebalance_requirements()

            if rebalance_requirements:
                self.logger.info(f"Executing rebalancing for {len(rebalance_requirements)} strategies")

                for strategy_id, requirements in rebalance_requirements.items():
                    capital_adjustment = requirements['capital_adjustment']

                    # Update current allocation
                    allocation = self.allocation_targets[strategy_id]
                    allocation.current_capital = allocation.target_capital
                    allocation.current_weight = allocation.target_weight
                    allocation.last_rebalance = datetime.now()

                    self.logger.info(f"Rebalanced {strategy_id}: {capital_adjustment:+.2f} -> {allocation.target_weight:.1%}")

                # Record rebalancing
                self.allocation_changes.append({
                    'timestamp': datetime.now(),
                    'rebalanced_strategies': list(rebalance_requirements.keys()),
                    'new_allocations': optimal_weights
                })

                self.last_rebalance = datetime.now()
                return True

            else:
                self.logger.info("No rebalancing required")
                return False

        except Exception as e:
            self.logger.error(f"Rebalancing execution error: {e}")
            return False

    async def start_allocation_engine(self):
        """Start the capital allocation engine"""
        try:
            self.allocation_active = True
            self.logger.info("Starting live capital allocation engine")

            while self.allocation_active:
                try:
                    # Check if rebalancing is needed
                    time_since_rebalance = datetime.now() - self.last_rebalance
                    rebalance_needed = time_since_rebalance.total_seconds() > (self.rebalance_frequency_minutes * 60)

                    if rebalance_needed:
                        await self.execute_rebalancing()

                    # Update portfolio performance
                    await self.update_portfolio_performance()

                    # Log status
                    await self.log_allocation_status()

                    # Sleep until next check
                    await asyncio.sleep(60)  # Check every minute

                except Exception as e:
                    self.logger.error(f"Allocation engine loop error: {e}")
                    await asyncio.sleep(60)

        except Exception as e:
            self.logger.error(f"Allocation engine error: {e}")

    async def update_portfolio_performance(self):
        """Update overall portfolio performance"""
        try:
            # Calculate current portfolio value
            current_value = sum(allocation.current_capital for allocation in self.allocation_targets.values())
            self.portfolio_values.append(current_value)

            # Calculate portfolio return
            if len(self.portfolio_values) > 1:
                portfolio_return = (current_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
                self.portfolio_returns.append(portfolio_return)

            # Update available capital
            self.available_capital = self.total_capital - current_value

        except Exception as e:
            self.logger.error(f"Portfolio performance update error: {e}")

    async def log_allocation_status(self):
        """Log current allocation status"""
        try:
            # Log every 10 minutes
            if datetime.now().minute % 10 == 0:
                self.logger.info("CAPITAL ALLOCATION STATUS:")
                self.logger.info(f"  Total Capital: ${self.total_capital:,.2f}")
                self.logger.info(f"  Available Capital: ${self.available_capital:,.2f}")

                for strategy_id, allocation in self.allocation_targets.items():
                    self.logger.info(f"  {strategy_id}: {allocation.current_weight:.1%} (${allocation.current_capital:,.2f})")

        except Exception as e:
            self.logger.error(f"Allocation status logging error: {e}")

    def get_allocation_summary(self) -> Dict:
        """Get current allocation summary"""
        try:
            allocations = {}
            total_allocated = 0

            for strategy_id, allocation in self.allocation_targets.items():
                allocations[strategy_id] = {
                    'target_weight': allocation.target_weight,
                    'current_weight': allocation.current_weight,
                    'target_capital': allocation.target_capital,
                    'current_capital': allocation.current_capital,
                    'last_rebalance': allocation.last_rebalance.isoformat()
                }
                total_allocated += allocation.current_capital

            return {
                'total_capital': self.total_capital,
                'total_allocated': total_allocated,
                'available_capital': self.available_capital,
                'emergency_reserve': self.total_capital * self.emergency_reserve_pct,
                'allocation_method': self.allocation_method.value,
                'last_rebalance': self.last_rebalance.isoformat(),
                'allocations': allocations,
                'portfolio_value': self.portfolio_values[-1] if self.portfolio_values else self.total_capital,
                'total_rebalances': len(self.allocation_changes)
            }

        except Exception as e:
            self.logger.error(f"Error getting allocation summary: {e}")
            return {}

    def stop_allocation_engine(self):
        """Stop the allocation engine"""
        self.allocation_active = False
        self.logger.info("Capital allocation engine stopped")

async def demo_capital_allocation():
    """Demo the capital allocation engine"""
    print("="*80)
    print("LIVE CAPITAL ALLOCATION ENGINE DEMO")
    print("Intelligent capital allocation across trading strategies")
    print("="*80)

    # Initialize allocation engine
    engine = LiveCapitalAllocationEngine(total_capital=500000.0)

    # Register demo strategies
    strategies = [
        ("GPU_AI_AGENT", {"total_return": 0.15, "volatility": 0.12, "sharpe_ratio": 1.25, "win_rate": 0.65}),
        ("GPU_PATTERN_RECOGNITION", {"total_return": 0.22, "volatility": 0.18, "sharpe_ratio": 1.22, "win_rate": 0.62}),
        ("GPU_MOMENTUM_SCANNER", {"total_return": 0.18, "volatility": 0.15, "sharpe_ratio": 1.20, "win_rate": 0.58}),
        ("GPU_OPTIONS_ENGINE", {"total_return": 0.25, "volatility": 0.20, "sharpe_ratio": 1.25, "win_rate": 0.55}),
        ("RD_ENHANCED_SIGNALS", {"total_return": 0.12, "volatility": 0.10, "sharpe_ratio": 1.20, "win_rate": 0.68})
    ]

    print(f"\nRegistering {len(strategies)} trading strategies...")
    for strategy_id, performance in strategies:
        engine.register_strategy(strategy_id, performance)

    # Show initial allocation
    print(f"\nInitial allocation summary:")
    summary = engine.get_allocation_summary()
    for strategy_id, allocation in summary['allocations'].items():
        print(f"  {strategy_id}: {allocation['target_weight']:.1%} (${allocation['target_capital']:,.2f})")

    # Test different allocation methods
    allocation_methods = [
        AllocationMethod.EQUAL_WEIGHT,
        AllocationMethod.RISK_PARITY,
        AllocationMethod.SHARPE_OPTIMIZATION,
        AllocationMethod.KELLY_CRITERION
    ]

    print(f"\nTesting allocation methods:")
    for method in allocation_methods:
        engine.allocation_method = method
        optimal_allocation = engine.calculate_optimal_allocation()
        print(f"\n{method.value}:")
        for strategy_id, weight in optimal_allocation.items():
            print(f"  {strategy_id}: {weight:.1%}")

    # Start allocation engine for demo
    print(f"\nStarting capital allocation demo for 15 seconds...")
    try:
        await asyncio.wait_for(engine.start_allocation_engine(), timeout=15)
    except asyncio.TimeoutError:
        print("\nDemo completed")
    finally:
        engine.stop_allocation_engine()

        # Show final status
        final_summary = engine.get_allocation_summary()
        print(f"\nFinal allocation status:")
        print(f"  Total capital: ${final_summary['total_capital']:,.2f}")
        print(f"  Portfolio value: ${final_summary['portfolio_value']:,.2f}")
        print(f"  Available capital: ${final_summary['available_capital']:,.2f}")
        print(f"  Total rebalances: {final_summary['total_rebalances']}")

    print(f"\nLive Capital Allocation Engine ready for autonomous trading!")

if __name__ == "__main__":
    asyncio.run(demo_capital_allocation())