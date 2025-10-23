#!/usr/bin/env python3
"""
OPTIONS LEARNING INTEGRATION
=============================
Connects the Week 3 Options Scanner with the Continuous Learning System
to enable adaptive strategy optimization and win rate improvement.

Goal: Improve options win rate from 55% → 65%+ through continuous feedback loops

Key Features:
- Tracks every options trade execution with outcomes
- Logs Greeks performance (delta ranges, strike selection)
- Feeds performance data to ContinuousLearningSystem
- Receives optimized parameters from learning cycles
- Applies updates to confidence threshold, strike selection, position sizing
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - OPTIONS_LEARNING - %(message)s')
logger = logging.getLogger(__name__)

# Import learning system components
try:
    from core.continuous_learning_system import (
        ContinuousLearningSystem,
        FeedbackEvent,
        FeedbackType,
        LearningObjective,
        StrategyPerformance
    )
    CONTINUOUS_LEARNING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Continuous learning system not available: {e}")
    CONTINUOUS_LEARNING_AVAILABLE = False
    # Create stub classes for compatibility
    class FeedbackType:
        TRADE_EXECUTION = "trade_execution"

    class LearningObjective:
        MAXIMIZE_RETURN = "maximize_return"
        MAXIMIZE_SHARPE = "maximize_sharpe"
        OPTIMIZE_RISK_ADJUSTED = "optimize_risk_adjusted"

    class FeedbackEvent:
        pass

    class ContinuousLearningSystem:
        pass


@dataclass
class OptionsTrade:
    """Complete options trade record for learning"""
    trade_id: str
    timestamp: datetime
    symbol: str
    strategy_type: str  # DUAL_OPTIONS, BULL_PUT_SPREAD, BUTTERFLY

    # Entry data
    entry_price: float
    contracts: int
    put_strike: float
    call_strike: Optional[float]
    expiration_date: str

    # Greeks at entry (if available)
    put_delta: Optional[float] = None
    call_delta: Optional[float] = None
    put_theta: Optional[float] = None
    call_theta: Optional[float] = None
    put_vega: Optional[float] = None
    call_vega: Optional[float] = None

    # Market regime
    market_regime: str = 'neutral'
    volatility: float = 0.0
    momentum: float = 0.0

    # Selection parameters
    confidence_threshold: float = 4.0
    strike_selection_method: str = 'PERCENTAGE_BASED'

    # Exit data (filled later)
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    win_loss: Optional[str] = None  # 'WIN', 'LOSS', 'BREAK_EVEN'

    # Performance metrics
    return_pct: Optional[float] = None
    hold_duration_hours: Optional[float] = None
    max_drawdown: Optional[float] = None

    # Learning insights
    greek_performance: Optional[Dict[str, Any]] = None
    strike_accuracy: Optional[Dict[str, Any]] = None
    regime_fit: Optional[Dict[str, Any]] = None


class OptionsLearningTracker:
    """Tracks options trades and provides learning feedback"""

    def __init__(self, config_path: str = "options_learning_config.json"):
        self.config = self._load_config(config_path)
        self.active_trades: Dict[str, OptionsTrade] = {}
        self.completed_trades: List[OptionsTrade] = []
        self.strategy_stats: Dict[str, Dict[str, Any]] = {}

        # Learning system integration
        self.learning_system: Optional[ContinuousLearningSystem] = None
        self.learning_enabled = self.config.get('learning_enabled', True)

        # Current optimized parameters
        self.optimized_params = {
            'confidence_threshold': self.config.get('base_confidence_threshold', 4.0),
            'put_delta_target': self.config.get('put_delta_target', -0.35),
            'call_delta_target': self.config.get('call_delta_target', 0.35),
            'position_size_multiplier': self.config.get('position_size_multiplier', 1.0),
            'bull_put_momentum_threshold': self.config.get('bull_put_momentum_threshold', 0.03),
        }

        # Load existing trades from disk
        self._load_completed_trades()
        self._load_active_trades()

        logger.info("OptionsLearningTracker initialized")
        logger.info(f"Learning enabled: {self.learning_enabled}")
        logger.info(f"Initial parameters: {self.optimized_params}")
        logger.info(f"Loaded {len(self.completed_trades)} completed trades from disk")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded config from {config_path}")
                    return config
            else:
                logger.warning(f"Config file {config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'learning_enabled': True,
            'learning_frequency': 'weekly',  # daily, weekly, manual
            'min_feedback_samples': 20,
            'max_parameter_change': 0.20,  # 20% max change for safety
            'base_confidence_threshold': 4.0,
            'put_delta_target': -0.35,
            'call_delta_target': 0.35,
            'position_size_multiplier': 1.0,
            'bull_put_momentum_threshold': 0.03,
            'learning_objectives': [
                'maximize_win_rate',
                'maximize_profit_factor',
                'maximize_sharpe_ratio'
            ]
        }

    async def initialize_learning_system(self) -> bool:
        """Initialize the continuous learning system"""
        try:
            if not self.learning_enabled:
                logger.info("Learning system disabled by config")
                return False

            if not CONTINUOUS_LEARNING_AVAILABLE:
                logger.warning("Continuous learning system not available - using basic tracking only")
                return False

            learning_config = {
                'learning_frequency_minutes': 10080 if self.config.get('learning_frequency') == 'weekly' else 1440,  # Weekly or daily
                'min_feedback_samples': self.config.get('min_feedback_samples', 20),
                'max_parameter_change': self.config.get('max_parameter_change', 0.20)
            }

            self.learning_system = ContinuousLearningSystem(learning_config)
            success = await self.learning_system.initialize()

            if success:
                logger.info("Continuous learning system initialized successfully")
                await self.learning_system.start()
            else:
                logger.error("Failed to initialize learning system")

            return success

        except Exception as e:
            logger.error(f"Error initializing learning system: {e}")
            return False

    def log_trade_entry(self, trade: OptionsTrade) -> None:
        """Log a new options trade entry"""
        try:
            self.active_trades[trade.trade_id] = trade

            logger.info(f"Logged trade entry: {trade.trade_id}")
            logger.info(f"  Symbol: {trade.symbol}")
            logger.info(f"  Strategy: {trade.strategy_type}")
            logger.info(f"  Put Strike: ${trade.put_strike:.2f}")
            if trade.call_strike:
                logger.info(f"  Call Strike: ${trade.call_strike:.2f}")
            logger.info(f"  Contracts: {trade.contracts}")

            # Save to disk
            self._save_active_trades()

        except Exception as e:
            logger.error(f"Error logging trade entry: {e}")

    def log_trade_exit(self, trade_id: str, exit_price: float, realized_pnl: float) -> None:
        """Log options trade exit and calculate performance"""
        try:
            if trade_id not in self.active_trades:
                logger.warning(f"Trade {trade_id} not found in active trades")
                return

            trade = self.active_trades[trade_id]

            # Update exit data
            trade.exit_timestamp = datetime.now(timezone.utc)
            trade.exit_price = exit_price
            trade.realized_pnl = realized_pnl

            # Calculate metrics
            trade.return_pct = (realized_pnl / (trade.entry_price * trade.contracts * 100)) if trade.entry_price > 0 else 0.0
            trade.hold_duration_hours = (trade.exit_timestamp - trade.timestamp).total_seconds() / 3600

            # Determine win/loss
            if realized_pnl > 0:
                trade.win_loss = 'WIN'
            elif realized_pnl < 0:
                trade.win_loss = 'LOSS'
            else:
                trade.win_loss = 'BREAK_EVEN'

            # Analyze Greeks performance
            trade.greek_performance = self._analyze_greek_performance(trade)

            # Analyze strike selection
            trade.strike_accuracy = self._analyze_strike_accuracy(trade)

            # Analyze regime fit
            trade.regime_fit = self._analyze_regime_fit(trade)

            # Move to completed
            self.completed_trades.append(trade)
            del self.active_trades[trade_id]

            logger.info(f"Logged trade exit: {trade_id}")
            logger.info(f"  P&L: ${realized_pnl:.2f}")
            logger.info(f"  Return: {trade.return_pct:.2%}")
            logger.info(f"  Result: {trade.win_loss}")
            logger.info(f"  Hold duration: {trade.hold_duration_hours:.1f} hours")

            # Update strategy statistics
            self._update_strategy_stats(trade)

            # Send feedback to learning system
            if self.learning_enabled and self.learning_system:
                asyncio.create_task(self._send_learning_feedback(trade))

            # Save to disk
            self._save_active_trades()
            self._save_completed_trades()

        except Exception as e:
            logger.error(f"Error logging trade exit: {e}")

    def _analyze_greek_performance(self, trade: OptionsTrade) -> Dict[str, Any]:
        """Analyze how Greeks performed during the trade"""
        try:
            analysis = {
                'put_delta_effective': True,
                'call_delta_effective': True,
                'theta_decay_favorable': True,
                'vega_impact': 'neutral'
            }

            # Analyze put delta (for cash-secured puts, we want -0.30 to -0.40)
            if trade.put_delta is not None:
                if trade.win_loss == 'WIN':
                    # Did the delta range work well?
                    if -0.40 <= trade.put_delta <= -0.30:
                        analysis['put_delta_effective'] = True
                    else:
                        analysis['put_delta_effective'] = False
                        analysis['put_delta_recommendation'] = 'adjust_closer_to_target'

            # Analyze call delta (for long calls, we want 0.30 to 0.40)
            if trade.call_delta is not None:
                if trade.win_loss == 'WIN':
                    if 0.30 <= trade.call_delta <= 0.40:
                        analysis['call_delta_effective'] = True
                    else:
                        analysis['call_delta_effective'] = False
                        analysis['call_delta_recommendation'] = 'adjust_closer_to_target'

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Greek performance: {e}")
            return {}

    def _analyze_strike_accuracy(self, trade: OptionsTrade) -> Dict[str, Any]:
        """Analyze how accurate strike selection was"""
        try:
            analysis = {
                'put_strike_optimal': True,
                'call_strike_optimal': True,
                'method_effective': trade.strike_selection_method
            }

            # For winning trades, strikes were likely good
            # For losing trades, we may need to adjust
            if trade.win_loss == 'WIN':
                analysis['put_strike_optimal'] = True
                analysis['call_strike_optimal'] = True
            else:
                # Analyze if strikes were too aggressive or too conservative
                # This would require more market data analysis
                analysis['requires_adjustment'] = True

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing strike accuracy: {e}")
            return {}

    def _analyze_regime_fit(self, trade: OptionsTrade) -> Dict[str, Any]:
        """Analyze if strategy fit the market regime"""
        try:
            analysis = {
                'regime': trade.market_regime,
                'strategy': trade.strategy_type,
                'fit_score': 0.5
            }

            # Bull Put Spreads work best in neutral/slightly bullish markets
            if trade.strategy_type == 'BULL_PUT_SPREAD':
                if trade.market_regime in ['neutral', 'bull'] and abs(trade.momentum) < 0.03:
                    analysis['fit_score'] = 0.9
                else:
                    analysis['fit_score'] = 0.3

            # Dual Options work best in trending markets
            elif trade.strategy_type == 'DUAL_OPTIONS':
                if abs(trade.momentum) > 0.05:
                    analysis['fit_score'] = 0.9
                else:
                    analysis['fit_score'] = 0.4

            # Butterfly spreads work best in low volatility
            elif trade.strategy_type == 'BUTTERFLY':
                if trade.volatility < 0.02:
                    analysis['fit_score'] = 0.9
                else:
                    analysis['fit_score'] = 0.3

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing regime fit: {e}")
            return {}

    def _update_strategy_stats(self, trade: OptionsTrade) -> None:
        """Update strategy statistics"""
        try:
            strategy = trade.strategy_type

            if strategy not in self.strategy_stats:
                self.strategy_stats[strategy] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'best_market_regime': 'unknown',
                    'optimal_delta_range': {'put': (-0.35, -0.35), 'call': (0.35, 0.35)}
                }

            stats = self.strategy_stats[strategy]
            stats['total_trades'] += 1
            stats['total_pnl'] += trade.realized_pnl or 0.0

            if trade.win_loss == 'WIN':
                stats['winning_trades'] += 1
                stats['avg_win'] = (stats['avg_win'] * (stats['winning_trades'] - 1) + trade.realized_pnl) / stats['winning_trades']
            elif trade.win_loss == 'LOSS':
                stats['losing_trades'] += 1
                stats['avg_loss'] = (stats['avg_loss'] * (stats['losing_trades'] - 1) + abs(trade.realized_pnl or 0.0)) / stats['losing_trades']

            # Calculate derived metrics
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades'] if stats['total_trades'] > 0 else 0.0
            stats['profit_factor'] = abs(stats['avg_win'] / stats['avg_loss']) if stats['avg_loss'] != 0 else float('inf')

            logger.info(f"Updated {strategy} stats:")
            logger.info(f"  Total trades: {stats['total_trades']}")
            logger.info(f"  Win rate: {stats['win_rate']:.1%}")
            logger.info(f"  Profit factor: {stats['profit_factor']:.2f}")

        except Exception as e:
            logger.error(f"Error updating strategy stats: {e}")

    async def _send_learning_feedback(self, trade: OptionsTrade) -> None:
        """Send trade feedback to learning system"""
        try:
            if not self.learning_system or not CONTINUOUS_LEARNING_AVAILABLE:
                return

            # Create feedback event
            feedback = FeedbackEvent(
                id=f"options_feedback_{trade.trade_id}",
                feedback_type=FeedbackType.TRADE_EXECUTION,
                timestamp=trade.exit_timestamp or datetime.now(timezone.utc),
                symbol=trade.symbol,
                strategy_id=f"options_{trade.strategy_type.lower()}",
                execution_data={
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price or 0.0,
                    'contracts': trade.contracts,
                    'put_strike': trade.put_strike,
                    'call_strike': trade.call_strike,
                    'fill_rate': 1.0,  # Assume full fill for now
                    'slippage_bps': 5.0,  # Estimate
                    'execution_time_ms': 200.0  # Estimate
                },
                performance_metrics={
                    'return': trade.return_pct or 0.0,
                    'sharpe_ratio': (trade.return_pct or 0.0) / max(0.01, trade.volatility),
                    'volatility': trade.volatility,
                    'drawdown': trade.max_drawdown or 0.0,
                    'win_loss': 1.0 if trade.win_loss == 'WIN' else 0.0
                },
                market_context={
                    'volatility': trade.volatility,
                    'trend_strength': trade.momentum,
                    'volume_ratio': 1.0,  # Would need actual data
                    'market_regime': trade.market_regime
                },
                metadata={
                    'parameters': {
                        'confidence_threshold': trade.confidence_threshold,
                        'put_delta_target': self.optimized_params['put_delta_target'],
                        'call_delta_target': self.optimized_params['call_delta_target'],
                        'strike_selection_method': trade.strike_selection_method
                    },
                    'greeks': {
                        'put_delta': trade.put_delta,
                        'call_delta': trade.call_delta,
                        'put_theta': trade.put_theta,
                        'call_theta': trade.call_theta
                    },
                    'analysis': {
                        'greek_performance': trade.greek_performance,
                        'strike_accuracy': trade.strike_accuracy,
                        'regime_fit': trade.regime_fit
                    }
                }
            )

            # Send to learning system
            await self.learning_system.process_feedback(feedback)
            logger.info(f"Sent learning feedback for trade {trade.trade_id}")

        except Exception as e:
            logger.error(f"Error sending learning feedback: {e}")

    async def run_learning_cycle(self, objective: str = 'maximize_win_rate') -> Dict[str, Any]:
        """Run a learning cycle and get optimized parameters"""
        try:
            if not self.learning_enabled or not self.learning_system or not CONTINUOUS_LEARNING_AVAILABLE:
                logger.warning("Learning system not enabled or not initialized")
                return self.optimized_params

            # Check if we have enough data
            if len(self.completed_trades) < self.config.get('min_feedback_samples', 20):
                logger.info(f"Insufficient trades for learning: {len(self.completed_trades)} < {self.config.get('min_feedback_samples', 20)}")
                return self.optimized_params

            # Map objective
            objective_map = {
                'maximize_win_rate': LearningObjective.MAXIMIZE_RETURN,  # Higher returns correlate with win rate
                'maximize_profit_factor': LearningObjective.OPTIMIZE_RISK_ADJUSTED,
                'maximize_sharpe_ratio': LearningObjective.MAXIMIZE_SHARPE
            }

            learning_objective = objective_map.get(objective, LearningObjective.MAXIMIZE_SHARPE)

            # Run learning cycle for each strategy type
            results = {}
            for strategy_type in ['DUAL_OPTIONS', 'BULL_PUT_SPREAD', 'BUTTERFLY']:
                strategy_id = f"options_{strategy_type.lower()}"

                result = await self.learning_system.run_learning_cycle(strategy_id, learning_objective)

                if result.deployment_ready and result.performance_improvement > 0:
                    logger.info(f"Learning cycle complete for {strategy_type}")
                    logger.info(f"  Performance improvement: {result.performance_improvement:.2%}")
                    logger.info(f"  Confidence: {result.confidence_score:.2f}")

                    # Update optimized parameters
                    self._apply_optimized_parameters(result.optimized_parameters)
                    results[strategy_type] = result.optimized_parameters

            return self.optimized_params

        except Exception as e:
            logger.error(f"Error running learning cycle: {e}")
            return self.optimized_params

    def _apply_optimized_parameters(self, optimized_params: Dict[str, Any]) -> None:
        """Apply optimized parameters with safety limits"""
        try:
            max_change = self.config.get('max_parameter_change', 0.20)

            for key, new_value in optimized_params.items():
                if key in self.optimized_params:
                    old_value = self.optimized_params[key]

                    # Apply safety limit (max 20% change)
                    if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
                        max_change_abs = abs(old_value * max_change)
                        change = new_value - old_value

                        if abs(change) > max_change_abs:
                            # Limit the change
                            limited_value = old_value + (max_change_abs if change > 0 else -max_change_abs)
                            logger.warning(f"Limited {key} change: {new_value:.3f} -> {limited_value:.3f}")
                            self.optimized_params[key] = limited_value
                        else:
                            self.optimized_params[key] = new_value
                            logger.info(f"Updated {key}: {old_value:.3f} -> {new_value:.3f}")

            # Save updated parameters
            self._save_optimized_parameters()

        except Exception as e:
            logger.error(f"Error applying optimized parameters: {e}")

    def get_optimized_parameters(self) -> Dict[str, Any]:
        """Get current optimized parameters"""
        return self.optimized_params.copy()

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics"""
        return {
            'strategy_stats': self.strategy_stats,
            'total_trades': len(self.completed_trades),
            'active_trades': len(self.active_trades),
            'overall_win_rate': self._calculate_overall_win_rate(),
            'overall_profit_factor': self._calculate_overall_profit_factor(),
            'learning_enabled': self.learning_enabled,
            'optimized_parameters': self.optimized_params
        }

    def _calculate_overall_win_rate(self) -> float:
        """Calculate overall win rate across all strategies"""
        if not self.completed_trades:
            return 0.0

        wins = sum(1 for trade in self.completed_trades if trade.win_loss == 'WIN')
        return wins / len(self.completed_trades)

    def _calculate_overall_profit_factor(self) -> float:
        """Calculate overall profit factor"""
        if not self.completed_trades:
            return 0.0

        total_wins = sum(trade.realized_pnl or 0.0 for trade in self.completed_trades if trade.win_loss == 'WIN')
        total_losses = sum(abs(trade.realized_pnl or 0.0) for trade in self.completed_trades if trade.win_loss == 'LOSS')

        return total_wins / total_losses if total_losses > 0 else float('inf')

    def _load_completed_trades(self) -> None:
        """Load completed trades from disk"""
        try:
            filename = "data/options_completed_trades.json"
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)

                # Convert back to OptionsTrade objects
                for trade_dict in data:
                    # Convert ISO datetime strings back to datetime objects
                    if 'timestamp' in trade_dict and isinstance(trade_dict['timestamp'], str):
                        trade_dict['timestamp'] = datetime.fromisoformat(trade_dict['timestamp'].replace('Z', '+00:00'))
                    if 'exit_timestamp' in trade_dict and isinstance(trade_dict['exit_timestamp'], str):
                        trade_dict['exit_timestamp'] = datetime.fromisoformat(trade_dict['exit_timestamp'].replace('Z', '+00:00'))

                    trade = OptionsTrade(**trade_dict)
                    self.completed_trades.append(trade)

                    # Update strategy stats
                    self._update_strategy_stats(trade)

                logger.info(f"Loaded {len(self.completed_trades)} completed trades from {filename}")
        except Exception as e:
            logger.error(f"Error loading completed trades: {e}")

    def _load_active_trades(self) -> None:
        """Load active trades from disk"""
        try:
            filename = "data/options_active_trades.json"
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)

                # Convert back to OptionsTrade objects
                for trade_dict in data:
                    # Convert ISO datetime strings back to datetime objects
                    if 'timestamp' in trade_dict and isinstance(trade_dict['timestamp'], str):
                        trade_dict['timestamp'] = datetime.fromisoformat(trade_dict['timestamp'].replace('Z', '+00:00'))
                    if 'exit_timestamp' in trade_dict and isinstance(trade_dict['exit_timestamp'], str):
                        trade_dict['exit_timestamp'] = datetime.fromisoformat(trade_dict['exit_timestamp'].replace('Z', '+00:00'))

                    trade = OptionsTrade(**trade_dict)
                    self.active_trades[trade.trade_id] = trade

                logger.info(f"Loaded {len(self.active_trades)} active trades from {filename}")
        except Exception as e:
            logger.error(f"Error loading active trades: {e}")

    def _save_active_trades(self) -> None:
        """Save active trades to disk"""
        try:
            filename = "data/options_active_trades.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            data = [asdict(trade) for trade in self.active_trades.values()]

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving active trades: {e}")

    def _save_completed_trades(self) -> None:
        """Save completed trades to disk"""
        try:
            filename = "data/options_completed_trades.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            data = [asdict(trade) for trade in self.completed_trades]

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving completed trades: {e}")

    def _save_optimized_parameters(self) -> None:
        """Save optimized parameters to disk"""
        try:
            filename = "data/options_optimized_parameters.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'parameters': self.optimized_params,
                'strategy_stats': self.strategy_stats
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved optimized parameters to {filename}")

        except Exception as e:
            logger.error(f"Error saving optimized parameters: {e}")


# Singleton instance
_tracker_instance: Optional[OptionsLearningTracker] = None


def get_tracker() -> OptionsLearningTracker:
    """Get singleton tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = OptionsLearningTracker()
    return _tracker_instance


async def initialize_learning() -> bool:
    """Initialize the learning system (call this on startup)"""
    tracker = get_tracker()
    return await tracker.initialize_learning_system()


if __name__ == "__main__":
    async def test_integration():
        """Test the options learning integration"""
        print("Testing Options Learning Integration")
        print("=" * 70)

        # Initialize tracker
        tracker = OptionsLearningTracker()
        success = await tracker.initialize_learning_system()

        print(f"Learning system initialized: {success}")

        # Simulate a trade
        test_trade = OptionsTrade(
            trade_id="test_001",
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            strategy_type="DUAL_OPTIONS",
            entry_price=150.0,
            contracts=2,
            put_strike=145.0,
            call_strike=155.0,
            expiration_date="251024",
            put_delta=-0.35,
            call_delta=0.35,
            market_regime="bull",
            volatility=0.25,
            momentum=0.05,
            confidence_threshold=4.5,
            strike_selection_method="GREEKS_DELTA_TARGETING"
        )

        tracker.log_trade_entry(test_trade)

        # Simulate exit
        tracker.log_trade_exit("test_001", 155.0, 500.0)

        # Get statistics
        stats = tracker.get_strategy_statistics()
        print(f"\nStrategy Statistics:")
        print(f"  Total trades: {stats['total_trades']}")
        print(f"  Win rate: {stats['overall_win_rate']:.1%}")
        print(f"  Profit factor: {stats['overall_profit_factor']:.2f}")

        print("\nTest complete!")

    asyncio.run(test_integration())
