#!/usr/bin/env python3
"""
Learning Engine for Options Trading Bot
Tracks performance, learns from mistakes, and adapts strategies
"""
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import statistics
import numpy as np
from pathlib import Path

@dataclass
class TradeRecord:
    """Individual trade record for learning"""
    trade_id: str
    symbol: str
    strategy: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_confidence: float
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: Optional[float]
    max_profit: float
    max_loss: float
    market_conditions: Dict[str, Any]
    exit_reason: str = "open"
    days_held: int = 0
    win: Optional[bool] = None
    
    def to_dict(self):
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert datetime objects to strings
        data['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        data['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        data['market_conditions'] = json.dumps(self.market_conditions)
        return data
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        # Convert string datetimes back to datetime objects
        if data['entry_time']:
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if data['exit_time']:
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        if isinstance(data['market_conditions'], str):
            data['market_conditions'] = json.loads(data['market_conditions'])
        return cls(**data)

@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    strategy: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_hold_time: float = 0.0
    confidence_accuracy: Dict[str, float] = None  # confidence_range -> actual_win_rate
    
    def __post_init__(self):
        if self.confidence_accuracy is None:
            self.confidence_accuracy = {}

class LearningEngine:
    """Machine learning system for trading bot"""
    
    def __init__(self, db_path: str = "trading_performance.db"):
        self.db_path = db_path
        self.trade_history: List[TradeRecord] = []
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.symbol_performance: Dict[str, Dict] = {}
        self.market_regime_performance: Dict[str, Dict] = {}
        self.confidence_calibration: Dict[str, float] = {}
        
        # Learning parameters
        self.min_trades_for_learning = 10  # Minimum trades before making adjustments
        self.confidence_buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        self.lookback_days = 30  # Days to look back for recent performance
        
        self.initialize_database()
        self.load_historical_data()
    
    def initialize_database(self):
        """Initialize SQLite database for trade storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                strategy TEXT,
                entry_time TEXT,
                exit_time TEXT,
                entry_confidence REAL,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                pnl REAL,
                max_profit REAL,
                max_loss REAL,
                market_conditions TEXT,
                exit_reason TEXT,
                days_held INTEGER,
                win INTEGER
            )
        ''')
        
        # Create performance summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy TEXT PRIMARY KEY,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                total_pnl REAL,
                avg_win REAL,
                avg_loss REAL,
                win_rate REAL,
                profit_factor REAL,
                avg_hold_time REAL,
                confidence_accuracy TEXT,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_trade_entry(self, trade_id: str, symbol: str, strategy: str, 
                          confidence: float, entry_price: float, quantity: int,
                          max_profit: float, max_loss: float, market_conditions: Dict):
        """Record a new trade entry"""
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            strategy=strategy,
            entry_time=datetime.now(),
            exit_time=None,
            entry_confidence=confidence,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            pnl=None,
            max_profit=max_profit,
            max_loss=max_loss,
            market_conditions=market_conditions
        )
        
        self.trade_history.append(trade)
        self._save_trade_to_db(trade)
        
        return trade
    
    def record_trade_exit(self, trade_id: str, exit_price: float, exit_reason: str):
        """Record trade exit and calculate performance"""
        # Find the trade
        trade = None
        for t in self.trade_history:
            if t.trade_id == trade_id:
                trade = t
                break
        
        if not trade:
            return None
        
        # Update trade record
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.days_held = (trade.exit_time - trade.entry_time).days
        
        # Calculate P&L
        if trade.strategy in ['LONG_CALL', 'LONG_PUT']:
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity * 100
        else:
            # Handle spreads if needed
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity * 100
        
        trade.win = trade.pnl > 0
        
        # Update database
        self._update_trade_in_db(trade)
        
        # Update performance metrics
        self._update_strategy_performance(trade)
        
        return trade
    
    def get_strategy_multiplier(self, strategy: str) -> float:
        """Get performance-based multiplier for strategy confidence"""
        if strategy not in self.strategy_performance:
            return 1.0  # No adjustment if no history
        
        perf = self.strategy_performance[strategy]
        if perf.total_trades < self.min_trades_for_learning:
            return 1.0  # Not enough data yet
        
        # Adjust based on win rate and profit factor
        base_multiplier = 1.0
        
        # Win rate adjustment (-20% to +20%)
        if perf.win_rate > 0.65:  # Good win rate
            base_multiplier += 0.15
        elif perf.win_rate < 0.45:  # Poor win rate
            base_multiplier -= 0.20
        
        # Profit factor adjustment
        if perf.profit_factor > 1.5:  # Profitable
            base_multiplier += 0.10
        elif perf.profit_factor < 0.8:  # Losing money
            base_multiplier -= 0.15
        
        # Cap the multiplier
        return max(0.5, min(1.5, base_multiplier))
    
    def get_symbol_confidence_adjustment(self, symbol: str, strategy: str) -> float:
        """Get symbol-specific confidence adjustment"""
        key = f"{symbol}_{strategy}"
        if key not in self.symbol_performance:
            return 0.0  # No adjustment
        
        perf = self.symbol_performance[key]
        if perf['total_trades'] < 5:  # Need some history
            return 0.0
        
        # Adjust based on recent performance
        if perf['win_rate'] > 0.7:
            return 0.05  # Boost confidence
        elif perf['win_rate'] < 0.3:
            return -0.10  # Reduce confidence
        
        return 0.0
    
    def should_avoid_strategy(self, strategy: str) -> bool:
        """Check if strategy should be avoided due to poor performance"""
        if strategy not in self.strategy_performance:
            return False
        
        perf = self.strategy_performance[strategy]
        if perf.total_trades < self.min_trades_for_learning:
            return False
        
        # Avoid if consistently losing money
        return (perf.win_rate < 0.35 and perf.profit_factor < 0.7 and 
                perf.total_trades >= 15)
    
    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on recent performance"""
        if len(self.trade_history) < 10:
            return 1.0
        
        # Look at last 20 trades
        recent_trades = self.trade_history[-20:]
        completed_trades = [t for t in recent_trades if t.pnl is not None]
        
        if len(completed_trades) < 5:
            return 1.0
        
        # Calculate recent performance
        recent_wins = sum(1 for t in completed_trades if t.win)
        recent_win_rate = recent_wins / len(completed_trades)
        recent_pnl = sum(t.pnl for t in completed_trades)
        
        # Adjust position sizing
        if recent_win_rate > 0.65 and recent_pnl > 0:
            return 1.2  # Increase size when winning
        elif recent_win_rate < 0.35 or recent_pnl < -500:
            return 0.7  # Reduce size when losing
        
        return 1.0
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get current learning insights and statistics"""
        insights = {
            'total_trades': len([t for t in self.trade_history if t.pnl is not None]),
            'strategy_performance': {},
            'confidence_calibration': {},
            'recent_performance': {},
            'recommendations': []
        }
        
        # Strategy performance summary
        for strategy, perf in self.strategy_performance.items():
            if perf.total_trades >= 5:
                insights['strategy_performance'][strategy] = {
                    'trades': perf.total_trades,
                    'win_rate': perf.win_rate,
                    'avg_pnl': perf.total_pnl / perf.total_trades if perf.total_trades > 0 else 0,
                    'profit_factor': perf.profit_factor,
                    'multiplier': self.get_strategy_multiplier(strategy)
                }
        
        # Confidence calibration
        for bucket_range, accuracy in self.confidence_calibration.items():
            insights['confidence_calibration'][bucket_range] = accuracy
        
        # Recent performance (last 30 days)
        recent_trades = [t for t in self.trade_history 
                        if t.exit_time and (datetime.now() - t.exit_time).days <= 30]
        
        if recent_trades:
            insights['recent_performance'] = {
                'trades': len(recent_trades),
                'win_rate': sum(1 for t in recent_trades if t.win) / len(recent_trades),
                'total_pnl': sum(t.pnl for t in recent_trades if t.pnl),
                'position_size_multiplier': self.get_position_size_multiplier()
            }
        
        # Generate recommendations
        insights['recommendations'] = self._generate_recommendations()
        
        return insights
    
    def calibrate_confidence(self, predicted_confidence: float, strategy: str, 
                           symbol: str, market_conditions: Dict) -> float:
        """Calibrate confidence based on historical accuracy"""
        base_confidence = predicted_confidence
        
        # Apply strategy performance adjustment
        strategy_multiplier = self.get_strategy_multiplier(strategy)
        adjusted_confidence = base_confidence * strategy_multiplier
        
        # Apply symbol-specific adjustment
        symbol_adjustment = self.get_symbol_confidence_adjustment(symbol, strategy)
        adjusted_confidence += symbol_adjustment
        
        # Apply market regime adjustment if we have learned patterns
        regime = market_conditions.get('market_regime', 'NEUTRAL')
        if regime in self.market_regime_performance:
            regime_perf = self.market_regime_performance[regime]
            if regime_perf.get('total_trades', 0) >= 10:
                regime_multiplier = regime_perf.get('win_rate', 0.5) / 0.5
                adjusted_confidence *= max(0.8, min(1.2, regime_multiplier))
        
        # Ensure confidence stays within bounds
        return max(0.3, min(0.95, adjusted_confidence))
    
    def _save_trade_to_db(self, trade: TradeRecord):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        trade_dict = trade.to_dict()
        placeholders = ', '.join(['?' for _ in trade_dict])
        columns = ', '.join(trade_dict.keys())
        
        cursor.execute(f'INSERT OR REPLACE INTO trades ({columns}) VALUES ({placeholders})',
                      list(trade_dict.values()))
        
        conn.commit()
        conn.close()
    
    def _update_trade_in_db(self, trade: TradeRecord):
        """Update completed trade in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE trades SET 
            exit_time = ?, exit_price = ?, pnl = ?, exit_reason = ?, 
            days_held = ?, win = ?
            WHERE trade_id = ?
        ''', (
            trade.exit_time.isoformat() if trade.exit_time else None,
            trade.exit_price, trade.pnl, trade.exit_reason,
            trade.days_held, 1 if trade.win else 0, trade.trade_id
        ))
        
        conn.commit()
        conn.close()
    
    def _update_strategy_performance(self, trade: TradeRecord):
        """Update strategy performance metrics"""
        strategy = trade.strategy
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = StrategyPerformance(strategy)
        
        perf = self.strategy_performance[strategy]
        perf.total_trades += 1
        
        if trade.win:
            perf.wins += 1
        else:
            perf.losses += 1
        
        if trade.pnl:
            perf.total_pnl += trade.pnl
        
        # Calculate derived metrics
        if perf.total_trades > 0:
            perf.win_rate = perf.wins / perf.total_trades
        
        # Update symbol performance
        symbol_key = f"{trade.symbol}_{strategy}"
        if symbol_key not in self.symbol_performance:
            self.symbol_performance[symbol_key] = {
                'total_trades': 0, 'wins': 0, 'total_pnl': 0
            }
        
        sym_perf = self.symbol_performance[symbol_key]
        sym_perf['total_trades'] += 1
        if trade.win:
            sym_perf['wins'] += 1
        if trade.pnl:
            sym_perf['total_pnl'] += trade.pnl
        
        if sym_perf['total_trades'] > 0:
            sym_perf['win_rate'] = sym_perf['wins'] / sym_perf['total_trades']
    
    def load_historical_data(self):
        """Load historical data from database"""
        if not Path(self.db_path).exists():
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load all completed trades
        cursor.execute('SELECT * FROM trades WHERE pnl IS NOT NULL')
        rows = cursor.fetchall()
        
        # Get column names
        cursor.execute('PRAGMA table_info(trades)')
        columns = [col[1] for col in cursor.fetchall()]
        
        # Convert to TradeRecord objects
        for row in rows:
            trade_dict = dict(zip(columns, row))
            try:
                trade = TradeRecord.from_dict(trade_dict)
                self.trade_history.append(trade)
                self._update_strategy_performance(trade)
            except Exception as e:
                continue  # Skip invalid records
        
        conn.close()
    
    def _generate_recommendations(self) -> List[str]:
        """Generate trading recommendations based on learned patterns"""
        recommendations = []
        
        # Strategy recommendations
        for strategy, perf in self.strategy_performance.items():
            if perf.total_trades >= 10:
                if perf.win_rate > 0.7:
                    recommendations.append(f"✅ {strategy} performing well ({perf.win_rate:.1%} win rate)")
                elif perf.win_rate < 0.4:
                    recommendations.append(f"⚠️ Consider reducing {strategy} usage ({perf.win_rate:.1%} win rate)")
        
        # Position sizing recommendation
        size_mult = self.get_position_size_multiplier()
        if size_mult > 1.1:
            recommendations.append("📈 Recent performance good - consider larger positions")
        elif size_mult < 0.9:
            recommendations.append("📉 Recent losses - using smaller position sizes")
        
        return recommendations

# Global learning engine instance
learning_engine = LearningEngine()