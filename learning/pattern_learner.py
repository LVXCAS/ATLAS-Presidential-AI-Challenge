import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import json
import pickle
from pathlib import Path
from collections import deque, defaultdict
import threading
import uuid
import hashlib
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

from event_bus import TradingEventBus, Event, Priority


class PatternType(Enum):
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    VOLUME_SPIKE = "volume_spike"
    GAP_PATTERN = "gap_pattern"
    SUPPORT_RESISTANCE = "support_resistance"
    TREND_ACCELERATION = "trend_acceleration"
    CONSOLIDATION = "consolidation"


class PatternSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    spread: float
    change_percent: float
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class PatternFeatures:
    """Features extracted from market data for pattern recognition"""
    # Price-based features
    price_velocity: float = 0.0  # Rate of price change
    price_acceleration: float = 0.0  # Rate of velocity change
    volatility: float = 0.0  # Price volatility
    momentum: float = 0.0  # Price momentum
    
    # Volume-based features
    volume_velocity: float = 0.0  # Rate of volume change
    volume_ratio: float = 1.0  # Current vs average volume
    
    # Technical indicators
    rsi: float = 50.0
    bollinger_position: float = 0.5  # Position within Bollinger Bands
    ma_slope_short: float = 0.0  # Short MA slope
    ma_slope_long: float = 0.0  # Long MA slope
    ma_convergence: float = 0.0  # MA convergence/divergence
    
    # Market microstructure
    bid_ask_spread: float = 0.0
    order_flow_imbalance: float = 0.0
    tick_direction: int = 0  # +1, 0, -1
    
    # Pattern-specific
    support_strength: float = 0.0
    resistance_strength: float = 0.0
    trend_strength: float = 0.0
    consolidation_score: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numpy array for ML"""
        return np.array([
            self.price_velocity, self.price_acceleration, self.volatility, self.momentum,
            self.volume_velocity, self.volume_ratio, self.rsi, self.bollinger_position,
            self.ma_slope_short, self.ma_slope_long, self.ma_convergence,
            self.bid_ask_spread, self.order_flow_imbalance, self.tick_direction,
            self.support_strength, self.resistance_strength, self.trend_strength,
            self.consolidation_score
        ])
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PatternInstance:
    """Identified pattern instance"""
    pattern_id: str
    symbol: str
    pattern_type: PatternType
    features: PatternFeatures
    start_time: datetime
    end_time: Optional[datetime]
    start_price: float
    current_price: float
    signal: PatternSignal
    confidence: float
    expected_return: float
    risk_score: float
    timeframe_minutes: int
    
    # Learning data
    actual_return: Optional[float] = None
    outcome_time: Optional[datetime] = None
    success: Optional[bool] = None
    pattern_hash: str = field(default="")
    
    def __post_init__(self):
        if not self.pattern_hash:
            self.pattern_hash = self._calculate_hash()
        if isinstance(self.start_time, str):
            self.start_time = datetime.fromisoformat(self.start_time)
        if isinstance(self.end_time, str) and self.end_time:
            self.end_time = datetime.fromisoformat(self.end_time)
        if isinstance(self.outcome_time, str) and self.outcome_time:
            self.outcome_time = datetime.fromisoformat(self.outcome_time)
    
    def _calculate_hash(self) -> str:
        """Calculate unique hash for pattern matching"""
        feature_str = str(self.features.to_vector())
        hash_input = f"{self.pattern_type.value}_{feature_str}_{self.timeframe_minutes}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'symbol': self.symbol,
            'pattern_type': self.pattern_type.value,
            'features': self.features.to_dict(),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'start_price': self.start_price,
            'current_price': self.current_price,
            'signal': self.signal.value,
            'confidence': self.confidence,
            'expected_return': self.expected_return,
            'risk_score': self.risk_score,
            'timeframe_minutes': self.timeframe_minutes,
            'actual_return': self.actual_return,
            'outcome_time': self.outcome_time.isoformat() if self.outcome_time else None,
            'success': self.success,
            'pattern_hash': self.pattern_hash
        }


class FeatureExtractor:
    """Extract features from market tick data"""
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        self.logger = logging.getLogger(f"{__name__}.FeatureExtractor")
        self.lookback_periods = lookback_periods or {
            'short': 20,
            'medium': 50,
            'long': 200
        }
        
        # Data buffers for feature calculation
        self.tick_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.price_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.volume_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        self.scaler = MinMaxScaler()
    
    async def extract_features(self, tick: MarketTick) -> PatternFeatures:
        """Extract comprehensive features from market tick"""
        symbol = tick.symbol
        
        # Update buffers
        self.tick_buffers[symbol].append(tick)
        self.price_buffers[symbol].append(tick.price)
        self.volume_buffers[symbol].append(tick.volume)
        
        if len(self.price_buffers[symbol]) < 20:
            return PatternFeatures()  # Return default features
        
        prices = np.array(list(self.price_buffers[symbol]))
        volumes = np.array(list(self.volume_buffers[symbol]))
        
        # Price-based features
        price_velocity = self._calculate_velocity(prices, periods=5)
        price_acceleration = self._calculate_acceleration(prices, periods=5)
        volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
        momentum = self._calculate_momentum(prices, periods=10)
        
        # Volume-based features
        volume_velocity = self._calculate_velocity(volumes, periods=5)
        volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1.0
        
        # Technical indicators
        rsi = self._calculate_rsi(prices)
        bollinger_position = self._calculate_bollinger_position(prices)
        ma_slopes = self._calculate_ma_slopes(prices)
        ma_convergence = self._calculate_ma_convergence(prices)
        
        # Market microstructure
        bid_ask_spread = tick.spread / tick.price if tick.price > 0 else 0
        order_flow_imbalance = self._estimate_order_flow_imbalance(tick)
        tick_direction = self._calculate_tick_direction(prices)
        
        # Pattern-specific features
        support_strength = self._calculate_support_strength(prices)
        resistance_strength = self._calculate_resistance_strength(prices)
        trend_strength = self._calculate_trend_strength(prices)
        consolidation_score = self._calculate_consolidation_score(prices)
        
        return PatternFeatures(
            price_velocity=price_velocity,
            price_acceleration=price_acceleration,
            volatility=volatility,
            momentum=momentum,
            volume_velocity=volume_velocity,
            volume_ratio=volume_ratio,
            rsi=rsi,
            bollinger_position=bollinger_position,
            ma_slope_short=ma_slopes['short'],
            ma_slope_long=ma_slopes['long'],
            ma_convergence=ma_convergence,
            bid_ask_spread=bid_ask_spread,
            order_flow_imbalance=order_flow_imbalance,
            tick_direction=tick_direction,
            support_strength=support_strength,
            resistance_strength=resistance_strength,
            trend_strength=trend_strength,
            consolidation_score=consolidation_score
        )
    
    def _calculate_velocity(self, data: np.ndarray, periods: int = 5) -> float:
        """Calculate rate of change (velocity)"""
        if len(data) < periods + 1:
            return 0.0
        return (data[-1] - data[-periods-1]) / periods
    
    def _calculate_acceleration(self, data: np.ndarray, periods: int = 5) -> float:
        """Calculate rate of velocity change (acceleration)"""
        if len(data) < periods * 2:
            return 0.0
        
        velocity_recent = self._calculate_velocity(data[-periods:], periods // 2)
        velocity_past = self._calculate_velocity(data[-periods*2:-periods], periods // 2)
        
        return velocity_recent - velocity_past
    
    def _calculate_momentum(self, prices: np.ndarray, periods: int = 10) -> float:
        """Calculate price momentum"""
        if len(prices) < periods + 1:
            return 0.0
        return (prices[-1] - prices[-periods-1]) / prices[-periods-1]
    
    def _calculate_rsi(self, prices: np.ndarray, periods: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < periods + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-periods:])
        avg_loss = np.mean(losses[-periods:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_position(self, prices: np.ndarray, periods: int = 20, std_dev: float = 2) -> float:
        """Calculate position within Bollinger Bands"""
        if len(prices) < periods:
            return 0.5
        
        ma = np.mean(prices[-periods:])
        std = np.std(prices[-periods:])
        
        upper_band = ma + (std_dev * std)
        lower_band = ma - (std_dev * std)
        
        if upper_band == lower_band:
            return 0.5
        
        return (prices[-1] - lower_band) / (upper_band - lower_band)
    
    def _calculate_ma_slopes(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate moving average slopes"""
        slopes = {}
        
        for name, period in self.lookback_periods.items():
            if len(prices) >= period:
                ma_values = [np.mean(prices[i-period+1:i+1]) for i in range(period-1, len(prices))]
                if len(ma_values) >= 2:
                    slopes[name] = ma_values[-1] - ma_values[-2]
                else:
                    slopes[name] = 0.0
            else:
                slopes[name] = 0.0
        
        return slopes
    
    def _calculate_ma_convergence(self, prices: np.ndarray) -> float:
        """Calculate moving average convergence/divergence"""
        if len(prices) < max(self.lookback_periods.values()):
            return 0.0
        
        short_ma = np.mean(prices[-self.lookback_periods['short']:])
        long_ma = np.mean(prices[-self.lookback_periods['long']:])
        
        return (short_ma - long_ma) / long_ma if long_ma != 0 else 0.0
    
    def _estimate_order_flow_imbalance(self, tick: MarketTick) -> float:
        """Estimate order flow imbalance from bid/ask"""
        if tick.bid == 0 or tick.ask == 0:
            return 0.0
        
        mid_price = (tick.bid + tick.ask) / 2
        return (tick.price - mid_price) / mid_price if mid_price != 0 else 0.0
    
    def _calculate_tick_direction(self, prices: np.ndarray) -> int:
        """Calculate tick direction (+1 up, 0 unchanged, -1 down)"""
        if len(prices) < 2:
            return 0
        
        if prices[-1] > prices[-2]:
            return 1
        elif prices[-1] < prices[-2]:
            return -1
        else:
            return 0
    
    def _calculate_support_strength(self, prices: np.ndarray, lookback: int = 50) -> float:
        """Calculate support level strength"""
        if len(prices) < lookback:
            return 0.0
        
        recent_prices = prices[-lookback:]
        current_price = prices[-1]
        
        # Find potential support levels (local minima)
        support_touches = 0
        for i in range(2, len(recent_prices) - 2):
            if (recent_prices[i] <= recent_prices[i-1] and 
                recent_prices[i] <= recent_prices[i-2] and
                recent_prices[i] <= recent_prices[i+1] and 
                recent_prices[i] <= recent_prices[i+2]):
                
                # Check if current price is near this support
                if abs(current_price - recent_prices[i]) / current_price < 0.02:
                    support_touches += 1
        
        return min(1.0, support_touches / 5.0)
    
    def _calculate_resistance_strength(self, prices: np.ndarray, lookback: int = 50) -> float:
        """Calculate resistance level strength"""
        if len(prices) < lookback:
            return 0.0
        
        recent_prices = prices[-lookback:]
        current_price = prices[-1]
        
        # Find potential resistance levels (local maxima)
        resistance_touches = 0
        for i in range(2, len(recent_prices) - 2):
            if (recent_prices[i] >= recent_prices[i-1] and 
                recent_prices[i] >= recent_prices[i-2] and
                recent_prices[i] >= recent_prices[i+1] and 
                recent_prices[i] >= recent_prices[i+2]):
                
                # Check if current price is near this resistance
                if abs(current_price - recent_prices[i]) / current_price < 0.02:
                    resistance_touches += 1
        
        return min(1.0, resistance_touches / 5.0)
    
    def _calculate_trend_strength(self, prices: np.ndarray, lookback: int = 30) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < lookback:
            return 0.0
        
        recent_prices = prices[-lookback:]
        x = np.arange(len(recent_prices))
        
        slope, _, r_value, _, _ = stats.linregress(x, recent_prices)
        
        # Normalize slope by price level
        normalized_slope = slope / np.mean(recent_prices)
        
        # Weight by R-squared (trend consistency)
        trend_strength = normalized_slope * (r_value ** 2)
        
        return np.tanh(trend_strength * 100)  # Bound between -1 and 1
    
    def _calculate_consolidation_score(self, prices: np.ndarray, lookback: int = 20) -> float:
        """Calculate consolidation/sideways movement score"""
        if len(prices) < lookback:
            return 0.0
        
        recent_prices = prices[-lookback:]
        
        # Calculate price range
        price_range = (np.max(recent_prices) - np.min(recent_prices)) / np.mean(recent_prices)
        
        # Calculate trend strength
        trend_strength = abs(self._calculate_trend_strength(prices, lookback))
        
        # High consolidation = low range and low trend strength
        consolidation_score = (1 - min(1.0, price_range * 50)) * (1 - trend_strength)
        
        return consolidation_score


class PatternDetector:
    """Detect trading patterns from features"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PatternDetector")
        
        # Pattern detection thresholds
        self.thresholds = {
            PatternType.BREAKOUT: {'volatility': 0.02, 'volume_ratio': 1.5, 'momentum': 0.03},
            PatternType.REVERSAL: {'rsi': 30, 'bollinger_position': 0.1, 'support_strength': 0.3},
            PatternType.CONTINUATION: {'trend_strength': 0.3, 'momentum': 0.02, 'volume_ratio': 1.2},
            PatternType.MEAN_REVERSION: {'rsi': 70, 'bollinger_position': 0.9, 'momentum': -0.02},
            PatternType.MOMENTUM: {'momentum': 0.05, 'volume_ratio': 2.0, 'trend_strength': 0.5},
            PatternType.VOLATILITY_EXPANSION: {'volatility': 0.05, 'price_acceleration': 0.01},
            PatternType.VOLATILITY_CONTRACTION: {'consolidation_score': 0.7, 'volatility': 0.01},
            PatternType.VOLUME_SPIKE: {'volume_ratio': 3.0, 'volume_velocity': 1000000},
            PatternType.GAP_PATTERN: {'price_velocity': 0.05, 'volume_ratio': 2.0},
            PatternType.SUPPORT_RESISTANCE: {'support_strength': 0.5, 'resistance_strength': 0.5},
            PatternType.TREND_ACCELERATION: {'price_acceleration': 0.02, 'trend_strength': 0.4},
            PatternType.CONSOLIDATION: {'consolidation_score': 0.8, 'volatility': 0.015}
        }
    
    async def detect_patterns(self, symbol: str, features: PatternFeatures, tick: MarketTick) -> List[PatternInstance]:
        """Detect patterns from features"""
        patterns = []
        
        # Test each pattern type
        for pattern_type, thresholds in self.thresholds.items():
            confidence = await self._test_pattern(pattern_type, features, thresholds)
            
            if confidence > 0.6:  # Minimum confidence threshold
                signal = self._determine_signal(pattern_type, features)
                expected_return = self._estimate_return(pattern_type, features, confidence)
                risk_score = self._calculate_risk_score(pattern_type, features)
                
                pattern = PatternInstance(
                    pattern_id=str(uuid.uuid4()),
                    symbol=symbol,
                    pattern_type=pattern_type,
                    features=features,
                    start_time=tick.timestamp,
                    end_time=None,
                    start_price=tick.price,
                    current_price=tick.price,
                    signal=signal,
                    confidence=confidence,
                    expected_return=expected_return,
                    risk_score=risk_score,
                    timeframe_minutes=self._estimate_timeframe(pattern_type, features)
                )
                
                patterns.append(pattern)
        
        return patterns
    
    async def _test_pattern(self, pattern_type: PatternType, features: PatternFeatures, thresholds: Dict[str, float]) -> float:
        """Test if features match pattern criteria"""
        feature_dict = features.to_dict()
        
        scores = []
        for feature_name, threshold in thresholds.items():
            if feature_name in feature_dict:
                feature_value = feature_dict[feature_name]
                
                # Different scoring methods based on pattern type
                if pattern_type in [PatternType.BREAKOUT, PatternType.MOMENTUM, PatternType.VOLUME_SPIKE]:
                    # For these patterns, higher values are better
                    score = min(1.0, feature_value / threshold) if threshold > 0 else 0.0
                elif pattern_type in [PatternType.REVERSAL, PatternType.MEAN_REVERSION]:
                    # For these patterns, specific ranges are better
                    if feature_name == 'rsi':
                        if threshold == 30:  # Oversold
                            score = max(0.0, (30 - feature_value) / 30) if feature_value <= 30 else 0.0
                        else:  # Overbought
                            score = max(0.0, (feature_value - 70) / 30) if feature_value >= 70 else 0.0
                    else:
                        score = min(1.0, feature_value / threshold) if threshold > 0 else 0.0
                else:
                    # Default scoring
                    score = min(1.0, feature_value / threshold) if threshold > 0 else 0.0
                
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _determine_signal(self, pattern_type: PatternType, features: PatternFeatures) -> PatternSignal:
        """Determine trading signal based on pattern type and features"""
        
        # Buy signals
        if pattern_type in [PatternType.BREAKOUT, PatternType.MOMENTUM, PatternType.REVERSAL]:
            if features.trend_strength > 0.5 and features.momentum > 0.03:
                return PatternSignal.STRONG_BUY
            elif features.momentum > 0.02:
                return PatternSignal.BUY
            elif features.momentum > 0.01:
                return PatternSignal.WEAK_BUY
        
        # Sell signals
        elif pattern_type in [PatternType.MEAN_REVERSION]:
            if features.rsi > 80 and features.momentum < -0.02:
                return PatternSignal.STRONG_SELL
            elif features.rsi > 70:
                return PatternSignal.SELL
            elif features.rsi > 60:
                return PatternSignal.WEAK_SELL
        
        return PatternSignal.NEUTRAL
    
    def _estimate_return(self, pattern_type: PatternType, features: PatternFeatures, confidence: float) -> float:
        """Estimate expected return for pattern"""
        base_returns = {
            PatternType.BREAKOUT: 0.05,
            PatternType.MOMENTUM: 0.04,
            PatternType.REVERSAL: 0.03,
            PatternType.MEAN_REVERSION: 0.025,
            PatternType.CONTINUATION: 0.035,
            PatternType.VOLATILITY_EXPANSION: 0.045,
            PatternType.VOLUME_SPIKE: 0.03,
            PatternType.GAP_PATTERN: 0.04,
            PatternType.TREND_ACCELERATION: 0.06
        }
        
        base_return = base_returns.get(pattern_type, 0.02)
        
        # Adjust based on features and confidence
        volatility_factor = min(2.0, 1 + features.volatility * 10)
        momentum_factor = min(2.0, 1 + abs(features.momentum) * 20)
        confidence_factor = confidence
        
        return base_return * volatility_factor * momentum_factor * confidence_factor
    
    def _calculate_risk_score(self, pattern_type: PatternType, features: PatternFeatures) -> float:
        """Calculate risk score for pattern"""
        base_risk = {
            PatternType.BREAKOUT: 0.4,
            PatternType.MOMENTUM: 0.5,
            PatternType.REVERSAL: 0.3,
            PatternType.MEAN_REVERSION: 0.25,
            PatternType.CONTINUATION: 0.35,
            PatternType.VOLATILITY_EXPANSION: 0.6,
            PatternType.VOLUME_SPIKE: 0.45,
            PatternType.GAP_PATTERN: 0.5,
            PatternType.TREND_ACCELERATION: 0.55
        }
        
        risk = base_risk.get(pattern_type, 0.4)
        
        # Adjust based on volatility and momentum
        volatility_risk = min(1.0, features.volatility * 20)
        momentum_risk = min(1.0, abs(features.momentum) * 10)
        
        return min(1.0, risk + volatility_risk * 0.3 + momentum_risk * 0.2)
    
    def _estimate_timeframe(self, pattern_type: PatternType, features: PatternFeatures) -> int:
        """Estimate pattern timeframe in minutes"""
        base_timeframes = {
            PatternType.BREAKOUT: 60,
            PatternType.MOMENTUM: 30,
            PatternType.REVERSAL: 120,
            PatternType.MEAN_REVERSION: 180,
            PatternType.CONTINUATION: 45,
            PatternType.VOLATILITY_EXPANSION: 30,
            PatternType.VOLUME_SPIKE: 15,
            PatternType.GAP_PATTERN: 60,
            PatternType.TREND_ACCELERATION: 90
        }
        
        base_timeframe = base_timeframes.get(pattern_type, 60)
        
        # Adjust based on volatility (higher volatility = shorter timeframe)
        volatility_factor = max(0.5, 1 - features.volatility * 5)
        
        return int(base_timeframe * volatility_factor)


class PatternKnowledgeBase:
    """Knowledge base for storing and retrieving patterns"""
    
    def __init__(self, db_path: str = "pattern_knowledge.db"):
        self.logger = logging.getLogger(f"{__name__}.PatternKnowledgeBase")
        self.db_path = db_path
        self.lock = threading.RLock()
        
        # In-memory cache
        self.pattern_cache: Dict[str, PatternInstance] = {}
        self.confidence_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Patterns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS patterns (
                        pattern_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        pattern_type TEXT NOT NULL,
                        pattern_hash TEXT NOT NULL,
                        features TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        start_price REAL NOT NULL,
                        current_price REAL NOT NULL,
                        signal TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        expected_return REAL NOT NULL,
                        risk_score REAL NOT NULL,
                        timeframe_minutes INTEGER NOT NULL,
                        actual_return REAL,
                        outcome_time TEXT,
                        success INTEGER,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Pattern outcomes table for learning
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS pattern_outcomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_hash TEXT NOT NULL,
                        pattern_type TEXT NOT NULL,
                        initial_confidence REAL NOT NULL,
                        actual_return REAL NOT NULL,
                        success INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        features TEXT NOT NULL
                    )
                ''')
                
                # Confidence tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS confidence_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_hash TEXT NOT NULL,
                        pattern_type TEXT NOT NULL,
                        old_confidence REAL NOT NULL,
                        new_confidence REAL NOT NULL,
                        update_reason TEXT NOT NULL,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indices
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_hash ON patterns (pattern_hash)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_type ON patterns (pattern_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_time ON patterns (symbol, start_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcomes_hash ON pattern_outcomes (pattern_hash)')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    async def store_pattern(self, pattern: PatternInstance):
        """Store pattern in knowledge base"""
        try:
            with self.lock:
                # Add to cache
                self.pattern_cache[pattern.pattern_id] = pattern
                
                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO patterns (
                            pattern_id, symbol, pattern_type, pattern_hash, features,
                            start_time, end_time, start_price, current_price, signal,
                            confidence, expected_return, risk_score, timeframe_minutes,
                            actual_return, outcome_time, success
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        pattern.pattern_id, pattern.symbol, pattern.pattern_type.value,
                        pattern.pattern_hash, json.dumps(pattern.features.to_dict()),
                        pattern.start_time.isoformat(),
                        pattern.end_time.isoformat() if pattern.end_time else None,
                        pattern.start_price, pattern.current_price, pattern.signal.value,
                        pattern.confidence, pattern.expected_return, pattern.risk_score,
                        pattern.timeframe_minutes, pattern.actual_return,
                        pattern.outcome_time.isoformat() if pattern.outcome_time else None,
                        pattern.success
                    ))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error storing pattern {pattern.pattern_id}: {e}")
    
    async def update_pattern_outcome(self, pattern_id: str, actual_return: float, success: bool):
        """Update pattern outcome for learning"""
        try:
            with self.lock:
                if pattern_id in self.pattern_cache:
                    pattern = self.pattern_cache[pattern_id]
                    pattern.actual_return = actual_return
                    pattern.success = success
                    pattern.outcome_time = datetime.now()
                    
                    # Update database
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE patterns 
                            SET actual_return = ?, success = ?, outcome_time = ?
                            WHERE pattern_id = ?
                        ''', (actual_return, success, pattern.outcome_time.isoformat(), pattern_id))
                        
                        # Store outcome for learning
                        cursor.execute('''
                            INSERT INTO pattern_outcomes (
                                pattern_hash, pattern_type, initial_confidence,
                                actual_return, success, timestamp, features
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            pattern.pattern_hash, pattern.pattern_type.value,
                            pattern.confidence, actual_return, success,
                            datetime.now().isoformat(), json.dumps(pattern.features.to_dict())
                        ))
                        
                        conn.commit()
                    
                    # Update confidence based on outcome
                    await self.update_pattern_confidence(pattern.pattern_hash, actual_return, success)
                    
        except Exception as e:
            self.logger.error(f"Error updating pattern outcome {pattern_id}: {e}")
    
    async def update_pattern_confidence(self, pattern_hash: str, actual_return: float, success: bool):
        """Update pattern confidence based on outcomes"""
        try:
            with self.lock:
                # Get historical outcomes for this pattern
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT initial_confidence, actual_return, success 
                        FROM pattern_outcomes 
                        WHERE pattern_hash = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 50
                    ''', (pattern_hash,))
                    
                    outcomes = cursor.fetchall()
                
                if not outcomes:
                    return
                
                # Calculate new confidence
                success_rate = sum(1 for _, _, success in outcomes if success) / len(outcomes)
                avg_return = np.mean([return_val for _, return_val, _ in outcomes])
                
                # Weight recent outcomes more heavily
                weights = np.exp(np.linspace(-2, 0, len(outcomes)))
                weighted_success = np.average([success for _, _, success in outcomes], weights=weights)
                weighted_return = np.average([return_val for _, return_val, _ in outcomes], weights=weights)
                
                # Calculate new confidence (blend of success rate and return performance)
                return_factor = min(2.0, max(0.5, 1 + weighted_return * 10))
                new_confidence = min(0.95, max(0.05, weighted_success * return_factor))
                
                # Update confidence cache
                self.confidence_cache[pattern_hash].append(new_confidence)
                
                # Log confidence update
                old_confidence = outcomes[0][0] if outcomes else 0.5
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO confidence_history (
                            pattern_hash, pattern_type, old_confidence, new_confidence, update_reason
                        ) VALUES (?, ?, ?, ?, ?)
                    ''', (
                        pattern_hash, 'unknown', old_confidence, new_confidence,
                        f"success_rate:{success_rate:.2f}, avg_return:{avg_return:.4f}"
                    ))
                    conn.commit()
                
                self.logger.info(f"Updated confidence for {pattern_hash}: {old_confidence:.3f} -> {new_confidence:.3f}")
                
        except Exception as e:
            self.logger.error(f"Error updating pattern confidence: {e}")
    
    async def get_pattern_confidence(self, pattern_hash: str) -> float:
        """Get current confidence for a pattern"""
        try:
            if pattern_hash in self.confidence_cache and self.confidence_cache[pattern_hash]:
                return self.confidence_cache[pattern_hash][-1]
            
            # Query from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT new_confidence 
                    FROM confidence_history 
                    WHERE pattern_hash = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (pattern_hash,))
                
                result = cursor.fetchone()
                if result:
                    return result[0]
                
            return 0.5  # Default confidence
            
        except Exception as e:
            self.logger.error(f"Error getting pattern confidence: {e}")
            return 0.5
    
    async def get_similar_patterns(self, features: PatternFeatures, pattern_type: PatternType, limit: int = 10) -> List[PatternInstance]:
        """Find similar patterns based on features"""
        try:
            target_vector = features.to_vector()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM patterns 
                    WHERE pattern_type = ? AND success IS NOT NULL 
                    ORDER BY start_time DESC 
                    LIMIT 100
                ''', (pattern_type.value,))
                
                patterns = []
                for row in cursor.fetchall():
                    try:
                        # Reconstruct pattern
                        feature_dict = json.loads(row[4])  # features column
                        pattern_features = PatternFeatures(**feature_dict)
                        pattern_vector = pattern_features.to_vector()
                        
                        # Calculate similarity
                        similarity = self._calculate_similarity(target_vector, pattern_vector)
                        
                        if similarity > 0.8:  # High similarity threshold
                            pattern = PatternInstance(
                                pattern_id=row[0],
                                symbol=row[1],
                                pattern_type=PatternType(row[2]),
                                features=pattern_features,
                                start_time=datetime.fromisoformat(row[5]),
                                end_time=datetime.fromisoformat(row[6]) if row[6] else None,
                                start_price=row[7],
                                current_price=row[8],
                                signal=PatternSignal(row[9]),
                                confidence=row[10],
                                expected_return=row[11],
                                risk_score=row[12],
                                timeframe_minutes=row[13],
                                actual_return=row[14],
                                outcome_time=datetime.fromisoformat(row[15]) if row[15] else None,
                                success=bool(row[16]) if row[16] is not None else None,
                                pattern_hash=row[3]
                            )
                            patterns.append((pattern, similarity))
                    
                    except Exception as e:
                        self.logger.warning(f"Error processing pattern row: {e}")
                        continue
                
                # Sort by similarity and return top matches
                patterns.sort(key=lambda x: x[1], reverse=True)
                return [pattern for pattern, _ in patterns[:limit]]
                
        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {e}")
            return []
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors"""
        try:
            # Handle zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception:
            return 0.0
    
    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total patterns
                cursor.execute('SELECT COUNT(*) FROM patterns')
                total_patterns = cursor.fetchone()[0]
                
                # Patterns by type
                cursor.execute('''
                    SELECT pattern_type, COUNT(*) 
                    FROM patterns 
                    GROUP BY pattern_type
                ''')
                patterns_by_type = dict(cursor.fetchall())
                
                # Success rates by type
                cursor.execute('''
                    SELECT pattern_type, 
                           AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                           AVG(actual_return) as avg_return
                    FROM patterns 
                    WHERE success IS NOT NULL 
                    GROUP BY pattern_type
                ''')
                success_stats = cursor.fetchall()
                
                return {
                    'total_patterns': total_patterns,
                    'patterns_by_type': patterns_by_type,
                    'success_statistics': [
                        {
                            'pattern_type': row[0],
                            'success_rate': row[1],
                            'avg_return': row[2]
                        }
                        for row in success_stats
                    ],
                    'cache_size': len(self.pattern_cache)
                }
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}


class PatternLearner:
    """Main pattern learning system"""
    
    def __init__(self, event_bus: TradingEventBus, learning_rate: float = 0.01):
        self.logger = logging.getLogger(__name__)
        self.event_bus = event_bus
        self.learning_rate = learning_rate
        
        # Components
        self.feature_extractor = FeatureExtractor()
        self.pattern_detector = PatternDetector()
        self.knowledge_base = PatternKnowledgeBase()
        
        # Active patterns being tracked
        self.active_patterns: Dict[str, PatternInstance] = {}
        
        # Learning metrics
        self.learning_stats = {
            'ticks_processed': 0,
            'patterns_detected': 0,
            'patterns_learned': 0,
            'confidence_updates': 0
        }
        
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event bus handlers"""
        if self.event_bus:
            self.event_bus.subscribe("market_data", self._handle_market_tick)
            self.event_bus.subscribe("position_closed", self._handle_position_closed)
            self.event_bus.subscribe("autonomous_trade_executed", self._handle_trade_executed)
    
    async def _handle_market_tick(self, event: Event):
        """Handle incoming market tick data"""
        try:
            data = event.data
            
            # Convert to MarketTick
            tick = MarketTick(
                symbol=data.get('symbol'),
                timestamp=datetime.now(),
                price=data.get('price', 0),
                volume=data.get('volume', 0),
                bid=data.get('bid', data.get('price', 0)),
                ask=data.get('ask', data.get('price', 0)),
                spread=data.get('spread', 0),
                change_percent=data.get('change_percent', 0)
            )
            
            await self.learn_from_tick(tick)
            
        except Exception as e:
            self.logger.error(f"Error handling market tick: {e}")
    
    async def _handle_position_closed(self, event: Event):
        """Handle position closure for learning"""
        try:
            data = event.data
            symbol = data.get('symbol')
            actual_return = data.get('pnl', 0) / data.get('cost_basis', 1)
            
            # Update outcomes for active patterns
            await self._update_pattern_outcomes(symbol, actual_return)
            
        except Exception as e:
            self.logger.error(f"Error handling position closed: {e}")
    
    async def _handle_trade_executed(self, event: Event):
        """Handle trade execution for learning"""
        try:
            data = event.data
            decision_id = data.get('decision_id')
            
            # Link executed trades to patterns if possible
            # This would require more sophisticated tracking
            
        except Exception as e:
            self.logger.error(f"Error handling trade executed: {e}")
    
    async def learn_from_tick(self, tick: MarketTick):
        """Main learning function - process each market tick"""
        try:
            self.learning_stats['ticks_processed'] += 1
            
            # Extract features
            features = await self.feature_extractor.extract_features(tick)
            
            # Detect patterns
            patterns = await self.pattern_detector.detect_patterns(tick.symbol, features, tick)
            
            for pattern in patterns:
                # Adjust confidence based on historical data
                historical_confidence = await self.knowledge_base.get_pattern_confidence(pattern.pattern_hash)
                pattern.confidence = (pattern.confidence + historical_confidence) / 2
                
                # Store pattern
                await self.knowledge_base.store_pattern(pattern)
                
                # Add to active tracking
                self.active_patterns[pattern.pattern_id] = pattern
                
                # Publish pattern discovery
                if self.event_bus and pattern.confidence > 0.75:
                    await self.event_bus.publish(
                        "pattern_discovered",
                        {
                            'pattern_id': pattern.pattern_id,
                            'symbol': pattern.symbol,
                            'pattern_type': pattern.pattern_type.value,
                            'confidence': pattern.confidence,
                            'expected_return': pattern.expected_return,
                            'signal': pattern.signal.value,
                            'features': pattern.features.to_dict()
                        },
                        priority=Priority.HIGH if pattern.confidence > 0.85 else Priority.NORMAL
                    )
                
                self.learning_stats['patterns_detected'] += 1
            
            # Clean up old active patterns
            await self._cleanup_expired_patterns()
            
        except Exception as e:
            self.logger.error(f"Error learning from tick: {e}")
    
    async def _update_pattern_outcomes(self, symbol: str, actual_return: float):
        """Update outcomes for patterns associated with a symbol"""
        try:
            patterns_to_update = [
                p for p in self.active_patterns.values()
                if p.symbol == symbol and p.success is None
            ]
            
            for pattern in patterns_to_update:
                success = actual_return > pattern.expected_return * 0.5  # 50% of expected
                
                await self.knowledge_base.update_pattern_outcome(
                    pattern.pattern_id, actual_return, success
                )
                
                pattern.actual_return = actual_return
                pattern.success = success
                pattern.outcome_time = datetime.now()
                
                self.learning_stats['patterns_learned'] += 1
                
        except Exception as e:
            self.logger.error(f"Error updating pattern outcomes: {e}")
    
    async def _cleanup_expired_patterns(self):
        """Remove expired patterns from active tracking"""
        try:
            now = datetime.now()
            expired_patterns = []
            
            for pattern_id, pattern in self.active_patterns.items():
                time_elapsed = (now - pattern.start_time).total_seconds() / 60  # minutes
                
                if time_elapsed > pattern.timeframe_minutes * 2:  # 2x timeframe
                    expired_patterns.append(pattern_id)
            
            for pattern_id in expired_patterns:
                del self.active_patterns[pattern_id]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up expired patterns: {e}")
    
    async def get_pattern_insights(self, symbol: str) -> Dict[str, Any]:
        """Get pattern-based insights for a symbol"""
        try:
            # Get recent patterns for symbol
            recent_patterns = [
                p for p in self.active_patterns.values()
                if p.symbol == symbol and (datetime.now() - p.start_time).seconds < 3600
            ]
            
            if not recent_patterns:
                return {'insights': 'No recent patterns detected'}
            
            # Aggregate insights
            avg_confidence = np.mean([p.confidence for p in recent_patterns])
            avg_expected_return = np.mean([p.expected_return for p in recent_patterns])
            dominant_signal = max(set([p.signal for p in recent_patterns]), 
                                key=[p.signal for p in recent_patterns].count)
            
            pattern_types = [p.pattern_type.value for p in recent_patterns]
            pattern_counts = {pt: pattern_types.count(pt) for pt in set(pattern_types)}
            
            return {
                'symbol': symbol,
                'recent_patterns_count': len(recent_patterns),
                'average_confidence': avg_confidence,
                'average_expected_return': avg_expected_return,
                'dominant_signal': dominant_signal.value,
                'pattern_distribution': pattern_counts,
                'recommendation': self._generate_recommendation(recent_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pattern insights: {e}")
            return {'error': str(e)}
    
    def _generate_recommendation(self, patterns: List[PatternInstance]) -> str:
        """Generate trading recommendation based on patterns"""
        if not patterns:
            return "No recommendation available"
        
        buy_signals = sum(1 for p in patterns if p.signal in [PatternSignal.BUY, PatternSignal.STRONG_BUY])
        sell_signals = sum(1 for p in patterns if p.signal in [PatternSignal.SELL, PatternSignal.STRONG_SELL])
        
        avg_confidence = np.mean([p.confidence for p in patterns])
        
        if buy_signals > sell_signals and avg_confidence > 0.7:
            return f"BULLISH - {buy_signals} buy signals with {avg_confidence:.1%} avg confidence"
        elif sell_signals > buy_signals and avg_confidence > 0.7:
            return f"BEARISH - {sell_signals} sell signals with {avg_confidence:.1%} avg confidence"
        else:
            return f"NEUTRAL - Mixed signals with {avg_confidence:.1%} avg confidence"
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        try:
            kb_stats = await self.knowledge_base.get_pattern_statistics()
            
            return {
                **self.learning_stats,
                'active_patterns': len(self.active_patterns),
                'knowledge_base': kb_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting learning statistics: {e}")
            return self.learning_stats


# Integration with autonomous_brain.py
class PatternLearnerBridgeInterface:
    """Interface for integrating PatternLearner with AutonomousTradingBrain"""
    
    def __init__(self, pattern_learner: PatternLearner):
        self.pattern_learner = pattern_learner
        self.logger = logging.getLogger(f"{__name__}.PatternLearnerBridge")
    
    async def enhance_trading_decision(self, symbol: str, base_confidence: float) -> float:
        """Enhance trading decision confidence using pattern insights"""
        try:
            insights = await self.pattern_learner.get_pattern_insights(symbol)
            pattern_confidence = insights.get('average_confidence', 0.5)
            
            # Blend pattern confidence with base confidence
            enhanced_confidence = (base_confidence * 0.7) + (pattern_confidence * 0.3)
            
            return min(0.95, max(0.05, enhanced_confidence))
            
        except Exception as e:
            self.logger.error(f"Error enhancing trading decision: {e}")
            return base_confidence
    
    async def get_pattern_based_signals(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get pattern-based trading signals for symbols"""
        signals = {}
        
        for symbol in symbols:
            try:
                insights = await self.pattern_learner.get_pattern_insights(symbol)
                signals[symbol] = insights
            except Exception as e:
                self.logger.error(f"Error getting signals for {symbol}: {e}")
                signals[symbol] = {'error': str(e)}
        
        return signals


# Example usage
async def main():
    """Example usage of the pattern learning system"""
    logging.basicConfig(level=logging.INFO)
    
    # Create event bus
    event_bus = TradingEventBus()
    await event_bus.start()
    
    # Create pattern learner
    pattern_learner = PatternLearner(event_bus)
    
    try:
        # Simulate market ticks
        for i in range(100):
            tick = MarketTick(
                symbol="AAPL",
                timestamp=datetime.now(),
                price=150.0 + np.random.normal(0, 2),
                volume=int(1000000 + np.random.normal(0, 100000)),
                bid=149.95,
                ask=150.05,
                spread=0.10,
                change_percent=np.random.normal(0, 1)
            )
            
            await pattern_learner.learn_from_tick(tick)
            await asyncio.sleep(0.1)
        
        # Get insights
        insights = await pattern_learner.get_pattern_insights("AAPL")
        stats = await pattern_learner.get_learning_statistics()
        
        print("Pattern Insights:", insights)
        print("Learning Stats:", stats)
        
    finally:
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(main())