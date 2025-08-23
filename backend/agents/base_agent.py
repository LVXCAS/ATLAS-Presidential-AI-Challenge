"""
Base Agent Framework for Bloomberg Terminal Trading System
High-performance foundation for all trading agents with institutional-grade features.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import numpy as np
import pandas as pd

from core.redis_manager import get_redis_manager
from core.database import DatabaseService
from services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class AgentStatus(Enum):
    """Agent operational status."""
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    ERROR = "ERROR"
    PAUSED = "PAUSED"
    LEARNING = "LEARNING"


@dataclass
class TradingSignal:
    """Standardized trading signal structure."""
    id: str
    agent_name: str
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    strength: float    # 0.0 to 1.0
    reasoning: Dict[str, Any]
    features_used: Dict[str, float]
    prediction_horizon: int  # minutes
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_score: Optional[float] = None
    expected_return: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['signal_type'] = self.signal_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AgentMetrics:
    """Agent performance and operational metrics."""
    total_signals: int = 0
    accurate_signals: int = 0
    total_trades: int = 0
    profitable_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_holding_period: float = 0.0
    last_signal_time: Optional[datetime] = None
    execution_time_ms: float = 0.0
    error_count: int = 0
    uptime_seconds: float = 0.0


class BaseAgent(ABC):
    """
    Base class for all trading agents with institutional-grade capabilities.
    
    Features:
    - High-performance async processing
    - Real-time market data integration
    - Performance tracking and metrics
    - Risk management integration
    - Feature caching and optimization
    - Error handling and recovery
    - Signal validation and filtering
    """
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        config: Dict[str, Any] = None
    ):
        self.name = name
        self.symbols = [s.upper() for s in symbols]
        self.config = config or {}
        self.status = AgentStatus.INACTIVE
        self.metrics = AgentMetrics()
        
        # Core services
        self.redis_manager = get_redis_manager()
        self.market_data_service: Optional[MarketDataService] = None
        
        # Performance tracking
        self.start_time = time.time()
        self.last_execution_time = 0.0
        self.signal_history: List[TradingSignal] = []
        self.feature_cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, float] = {}
        
        # Callbacks and subscribers
        self.signal_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Async tasks
        self._tasks: List[asyncio.Task] = []
        self.is_running = False
        
        logger.info(f"Initialized agent: {self.name} for symbols: {self.symbols}")
    
    async def start(self) -> None:
        """Start the agent with full initialization."""
        if self.is_running:
            logger.warning(f"Agent {self.name} is already running")
            return
        
        try:
            logger.info(f"Starting agent: {self.name}")
            
            # Initialize Redis connection
            await self.redis_manager.initialize()
            
            # Initialize market data service
            self.market_data_service = MarketDataService()
            
            # Perform agent-specific initialization
            await self.initialize()
            
            # Start background tasks
            self._tasks = [
                asyncio.create_task(self._signal_generation_loop()),
                asyncio.create_task(self._performance_monitoring_loop()),
                asyncio.create_task(self._cache_cleanup_loop()),
                asyncio.create_task(self._health_check_loop())
            ]
            
            self.is_running = True
            self.status = AgentStatus.ACTIVE
            self.start_time = time.time()
            
            # Subscribe to market data for our symbols
            if self.market_data_service:
                await self.market_data_service.subscribe_to_symbols(self.symbols)
            
            logger.info(f"Agent {self.name} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start agent {self.name}: {e}")
            self.status = AgentStatus.ERROR
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        logger.info(f"Stopping agent: {self.name}")
        
        self.is_running = False
        self.status = AgentStatus.INACTIVE
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Cleanup resources
        await self.cleanup()
        
        logger.info(f"Agent {self.name} stopped")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Agent-specific initialization logic."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Agent-specific cleanup logic."""
        pass
    
    @abstractmethod
    async def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate trading signal for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None if no signal
        """
        pass
    
    @abstractmethod
    async def calculate_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate features used by this agent.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of feature names to values
        """
        pass
    
    async def _signal_generation_loop(self) -> None:
        """Main signal generation loop."""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Generate signals for all symbols
                tasks = [self._process_symbol(symbol) for symbol in self.symbols]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update execution time metrics
                execution_time = (time.time() - start_time) * 1000  # milliseconds
                self.metrics.execution_time_ms = execution_time
                self.last_execution_time = time.time()
                
                # Sleep based on agent configuration
                sleep_interval = self.config.get('signal_interval', 60)  # Default 1 minute
                await asyncio.sleep(sleep_interval)
                
            except Exception as e:
                logger.error(f"Error in signal generation loop for {self.name}: {e}")
                self.metrics.error_count += 1
                await self._handle_error(e)
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _process_symbol(self, symbol: str) -> None:
        """Process a single symbol for signal generation."""
        try:
            # Generate signal
            signal = await self.generate_signal(symbol)
            
            if signal:
                # Validate signal
                if await self._validate_signal(signal):
                    # Store signal
                    await self._store_signal(signal)
                    
                    # Notify callbacks
                    await self._notify_signal_callbacks(signal)
                    
                    # Update metrics
                    self.metrics.total_signals += 1
                    self.metrics.last_signal_time = signal.timestamp
                    
                    # Add to history
                    self.signal_history.append(signal)
                    # Keep only last 1000 signals
                    if len(self.signal_history) > 1000:
                        self.signal_history = self.signal_history[-1000:]
                    
                    logger.info(f"Generated signal: {signal.agent_name} {signal.symbol} "
                              f"{signal.signal_type.value} (confidence: {signal.confidence:.2f})")
                else:
                    logger.debug(f"Signal validation failed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol} in {self.name}: {e}")
            self.metrics.error_count += 1
    
    async def _validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate signal quality and constraints.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid
        """
        # Basic validation
        if not (0.0 <= signal.confidence <= 1.0):
            logger.warning(f"Invalid confidence: {signal.confidence}")
            return False
        
        if not (0.0 <= signal.strength <= 1.0):
            logger.warning(f"Invalid strength: {signal.strength}")
            return False
        
        # Minimum confidence threshold
        min_confidence = self.config.get('min_confidence', 0.6)
        if signal.confidence < min_confidence:
            logger.debug(f"Signal below minimum confidence: {signal.confidence} < {min_confidence}")
            return False
        
        # Rate limiting - prevent too many signals for same symbol
        cache_key = f"last_signal:{self.name}:{signal.symbol}"
        redis_client = await self.redis_manager.get_client()
        last_signal_time = await redis_client.get(cache_key)
        
        if last_signal_time:
            min_interval = self.config.get('min_signal_interval', 300)  # 5 minutes
            if time.time() - float(last_signal_time) < min_interval:
                logger.debug(f"Signal too frequent for {signal.symbol}")
                return False
        
        # Store timestamp for rate limiting
        await redis_client.setex(cache_key, 3600, str(time.time()))  # 1 hour expiry
        
        return True
    
    async def _store_signal(self, signal: TradingSignal) -> None:
        """Store signal in database and cache."""
        try:
            # Store in database
            await DatabaseService.insert_agent_signal({
                "agent_name": signal.agent_name,
                "symbol": signal.symbol,
                "timestamp": signal.timestamp,
                "signal_type": signal.signal_type.value,
                "confidence": signal.confidence,
                "strength": signal.strength,
                "reasoning": signal.reasoning,
                "features_used": signal.features_used,
                "prediction_horizon": signal.prediction_horizon,
                "target_price": signal.target_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit
            })
            
            # Cache in Redis for fast access
            await self.redis_manager.cache_agent_signal(
                signal.agent_name,
                signal.symbol,
                signal.to_dict()
            )
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
    
    async def _notify_signal_callbacks(self, signal: TradingSignal) -> None:
        """Notify all registered signal callbacks."""
        for callback in self.signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")
    
    async def _performance_monitoring_loop(self) -> None:
        """Monitor and update agent performance metrics."""
        while self.is_running:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(300)  # Update every 5 minutes
            except Exception as e:
                logger.error(f"Error in performance monitoring for {self.name}: {e}")
                await asyncio.sleep(60)
    
    async def _update_performance_metrics(self) -> None:
        """Calculate and update performance metrics."""
        try:
            # Calculate uptime
            self.metrics.uptime_seconds = time.time() - self.start_time
            
            # Calculate accuracy and win rate from recent signals
            if self.signal_history:
                # This would be enhanced with actual trade outcomes
                # For now, we'll use a simplified calculation
                recent_signals = self.signal_history[-100:]  # Last 100 signals
                
                # Mock accuracy calculation (would be based on actual outcomes)
                self.metrics.accuracy = min(0.95, 0.5 + (len(recent_signals) * 0.001))
                
                # Update signal count
                self.metrics.total_signals = len(self.signal_history)
            
            # Store metrics in Redis for monitoring
            redis_client = await self.redis_manager.get_client()
            await redis_client.hset(
                f"agent_metrics:{self.name}",
                mapping={
                    "total_signals": str(self.metrics.total_signals),
                    "accuracy": str(self.metrics.accuracy),
                    "execution_time_ms": str(self.metrics.execution_time_ms),
                    "error_count": str(self.metrics.error_count),
                    "uptime_seconds": str(self.metrics.uptime_seconds),
                    "status": self.status.value,
                    "last_update": str(time.time())
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _cache_cleanup_loop(self) -> None:
        """Clean up expired cache entries."""
        while self.is_running:
            try:
                current_time = time.time()
                expired_keys = [
                    key for key, expiry in self.cache_expiry.items()
                    if expiry < current_time
                ]
                
                for key in expired_keys:
                    self.feature_cache.pop(key, None)
                    self.cache_expiry.pop(key, None)
                
                await asyncio.sleep(600)  # Clean every 10 minutes
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_loop(self) -> None:
        """Perform periodic health checks."""
        while self.is_running:
            try:
                # Check if we're generating signals
                time_since_last_signal = time.time() - self.last_execution_time
                max_silence = self.config.get('max_silence_seconds', 600)  # 10 minutes
                
                if time_since_last_signal > max_silence:
                    logger.warning(f"Agent {self.name} hasn't generated signals for {time_since_last_signal:.0f}s")
                    self.status = AgentStatus.ERROR
                
                # Check error rate
                error_threshold = self.config.get('max_error_rate', 0.1)
                if self.metrics.total_signals > 0:
                    error_rate = self.metrics.error_count / self.metrics.total_signals
                    if error_rate > error_threshold:
                        logger.error(f"High error rate for {self.name}: {error_rate:.2%}")
                        self.status = AgentStatus.ERROR
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(60)
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle errors with recovery strategies."""
        logger.error(f"Error in agent {self.name}: {error}")
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self, error)
                else:
                    callback(self, error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
        
        # Update status
        self.status = AgentStatus.ERROR
    
    # Utility methods for subclasses
    
    async def get_cached_feature(self, key: str, ttl: int = 300) -> Optional[Any]:
        """Get cached feature with TTL."""
        if key in self.feature_cache:
            if key in self.cache_expiry and self.cache_expiry[key] > time.time():
                return self.feature_cache[key]
            else:
                # Remove expired entry
                self.feature_cache.pop(key, None)
                self.cache_expiry.pop(key, None)
        
        return None
    
    async def cache_feature(self, key: str, value: Any, ttl: int = 300) -> None:
        """Cache feature with TTL."""
        self.feature_cache[key] = value
        self.cache_expiry[key] = time.time() + ttl
    
    async def get_market_data(self, symbol: str, lookback_periods: int = 100) -> Optional[pd.DataFrame]:
        """Get market data for a symbol."""
        try:
            # Try cache first
            cache_key = f"market_data:{symbol}:{lookback_periods}"
            cached_data = await self.get_cached_feature(cache_key, ttl=60)  # 1 minute cache
            
            if cached_data is not None:
                return cached_data
            
            # Get from market data service
            if self.market_data_service:
                price_history = await self.redis_manager.get_price_history(symbol, lookback_periods)
                
                if price_history:
                    df = pd.DataFrame(price_history)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Cache the result
                    await self.cache_feature(cache_key, df, ttl=60)
                    
                    return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate common technical indicators."""
        if df is None or df.empty:
            return {}
        
        indicators = {}
        
        try:
            # Simple Moving Averages
            indicators['sma_20'] = df['price'].tail(20).mean()
            indicators['sma_50'] = df['price'].tail(50).mean()
            
            # Exponential Moving Averages
            indicators['ema_12'] = df['price'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = df['price'].ewm(span=26).mean().iloc[-1]
            
            # RSI calculation
            if len(df) >= 14:
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            ema_12 = df['price'].ewm(span=12).mean()
            ema_26 = df['price'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            
            indicators['macd'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = (macd_line - signal_line).iloc[-1]
            
            # Bollinger Bands
            if len(df) >= 20:
                bb_middle = df['price'].rolling(window=20).mean()
                bb_std = df['price'].rolling(window=20).std()
                indicators['bb_upper'] = (bb_middle + (bb_std * 2)).iloc[-1]
                indicators['bb_lower'] = (bb_middle - (bb_std * 2)).iloc[-1]
                indicators['bb_position'] = (df['price'].iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            # Volume indicators
            if 'volume' in df.columns:
                indicators['volume_sma'] = df['volume'].tail(20).mean()
                indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    # Public API methods
    
    def add_signal_callback(self, callback: Callable) -> None:
        """Add callback for signal notifications."""
        self.signal_callbacks.append(callback)
    
    def remove_signal_callback(self, callback: Callable) -> None:
        """Remove signal callback."""
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)
    
    def add_error_callback(self, callback: Callable) -> None:
        """Add callback for error notifications."""
        self.error_callbacks.append(callback)
    
    def get_metrics(self) -> AgentMetrics:
        """Get current agent metrics."""
        return self.metrics
    
    def get_recent_signals(self, count: int = 10) -> List[TradingSignal]:
        """Get recent signals."""
        return self.signal_history[-count:] if self.signal_history else []
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        return (
            self.is_running and 
            self.status in [AgentStatus.ACTIVE, AgentStatus.LEARNING] and
            time.time() - self.last_execution_time < 600  # Active within 10 minutes
        )