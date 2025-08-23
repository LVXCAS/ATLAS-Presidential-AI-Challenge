"""
Momentum Trading Agent for Bloomberg Terminal
Advanced momentum detection using multiple timeframes and indicators.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from agents.base_agent import BaseAgent, TradingSignal, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class MomentumAgent(BaseAgent):
    """
    Advanced momentum trading agent using:
    - Multi-timeframe momentum analysis
    - Price rate of change (ROC) 
    - Moving average convergence/divergence
    - Volume-weighted momentum
    - Trend strength indicators
    """
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        default_config = {
            'lookback_periods': 200,
            'short_ma': 12,
            'long_ma': 26,
            'signal_ma': 9,
            'roc_period': 14,
            'volume_threshold': 1.5,
            'momentum_threshold': 0.02,  # 2% momentum required
            'min_confidence': 0.65,
            'signal_interval': 60,  # 1 minute
            'trend_strength_period': 20
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(
            name="MomentumAgent",
            symbols=symbols,
            config=default_config
        )
        
        # Agent-specific state
        self.momentum_cache: Dict[str, Dict] = {}
        self.trend_cache: Dict[str, Dict] = {}
        
    async def initialize(self) -> None:
        """Initialize momentum-specific components."""
        logger.info(f"Initializing {self.name} for momentum trading")
        
        # Initialize momentum cache for all symbols
        for symbol in self.symbols:
            self.momentum_cache[symbol] = {}
            self.trend_cache[symbol] = {}
        
        logger.info(f"{self.name} initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup momentum-specific resources."""
        self.momentum_cache.clear()
        self.trend_cache.clear()
        logger.info(f"{self.name} cleanup completed")
    
    async def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate momentum-based trading signal.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None if no signal
        """
        try:
            # Get market data
            df = await self.get_market_data(symbol, self.config['lookback_periods'])
            if df is None or len(df) < 50:
                return None
            
            # Calculate features
            features = await self.calculate_features(symbol)
            if not features:
                return None
            
            # Analyze momentum patterns
            momentum_analysis = await self._analyze_momentum(df, features)
            if not momentum_analysis:
                return None
            
            # Generate signal based on analysis
            signal = await self._generate_momentum_signal(symbol, momentum_analysis, features)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating momentum signal for {symbol}: {e}")
            return None
    
    async def calculate_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate momentum-specific features.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of feature names to values
        """
        try:
            # Check cache first
            cache_key = f"momentum_features:{symbol}"
            cached_features = await self.get_cached_feature(cache_key, ttl=60)
            if cached_features:
                return cached_features
            
            # Get market data
            df = await self.get_market_data(symbol, self.config['lookback_periods'])
            if df is None or df.empty:
                return {}
            
            features = {}
            
            # Basic technical indicators
            tech_indicators = await self.calculate_technical_indicators(df)
            features.update(tech_indicators)
            
            # Momentum-specific calculations
            prices = df['price'].values
            
            # Rate of Change (ROC)
            roc_period = self.config['roc_period']
            if len(prices) > roc_period:
                current_price = prices[-1]
                past_price = prices[-roc_period-1]
                features['roc'] = (current_price - past_price) / past_price if past_price > 0 else 0
            
            # Price momentum (multiple timeframes)
            features['momentum_5'] = self._calculate_price_momentum(prices, 5)
            features['momentum_10'] = self._calculate_price_momentum(prices, 10)
            features['momentum_20'] = self._calculate_price_momentum(prices, 20)
            
            # Moving Average relationships
            if 'sma_20' in features and 'sma_50' in features:
                features['ma_ratio'] = features['sma_20'] / features['sma_50'] if features['sma_50'] > 0 else 1
                features['ma_spread'] = (features['sma_20'] - features['sma_50']) / features['sma_50'] if features['sma_50'] > 0 else 0
            
            # MACD momentum
            if all(k in features for k in ['macd', 'macd_signal']):
                features['macd_momentum'] = features['macd'] - features['macd_signal']
                features['macd_strength'] = abs(features['macd_momentum'])
            
            # Trend strength
            features['trend_strength'] = self._calculate_trend_strength(df)
            
            # Volume momentum
            if 'volume' in df.columns:
                features['volume_momentum'] = self._calculate_volume_momentum(df)
            
            # Price acceleration
            features['price_acceleration'] = self._calculate_price_acceleration(prices)
            
            # Support/Resistance momentum
            features['breakout_strength'] = self._calculate_breakout_strength(df)
            
            # Volatility-adjusted momentum
            features['vol_adj_momentum'] = self._calculate_volatility_adjusted_momentum(df)
            
            # Cache the features
            await self.cache_feature(cache_key, features, ttl=60)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating momentum features for {symbol}: {e}")
            return {}
    
    def _calculate_price_momentum(self, prices: np.ndarray, period: int) -> float:
        """Calculate price momentum over specified period."""
        if len(prices) <= period:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-period-1]
        
        return (current_price - past_price) / past_price if past_price > 0 else 0.0
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using linear regression slope."""
        if len(df) < self.config['trend_strength_period']:
            return 0.0
        
        try:
            # Use recent data for trend calculation
            recent_data = df.tail(self.config['trend_strength_period'])
            x = np.arange(len(recent_data))
            y = recent_data['price'].values
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Normalize slope relative to current price
            normalized_slope = slope / y[-1] if y[-1] > 0 else 0
            
            return normalized_slope
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _calculate_volume_momentum(self, df: pd.DataFrame) -> float:
        """Calculate volume-weighted momentum."""
        if 'volume' not in df.columns or len(df) < 20:
            return 0.0
        
        try:
            # Recent volume vs average
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].tail(20).mean()
            
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price momentum weighted by volume
            price_momentum = self._calculate_price_momentum(df['price'].values, 5)
            
            return price_momentum * min(volume_ratio, 3.0)  # Cap at 3x volume
            
        except Exception as e:
            logger.error(f"Error calculating volume momentum: {e}")
            return 0.0
    
    def _calculate_price_acceleration(self, prices: np.ndarray) -> float:
        """Calculate price acceleration (second derivative)."""
        if len(prices) < 10:
            return 0.0
        
        try:
            # Calculate short-term and medium-term momentum
            short_momentum = self._calculate_price_momentum(prices, 3)
            medium_momentum = self._calculate_price_momentum(prices, 6)
            
            # Acceleration is the difference
            acceleration = short_momentum - medium_momentum
            
            return acceleration
            
        except Exception as e:
            logger.error(f"Error calculating price acceleration: {e}")
            return 0.0
    
    def _calculate_breakout_strength(self, df: pd.DataFrame) -> float:
        """Calculate strength of price breakouts."""
        if len(df) < 50:
            return 0.0
        
        try:
            current_price = df['price'].iloc[-1]
            
            # Calculate recent high/low
            recent_high = df['price'].tail(20).max()
            recent_low = df['price'].tail(20).min()
            
            # Calculate longer-term high/low
            long_high = df['price'].tail(50).max()
            long_low = df['price'].tail(50).min()
            
            # Breakout strength
            if current_price > recent_high:
                # Upward breakout
                breakout_strength = (current_price - recent_high) / (long_high - long_low)
            elif current_price < recent_low:
                # Downward breakout
                breakout_strength = (recent_low - current_price) / (long_high - long_low)
            else:
                breakout_strength = 0.0
            
            return min(abs(breakout_strength), 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating breakout strength: {e}")
            return 0.0
    
    def _calculate_volatility_adjusted_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum adjusted for volatility."""
        if len(df) < 20:
            return 0.0
        
        try:
            # Calculate raw momentum
            raw_momentum = self._calculate_price_momentum(df['price'].values, 10)
            
            # Calculate volatility (standard deviation)
            volatility = df['price'].tail(20).std()
            avg_price = df['price'].tail(20).mean()
            
            # Normalize volatility
            vol_ratio = volatility / avg_price if avg_price > 0 else 0
            
            # Adjust momentum for volatility
            vol_adj_momentum = raw_momentum / (1 + vol_ratio) if vol_ratio > 0 else raw_momentum
            
            return vol_adj_momentum
            
        except Exception as e:
            logger.error(f"Error calculating volatility-adjusted momentum: {e}")
            return 0.0
    
    async def _analyze_momentum(self, df: pd.DataFrame, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Comprehensive momentum analysis.
        
        Args:
            df: Market data DataFrame
            features: Calculated features
            
        Returns:
            Momentum analysis results
        """
        try:
            analysis = {
                'overall_momentum': 0.0,
                'trend_direction': 'NEUTRAL',
                'momentum_strength': 0.0,
                'confidence_factors': [],
                'risk_factors': [],
                'supporting_indicators': 0
            }
            
            # Analyze momentum indicators
            momentum_scores = []
            
            # 1. Price momentum analysis
            momentum_5 = features.get('momentum_5', 0)
            momentum_10 = features.get('momentum_10', 0)
            momentum_20 = features.get('momentum_20', 0)
            
            if momentum_5 > self.config['momentum_threshold']:
                momentum_scores.append(1.0)
                analysis['confidence_factors'].append("Strong short-term momentum")
            elif momentum_5 < -self.config['momentum_threshold']:
                momentum_scores.append(-1.0)
                analysis['confidence_factors'].append("Strong short-term reversal momentum")
            
            # 2. Moving average analysis
            ma_ratio = features.get('ma_ratio', 1.0)
            if ma_ratio > 1.02:  # 2% above
                momentum_scores.append(0.5)
                analysis['confidence_factors'].append("Moving averages bullish")
            elif ma_ratio < 0.98:  # 2% below
                momentum_scores.append(-0.5)
                analysis['confidence_factors'].append("Moving averages bearish")
            
            # 3. MACD momentum
            macd_momentum = features.get('macd_momentum', 0)
            if abs(macd_momentum) > 0.1:  # Significant MACD momentum
                score = 0.7 if macd_momentum > 0 else -0.7
                momentum_scores.append(score)
                direction = "bullish" if macd_momentum > 0 else "bearish"
                analysis['confidence_factors'].append(f"MACD momentum {direction}")
            
            # 4. Volume confirmation
            volume_momentum = features.get('volume_momentum', 0)
            if abs(volume_momentum) > 0.02:  # 2% volume-weighted momentum
                score = 0.6 if volume_momentum > 0 else -0.6
                momentum_scores.append(score)
                analysis['confidence_factors'].append("Volume confirms momentum")
            
            # 5. Trend strength
            trend_strength = features.get('trend_strength', 0)
            if abs(trend_strength) > 0.01:  # 1% trend strength
                score = 0.5 if trend_strength > 0 else -0.5
                momentum_scores.append(score)
                analysis['confidence_factors'].append("Strong trend detected")
            
            # 6. Breakout analysis
            breakout_strength = features.get('breakout_strength', 0)
            if breakout_strength > 0.1:  # 10% breakout
                momentum_scores.append(0.8)
                analysis['confidence_factors'].append("Price breakout detected")
            
            # Calculate overall momentum
            if momentum_scores:
                analysis['overall_momentum'] = np.mean(momentum_scores)
                analysis['momentum_strength'] = abs(analysis['overall_momentum'])
                analysis['supporting_indicators'] = len([s for s in momentum_scores if abs(s) > 0.3])
            
            # Determine trend direction
            if analysis['overall_momentum'] > 0.3:
                analysis['trend_direction'] = 'BULLISH'
            elif analysis['overall_momentum'] < -0.3:
                analysis['trend_direction'] = 'BEARISH'
            else:
                analysis['trend_direction'] = 'NEUTRAL'
            
            # Risk factors
            rsi = features.get('rsi', 50)
            if rsi > 80:
                analysis['risk_factors'].append("Overbought conditions (RSI > 80)")
            elif rsi < 20:
                analysis['risk_factors'].append("Oversold conditions (RSI < 20)")
            
            vol_adj_momentum = features.get('vol_adj_momentum', 0)
            if abs(vol_adj_momentum) < 0.005:  # Very low momentum
                analysis['risk_factors'].append("Low volatility-adjusted momentum")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return None
    
    async def _generate_momentum_signal(
        self, 
        symbol: str, 
        analysis: Dict[str, Any], 
        features: Dict[str, float]
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal based on momentum analysis.
        
        Args:
            symbol: Trading symbol
            analysis: Momentum analysis results
            features: Calculated features
            
        Returns:
            TradingSignal or None
        """
        try:
            overall_momentum = analysis['overall_momentum']
            momentum_strength = analysis['momentum_strength']
            trend_direction = analysis['trend_direction']
            
            # Require minimum momentum strength
            if momentum_strength < 0.4:
                return None
            
            # Determine signal type
            if overall_momentum > 0.6 and trend_direction == 'BULLISH':
                signal_type = SignalType.STRONG_BUY
            elif overall_momentum > 0.3 and trend_direction == 'BULLISH':
                signal_type = SignalType.BUY
            elif overall_momentum < -0.6 and trend_direction == 'BEARISH':
                signal_type = SignalType.STRONG_SELL
            elif overall_momentum < -0.3 and trend_direction == 'BEARISH':
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Skip HOLD signals
            if signal_type == SignalType.HOLD:
                return None
            
            # Calculate confidence
            base_confidence = min(momentum_strength * 1.2, 0.95)
            
            # Adjust confidence based on supporting indicators
            supporting_indicators = analysis['supporting_indicators']
            confidence_boost = min(supporting_indicators * 0.05, 0.15)
            confidence = min(base_confidence + confidence_boost, 0.95)
            
            # Reduce confidence for risk factors
            risk_factor_count = len(analysis['risk_factors'])
            confidence_penalty = risk_factor_count * 0.1
            confidence = max(confidence - confidence_penalty, 0.3)
            
            # Skip if confidence too low
            if confidence < self.config['min_confidence']:
                return None
            
            # Calculate target price and stops
            current_price = features.get('last_price', 0)
            if current_price == 0:
                return None
            
            # Dynamic target based on momentum strength
            price_target_pct = momentum_strength * 0.05  # Up to 5% target
            stop_loss_pct = 0.02  # 2% stop loss
            
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                target_price = current_price * (1 + price_target_pct)
                stop_loss = current_price * (1 - stop_loss_pct)
            else:
                target_price = current_price * (1 - price_target_pct)
                stop_loss = current_price * (1 + stop_loss_pct)
            
            # Create signal
            signal = TradingSignal(
                id=str(uuid.uuid4()),
                agent_name=self.name,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                signal_type=signal_type,
                confidence=confidence,
                strength=momentum_strength,
                reasoning={
                    'analysis': analysis,
                    'momentum_indicators': {
                        'overall_momentum': overall_momentum,
                        'trend_direction': trend_direction,
                        'supporting_indicators': supporting_indicators
                    },
                    'risk_assessment': {
                        'risk_factors': analysis['risk_factors'],
                        'confidence_factors': analysis['confidence_factors']
                    }
                },
                features_used=features,
                prediction_horizon=30,  # 30 minutes
                target_price=target_price,
                stop_loss=stop_loss,
                risk_score=risk_factor_count / 10.0,
                expected_return=price_target_pct
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating momentum signal: {e}")
            return None


# Convenience function for creating momentum agent
def create_momentum_agent(symbols: List[str], **kwargs) -> MomentumAgent:
    """Create a momentum trading agent with default configuration."""
    return MomentumAgent(symbols, kwargs)