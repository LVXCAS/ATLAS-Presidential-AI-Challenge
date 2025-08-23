"""
Mean Reversion Trading Agent for Bloomberg Terminal
Advanced mean reversion detection using statistical analysis and multiple indicators.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from scipy import stats

from agents.base_agent import BaseAgent, TradingSignal, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class MeanReversionAgent(BaseAgent):
    """
    Advanced mean reversion trading agent using:
    - Statistical analysis (Z-score, percentiles)
    - Bollinger Bands with dynamic adjustment
    - RSI divergence analysis
    - Volume profile analysis
    - Support/resistance levels
    - Reversion strength indicators
    """
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        default_config = {
            'lookback_periods': 200,
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'zscore_threshold': 1.5,
            'percentile_threshold': 10,  # 10th/90th percentile
            'min_confidence': 0.60,
            'signal_interval': 120,  # 2 minutes
            'volume_confirmation': True,
            'divergence_lookback': 10,
            'reversion_strength_period': 50
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(
            name="MeanReversionAgent",
            symbols=symbols,
            config=default_config
        )
        
        # Agent-specific state
        self.reversion_cache: Dict[str, Dict] = {}
        self.support_resistance: Dict[str, Dict] = {}
        
    async def initialize(self) -> None:
        """Initialize mean reversion specific components."""
        logger.info(f"Initializing {self.name} for mean reversion trading")
        
        # Initialize caches for all symbols
        for symbol in self.symbols:
            self.reversion_cache[symbol] = {}
            self.support_resistance[symbol] = {'support': [], 'resistance': []}
        
        logger.info(f"{self.name} initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup mean reversion specific resources."""
        self.reversion_cache.clear()
        self.support_resistance.clear()
        logger.info(f"{self.name} cleanup completed")
    
    async def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate mean reversion trading signal.
        
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
            
            # Analyze mean reversion patterns
            reversion_analysis = await self._analyze_mean_reversion(df, features)
            if not reversion_analysis:
                return None
            
            # Generate signal based on analysis
            signal = await self._generate_reversion_signal(symbol, reversion_analysis, features)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signal for {symbol}: {e}")
            return None
    
    async def calculate_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate mean reversion specific features.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of feature names to values
        """
        try:
            # Check cache first
            cache_key = f"reversion_features:{symbol}"
            cached_features = await self.get_cached_feature(cache_key, ttl=120)
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
            
            # Mean reversion specific calculations
            prices = df['price'].values
            current_price = prices[-1]
            
            # Statistical measures
            features.update(self._calculate_statistical_features(prices))
            
            # Bollinger Bands analysis
            features.update(self._calculate_bollinger_features(df))
            
            # RSI analysis
            features.update(self._calculate_rsi_features(features))
            
            # Support/Resistance analysis
            features.update(await self._calculate_support_resistance_features(df, symbol))
            
            # Volume profile analysis
            if 'volume' in df.columns:
                features.update(self._calculate_volume_features(df))
            
            # Divergence analysis
            features.update(self._calculate_divergence_features(df))
            
            # Reversion strength
            features.update(self._calculate_reversion_strength(df))
            
            # Price extremes
            features.update(self._calculate_price_extremes(df))
            
            # Market microstructure
            features.update(self._calculate_microstructure_features(df))
            
            # Cache the features
            await self.cache_feature(cache_key, features, ttl=120)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating reversion features for {symbol}: {e}")
            return {}
    
    def _calculate_statistical_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate statistical features for mean reversion."""
        features = {}
        
        try:
            current_price = prices[-1]
            
            # Z-score calculation
            if len(prices) >= 50:
                mean_price = np.mean(prices[-50:])
                std_price = np.std(prices[-50:])
                features['zscore'] = (current_price - mean_price) / std_price if std_price > 0 else 0
                features['zscore_abs'] = abs(features['zscore'])
            
            # Percentile ranking
            if len(prices) >= 100:
                percentile_rank = stats.percentileofscore(prices[-100:], current_price)
                features['percentile_rank'] = percentile_rank
                features['extreme_percentile'] = (
                    percentile_rank <= self.config['percentile_threshold'] or
                    percentile_rank >= (100 - self.config['percentile_threshold'])
                )
            
            # Distance from various moving averages
            if len(prices) >= 20:
                sma_20 = np.mean(prices[-20:])
                features['distance_from_sma20'] = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
            
            if len(prices) >= 50:
                sma_50 = np.mean(prices[-50:])
                features['distance_from_sma50'] = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0
            
            # Price volatility
            if len(prices) >= 20:
                returns = np.diff(np.log(prices[-21:]))  # 20 returns
                features['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
                features['volatility_normalized'] = features['volatility'] / np.mean(prices[-20:]) if np.mean(prices[-20:]) > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating statistical features: {e}")
        
        return features
    
    def _calculate_bollinger_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Bollinger Bands features."""
        features = {}
        
        try:
            bb_period = self.config['bb_period']
            bb_std = self.config['bb_std_dev']
            
            if len(df) >= bb_period:
                # Calculate Bollinger Bands
                bb_middle = df['price'].rolling(window=bb_period).mean()
                bb_std_dev = df['price'].rolling(window=bb_period).std()
                bb_upper = bb_middle + (bb_std_dev * bb_std)
                bb_lower = bb_middle - (bb_std_dev * bb_std)
                
                current_price = df['price'].iloc[-1]
                current_upper = bb_upper.iloc[-1]
                current_lower = bb_lower.iloc[-1]
                current_middle = bb_middle.iloc[-1]
                
                # BB position (0 = lower band, 1 = upper band)
                if current_upper != current_lower:
                    features['bb_position'] = (current_price - current_lower) / (current_upper - current_lower)
                else:
                    features['bb_position'] = 0.5
                
                # Distance from bands
                features['distance_from_upper'] = (current_upper - current_price) / current_price if current_price > 0 else 0
                features['distance_from_lower'] = (current_price - current_lower) / current_price if current_price > 0 else 0
                
                # BB width (volatility measure)
                features['bb_width'] = (current_upper - current_lower) / current_middle if current_middle > 0 else 0
                
                # BB squeeze detection
                if len(bb_std_dev) >= 50:
                    avg_width = ((bb_upper - bb_lower) / bb_middle).tail(50).mean()
                    current_width = features['bb_width']
                    features['bb_squeeze'] = current_width < (avg_width * 0.8)  # 20% below average
                
                # Band touches/breaks
                features['near_upper_band'] = features['bb_position'] > 0.9
                features['near_lower_band'] = features['bb_position'] < 0.1
                features['outside_bands'] = features['bb_position'] > 1.0 or features['bb_position'] < 0.0
                
        except Exception as e:
            logger.error(f"Error calculating Bollinger features: {e}")
        
        return features
    
    def _calculate_rsi_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate RSI-specific features for mean reversion."""
        rsi_features = {}
        
        try:
            rsi = features.get('rsi', 50)
            rsi_oversold = self.config['rsi_oversold']
            rsi_overbought = self.config['rsi_overbought']
            
            # RSI levels
            rsi_features['rsi_oversold'] = rsi <= rsi_oversold
            rsi_features['rsi_overbought'] = rsi >= rsi_overbought
            rsi_features['rsi_extreme'] = rsi <= 20 or rsi >= 80
            
            # RSI distance from extremes
            rsi_features['rsi_distance_from_50'] = abs(rsi - 50) / 50
            rsi_features['rsi_reversion_potential'] = max(0, (abs(rsi - 50) - 20) / 30) if abs(rsi - 50) > 20 else 0
            
        except Exception as e:
            logger.error(f"Error calculating RSI features: {e}")
        
        return rsi_features
    
    async def _calculate_support_resistance_features(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        features = {}
        
        try:
            current_price = df['price'].iloc[-1]
            
            # Simple support/resistance calculation using recent highs/lows
            if len(df) >= 50:
                recent_data = df.tail(50)
                
                # Find local maxima and minima
                highs = recent_data['price'].rolling(window=5, center=True).max()
                lows = recent_data['price'].rolling(window=5, center=True).min()
                
                # Identify significant levels
                resistance_levels = []
                support_levels = []
                
                for i in range(2, len(recent_data) - 2):
                    price = recent_data['price'].iloc[i]
                    
                    if price == highs.iloc[i]:  # Local high
                        resistance_levels.append(price)
                    elif price == lows.iloc[i]:  # Local low
                        support_levels.append(price)
                
                # Store levels
                self.support_resistance[symbol]['resistance'] = sorted(set(resistance_levels), reverse=True)[:5]
                self.support_resistance[symbol]['support'] = sorted(set(support_levels))[-5:]
                
                # Calculate distances to nearest levels
                if self.support_resistance[symbol]['resistance']:
                    nearest_resistance = min([r for r in self.support_resistance[symbol]['resistance'] if r >= current_price], 
                                           default=current_price * 1.1)
                    features['distance_to_resistance'] = (nearest_resistance - current_price) / current_price
                else:
                    features['distance_to_resistance'] = 0.1
                
                if self.support_resistance[symbol]['support']:
                    nearest_support = max([s for s in self.support_resistance[symbol]['support'] if s <= current_price], 
                                        default=current_price * 0.9)
                    features['distance_to_support'] = (current_price - nearest_support) / current_price
                else:
                    features['distance_to_support'] = 0.1
                
                # Near support/resistance
                features['near_resistance'] = features['distance_to_resistance'] < 0.02  # Within 2%
                features['near_support'] = features['distance_to_support'] < 0.02  # Within 2%
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance features: {e}")
        
        return features
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based features."""
        features = {}
        
        try:
            if 'volume' not in df.columns:
                return features
            
            current_volume = df['volume'].iloc[-1]
            
            # Volume moving averages
            if len(df) >= 20:
                vol_sma_20 = df['volume'].tail(20).mean()
                features['volume_ratio'] = current_volume / vol_sma_20 if vol_sma_20 > 0 else 1
                features['high_volume'] = features['volume_ratio'] > 1.5
            
            # Volume trend
            if len(df) >= 10:
                recent_volume = df['volume'].tail(5).mean()
                past_volume = df['volume'].iloc[-10:-5].mean()
                features['volume_trend'] = (recent_volume - past_volume) / past_volume if past_volume > 0 else 0
            
            # Volume-Price Trend (VPT)
            if len(df) >= 20:
                price_changes = df['price'].pct_change()
                vpt = (price_changes * df['volume']).cumsum()
                features['vpt_trend'] = (vpt.iloc[-1] - vpt.iloc[-10]) / abs(vpt.iloc[-10]) if vpt.iloc[-10] != 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating volume features: {e}")
        
        return features
    
    def _calculate_divergence_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price-indicator divergences."""
        features = {}
        
        try:
            if len(df) < self.config['divergence_lookback']:
                return features
            
            recent_data = df.tail(self.config['divergence_lookback'])
            
            # Price trend
            price_trend = (recent_data['price'].iloc[-1] - recent_data['price'].iloc[0]) / recent_data['price'].iloc[0]
            
            # Calculate RSI for divergence
            delta = recent_data['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            
            if len(rsi_series.dropna()) >= 2:
                rsi_trend = rsi_series.iloc[-1] - rsi_series.iloc[0]
                
                # Bullish divergence: price down, RSI up
                features['bullish_divergence'] = price_trend < -0.02 and rsi_trend > 2
                
                # Bearish divergence: price up, RSI down
                features['bearish_divergence'] = price_trend > 0.02 and rsi_trend < -2
            
        except Exception as e:
            logger.error(f"Error calculating divergence features: {e}")
        
        return features
    
    def _calculate_reversion_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate mean reversion strength indicators."""
        features = {}
        
        try:
            reversion_period = self.config['reversion_strength_period']
            if len(df) < reversion_period:
                return features
            
            prices = df['price'].values[-reversion_period:]
            
            # Calculate autocorrelation (negative indicates mean reversion)
            if len(prices) >= 10:
                returns = np.diff(np.log(prices))
                if len(returns) > 1:
                    features['autocorrelation'] = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                    features['mean_reversion_strength'] = -features['autocorrelation'] if features['autocorrelation'] < 0 else 0
            
            # Half-life of mean reversion
            features['reversion_half_life'] = self._calculate_half_life(prices)
            
        except Exception as e:
            logger.error(f"Error calculating reversion strength: {e}")
        
        return features
    
    def _calculate_half_life(self, prices: np.ndarray) -> float:
        """Calculate half-life of mean reversion using Ornstein-Uhlenbeck process."""
        try:
            if len(prices) < 20:
                return 0.0
            
            # Calculate log prices and their lagged values
            log_prices = np.log(prices)
            lagged_prices = log_prices[:-1]
            price_diff = np.diff(log_prices)
            
            # Simple regression to estimate mean reversion speed
            if len(lagged_prices) > 0 and np.std(lagged_prices) > 0:
                slope, _ = np.polyfit(lagged_prices, price_diff, 1)
                half_life = -np.log(2) / slope if slope < 0 else 100  # Cap at 100
                return min(abs(half_life), 100)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating half-life: {e}")
            return 0.0
    
    def _calculate_price_extremes(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price extreme indicators."""
        features = {}
        
        try:
            if len(df) < 50:
                return features
            
            current_price = df['price'].iloc[-1]
            
            # Recent extremes
            recent_high = df['price'].tail(20).max()
            recent_low = df['price'].tail(20).min()
            
            # Long-term extremes
            long_high = df['price'].tail(50).max()
            long_low = df['price'].tail(50).min()
            
            # Extreme positions
            features['at_recent_high'] = current_price >= recent_high * 0.995  # Within 0.5%
            features['at_recent_low'] = current_price <= recent_low * 1.005   # Within 0.5%
            features['at_long_high'] = current_price >= long_high * 0.99      # Within 1%
            features['at_long_low'] = current_price <= long_low * 1.01        # Within 1%
            
            # Distance from extremes
            features['distance_from_recent_high'] = (recent_high - current_price) / current_price
            features['distance_from_recent_low'] = (current_price - recent_low) / current_price
            
        except Exception as e:
            logger.error(f"Error calculating price extremes: {e}")
        
        return features
    
    def _calculate_microstructure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market microstructure features."""
        features = {}
        
        try:
            if len(df) < 20:
                return features
            
            # Price impact and momentum
            recent_returns = df['price'].pct_change().tail(10)
            features['recent_volatility'] = recent_returns.std()
            features['price_momentum_short'] = recent_returns.mean()
            
            # Trend consistency
            positive_returns = (recent_returns > 0).sum()
            features['trend_consistency'] = abs(positive_returns - 5) / 5  # Deviation from 50%
            
        except Exception as e:
            logger.error(f"Error calculating microstructure features: {e}")
        
        return features
    
    async def _analyze_mean_reversion(self, df: pd.DataFrame, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Comprehensive mean reversion analysis.
        
        Args:
            df: Market data DataFrame
            features: Calculated features
            
        Returns:
            Mean reversion analysis results
        """
        try:
            analysis = {
                'reversion_signal': 'NONE',
                'reversion_strength': 0.0,
                'confidence_factors': [],
                'risk_factors': [],
                'statistical_evidence': 0,
                'technical_evidence': 0,
                'volume_confirmation': False
            }
            
            evidence_count = 0
            strength_scores = []
            
            # 1. Statistical Evidence
            zscore = features.get('zscore', 0)
            if abs(zscore) >= self.config['zscore_threshold']:
                evidence_count += 1
                strength_scores.append(min(abs(zscore) / 3.0, 1.0))  # Normalize to 1.0
                direction = "oversold" if zscore < 0 else "overbought"
                analysis['confidence_factors'].append(f"Significant Z-score: {direction}")
            
            # 2. Percentile Evidence
            if features.get('extreme_percentile', False):
                evidence_count += 1
                percentile = features.get('percentile_rank', 50)
                if percentile <= 10:
                    strength_scores.append(0.8)
                    analysis['confidence_factors'].append("Price in bottom 10th percentile")
                elif percentile >= 90:
                    strength_scores.append(0.8)
                    analysis['confidence_factors'].append("Price in top 10th percentile")
            
            # 3. Bollinger Bands Evidence
            bb_position = features.get('bb_position', 0.5)
            if bb_position >= 1.0:  # Above upper band
                evidence_count += 1
                strength_scores.append(0.7)
                analysis['confidence_factors'].append("Price above Bollinger upper band")
            elif bb_position <= 0.0:  # Below lower band
                evidence_count += 1
                strength_scores.append(0.7)
                analysis['confidence_factors'].append("Price below Bollinger lower band")
            elif bb_position > 0.9:  # Near upper band
                evidence_count += 1
                strength_scores.append(0.5)
                analysis['confidence_factors'].append("Price near Bollinger upper band")
            elif bb_position < 0.1:  # Near lower band
                evidence_count += 1
                strength_scores.append(0.5)
                analysis['confidence_factors'].append("Price near Bollinger lower band")
            
            # 4. RSI Evidence
            if features.get('rsi_extreme', False):
                evidence_count += 1
                rsi = features.get('rsi', 50)
                if rsi <= 20:
                    strength_scores.append(0.8)
                    analysis['confidence_factors'].append("RSI extremely oversold")
                elif rsi >= 80:
                    strength_scores.append(0.8)
                    analysis['confidence_factors'].append("RSI extremely overbought")
            elif features.get('rsi_oversold', False) or features.get('rsi_overbought', False):
                evidence_count += 1
                strength_scores.append(0.6)
                status = "oversold" if features.get('rsi_oversold') else "overbought"
                analysis['confidence_factors'].append(f"RSI {status}")
            
            # 5. Divergence Evidence
            if features.get('bullish_divergence', False):
                evidence_count += 1
                strength_scores.append(0.7)
                analysis['confidence_factors'].append("Bullish divergence detected")
            elif features.get('bearish_divergence', False):
                evidence_count += 1
                strength_scores.append(0.7)
                analysis['confidence_factors'].append("Bearish divergence detected")
            
            # 6. Support/Resistance Evidence
            if features.get('near_support', False):
                evidence_count += 1
                strength_scores.append(0.6)
                analysis['confidence_factors'].append("Price near support level")
            elif features.get('near_resistance', False):
                evidence_count += 1
                strength_scores.append(0.6)
                analysis['confidence_factors'].append("Price near resistance level")
            
            # 7. Volume Confirmation
            if self.config['volume_confirmation']:
                if features.get('high_volume', False):
                    analysis['volume_confirmation'] = True
                    strength_scores.append(0.3)
                    analysis['confidence_factors'].append("High volume confirms signal")
                else:
                    analysis['risk_factors'].append("Low volume - weak confirmation")
            
            # 8. Mean Reversion Strength
            reversion_strength = features.get('mean_reversion_strength', 0)
            if reversion_strength > 0.3:
                evidence_count += 1
                strength_scores.append(reversion_strength)
                analysis['confidence_factors'].append("Strong mean reversion characteristics")
            
            # Determine signal direction
            if zscore < -self.config['zscore_threshold'] or bb_position < 0.1 or features.get('rsi_oversold', False):
                analysis['reversion_signal'] = 'BUY'  # Oversold, expect reversion up
            elif zscore > self.config['zscore_threshold'] or bb_position > 0.9 or features.get('rsi_overbought', False):
                analysis['reversion_signal'] = 'SELL'  # Overbought, expect reversion down
            
            # Calculate overall strength
            if strength_scores:
                analysis['reversion_strength'] = min(np.mean(strength_scores), 1.0)
            
            analysis['statistical_evidence'] = evidence_count
            analysis['technical_evidence'] = len([f for f in analysis['confidence_factors'] if 'band' in f or 'RSI' in f])
            
            # Risk factors
            if features.get('bb_squeeze', False):
                analysis['risk_factors'].append("Bollinger Band squeeze - low volatility")
            
            recent_volatility = features.get('recent_volatility', 0)
            if recent_volatility < 0.01:  # Very low volatility
                analysis['risk_factors'].append("Very low volatility - limited reversion potential")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in mean reversion analysis: {e}")
            return None
    
    async def _generate_reversion_signal(
        self, 
        symbol: str, 
        analysis: Dict[str, Any], 
        features: Dict[str, float]
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal based on mean reversion analysis.
        
        Args:
            symbol: Trading symbol
            analysis: Mean reversion analysis results
            features: Calculated features
            
        Returns:
            TradingSignal or None
        """
        try:
            reversion_signal = analysis['reversion_signal']
            reversion_strength = analysis['reversion_strength']
            statistical_evidence = analysis['statistical_evidence']
            
            # Require minimum evidence
            if statistical_evidence < 2 or reversion_strength < 0.4:
                return None
            
            # Determine signal type
            if reversion_signal == 'BUY' and reversion_strength > 0.7:
                signal_type = SignalType.STRONG_BUY
            elif reversion_signal == 'BUY':
                signal_type = SignalType.BUY
            elif reversion_signal == 'SELL' and reversion_strength > 0.7:
                signal_type = SignalType.STRONG_SELL
            elif reversion_signal == 'SELL':
                signal_type = SignalType.SELL
            else:
                return None
            
            # Calculate confidence
            base_confidence = min(reversion_strength * 1.1, 0.90)
            
            # Boost confidence for multiple evidence types
            evidence_boost = min(statistical_evidence * 0.05, 0.15)
            confidence = min(base_confidence + evidence_boost, 0.90)
            
            # Volume confirmation boost
            if analysis['volume_confirmation']:
                confidence = min(confidence + 0.05, 0.90)
            
            # Risk factor penalty
            risk_penalty = len(analysis['risk_factors']) * 0.08
            confidence = max(confidence - risk_penalty, 0.3)
            
            # Skip if confidence too low
            if confidence < self.config['min_confidence']:
                return None
            
            # Calculate targets and stops
            current_price = features.get('last_price', 0)
            if current_price == 0:
                return None
            
            # Mean reversion targets are typically smaller
            price_target_pct = reversion_strength * 0.03  # Up to 3% target
            stop_loss_pct = 0.015  # 1.5% stop loss
            
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
                strength=reversion_strength,
                reasoning={
                    'analysis': analysis,
                    'reversion_indicators': {
                        'zscore': features.get('zscore', 0),
                        'bb_position': features.get('bb_position', 0.5),
                        'rsi': features.get('rsi', 50),
                        'percentile_rank': features.get('percentile_rank', 50)
                    },
                    'evidence_summary': {
                        'statistical_evidence': statistical_evidence,
                        'technical_evidence': analysis['technical_evidence'],
                        'volume_confirmation': analysis['volume_confirmation']
                    }
                },
                features_used=features,
                prediction_horizon=20,  # 20 minutes for mean reversion
                target_price=target_price,
                stop_loss=stop_loss,
                risk_score=len(analysis['risk_factors']) / 5.0,
                expected_return=price_target_pct
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating reversion signal: {e}")
            return None


# Convenience function for creating mean reversion agent
def create_mean_reversion_agent(symbols: List[str], **kwargs) -> MeanReversionAgent:
    """Create a mean reversion trading agent with default configuration."""
    return MeanReversionAgent(symbols, kwargs)