"""
Mean Reversion Trading Agent - LangGraph Implementation

This agent implements a comprehensive mean reversion trading strategy that combines:
- Bollinger Band reversions and Z-score analysis
- Fibonacci extension targets for exits
- Pairs trading with cointegration detection
- Sentiment divergence detection
- Explainable AI with top-3 reasoning factors

The agent operates autonomously using LangGraph state management and provides
structured signals with confidence scores and detailed explanations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
import pandas as pd
from decimal import Decimal
import warnings

# Statistical analysis imports
from scipy import stats
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression

# LangGraph imports
from langgraph.graph import StateGraph, END

# Technical analysis imports
from strategies.technical_indicators import (
    IndicatorLibrary, calculate_bollinger_bands, calculate_z_score
)
from strategies.fibonacci_analysis import (
    FibonacciAnalyzer, analyze_fibonacci_levels
)

# ENHANCEMENT: Import dynamic threshold calculators
from agents.mean_reversion_agent_enhancements import MeanReversionEnhancements

# Database imports
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Configuration
from config.settings import settings
from config.secure_config import get_api_keys

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of mean reversion signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9


class MarketRegime(Enum):
    """Market regime types"""
    MEAN_REVERTING = "mean_reverting"
    TRENDING = "trending"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    SIDEWAYS = "sideways"


@dataclass
class MarketData:
    """Market data structure for mean reversion analysis"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SentimentData:
    """Sentiment data structure"""
    symbol: str
    overall_sentiment: float  # -1 to 1
    confidence: float  # 0 to 1
    news_count: int
    social_sentiment: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TechnicalSignal:
    """Individual technical signal"""
    indicator: str
    signal_type: SignalType
    strength: float
    confidence: float
    value: float
    explanation: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'indicator': self.indicator,
            'signal_type': self.signal_type.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'value': self.value,
            'explanation': self.explanation,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PairsSignal:
    """Pairs trading signal"""
    symbol_a: str
    symbol_b: str
    spread: float
    z_score: float
    cointegration_pvalue: float
    hedge_ratio: float
    signal_type: SignalType
    confidence: float
    explanation: str
    
    def to_dict(self) -> Dict:
        return {
            'symbol_a': self.symbol_a,
            'symbol_b': self.symbol_b,
            'spread': self.spread,
            'z_score': self.z_score,
            'cointegration_pvalue': self.cointegration_pvalue,
            'hedge_ratio': self.hedge_ratio,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'explanation': self.explanation
        }


@dataclass
class FibonacciTarget:
    """Fibonacci extension target for exits"""
    level_name: str
    target_price: float
    distance_pct: float
    confidence: float
    explanation: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Reason:
    """Explanation reason for signal generation"""
    rank: int
    factor: str
    contribution: float
    explanation: str
    confidence: float
    supporting_data: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MeanReversionSignal:
    """Final mean reversion trading signal with explainability"""
    symbol: str
    signal_type: SignalType
    value: float  # -1 to 1 (negative = sell, positive = buy)
    confidence: float  # 0 to 1
    top_3_reasons: List[Reason]
    timestamp: datetime
    model_version: str
    
    # Technical components
    bollinger_signals: List[TechnicalSignal]
    zscore_signals: List[TechnicalSignal]
    pairs_signals: List[PairsSignal]
    fibonacci_targets: List[FibonacciTarget]
    
    # Market context
    sentiment_divergence: Optional[float] = None
    market_regime: Optional[MarketRegime] = None
    position_size_pct: Optional[float] = None
    
    # Risk metrics
    stop_loss_pct: Optional[float] = None
    take_profit_targets: Optional[List[float]] = None
    max_holding_period: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'value': self.value,
            'confidence': self.confidence,
            'top_3_reasons': [reason.to_dict() for reason in self.top_3_reasons],
            'timestamp': self.timestamp.isoformat(),
            'model_version': self.model_version,
            'bollinger_signals': [signal.to_dict() for signal in self.bollinger_signals],
            'zscore_signals': [signal.to_dict() for signal in self.zscore_signals],
            'pairs_signals': [signal.to_dict() for signal in self.pairs_signals],
            'fibonacci_targets': [target.to_dict() for target in self.fibonacci_targets],
            'sentiment_divergence': self.sentiment_divergence,
            'market_regime': self.market_regime.value if self.market_regime else None,
            'position_size_pct': self.position_size_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_targets': self.take_profit_targets,
            'max_holding_period': self.max_holding_period
        }


class BollingerBandAnalyzer:
    """Analyzes Bollinger Band reversions for mean reversion signals"""
    
    def __init__(self):
        self.indicator_library = IndicatorLibrary()
    
    def calculate_bollinger_signals(self, price_data: np.ndarray,
                                  period: int = 20, std_dev: float = 2.0) -> List[TechnicalSignal]:
        """
        ENHANCED: Calculate Bollinger Band reversion signals with DYNAMIC widths

        Now adapts band width based on volatility:
        - High volatility (>30%): Wider bands (2.6 std)
        - Low volatility (<15%): Tighter bands (1.6 std)
        """
        signals = []

        try:
            # ENHANCEMENT: Use dynamic Bollinger Bands that adapt to volatility
            df = pd.DataFrame({'close': price_data})
            bb_upper, bb_middle, bb_lower, std_mult, volatility = (
                MeanReversionEnhancements.calculate_dynamic_bollinger_bands(df, period, std_dev)
            )

            # Convert to numpy arrays
            upper_band = bb_upper.values if hasattr(bb_upper, 'values') else bb_upper
            middle_band = bb_middle.values if hasattr(bb_middle, 'values') else bb_middle
            lower_band = bb_lower.values if hasattr(bb_lower, 'values') else bb_lower

            logger.info(f"✅ Dynamic BB: volatility={volatility:.1%}, using {std_mult:.2f} std (adaptive!)")

            if len(upper_band) < 2:
                return signals
            
            current_price = price_data[-1]
            prev_price = price_data[-2] if len(price_data) > 1 else current_price
            
            current_upper = upper_band[-1]
            current_lower = lower_band[-1]
            current_middle = middle_band[-1]
            
            # Skip NaN values
            if np.isnan(current_upper) or np.isnan(current_lower) or np.isnan(current_middle):
                return signals
            
            # Calculate band position (0 = lower band, 1 = upper band)
            band_width = current_upper - current_lower
            if band_width > 0:
                band_position = (current_price - current_lower) / band_width
            else:
                band_position = 0.5
            
            # Bollinger Band squeeze detection
            if len(bb_data) >= 20:
                recent_widths = []
                for i in range(-20, 0):
                    if not (np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
                        recent_widths.append(upper_band[i] - lower_band[i])
                
                if recent_widths:
                    avg_width = np.mean(recent_widths)
                    current_width = current_upper - current_lower
                    width_ratio = current_width / avg_width if avg_width > 0 else 1.0
                    
                    # Squeeze condition (narrow bands)
                    if width_ratio < 0.8:
                        signals.append(TechnicalSignal(
                            indicator="BB_Squeeze",
                            signal_type=SignalType.HOLD,
                            strength=0.3,
                            confidence=0.6,
                            value=0.0,
                            explanation=f"Bollinger Bands squeeze detected (width ratio: {width_ratio:.2f})",
                            timestamp=datetime.utcnow()
                        ))
            
            # Upper band reversal (bearish)
            if current_price >= current_upper * 0.98:  # Within 2% of upper band
                if prev_price < current_upper * 0.98:  # Just touched upper band
                    strength = min((current_price - current_upper) / current_upper * 100 + 1.0, 1.0)
                    signals.append(TechnicalSignal(
                        indicator="BB_Upper_Reversal",
                        signal_type=SignalType.SELL,
                        strength=strength,
                        confidence=0.8,
                        value=-strength,
                        explanation=f"Price touched upper Bollinger Band at {current_upper:.2f}",
                        timestamp=datetime.utcnow()
                    ))
                elif band_position > 0.95:  # Very close to upper band
                    strength = min(band_position - 0.5, 0.5) * 2  # Scale to 0-1
                    signals.append(TechnicalSignal(
                        indicator="BB_Upper_Approach",
                        signal_type=SignalType.SELL,
                        strength=strength,
                        confidence=0.6,
                        value=-strength,
                        explanation=f"Price near upper Bollinger Band (position: {band_position:.2f})",
                        timestamp=datetime.utcnow()
                    ))
            
            # Lower band reversal (bullish)
            elif current_price <= current_lower * 1.02:  # Within 2% of lower band
                if prev_price > current_lower * 1.02:  # Just touched lower band
                    strength = min((current_lower - current_price) / current_lower * 100 + 1.0, 1.0)
                    signals.append(TechnicalSignal(
                        indicator="BB_Lower_Reversal",
                        signal_type=SignalType.BUY,
                        strength=strength,
                        confidence=0.8,
                        value=strength,
                        explanation=f"Price touched lower Bollinger Band at {current_lower:.2f}",
                        timestamp=datetime.utcnow()
                    ))
                elif band_position < 0.05:  # Very close to lower band
                    strength = min(0.5 - band_position, 0.5) * 2  # Scale to 0-1
                    signals.append(TechnicalSignal(
                        indicator="BB_Lower_Approach",
                        signal_type=SignalType.BUY,
                        strength=strength,
                        confidence=0.6,
                        value=strength,
                        explanation=f"Price near lower Bollinger Band (position: {band_position:.2f})",
                        timestamp=datetime.utcnow()
                    ))
            
            # Mean reversion to middle band
            elif abs(current_price - current_middle) / current_middle > 0.02:  # >2% from middle
                distance_from_middle = (current_price - current_middle) / current_middle
                
                if distance_from_middle > 0:  # Above middle, expect reversion down
                    strength = min(abs(distance_from_middle) * 5, 0.6)  # Scale appropriately
                    signals.append(TechnicalSignal(
                        indicator="BB_Middle_Reversion",
                        signal_type=SignalType.SELL,
                        strength=strength,
                        confidence=0.4,
                        value=-strength,
                        explanation=f"Price {distance_from_middle*100:.1f}% above middle band",
                        timestamp=datetime.utcnow()
                    ))
                else:  # Below middle, expect reversion up
                    strength = min(abs(distance_from_middle) * 5, 0.6)
                    signals.append(TechnicalSignal(
                        indicator="BB_Middle_Reversion",
                        signal_type=SignalType.BUY,
                        strength=strength,
                        confidence=0.4,
                        value=strength,
                        explanation=f"Price {abs(distance_from_middle)*100:.1f}% below middle band",
                        timestamp=datetime.utcnow()
                    ))
        
        except Exception as e:
            logger.error(f"Error calculating Bollinger Band signals: {e}")

        return signals

    def calculate_mean_reversion_probability(self, price_data: np.ndarray) -> Dict:
        """
        ENHANCEMENT: Calculate statistical probability of mean reversion

        This gives you a probability score (0-100%) that price will revert!
        """
        try:
            df = pd.DataFrame({'close': price_data})
            mr_prob = MeanReversionEnhancements.calculate_mean_reversion_probability(df, price_data[-1])

            logger.info(f"✅ Mean reversion probability: {mr_prob['reversion_probability']:.1%}")
            return mr_prob

        except Exception as e:
            logger.error(f"Error calculating MR probability: {e}")
            return {'reversion_probability': 0.5, 'z_score': 0.0, 'distance_from_mean': 0.0}

    def calculate_dynamic_rsi_thresholds(self, price_data: np.ndarray) -> Dict:
        """
        ENHANCEMENT: Calculate dynamic RSI thresholds based on market conditions

        No more static 30/70! Adapts to trending vs ranging markets.
        """
        try:
            df = pd.DataFrame({'close': price_data})
            rsi_info = MeanReversionEnhancements.calculate_dynamic_rsi_thresholds(df)

            logger.info(f"✅ Dynamic RSI: {rsi_info['oversold_threshold']:.1f} / {rsi_info['overbought_threshold']:.1f}")
            return rsi_info

        except Exception as e:
            logger.error(f"Error calculating dynamic RSI: {e}")
            return {'current_rsi': 50, 'oversold_threshold': 30, 'overbought_threshold': 70}


class ZScoreAnalyzer:
    """Analyzes Z-score for mean reversion signals"""
    
    def __init__(self):
        self.indicator_library = IndicatorLibrary()
    
    def calculate_zscore_signals(self, price_data: np.ndarray, 
                               period: int = 20, entry_threshold: float = 2.0,
                               exit_threshold: float = 0.5) -> List[TechnicalSignal]:
        """Calculate Z-score mean reversion signals"""
        signals = []
        
        try:
            # Calculate Z-score
            zscore_result = calculate_z_score(price_data, period=period)
            zscore_values = zscore_result.values
            
            if len(zscore_values) < 2:
                return signals
            
            current_zscore = zscore_values[-1]
            prev_zscore = zscore_values[-2] if len(zscore_values) > 1 else current_zscore
            
            # Skip NaN values
            if np.isnan(current_zscore) or np.isnan(prev_zscore):
                return signals
            
            # Strong mean reversion signals (Z-score > entry_threshold)
            if current_zscore >= entry_threshold:
                strength = min((current_zscore - entry_threshold) / entry_threshold + 0.5, 1.0)
                signals.append(TechnicalSignal(
                    indicator="ZScore_High",
                    signal_type=SignalType.SELL,
                    strength=strength,
                    confidence=0.85,
                    value=-strength,
                    explanation=f"Z-score extremely high at {current_zscore:.2f} (threshold: {entry_threshold})",
                    timestamp=datetime.utcnow()
                ))
            
            elif current_zscore <= -entry_threshold:
                strength = min((abs(current_zscore) - entry_threshold) / entry_threshold + 0.5, 1.0)
                signals.append(TechnicalSignal(
                    indicator="ZScore_Low",
                    signal_type=SignalType.BUY,
                    strength=strength,
                    confidence=0.85,
                    value=strength,
                    explanation=f"Z-score extremely low at {current_zscore:.2f} (threshold: {-entry_threshold})",
                    timestamp=datetime.utcnow()
                ))
            
            # Moderate mean reversion signals
            elif current_zscore >= 1.0:
                strength = min(current_zscore / entry_threshold, 0.7)
                signals.append(TechnicalSignal(
                    indicator="ZScore_Moderate_High",
                    signal_type=SignalType.SELL,
                    strength=strength,
                    confidence=0.6,
                    value=-strength,
                    explanation=f"Z-score moderately high at {current_zscore:.2f}",
                    timestamp=datetime.utcnow()
                ))
            
            elif current_zscore <= -1.0:
                strength = min(abs(current_zscore) / entry_threshold, 0.7)
                signals.append(TechnicalSignal(
                    indicator="ZScore_Moderate_Low",
                    signal_type=SignalType.BUY,
                    strength=strength,
                    confidence=0.6,
                    value=strength,
                    explanation=f"Z-score moderately low at {current_zscore:.2f}",
                    timestamp=datetime.utcnow()
                ))
            
            # Mean reversion completion signals (Z-score approaching zero)
            if abs(prev_zscore) > exit_threshold and abs(current_zscore) <= exit_threshold:
                if prev_zscore > 0:  # Was high, now normalizing
                    signals.append(TechnicalSignal(
                        indicator="ZScore_Normalization",
                        signal_type=SignalType.HOLD,
                        strength=0.3,
                        confidence=0.7,
                        value=0.0,
                        explanation=f"Z-score normalizing from high ({prev_zscore:.2f} to {current_zscore:.2f})",
                        timestamp=datetime.utcnow()
                    ))
                else:  # Was low, now normalizing
                    signals.append(TechnicalSignal(
                        indicator="ZScore_Normalization",
                        signal_type=SignalType.HOLD,
                        strength=0.3,
                        confidence=0.7,
                        value=0.0,
                        explanation=f"Z-score normalizing from low ({prev_zscore:.2f} to {current_zscore:.2f})",
                        timestamp=datetime.utcnow()
                    ))
            
            # Z-score momentum (rate of change)
            zscore_momentum = current_zscore - prev_zscore
            if abs(zscore_momentum) > 0.5:  # Significant momentum
                if zscore_momentum > 0 and current_zscore > 0:  # Accelerating upward
                    signals.append(TechnicalSignal(
                        indicator="ZScore_Momentum",
                        signal_type=SignalType.SELL,
                        strength=min(zscore_momentum, 0.5),
                        confidence=0.5,
                        value=-min(zscore_momentum, 0.5),
                        explanation=f"Z-score momentum accelerating upward ({zscore_momentum:.2f})",
                        timestamp=datetime.utcnow()
                    ))
                elif zscore_momentum < 0 and current_zscore < 0:  # Accelerating downward
                    signals.append(TechnicalSignal(
                        indicator="ZScore_Momentum",
                        signal_type=SignalType.BUY,
                        strength=min(abs(zscore_momentum), 0.5),
                        confidence=0.5,
                        value=min(abs(zscore_momentum), 0.5),
                        explanation=f"Z-score momentum accelerating downward ({zscore_momentum:.2f})",
                        timestamp=datetime.utcnow()
                    ))
        
        except Exception as e:
            logger.error(f"Error calculating Z-score signals: {e}")
        
        return signals


class PairsTradingAnalyzer:
    """Analyzes pairs trading opportunities with cointegration detection"""
    
    def __init__(self, cointegration_threshold: float = 0.05):
        """
        Initialize pairs trading analyzer
        
        Args:
            cointegration_threshold: P-value threshold for cointegration test
        """
        self.cointegration_threshold = cointegration_threshold
    
    def test_cointegration(self, price_series_a: np.ndarray, 
                          price_series_b: np.ndarray) -> Tuple[float, float, float]:
        """
        Test for cointegration between two price series
        
        Args:
            price_series_a: Price series for asset A
            price_series_b: Price series for asset B
        
        Returns:
            Tuple of (cointegration_stat, p_value, hedge_ratio)
        """
        try:
            # Remove NaN values
            valid_indices = ~(np.isnan(price_series_a) | np.isnan(price_series_b))
            clean_a = price_series_a[valid_indices]
            clean_b = price_series_b[valid_indices]
            
            if len(clean_a) < 10:  # Need minimum data points
                return 0.0, 1.0, 1.0
            
            # Perform cointegration test
            coint_stat, p_value, _ = coint(clean_a, clean_b)
            
            # Calculate hedge ratio using linear regression
            reg = LinearRegression()
            reg.fit(clean_b.reshape(-1, 1), clean_a)
            hedge_ratio = reg.coef_[0]
            
            return coint_stat, p_value, hedge_ratio
        
        except Exception as e:
            logger.error(f"Error in cointegration test: {e}")
            return 0.0, 1.0, 1.0
    
    def calculate_spread(self, price_series_a: np.ndarray, 
                        price_series_b: np.ndarray, hedge_ratio: float) -> np.ndarray:
        """Calculate spread between two cointegrated assets"""
        return price_series_a - hedge_ratio * price_series_b
    
    def calculate_pairs_signals(self, symbol_a: str, price_series_a: np.ndarray,
                              symbol_b: str, price_series_b: np.ndarray,
                              lookback_period: int = 60) -> List[PairsSignal]:
        """Calculate pairs trading signals"""
        signals = []
        
        try:
            # Test for cointegration
            coint_stat, p_value, hedge_ratio = self.test_cointegration(
                price_series_a, price_series_b
            )
            
            # Only proceed if assets are cointegrated
            if p_value <= self.cointegration_threshold:
                # Calculate spread
                spread = self.calculate_spread(price_series_a, price_series_b, hedge_ratio)
                
                # Calculate Z-score of spread
                if len(spread) >= lookback_period:
                    recent_spread = spread[-lookback_period:]
                    spread_mean = np.mean(recent_spread)
                    spread_std = np.std(recent_spread)
                    
                    if spread_std > 0:
                        current_spread = spread[-1]
                        spread_zscore = (current_spread - spread_mean) / spread_std
                        
                        # Generate signals based on spread Z-score
                        if spread_zscore >= 2.0:  # Spread too high, short A, long B
                            confidence = min(0.9, 0.5 + (spread_zscore - 2.0) * 0.1)
                            signals.append(PairsSignal(
                                symbol_a=symbol_a,
                                symbol_b=symbol_b,
                                spread=current_spread,
                                z_score=spread_zscore,
                                cointegration_pvalue=p_value,
                                hedge_ratio=hedge_ratio,
                                signal_type=SignalType.SELL,  # Sell A, Buy B
                                confidence=confidence,
                                explanation=f"Spread Z-score high ({spread_zscore:.2f}): Short {symbol_a}, Long {symbol_b}"
                            ))
                        
                        elif spread_zscore <= -2.0:  # Spread too low, long A, short B
                            confidence = min(0.9, 0.5 + (abs(spread_zscore) - 2.0) * 0.1)
                            signals.append(PairsSignal(
                                symbol_a=symbol_a,
                                symbol_b=symbol_b,
                                spread=current_spread,
                                z_score=spread_zscore,
                                cointegration_pvalue=p_value,
                                hedge_ratio=hedge_ratio,
                                signal_type=SignalType.BUY,  # Buy A, Sell B
                                confidence=confidence,
                                explanation=f"Spread Z-score low ({spread_zscore:.2f}): Long {symbol_a}, Short {symbol_b}"
                            ))
                        
                        elif 1.0 <= abs(spread_zscore) < 2.0:  # Moderate signals
                            confidence = 0.3 + abs(spread_zscore) * 0.1
                            signal_type = SignalType.SELL if spread_zscore > 0 else SignalType.BUY
                            action = "Short A, Long B" if spread_zscore > 0 else "Long A, Short B"
                            
                            signals.append(PairsSignal(
                                symbol_a=symbol_a,
                                symbol_b=symbol_b,
                                spread=current_spread,
                                z_score=spread_zscore,
                                cointegration_pvalue=p_value,
                                hedge_ratio=hedge_ratio,
                                signal_type=signal_type,
                                confidence=confidence,
                                explanation=f"Moderate spread divergence ({spread_zscore:.2f}): {action}"
                            ))
                        
                        elif abs(spread_zscore) <= 0.5:  # Spread normalizing
                            signals.append(PairsSignal(
                                symbol_a=symbol_a,
                                symbol_b=symbol_b,
                                spread=current_spread,
                                z_score=spread_zscore,
                                cointegration_pvalue=p_value,
                                hedge_ratio=hedge_ratio,
                                signal_type=SignalType.HOLD,
                                confidence=0.6,
                                explanation=f"Spread normalizing ({spread_zscore:.2f}): Consider closing positions"
                            ))
        
        except Exception as e:
            logger.error(f"Error calculating pairs signals: {e}")
        
        return signals


class FibonacciTargetCalculator:
    """Calculates Fibonacci extension targets for mean reversion exits"""
    
    def __init__(self):
        self.fibonacci_analyzer = FibonacciAnalyzer()
    
    def calculate_fibonacci_targets(self, high_data: np.ndarray, low_data: np.ndarray,
                                  close_data: np.ndarray, current_price: float,
                                  signal_direction: str) -> List[FibonacciTarget]:
        """Calculate Fibonacci extension targets for exits"""
        targets = []
        
        try:
            # Perform Fibonacci analysis
            fib_analysis = analyze_fibonacci_levels(high_data, low_data, close_data)
            
            # Process Fibonacci extensions for exit targets
            for fib_ext in fib_analysis.get('fibonacci_extensions', []):
                extension_levels = fib_ext.get('extension_levels', {})
                
                for level_name, target_price in extension_levels.items():
                    distance_pct = abs(target_price - current_price) / current_price * 100
                    
                    # Only consider reasonable targets (within 10% for mean reversion)
                    if 1.0 <= distance_pct <= 10.0:
                        # Check if target aligns with signal direction
                        target_direction = "bullish" if target_price > current_price else "bearish"
                        
                        if ((signal_direction == "buy" and target_direction == "bullish") or
                            (signal_direction == "sell" and target_direction == "bearish")):
                            
                            # Calculate confidence based on distance and level importance
                            if level_name in ['ext_1272', 'ext_1618']:  # Key levels
                                base_confidence = 0.8
                            else:
                                base_confidence = 0.6
                            
                            # Adjust confidence based on distance
                            distance_factor = max(0.3, 1.0 - (distance_pct - 1.0) / 9.0)
                            final_confidence = base_confidence * distance_factor
                            
                            targets.append(FibonacciTarget(
                                level_name=level_name,
                                target_price=target_price,
                                distance_pct=distance_pct,
                                confidence=final_confidence,
                                explanation=f"Fibonacci {level_name} target at {target_price:.2f} ({distance_pct:.1f}% away)"
                            ))
            
            # Sort targets by confidence (highest first)
            targets.sort(key=lambda x: x.confidence, reverse=True)
            
            # Limit to top 3 targets
            return targets[:3]
        
        except Exception as e:
            logger.error(f"Error calculating Fibonacci targets: {e}")
            return []


class SentimentDivergenceDetector:
    """Detects sentiment divergence for mean reversion confirmation"""
    
    def calculate_sentiment_divergence(self, price_data: np.ndarray, 
                                     sentiment_data: List[float],
                                     lookback_period: int = 10) -> Optional[float]:
        """
        Calculate sentiment divergence score
        
        Args:
            price_data: Recent price data
            sentiment_data: Recent sentiment scores
            lookback_period: Number of periods to analyze
        
        Returns:
            Divergence score (-1 to 1, where negative indicates bearish divergence)
        """
        try:
            if len(price_data) < lookback_period or len(sentiment_data) < lookback_period:
                return None
            
            # Get recent data
            recent_prices = price_data[-lookback_period:]
            recent_sentiment = sentiment_data[-lookback_period:]
            
            # Calculate price trend
            price_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            price_trend = price_slope / np.mean(recent_prices)  # Normalize
            
            # Calculate sentiment trend
            sentiment_slope = np.polyfit(range(len(recent_sentiment)), recent_sentiment, 1)[0]
            
            # Calculate divergence
            # Positive divergence: price down, sentiment up (bullish for mean reversion)
            # Negative divergence: price up, sentiment down (bearish for mean reversion)
            
            if price_trend > 0.01 and sentiment_slope < -0.01:  # Price up, sentiment down
                divergence = -min(abs(price_trend) + abs(sentiment_slope), 1.0)
            elif price_trend < -0.01 and sentiment_slope > 0.01:  # Price down, sentiment up
                divergence = min(abs(price_trend) + abs(sentiment_slope), 1.0)
            else:
                # No significant divergence
                divergence = 0.0
            
            return divergence
        
        except Exception as e:
            logger.error(f"Error calculating sentiment divergence: {e}")
            return None


class MarketRegimeDetector:
    """Detects market regime for mean reversion strategy optimization"""
    
    def detect_regime(self, price_data: np.ndarray, volume_data: np.ndarray,
                     lookback_period: int = 30) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(price_data) < lookback_period:
                return MarketRegime.SIDEWAYS
            
            recent_prices = price_data[-lookback_period:]
            recent_volumes = volume_data[-lookback_period:] if len(volume_data) >= lookback_period else None
            
            # Calculate volatility
            returns = np.diff(np.log(recent_prices))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate trend strength
            trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            trend_strength = abs(trend_slope) / np.mean(recent_prices)
            
            # Determine regime
            if volatility > 0.4:  # High volatility
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.15:  # Low volatility
                return MarketRegime.LOW_VOLATILITY
            elif trend_strength > 0.02:  # Strong trend
                return MarketRegime.TRENDING
            else:
                return MarketRegime.MEAN_REVERTING
        
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.SIDEWAYS


class ExplainabilityEngine:
    """Generates explainable AI output with top-3 reasons"""
    
    def generate_top_3_reasons(self, bollinger_signals: List[TechnicalSignal],
                             zscore_signals: List[TechnicalSignal],
                             pairs_signals: List[PairsSignal],
                             fibonacci_targets: List[FibonacciTarget],
                             sentiment_divergence: Optional[float],
                             market_regime: MarketRegime,
                             final_signal_value: float) -> List[Reason]:
        """Generate top 3 reasons for the trading decision"""
        all_factors = []
        
        # Bollinger Band factors
        for signal in bollinger_signals:
            contribution = abs(signal.value) * signal.confidence
            all_factors.append({
                'factor': f"Bollinger Bands ({signal.indicator})",
                'contribution': contribution,
                'explanation': signal.explanation,
                'confidence': signal.confidence,
                'supporting_data': signal.to_dict()
            })
        
        # Z-score factors
        for signal in zscore_signals:
            contribution = abs(signal.value) * signal.confidence
            all_factors.append({
                'factor': f"Z-Score Analysis ({signal.indicator})",
                'contribution': contribution,
                'explanation': signal.explanation,
                'confidence': signal.confidence,
                'supporting_data': signal.to_dict()
            })
        
        # Pairs trading factors
        for signal in pairs_signals:
            contribution = abs(signal.z_score) * signal.confidence / 3.0  # Normalize
            all_factors.append({
                'factor': f"Pairs Trading ({signal.symbol_a}/{signal.symbol_b})",
                'contribution': contribution,
                'explanation': signal.explanation,
                'confidence': signal.confidence,
                'supporting_data': signal.to_dict()
            })
        
        # Fibonacci target factors
        if fibonacci_targets:
            avg_confidence = np.mean([target.confidence for target in fibonacci_targets])
            target_count = len(fibonacci_targets)
            all_factors.append({
                'factor': 'Fibonacci Extension Targets',
                'contribution': avg_confidence * 0.7,
                'explanation': f"{target_count} Fibonacci extension targets identified",
                'confidence': avg_confidence,
                'supporting_data': {'fibonacci_targets': [t.to_dict() for t in fibonacci_targets]}
            })
        
        # Sentiment divergence factor
        if sentiment_divergence is not None and abs(sentiment_divergence) > 0.1:
            divergence_contribution = abs(sentiment_divergence) * 0.8
            divergence_type = "bullish" if sentiment_divergence > 0 else "bearish"
            all_factors.append({
                'factor': 'Sentiment Divergence',
                'contribution': divergence_contribution,
                'explanation': f"{divergence_type.capitalize()} sentiment divergence detected ({sentiment_divergence:.2f})",
                'confidence': 0.7,
                'supporting_data': {'sentiment_divergence': sentiment_divergence}
            })
        
        # Market regime factor
        regime_contribution = 0.5 if market_regime == MarketRegime.MEAN_REVERTING else 0.3
        all_factors.append({
            'factor': 'Market Regime',
            'contribution': regime_contribution,
            'explanation': f"Market regime: {market_regime.value}",
            'confidence': 0.6,
            'supporting_data': {'market_regime': market_regime.value}
        })
        
        # Sort by contribution and take top 3
        all_factors.sort(key=lambda x: x['contribution'], reverse=True)
        top_3_factors = all_factors[:3]
        
        # Convert to Reason objects
        reasons = []
        for i, factor in enumerate(top_3_factors):
            reasons.append(Reason(
                rank=i + 1,
                factor=factor['factor'],
                contribution=factor['contribution'],
                explanation=factor['explanation'],
                confidence=factor['confidence'],
                supporting_data=factor['supporting_data']
            ))
        
        return reasons


class MeanReversionTradingAgent:
    """Main Mean Reversion Trading Agent using LangGraph"""
    
    def __init__(self):
        self.bollinger_analyzer = BollingerBandAnalyzer()
        self.zscore_analyzer = ZScoreAnalyzer()
        self.pairs_analyzer = PairsTradingAnalyzer()
        self.fibonacci_calculator = FibonacciTargetCalculator()
        self.sentiment_detector = SentimentDivergenceDetector()
        self.regime_detector = MarketRegimeDetector()
        self.explainability_engine = ExplainabilityEngine()
        self.model_version = "1.0.0"
        
        # Initialize LangGraph
        self.graph = self._create_langgraph()
    
    def _create_langgraph(self) -> StateGraph:
        """Create LangGraph state machine for mean reversion trading"""
        from typing import TypedDict
        
        class MeanReversionState(TypedDict):
            symbol: str
            market_data: List[MarketData]
            sentiment_data: Optional[SentimentData]
            pairs_data: Optional[Dict[str, List[MarketData]]]
            bollinger_signals: List[TechnicalSignal]
            zscore_signals: List[TechnicalSignal]
            pairs_signals: List[PairsSignal]
            fibonacci_targets: List[FibonacciTarget]
            sentiment_divergence: Optional[float]
            market_regime: Optional[MarketRegime]
            final_signal: Optional[MeanReversionSignal]
            error: Optional[str]
        
        # Create state graph
        workflow = StateGraph(MeanReversionState)
        
        # Add nodes
        workflow.add_node("analyze_bollinger", self._analyze_bollinger_bands)
        workflow.add_node("analyze_zscore", self._analyze_zscore)
        workflow.add_node("analyze_pairs", self._analyze_pairs_trading)
        workflow.add_node("calculate_fibonacci", self._calculate_fibonacci_targets)
        workflow.add_node("detect_sentiment_divergence", self._detect_sentiment_divergence)
        workflow.add_node("detect_regime", self._detect_market_regime)
        workflow.add_node("generate_signal", self._generate_final_signal)
        workflow.add_node("store_signal", self._store_signal)
        
        # Define edges
        workflow.add_edge("analyze_bollinger", "analyze_zscore")
        workflow.add_edge("analyze_zscore", "analyze_pairs")
        workflow.add_edge("analyze_pairs", "calculate_fibonacci")
        workflow.add_edge("calculate_fibonacci", "detect_sentiment_divergence")
        workflow.add_edge("detect_sentiment_divergence", "detect_regime")
        workflow.add_edge("detect_regime", "generate_signal")
        workflow.add_edge("generate_signal", "store_signal")
        workflow.add_edge("store_signal", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_bollinger")
        
        return workflow.compile()
    
    async def _analyze_bollinger_bands(self, state: Dict) -> Dict:
        """Analyze Bollinger Band signals"""
        try:
            market_data = state.get('market_data', [])
            if not market_data:
                state['error'] = "No market data available"
                return state
            
            # Extract price data
            close_prices = np.array([data.close for data in market_data])
            
            # Calculate Bollinger Band signals
            bollinger_signals = self.bollinger_analyzer.calculate_bollinger_signals(close_prices)
            state['bollinger_signals'] = bollinger_signals
            
            logger.info(f"Generated {len(bollinger_signals)} Bollinger Band signals")
            
        except Exception as e:
            logger.error(f"Error in Bollinger Band analysis: {e}")
            state['error'] = str(e)
        
        return state
    
    async def _analyze_zscore(self, state: Dict) -> Dict:
        """Analyze Z-score signals"""
        try:
            market_data = state.get('market_data', [])
            if not market_data:
                return state
            
            # Extract price data
            close_prices = np.array([data.close for data in market_data])
            
            # Calculate Z-score signals
            zscore_signals = self.zscore_analyzer.calculate_zscore_signals(close_prices)
            state['zscore_signals'] = zscore_signals
            
            logger.info(f"Generated {len(zscore_signals)} Z-score signals")
            
        except Exception as e:
            logger.error(f"Error in Z-score analysis: {e}")
            state['error'] = str(e)
        
        return state
    
    async def _analyze_pairs_trading(self, state: Dict) -> Dict:
        """Analyze pairs trading opportunities"""
        try:
            symbol = state.get('symbol', '')
            market_data = state.get('market_data', [])
            pairs_data = state.get('pairs_data', {})
            
            pairs_signals = []
            
            if pairs_data:
                # Extract price data for main symbol
                main_prices = np.array([data.close for data in market_data])
                
                # Analyze pairs with other symbols
                for pair_symbol, pair_market_data in pairs_data.items():
                    if pair_symbol != symbol and pair_market_data:
                        pair_prices = np.array([data.close for data in pair_market_data])
                        
                        # Ensure same length
                        min_length = min(len(main_prices), len(pair_prices))
                        if min_length >= 30:  # Minimum data for pairs analysis
                            main_subset = main_prices[-min_length:]
                            pair_subset = pair_prices[-min_length:]
                            
                            # Calculate pairs signals
                            pair_signals = self.pairs_analyzer.calculate_pairs_signals(
                                symbol, main_subset, pair_symbol, pair_subset
                            )
                            pairs_signals.extend(pair_signals)
            
            state['pairs_signals'] = pairs_signals
            logger.info(f"Generated {len(pairs_signals)} pairs trading signals")
            
        except Exception as e:
            logger.error(f"Error in pairs trading analysis: {e}")
            state['error'] = str(e)
        
        return state
    
    async def _calculate_fibonacci_targets(self, state: Dict) -> Dict:
        """Calculate Fibonacci extension targets"""
        try:
            market_data = state.get('market_data', [])
            if not market_data:
                return state
            
            # Extract OHLC data
            high_prices = np.array([data.high for data in market_data])
            low_prices = np.array([data.low for data in market_data])
            close_prices = np.array([data.close for data in market_data])
            current_price = close_prices[-1]
            
            # Determine signal direction from existing signals
            bollinger_signals = state.get('bollinger_signals', [])
            zscore_signals = state.get('zscore_signals', [])
            
            signal_direction = "hold"
            if bollinger_signals or zscore_signals:
                total_signal_value = 0
                for signal in bollinger_signals + zscore_signals:
                    total_signal_value += signal.value
                
                if total_signal_value > 0.1:
                    signal_direction = "buy"
                elif total_signal_value < -0.1:
                    signal_direction = "sell"
            
            # Calculate Fibonacci targets
            fibonacci_targets = self.fibonacci_calculator.calculate_fibonacci_targets(
                high_prices, low_prices, close_prices, current_price, signal_direction
            )
            
            state['fibonacci_targets'] = fibonacci_targets
            logger.info(f"Generated {len(fibonacci_targets)} Fibonacci targets")
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci targets: {e}")
            state['error'] = str(e)
        
        return state
    
    async def _detect_sentiment_divergence(self, state: Dict) -> Dict:
        """Detect sentiment divergence"""
        try:
            market_data = state.get('market_data', [])
            sentiment_data = state.get('sentiment_data')
            
            sentiment_divergence = None
            
            if sentiment_data and market_data:
                # For this implementation, we'll use a simplified approach
                # In a real system, you'd have historical sentiment data
                close_prices = np.array([data.close for data in market_data])
                
                # Mock sentiment history (in real implementation, fetch from database)
                sentiment_history = [sentiment_data.overall_sentiment] * min(10, len(close_prices))
                
                sentiment_divergence = self.sentiment_detector.calculate_sentiment_divergence(
                    close_prices, sentiment_history
                )
            
            state['sentiment_divergence'] = sentiment_divergence
            logger.info(f"Sentiment divergence: {sentiment_divergence}")
            
        except Exception as e:
            logger.error(f"Error detecting sentiment divergence: {e}")
            state['error'] = str(e)
        
        return state
    
    async def _detect_market_regime(self, state: Dict) -> Dict:
        """Detect market regime"""
        try:
            market_data = state.get('market_data', [])
            if not market_data:
                return state
            
            close_prices = np.array([data.close for data in market_data])
            volume_data = np.array([data.volume for data in market_data])
            
            market_regime = self.regime_detector.detect_regime(close_prices, volume_data)
            state['market_regime'] = market_regime
            
            logger.info(f"Detected market regime: {market_regime.value}")
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            state['error'] = str(e)
        
        return state
    
    async def _generate_final_signal(self, state: Dict) -> Dict:
        """Generate final mean reversion signal"""
        try:
            symbol = state.get('symbol', '')
            bollinger_signals = state.get('bollinger_signals', [])
            zscore_signals = state.get('zscore_signals', [])
            pairs_signals = state.get('pairs_signals', [])
            fibonacci_targets = state.get('fibonacci_targets', [])
            sentiment_divergence = state.get('sentiment_divergence')
            market_regime = state.get('market_regime', MarketRegime.SIDEWAYS)
            
            # Calculate overall signal value and confidence
            total_signal_value = 0.0
            total_weight = 0.0
            
            # Weight Bollinger Band signals
            for signal in bollinger_signals:
                weight = signal.confidence * 0.3  # 30% weight
                total_signal_value += signal.value * weight
                total_weight += weight
            
            # Weight Z-score signals (higher weight as they're more reliable for mean reversion)
            for signal in zscore_signals:
                weight = signal.confidence * 0.4  # 40% weight
                total_signal_value += signal.value * weight
                total_weight += weight
            
            # Weight pairs signals
            for signal in pairs_signals:
                weight = signal.confidence * 0.2  # 20% weight
                signal_value = 1.0 if signal.signal_type == SignalType.BUY else -1.0
                total_signal_value += signal_value * weight
                total_weight += weight
            
            # Apply sentiment divergence adjustment
            if sentiment_divergence is not None:
                divergence_weight = 0.1  # 10% weight
                total_signal_value += sentiment_divergence * divergence_weight
                total_weight += divergence_weight
            
            # Normalize signal value
            if total_weight > 0:
                final_signal_value = total_signal_value / total_weight
            else:
                final_signal_value = 0.0
            
            # Clamp to [-1, 1] range
            final_signal_value = max(-1.0, min(1.0, final_signal_value))
            
            # Calculate confidence
            base_confidence = min(total_weight, 1.0)
            
            # Adjust confidence based on market regime
            if market_regime == MarketRegime.MEAN_REVERTING:
                regime_multiplier = 1.2
            elif market_regime == MarketRegime.TRENDING:
                regime_multiplier = 0.7  # Mean reversion less reliable in trending markets
            else:
                regime_multiplier = 1.0
            
            final_confidence = min(base_confidence * regime_multiplier, 1.0)
            
            # Determine signal type
            if final_signal_value >= 0.3:
                signal_type = SignalType.STRONG_BUY if final_signal_value >= 0.7 else SignalType.BUY
            elif final_signal_value <= -0.3:
                signal_type = SignalType.STRONG_SELL if final_signal_value <= -0.7 else SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Generate top 3 reasons
            top_3_reasons = self.explainability_engine.generate_top_3_reasons(
                bollinger_signals, zscore_signals, pairs_signals, fibonacci_targets,
                sentiment_divergence, market_regime, final_signal_value
            )
            
            # Calculate risk metrics
            stop_loss_pct = 0.03 if market_regime == MarketRegime.HIGH_VOLATILITY else 0.02  # 2-3%
            
            # Take profit targets from Fibonacci levels
            take_profit_targets = []
            if fibonacci_targets:
                current_price = state.get('market_data', [])[-1].close if state.get('market_data') else 100
                for target in fibonacci_targets[:2]:  # Top 2 targets
                    profit_pct = abs(target.target_price - current_price) / current_price
                    take_profit_targets.append(profit_pct)
            
            if not take_profit_targets:
                take_profit_targets = [0.02, 0.04]  # Default 2% and 4%
            
            # Create final signal
            final_signal = MeanReversionSignal(
                symbol=symbol,
                signal_type=signal_type,
                value=final_signal_value,
                confidence=final_confidence,
                top_3_reasons=top_3_reasons,
                timestamp=datetime.utcnow(),
                model_version=self.model_version,
                bollinger_signals=bollinger_signals,
                zscore_signals=zscore_signals,
                pairs_signals=pairs_signals,
                fibonacci_targets=fibonacci_targets,
                sentiment_divergence=sentiment_divergence,
                market_regime=market_regime,
                position_size_pct=min(0.05 * final_confidence, 0.08),  # Max 8% position
                stop_loss_pct=stop_loss_pct,
                take_profit_targets=take_profit_targets,
                max_holding_period=5  # 5 days max for mean reversion
            )
            
            state['final_signal'] = final_signal
            logger.info(f"Generated final signal: {signal_type.value} with confidence {final_confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error generating final signal: {e}")
            state['error'] = str(e)
        
        return state
    
    async def _store_signal(self, state: Dict) -> Dict:
        """Store signal in database"""
        try:
            final_signal = state.get('final_signal')
            if not final_signal:
                return state
            
            # In a real implementation, store in database
            # For now, just log the signal
            logger.info(f"Storing mean reversion signal: {final_signal.to_dict()}")
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
            state['error'] = str(e)
        
        return state
    
    async def find_mean_reversion_opportunity(self, symbol: str, market_data: List[MarketData] = None,
                            sentiment_data: Optional[SentimentData] = None,
                            pairs_data: Optional[Dict[str, List[MarketData]]] = None) -> Optional[MeanReversionSignal]:
        """
        Generate mean reversion trading signal
        
        Args:
            symbol: Trading symbol
            market_data: Historical market data
            sentiment_data: Optional sentiment data
            pairs_data: Optional pairs trading data
        
        Returns:
            MeanReversionSignal or None if error
        """
        try:
            # Prepare initial state
            initial_state = {
                'symbol': symbol,
                'market_data': market_data,
                'sentiment_data': sentiment_data,
                'pairs_data': pairs_data or {},
                'bollinger_signals': [],
                'zscore_signals': [],
                'pairs_signals': [],
                'fibonacci_targets': [],
                'sentiment_divergence': None,
                'market_regime': None,
                'final_signal': None,
                'error': None
            }
            
            # Execute LangGraph workflow
            result = await self.graph.ainvoke(initial_state)
            
            if result.get('error'):
                logger.error(f"Error in mean reversion analysis: {result['error']}")
                return None
            
            return result.get('final_signal')
        
        except Exception as e:
            logger.error(f"Error in generate_signal: {e}")
            return None
    
    def generate_signal_sync(self, symbol: str, market_data: List[MarketData],
                           sentiment_data: Optional[SentimentData] = None,
                           pairs_data: Optional[Dict[str, List[MarketData]]] = None) -> Optional[MeanReversionSignal]:
        """Synchronous version of generate_signal"""
        return asyncio.run(self.generate_signal(symbol, market_data, sentiment_data, pairs_data))


# Convenience functions for direct access
async def analyze_mean_reversion(symbol: str, market_data: List[MarketData],
                               sentiment_data: Optional[SentimentData] = None,
                               pairs_data: Optional[Dict[str, List[MarketData]]] = None) -> Optional[MeanReversionSignal]:
    """
    Analyze mean reversion opportunities for a symbol
    
    Args:
        symbol: Trading symbol
        market_data: Historical market data
        sentiment_data: Optional sentiment data
        pairs_data: Optional pairs trading data
    
    Returns:
        MeanReversionSignal or None
    """
    agent = MeanReversionTradingAgent()
    return await agent.generate_signal(symbol, market_data, sentiment_data, pairs_data)


def analyze_mean_reversion_sync(symbol: str, market_data: List[MarketData],
                              sentiment_data: Optional[SentimentData] = None,
                              pairs_data: Optional[Dict[str, List[MarketData]]] = None) -> Optional[MeanReversionSignal]:
    """Synchronous version of analyze_mean_reversion"""
    return asyncio.run(analyze_mean_reversion(symbol, market_data, sentiment_data, pairs_data))


if __name__ == "__main__":
    # Example usage
    import random
    from datetime import datetime, timedelta
    
    # Generate sample market data
    sample_data = []
    base_price = 100.0
    
    for i in range(100):
        # Add some mean-reverting behavior
        if i > 0:
            prev_price = sample_data[-1].close
            # Mean reversion towards 100
            mean_reversion = (100 - prev_price) * 0.1
            random_change = random.gauss(0, 1)
            price_change = mean_reversion + random_change
            base_price = max(50, min(150, prev_price + price_change))
        
        sample_data.append(MarketData(
            symbol="AAPL",
            timestamp=datetime.now() - timedelta(days=100-i),
            open=base_price + random.gauss(0, 0.5),
            high=base_price + abs(random.gauss(0, 1)),
            low=base_price - abs(random.gauss(0, 1)),
            close=base_price,
            volume=random.randint(1000000, 5000000)
        ))
    
    # Generate sample sentiment data
    sample_sentiment = SentimentData(
        symbol="AAPL",
        overall_sentiment=random.gauss(0, 0.3),
        confidence=random.uniform(0.5, 0.9),
        news_count=random.randint(5, 20)
    )
    
    # Test the agent
    agent = MeanReversionTradingAgent()
    signal = agent.generate_signal_sync("AAPL", sample_data, sample_sentiment)
    
    if signal:
        print("Mean Reversion Signal Generated:")
        print(f"Symbol: {signal.symbol}")
        print(f"Signal Type: {signal.signal_type.value}")
        print(f"Value: {signal.value:.3f}")
        print(f"Confidence: {signal.confidence:.3f}")
        print(f"Market Regime: {signal.market_regime.value if signal.market_regime else 'Unknown'}")
        print("\nTop 3 Reasons:")
        for reason in signal.top_3_reasons:
            print(f"{reason.rank}. {reason.factor}: {reason.explanation}")
        print(f"\nFibonacci Targets: {len(signal.fibonacci_targets)}")
        print(f"Pairs Signals: {len(signal.pairs_signals)}")
        print(f"Position Size: {signal.position_size_pct:.1%}")
        print(f"Stop Loss: {signal.stop_loss_pct:.1%}")
        print(f"Take Profit Targets: {signal.take_profit_targets}")
    else:
        print("Failed to generate signal")

# Create singleton instance
mean_reversion_agent = MeanReversionTradingAgent()