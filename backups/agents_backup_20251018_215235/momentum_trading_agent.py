"""
Momentum Trading Agent - LangGraph Implementation

This agent implements a comprehensive momentum trading strategy that combines:
- EMA crossovers, RSI breakouts, MACD signals
- Fibonacci retracement levels for entry timing
- Sentiment confirmation for signal strength
- Volatility-adjusted position sizing
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

# LangGraph imports
from langgraph.graph import StateGraph, END

# Technical analysis imports
from strategies.technical_indicators import (
    IndicatorLibrary, calculate_ema, calculate_rsi, calculate_macd
)
from strategies.fibonacci_analysis import (
    FibonacciAnalyzer, analyze_fibonacci_levels
)

# Database imports
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Configuration
from config.settings import settings
from config.secure_config import get_api_keys

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of momentum signals"""
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
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class MarketData:
    """Market data structure for momentum analysis"""
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
class FibonacciSignal:
    """Fibonacci-based signal"""
    level_type: str  # 'retracement' or 'extension'
    level_name: str  # e.g., 'fib_618'
    level_price: float
    current_price: float
    distance_pct: float
    confluence_strength: float
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
class MomentumSignal:
    """Final momentum trading signal with explainability"""
    symbol: str
    signal_type: SignalType
    value: float  # -1 to 1 (negative = sell, positive = buy)
    confidence: float  # 0 to 1
    top_3_reasons: List[Reason]
    timestamp: datetime
    model_version: str
    
    # Technical components
    ema_signals: List[TechnicalSignal]
    rsi_signals: List[TechnicalSignal]
    macd_signals: List[TechnicalSignal]
    fibonacci_signals: List[FibonacciSignal]
    
    # Market context
    sentiment_score: Optional[float] = None
    volatility_adjustment: Optional[float] = None
    market_regime: Optional[MarketRegime] = None
    position_size_pct: Optional[float] = None
    
    # Risk metrics
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
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
            'ema_signals': [signal.to_dict() for signal in self.ema_signals],
            'rsi_signals': [signal.to_dict() for signal in self.rsi_signals],
            'macd_signals': [signal.to_dict() for signal in self.macd_signals],
            'fibonacci_signals': [signal.to_dict() for signal in self.fibonacci_signals],
            'sentiment_score': self.sentiment_score,
            'volatility_adjustment': self.volatility_adjustment,
            'market_regime': self.market_regime.value if self.market_regime else None,
            'position_size_pct': self.position_size_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_holding_period': self.max_holding_period
        }


class TechnicalAnalyzer:
    """Handles all technical indicator calculations"""
    
    def __init__(self):
        self.indicator_library = IndicatorLibrary()
    
    def calculate_ema_signals(self, price_data: np.ndarray, 
                            fast_period: int = 12, slow_period: int = 26) -> List[TechnicalSignal]:
        """Calculate EMA crossover signals"""
        signals = []
        
        try:
            # Calculate fast and slow EMAs
            fast_ema = calculate_ema(price_data, period=fast_period)
            slow_ema = calculate_ema(price_data, period=slow_period)
            
            fast_values = fast_ema.values
            slow_values = slow_ema.values
            
            # Skip NaN values
            valid_idx = max(fast_period, slow_period)
            if len(fast_values) <= valid_idx:
                return signals
            
            # Check for crossovers
            current_fast = fast_values[-1]
            current_slow = slow_values[-1]
            prev_fast = fast_values[-2] if len(fast_values) > 1 else current_fast
            prev_slow = slow_values[-2] if len(slow_values) > 1 else current_slow
            
            # Bullish crossover (fast EMA crosses above slow EMA)
            if prev_fast <= prev_slow and current_fast > current_slow:
                strength = min((current_fast - current_slow) / current_slow * 100, 1.0)
                signals.append(TechnicalSignal(
                    indicator="EMA_Crossover",
                    signal_type=SignalType.BUY,
                    strength=strength,
                    confidence=0.7,
                    value=strength,
                    explanation=f"Fast EMA ({fast_period}) crossed above Slow EMA ({slow_period})",
                    timestamp=datetime.utcnow()
                ))
            
            # Bearish crossover (fast EMA crosses below slow EMA)
            elif prev_fast >= prev_slow and current_fast < current_slow:
                strength = min((current_slow - current_fast) / current_slow * 100, 1.0)
                signals.append(TechnicalSignal(
                    indicator="EMA_Crossover",
                    signal_type=SignalType.SELL,
                    strength=strength,
                    confidence=0.7,
                    value=-strength,
                    explanation=f"Fast EMA ({fast_period}) crossed below Slow EMA ({slow_period})",
                    timestamp=datetime.utcnow()
                ))
            
            # Trend continuation signals
            elif current_fast > current_slow:
                # Uptrend continuation
                trend_strength = min((current_fast - current_slow) / current_slow * 10, 0.5)
                signals.append(TechnicalSignal(
                    indicator="EMA_Trend",
                    signal_type=SignalType.BUY,
                    strength=trend_strength,
                    confidence=0.4,
                    value=trend_strength,
                    explanation=f"Fast EMA above Slow EMA - uptrend continuation",
                    timestamp=datetime.utcnow()
                ))
            
            elif current_fast < current_slow:
                # Downtrend continuation
                trend_strength = min((current_slow - current_fast) / current_slow * 10, 0.5)
                signals.append(TechnicalSignal(
                    indicator="EMA_Trend",
                    signal_type=SignalType.SELL,
                    strength=trend_strength,
                    confidence=0.4,
                    value=-trend_strength,
                    explanation=f"Fast EMA below Slow EMA - downtrend continuation",
                    timestamp=datetime.utcnow()
                ))
        
        except Exception as e:
            logger.error(f"Error calculating EMA signals: {e}")
        
        return signals
    
    def calculate_rsi_signals(self, price_data: np.ndarray, 
                            period: int = 14, oversold: float = 30, 
                            overbought: float = 70) -> List[TechnicalSignal]:
        """Calculate RSI breakout signals"""
        signals = []
        
        try:
            rsi_result = calculate_rsi(price_data, period=period)
            rsi_values = rsi_result.values
            
            if len(rsi_values) < 2:
                return signals
            
            current_rsi = rsi_values[-1]
            prev_rsi = rsi_values[-2]
            
            # Skip NaN values
            if np.isnan(current_rsi) or np.isnan(prev_rsi):
                return signals
            
            # RSI breakout from oversold (bullish)
            if prev_rsi <= oversold and current_rsi > oversold:
                strength = min((current_rsi - oversold) / (50 - oversold), 1.0)
                signals.append(TechnicalSignal(
                    indicator="RSI_Breakout",
                    signal_type=SignalType.BUY,
                    strength=strength,
                    confidence=0.8,
                    value=strength,
                    explanation=f"RSI broke above oversold level ({oversold})",
                    timestamp=datetime.utcnow()
                ))
            
            # RSI breakout from overbought (bearish)
            elif prev_rsi >= overbought and current_rsi < overbought:
                strength = min((overbought - current_rsi) / (overbought - 50), 1.0)
                signals.append(TechnicalSignal(
                    indicator="RSI_Breakout",
                    signal_type=SignalType.SELL,
                    strength=strength,
                    confidence=0.8,
                    value=-strength,
                    explanation=f"RSI broke below overbought level ({overbought})",
                    timestamp=datetime.utcnow()
                ))
            
            # RSI momentum signals
            elif current_rsi > 50 and prev_rsi <= 50:
                # Bullish momentum
                strength = min((current_rsi - 50) / 50, 0.6)
                signals.append(TechnicalSignal(
                    indicator="RSI_Momentum",
                    signal_type=SignalType.BUY,
                    strength=strength,
                    confidence=0.5,
                    value=strength,
                    explanation=f"RSI crossed above 50 - bullish momentum",
                    timestamp=datetime.utcnow()
                ))
            
            elif current_rsi < 50 and prev_rsi >= 50:
                # Bearish momentum
                strength = min((50 - current_rsi) / 50, 0.6)
                signals.append(TechnicalSignal(
                    indicator="RSI_Momentum",
                    signal_type=SignalType.SELL,
                    strength=strength,
                    confidence=0.5,
                    value=-strength,
                    explanation=f"RSI crossed below 50 - bearish momentum",
                    timestamp=datetime.utcnow()
                ))
        
        except Exception as e:
            logger.error(f"Error calculating RSI signals: {e}")
        
        return signals
    
    def calculate_macd_signals(self, price_data: np.ndarray,
                             fast_period: int = 12, slow_period: int = 26,
                             signal_period: int = 9) -> List[TechnicalSignal]:
        """Calculate MACD signals"""
        signals = []
        
        try:
            macd_result = calculate_macd(price_data, fast_period, slow_period, signal_period)
            macd_data = macd_result.values
            
            if len(macd_data) < 2:
                return signals
            
            # Extract MACD line, signal line, and histogram
            current_macd = macd_data[-1, 0]  # MACD line
            current_signal = macd_data[-1, 1]  # Signal line
            current_histogram = macd_data[-1, 2]  # Histogram
            
            prev_macd = macd_data[-2, 0]
            prev_signal = macd_data[-2, 1]
            prev_histogram = macd_data[-2, 2]
            
            # MACD line crosses above signal line (bullish)
            if prev_macd <= prev_signal and current_macd > current_signal:
                strength = min(abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0, 1.0)
                signals.append(TechnicalSignal(
                    indicator="MACD_Crossover",
                    signal_type=SignalType.BUY,
                    strength=strength,
                    confidence=0.75,
                    value=strength,
                    explanation="MACD line crossed above signal line",
                    timestamp=datetime.utcnow()
                ))
            
            # MACD line crosses below signal line (bearish)
            elif prev_macd >= prev_signal and current_macd < current_signal:
                strength = min(abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0, 1.0)
                signals.append(TechnicalSignal(
                    indicator="MACD_Crossover",
                    signal_type=SignalType.SELL,
                    strength=strength,
                    confidence=0.75,
                    value=-strength,
                    explanation="MACD line crossed below signal line",
                    timestamp=datetime.utcnow()
                ))
            
            # MACD histogram momentum
            if current_histogram > 0 and prev_histogram <= 0:
                # Bullish momentum
                strength = min(abs(current_histogram) / 0.1, 0.6)  # Normalize
                signals.append(TechnicalSignal(
                    indicator="MACD_Histogram",
                    signal_type=SignalType.BUY,
                    strength=strength,
                    confidence=0.6,
                    value=strength,
                    explanation="MACD histogram turned positive",
                    timestamp=datetime.utcnow()
                ))
            
            elif current_histogram < 0 and prev_histogram >= 0:
                # Bearish momentum
                strength = min(abs(current_histogram) / 0.1, 0.6)  # Normalize
                signals.append(TechnicalSignal(
                    indicator="MACD_Histogram",
                    signal_type=SignalType.SELL,
                    strength=strength,
                    confidence=0.6,
                    value=-strength,
                    explanation="MACD histogram turned negative",
                    timestamp=datetime.utcnow()
                ))
        
        except Exception as e:
            logger.error(f"Error calculating MACD signals: {e}")
        
        return signals


class FibonacciIntegrator:
    """Integrates Fibonacci analysis with momentum signals"""
    
    def __init__(self):
        self.fibonacci_analyzer = FibonacciAnalyzer()
    
    def calculate_fibonacci_signals(self, high_data: np.ndarray, low_data: np.ndarray,
                                  close_data: np.ndarray, current_price: float) -> List[FibonacciSignal]:
        """Calculate Fibonacci-based signals for entry timing"""
        signals = []
        
        try:
            # Perform comprehensive Fibonacci analysis
            fib_analysis = analyze_fibonacci_levels(high_data, low_data, close_data)
            
            # Process Fibonacci retracement levels
            for fib_level in fib_analysis.get('fibonacci_levels', []):
                for level_name, level_price in fib_level.get('level_prices', {}).items():
                    distance_pct = abs(current_price - level_price) / current_price * 100
                    
                    # Only consider levels within reasonable distance (5%)
                    if distance_pct <= 5.0:
                        signals.append(FibonacciSignal(
                            level_type='retracement',
                            level_name=level_name,
                            level_price=level_price,
                            current_price=current_price,
                            distance_pct=distance_pct,
                            confluence_strength=1.0,  # Base strength
                            explanation=f"Price near Fibonacci {level_name} level at {level_price:.2f}"
                        ))
            
            # Process confluence zones (higher priority)
            for confluence in fib_analysis.get('confluence_zones', []):
                confluence_price = confluence.get('price', 0)
                distance_pct = abs(current_price - confluence_price) / current_price * 100
                
                if distance_pct <= 3.0:  # Tighter tolerance for confluence zones
                    signals.append(FibonacciSignal(
                        level_type='confluence',
                        level_name='confluence_zone',
                        level_price=confluence_price,
                        current_price=current_price,
                        distance_pct=distance_pct,
                        confluence_strength=confluence.get('strength', 1.0),
                        explanation=f"Price near confluence zone at {confluence_price:.2f} with {confluence.get('level_count', 0)} levels"
                    ))
        
        except Exception as e:
            logger.error(f"Error calculating Fibonacci signals: {e}")
        
        return signals
    
    def calculate_confluence_strength(self, current_price: float, 
                                    fibonacci_signals: List[FibonacciSignal]) -> float:
        """Calculate overall confluence strength for current price"""
        if not fibonacci_signals:
            return 0.0
        
        total_strength = 0.0
        weight_sum = 0.0
        
        for signal in fibonacci_signals:
            # Weight by inverse distance (closer levels have more influence)
            distance_weight = max(0.1, 1.0 - (signal.distance_pct / 5.0))
            confluence_weight = signal.confluence_strength
            
            total_strength += confluence_weight * distance_weight
            weight_sum += distance_weight
        
        return min(total_strength / weight_sum if weight_sum > 0 else 0.0, 1.0)


class SentimentIntegrator:
    """Integrates sentiment analysis with momentum signals"""
    
    def calculate_sentiment_alignment(self, signal_direction: float, 
                                    sentiment_score: float) -> float:
        """Calculate how well sentiment aligns with signal direction"""
        if sentiment_score is None:
            return 1.0  # Neutral if no sentiment data
        
        # Both positive
        if signal_direction > 0 and sentiment_score > 0:
            return 1.0 + min(sentiment_score * 0.5, 0.5)  # Boost up to 1.5x
        
        # Both negative
        elif signal_direction < 0 and sentiment_score < 0:
            return 1.0 + min(abs(sentiment_score) * 0.5, 0.5)  # Boost up to 1.5x
        
        # Conflicting signals
        elif (signal_direction > 0 and sentiment_score < 0) or (signal_direction < 0 and sentiment_score > 0):
            conflict_penalty = min(abs(sentiment_score) * 0.3, 0.3)
            return max(0.4, 1.0 - conflict_penalty)  # Reduce but don't eliminate
        
        # Neutral sentiment
        else:
            return 1.0


class VolatilityAdjuster:
    """Handles volatility-adjusted position sizing"""
    
    def calculate_volatility_adjustment(self, price_data: np.ndarray, 
                                      lookback_period: int = 20) -> Tuple[float, MarketRegime]:
        """Calculate volatility adjustment factor and market regime"""
        try:
            if len(price_data) < lookback_period:
                return 1.0, MarketRegime.SIDEWAYS
            
            # Calculate returns
            returns = np.diff(np.log(price_data[-lookback_period:]))
            
            # Calculate volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252)
            
            # Determine market regime
            if volatility > 0.4:  # 40% annualized volatility
                regime = MarketRegime.HIGH_VOLATILITY
                adjustment = 0.5  # Reduce position size in high volatility
            elif volatility < 0.15:  # 15% annualized volatility
                regime = MarketRegime.LOW_VOLATILITY
                adjustment = 1.2  # Increase position size in low volatility
            else:
                regime = MarketRegime.SIDEWAYS
                adjustment = 1.0
            
            # Check for trending behavior
            recent_prices = price_data[-lookback_period:]
            if len(recent_prices) >= 10:
                trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                trend_strength = abs(trend_slope) / np.mean(recent_prices)
                
                if trend_slope > 0 and trend_strength > 0.01:  # 1% trend strength
                    regime = MarketRegime.TRENDING_UP
                elif trend_slope < 0 and trend_strength > 0.01:
                    regime = MarketRegime.TRENDING_DOWN
            
            return adjustment, regime
        
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0, MarketRegime.SIDEWAYS
    
    def calculate_position_size(self, base_size_pct: float, volatility_adjustment: float,
                              confidence: float, account_balance: float) -> float:
        """Calculate optimal position size"""
        # Base position size adjusted for volatility and confidence
        adjusted_size = base_size_pct * volatility_adjustment * confidence
        
        # Apply risk limits
        max_position_pct = 0.1  # Maximum 10% of account per position
        min_position_pct = 0.01  # Minimum 1% of account per position
        
        final_size_pct = max(min_position_pct, min(adjusted_size, max_position_pct))
        
        return final_size_pct


class ExplainabilityEngine:
    """Generates explainable AI output with top-3 reasons"""
    
    def generate_top_3_reasons(self, technical_signals: List[TechnicalSignal],
                             fibonacci_signals: List[FibonacciSignal],
                             sentiment_score: Optional[float],
                             volatility_adjustment: float,
                             final_signal_value: float) -> List[Reason]:
        """Generate top 3 reasons for the trading decision"""
        all_factors = []
        
        # Technical indicator factors
        for signal in technical_signals:
            contribution = abs(signal.value) * signal.confidence
            all_factors.append({
                'factor': f"{signal.indicator}",
                'contribution': contribution,
                'explanation': signal.explanation,
                'confidence': signal.confidence,
                'supporting_data': signal.to_dict()
            })
        
        # Fibonacci factors
        if fibonacci_signals:
            fib_contribution = sum(1.0 / (signal.distance_pct + 1) * signal.confluence_strength 
                                 for signal in fibonacci_signals) / len(fibonacci_signals)
            all_factors.append({
                'factor': 'Fibonacci Confluence',
                'contribution': fib_contribution,
                'explanation': f"Price near {len(fibonacci_signals)} Fibonacci level(s)",
                'confidence': 0.8,
                'supporting_data': {'fibonacci_signals': [s.to_dict() for s in fibonacci_signals]}
            })
        
        # Sentiment factor
        if sentiment_score is not None:
            sentiment_contribution = abs(sentiment_score) * 0.7
            sentiment_direction = "positive" if sentiment_score > 0 else "negative"
            all_factors.append({
                'factor': 'Market Sentiment',
                'contribution': sentiment_contribution,
                'explanation': f"Market sentiment is {sentiment_direction} ({sentiment_score:.2f})",
                'confidence': 0.6,
                'supporting_data': {'sentiment_score': sentiment_score}
            })
        
        # Volatility factor
        if volatility_adjustment != 1.0:
            vol_contribution = abs(volatility_adjustment - 1.0)
            vol_description = "high" if volatility_adjustment < 1.0 else "low"
            all_factors.append({
                'factor': 'Market Volatility',
                'contribution': vol_contribution,
                'explanation': f"Market volatility is {vol_description} (adjustment: {volatility_adjustment:.2f})",
                'confidence': 0.5,
                'supporting_data': {'volatility_adjustment': volatility_adjustment}
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


class MomentumTradingAgent:
    """Main Momentum Trading Agent using LangGraph"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.fibonacci_integrator = FibonacciIntegrator()
        self.sentiment_integrator = SentimentIntegrator()
        self.volatility_adjuster = VolatilityAdjuster()
        self.explainability_engine = ExplainabilityEngine()
        self.model_version = "1.0.0"
        
        # Initialize LangGraph
        self.graph = self._create_langgraph()
    
    def _create_langgraph(self) -> StateGraph:
        """Create LangGraph state machine for momentum trading"""
        from typing import TypedDict
        
        class MomentumState(TypedDict):
            symbol: str
            market_data: List[MarketData]
            sentiment_data: Optional[SentimentData]
            technical_signals: List[TechnicalSignal]
            fibonacci_signals: List[FibonacciSignal]
            final_signal: Optional[MomentumSignal]
            error: Optional[str]
        
        # Create state graph
        workflow = StateGraph(MomentumState)
        
        # Add nodes
        workflow.add_node("analyze_technical", self._analyze_technical_indicators)
        workflow.add_node("analyze_fibonacci", self._analyze_fibonacci_levels)
        workflow.add_node("integrate_sentiment", self._integrate_sentiment)
        workflow.add_node("calculate_volatility", self._calculate_volatility_adjustment)
        workflow.add_node("generate_signal", self._generate_final_signal)
        workflow.add_node("store_signal", self._store_signal)
        
        # Define edges
        workflow.add_edge("analyze_technical", "analyze_fibonacci")
        workflow.add_edge("analyze_fibonacci", "integrate_sentiment")
        workflow.add_edge("integrate_sentiment", "calculate_volatility")
        workflow.add_edge("calculate_volatility", "generate_signal")
        workflow.add_edge("generate_signal", "store_signal")
        workflow.add_edge("store_signal", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_technical")
        
        return workflow.compile()
    
    async def _analyze_technical_indicators(self, state: Dict) -> Dict:
        """Analyze technical indicators (EMA, RSI, MACD)"""
        try:
            market_data = state['market_data']
            if not market_data:
                state['error'] = "No market data available"
                return state
            
            # Extract price data
            close_prices = np.array([data.close for data in market_data])
            
            # Calculate technical signals
            ema_signals = self.technical_analyzer.calculate_ema_signals(close_prices)
            rsi_signals = self.technical_analyzer.calculate_rsi_signals(close_prices)
            macd_signals = self.technical_analyzer.calculate_macd_signals(close_prices)
            
            # Combine all technical signals
            all_technical_signals = ema_signals + rsi_signals + macd_signals
            state['technical_signals'] = all_technical_signals
            
            logger.info(f"Generated {len(all_technical_signals)} technical signals for {state['symbol']}")
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            state['error'] = str(e)
        
        return state
    
    async def _analyze_fibonacci_levels(self, state: Dict) -> Dict:
        """Analyze Fibonacci retracement levels"""
        try:
            market_data = state['market_data']
            if not market_data:
                return state
            
            # Extract OHLC data
            high_prices = np.array([data.high for data in market_data])
            low_prices = np.array([data.low for data in market_data])
            close_prices = np.array([data.close for data in market_data])
            current_price = close_prices[-1]
            
            # Calculate Fibonacci signals
            fibonacci_signals = self.fibonacci_integrator.calculate_fibonacci_signals(
                high_prices, low_prices, close_prices, current_price
            )
            
            state['fibonacci_signals'] = fibonacci_signals
            
            logger.info(f"Generated {len(fibonacci_signals)} Fibonacci signals for {state['symbol']}")
            
        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")
            state['error'] = str(e)
        
        return state
    
    async def _integrate_sentiment(self, state: Dict) -> Dict:
        """Integrate sentiment analysis"""
        try:
            # Sentiment integration is handled in signal generation
            # This node can be expanded to fetch real-time sentiment data
            logger.info(f"Sentiment integration completed for {state['symbol']}")
            
        except Exception as e:
            logger.error(f"Error in sentiment integration: {e}")
            state['error'] = str(e)
        
        return state
    
    async def _calculate_volatility_adjustment(self, state: Dict) -> Dict:
        """Calculate volatility adjustment and market regime"""
        try:
            market_data = state['market_data']
            if not market_data:
                return state
            
            close_prices = np.array([data.close for data in market_data])
            
            volatility_adjustment, market_regime = self.volatility_adjuster.calculate_volatility_adjustment(close_prices)
            
            state['volatility_adjustment'] = volatility_adjustment
            state['market_regime'] = market_regime
            
            logger.info(f"Volatility adjustment: {volatility_adjustment:.2f}, Regime: {market_regime.value}")
            
        except Exception as e:
            logger.error(f"Error in volatility calculation: {e}")
            state['error'] = str(e)
        
        return state
    
    async def _generate_final_signal(self, state: Dict) -> Dict:
        """Generate final momentum signal with explainability"""
        try:
            technical_signals = state.get('technical_signals', [])
            fibonacci_signals = state.get('fibonacci_signals', [])
            sentiment_data = state.get('sentiment_data')
            volatility_adjustment = state.get('volatility_adjustment', 1.0)
            market_regime = state.get('market_regime', MarketRegime.SIDEWAYS)
            
            # Calculate overall signal value
            signal_value = 0.0
            total_weight = 0.0
            
            # Weight technical signals
            for signal in technical_signals:
                weight = signal.confidence
                signal_value += signal.value * weight
                total_weight += weight
            
            # Add Fibonacci confluence boost
            if fibonacci_signals:
                confluence_strength = self.fibonacci_integrator.calculate_confluence_strength(
                    state['market_data'][-1].close, fibonacci_signals
                )
                # Fibonacci provides directional bias based on level type
                fib_bias = 0.1 * confluence_strength  # Small bias
                signal_value += fib_bias
                total_weight += confluence_strength * 0.5
            
            # Normalize signal value
            if total_weight > 0:
                signal_value = signal_value / total_weight
            
            # Apply sentiment alignment
            sentiment_score = sentiment_data.overall_sentiment if sentiment_data else None
            if sentiment_score is not None:
                sentiment_alignment = self.sentiment_integrator.calculate_sentiment_alignment(
                    signal_value, sentiment_score
                )
                signal_value *= sentiment_alignment
            
            # Determine signal type and confidence
            signal_type = SignalType.HOLD
            confidence = 0.5
            
            if signal_value > 0.3:
                signal_type = SignalType.STRONG_BUY if signal_value > 0.7 else SignalType.BUY
                confidence = min(abs(signal_value), 0.9)
            elif signal_value < -0.3:
                signal_type = SignalType.STRONG_SELL if signal_value < -0.7 else SignalType.SELL
                confidence = min(abs(signal_value), 0.9)
            
            # Calculate position sizing
            base_position_size = 0.05  # 5% base position
            position_size_pct = self.volatility_adjuster.calculate_position_size(
                base_position_size, volatility_adjustment, confidence, 100000  # Assume $100k account
            )
            
            # Generate top-3 reasons
            top_3_reasons = self.explainability_engine.generate_top_3_reasons(
                technical_signals, fibonacci_signals, sentiment_score, 
                volatility_adjustment, signal_value
            )
            
            # Create final signal
            final_signal = MomentumSignal(
                symbol=state['symbol'],
                signal_type=signal_type,
                value=signal_value,
                confidence=confidence,
                top_3_reasons=top_3_reasons,
                timestamp=datetime.utcnow(),
                model_version=self.model_version,
                ema_signals=[s for s in technical_signals if 'EMA' in s.indicator],
                rsi_signals=[s for s in technical_signals if 'RSI' in s.indicator],
                macd_signals=[s for s in technical_signals if 'MACD' in s.indicator],
                fibonacci_signals=fibonacci_signals,
                sentiment_score=sentiment_score,
                volatility_adjustment=volatility_adjustment,
                market_regime=market_regime,
                position_size_pct=position_size_pct,
                stop_loss_pct=0.02,  # 2% stop loss
                take_profit_pct=0.06,  # 6% take profit (3:1 ratio)
                max_holding_period=5  # 5 days max holding
            )
            
            state['final_signal'] = final_signal
            
            logger.info(f"Generated final signal for {state['symbol']}: {signal_type.value} "
                       f"(value: {signal_value:.3f}, confidence: {confidence:.3f})")
            
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
            
            # Store signal in database (implementation depends on database setup)
            # For now, just log the signal
            logger.info(f"Storing signal: {final_signal.to_dict()}")
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
            state['error'] = str(e)
        
        return state
    
    async def analyze_momentum(self, symbol: str, market_data: List[MarketData] = None,
                                     sentiment_data: Optional[SentimentData] = None) -> Optional[MomentumSignal]:
        """Main entry point for generating momentum signals"""
        try:
            # Prepare initial state
            initial_state = {
                'symbol': symbol,
                'market_data': market_data,
                'sentiment_data': sentiment_data,
                'technical_signals': [],
                'fibonacci_signals': [],
                'final_signal': None,
                'error': None
            }
            
            # Run LangGraph workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            if final_state.get('error'):
                logger.error(f"Error in momentum signal generation: {final_state['error']}")
                return None
            
            return final_state.get('final_signal')
            
        except Exception as e:
            logger.error(f"Error in momentum signal generation: {e}")
            return None


# Convenience functions for testing and direct usage
async def generate_momentum_signal(symbol: str, market_data: List[Dict],
                                 sentiment_data: Optional[Dict] = None) -> Optional[Dict]:
    """Generate momentum signal from raw data"""
    agent = MomentumTradingAgent()
    
    # Convert raw data to structured objects
    structured_market_data = [
        MarketData(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            vwap=data.get('vwap')
        ) for data in market_data
    ]
    
    structured_sentiment_data = None
    if sentiment_data:
        structured_sentiment_data = SentimentData(
            symbol=sentiment_data['symbol'],
            overall_sentiment=sentiment_data['overall_sentiment'],
            confidence=sentiment_data['confidence'],
            news_count=sentiment_data['news_count'],
            social_sentiment=sentiment_data.get('social_sentiment'),
            timestamp=datetime.fromisoformat(sentiment_data['timestamp']) if sentiment_data.get('timestamp') else None
        )
    
    signal = await agent.generate_momentum_signal(symbol, structured_market_data, structured_sentiment_data)
    
    return signal.to_dict() if signal else None


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_momentum_agent():
        # Sample market data
        sample_data = []
        base_price = 100.0
        
        for i in range(50):
            price = base_price + np.random.normal(0, 2) + i * 0.1  # Slight uptrend with noise
            sample_data.append({
                'symbol': 'AAPL',
                'timestamp': datetime.utcnow() - timedelta(days=50-i),
                'open': price - 0.5,
                'high': price + 1.0,
                'low': price - 1.0,
                'close': price,
                'volume': 1000000 + np.random.randint(-100000, 100000)
            })
        
        # Sample sentiment data
        sentiment = {
            'symbol': 'AAPL',
            'overall_sentiment': 0.3,
            'confidence': 0.8,
            'news_count': 15,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Generate signal
        signal = await generate_momentum_signal('AAPL', sample_data, sentiment)
        
        if signal:
            print("Generated Momentum Signal:")
            print(json.dumps(signal, indent=2, default=str))
        else:
            print("Failed to generate signal")
    
    # Run test
    asyncio.run(test_momentum_agent())

# Create singleton instance
momentum_trading_agent = MomentumTradingAgent()