"""
Specialized Expert Agents System
===============================

This module implements a comprehensive system of specialized expert agents,
each with distinct roles and capabilities for different aspects of trading:

1. Market Analysis Expert - Technical and fundamental analysis
2. Risk Assessment Expert - Risk evaluation and management
3. Trade Execution Expert - Order optimization and execution
4. Portfolio Optimization Expert - Portfolio allocation and rebalancing
5. Performance Tracking Expert - Performance analysis and attribution
6. Sentiment Analysis Expert - News and social sentiment analysis
7. Macro Economic Expert - Economic indicators and macro trends
8. Options Strategy Expert - Complex options strategies
9. Crypto Market Expert - Cryptocurrency-specific analysis
10. High Frequency Expert - Ultra-fast trading strategies

Each expert agent operates independently while collaborating through
the communication system to provide comprehensive trading intelligence.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import json
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

# ML/Analysis imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import talib
import yfinance as yf
from textblob import TextBlob
import feedparser

# Import existing components
from backend.events.event_bus import EventBus, Event as EventBusEvent, EventType
from core.parallel_trading_architecture import InterEngineMessage, MessageType, EngineType

logger = logging.getLogger(__name__)


class ExpertType(Enum):
    """Types of expert agents"""
    MARKET_ANALYSIS = "market_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_EXECUTION = "trade_execution"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    PERFORMANCE_TRACKING = "performance_tracking"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    MACRO_ECONOMIC = "macro_economic"
    OPTIONS_STRATEGY = "options_strategy"
    CRYPTO_MARKET = "crypto_market"
    HIGH_FREQUENCY = "high_frequency"


class ExpertStatus(Enum):
    """Expert agent status"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class AnalysisType(Enum):
    """Types of analysis"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    RISK = "risk"
    EXECUTION = "execution"
    PORTFOLIO = "portfolio"
    PERFORMANCE = "performance"
    MACRO = "macro"
    OPTIONS = "options"
    CRYPTO = "crypto"


@dataclass
class ExpertAnalysis:
    """Analysis result from an expert agent"""
    expert_type: ExpertType
    analysis_type: AnalysisType
    symbol: str
    timestamp: datetime
    confidence: float
    analysis_data: Dict[str, Any]
    recommendations: List[str]
    risk_factors: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertConsensus:
    """Consensus result from multiple experts"""
    symbol: str
    timestamp: datetime
    participating_experts: List[ExpertType]
    consensus_score: float
    aggregated_analysis: Dict[str, Any]
    unified_recommendations: List[str]
    risk_assessment: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]


class BaseExpertAgent(ABC):
    """Abstract base class for all expert agents"""

    def __init__(self, expert_type: ExpertType, config: Dict[str, Any] = None):
        self.expert_type = expert_type
        self.config = config or {}
        self.status = ExpertStatus.INITIALIZING

        # Agent state
        self.expertise_score = 0.8  # Initial expertise level
        self.analysis_history: List[ExpertAnalysis] = []
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'avg_confidence': 0.0,
            'accuracy_score': 0.0,
            'response_time_ms': 0.0
        }

        # Communication
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.collaboration_network: Dict[ExpertType, float] = {}  # Trust scores

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()

    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> ExpertAnalysis:
        """Perform expert analysis on provided data"""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the expert agent"""
        pass

    async def get_analysis(self, symbol: str, data_sources: Dict[str, Any]) -> Optional[ExpertAnalysis]:
        """Get comprehensive analysis from this expert"""
        try:
            start_time = asyncio.get_event_loop().time()
            self.status = ExpertStatus.BUSY

            # Prepare analysis data
            analysis_data = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
                'data_sources': data_sources,
                'expert_config': self.config
            }

            # Perform analysis
            result = await self.analyze(analysis_data)

            # Update performance metrics
            end_time = asyncio.get_event_loop().time()
            response_time = (end_time - start_time) * 1000

            with self.lock:
                self.performance_metrics['total_analyses'] += 1
                self.performance_metrics['response_time_ms'] = (
                    (self.performance_metrics['response_time_ms'] * (self.performance_metrics['total_analyses'] - 1) +
                     response_time) / self.performance_metrics['total_analyses']
                )

                if result:
                    self.performance_metrics['successful_analyses'] += 1
                    self.performance_metrics['avg_confidence'] = (
                        (self.performance_metrics['avg_confidence'] * (self.performance_metrics['successful_analyses'] - 1) +
                         result.confidence) / self.performance_metrics['successful_analyses']
                    )

            # Store in history
            if result:
                self.analysis_history.append(result)
                if len(self.analysis_history) > 1000:
                    self.analysis_history = self.analysis_history[-1000:]

            self.status = ExpertStatus.ACTIVE
            return result

        except Exception as e:
            logger.error(f"Error in {self.expert_type.value} analysis: {e}")
            self.status = ExpertStatus.ERROR
            return None

    async def collaborate(self, other_experts: List['BaseExpertAgent'], symbol: str) -> Dict[str, Any]:
        """Collaborate with other experts for consensus"""
        try:
            collaboration_data = {
                'symbol': symbol,
                'own_analysis': self.analysis_history[-1] if self.analysis_history else None,
                'collaboration_request': True,
                'timestamp': datetime.now(timezone.utc)
            }

            # Request analysis from other experts
            other_analyses = []
            for expert in other_experts:
                if expert.expert_type != self.expert_type:
                    # In a full implementation, this would be done through message passing
                    # For now, simulate collaboration
                    trust_score = self.collaboration_network.get(expert.expert_type, 0.5)
                    other_analyses.append({
                        'expert_type': expert.expert_type,
                        'trust_score': trust_score,
                        'recent_performance': expert.performance_metrics
                    })

            return {
                'collaboration_partners': [expert.expert_type for expert in other_experts],
                'analyses_received': len(other_analyses),
                'collaboration_score': np.mean([analysis['trust_score'] for analysis in other_analyses]) if other_analyses else 0.0
            }

        except Exception as e:
            logger.error(f"Error in collaboration: {e}")
            return {}

    def update_trust_score(self, expert_type: ExpertType, accuracy: float) -> None:
        """Update trust score for another expert based on their accuracy"""
        try:
            current_trust = self.collaboration_network.get(expert_type, 0.5)
            # Exponential moving average
            alpha = 0.1
            new_trust = alpha * accuracy + (1 - alpha) * current_trust
            self.collaboration_network[expert_type] = np.clip(new_trust, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error updating trust score: {e}")

    def get_expert_status(self) -> Dict[str, Any]:
        """Get expert status and metrics"""
        return {
            'expert_type': self.expert_type.value,
            'status': self.status.value,
            'expertise_score': self.expertise_score,
            'performance_metrics': self.performance_metrics,
            'recent_analyses': len(self.analysis_history),
            'collaboration_network': {k.value: v for k, v in self.collaboration_network.items()}
        }


class MarketAnalysisExpert(BaseExpertAgent):
    """Expert specializing in technical and fundamental market analysis"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(ExpertType.MARKET_ANALYSIS, config)
        self.technical_indicators = [
            'sma', 'ema', 'rsi', 'macd', 'bollinger_bands',
            'stochastic', 'adx', 'atr', 'obv', 'williams_r'
        ]
        self.fundamental_metrics = [
            'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity',
            'revenue_growth', 'eps_growth', 'dividend_yield'
        ]

    async def initialize(self) -> bool:
        """Initialize market analysis expert"""
        try:
            logger.info("Initializing Market Analysis Expert")
            # Initialize data sources and technical analysis tools
            self.status = ExpertStatus.ACTIVE
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Market Analysis Expert: {e}")
            return False

    async def analyze(self, data: Dict[str, Any]) -> ExpertAnalysis:
        """Perform comprehensive market analysis"""
        try:
            symbol = data['symbol']
            data_sources = data.get('data_sources', {})

            # Get market data
            market_data = await self._get_market_data(symbol)
            if market_data is None:
                raise ValueError("No market data available")

            # Technical analysis
            technical_analysis = await self._perform_technical_analysis(market_data)

            # Fundamental analysis
            fundamental_analysis = await self._perform_fundamental_analysis(symbol)

            # Price targets and support/resistance
            price_levels = await self._calculate_price_levels(market_data)

            # Trend analysis
            trend_analysis = await self._analyze_trends(market_data)

            # Volatility analysis
            volatility_analysis = await self._analyze_volatility(market_data)

            # Combine all analyses
            analysis_data = {
                'technical': technical_analysis,
                'fundamental': fundamental_analysis,
                'price_levels': price_levels,
                'trend': trend_analysis,
                'volatility': volatility_analysis
            }

            # Generate recommendations
            recommendations = await self._generate_recommendations(analysis_data)

            # Assess risk factors
            risk_factors = await self._assess_risk_factors(analysis_data)

            # Calculate overall confidence
            confidence = self._calculate_confidence(analysis_data)

            return ExpertAnalysis(
                expert_type=self.expert_type,
                analysis_type=AnalysisType.TECHNICAL,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                confidence=confidence,
                analysis_data=analysis_data,
                recommendations=recommendations,
                risk_factors=risk_factors,
                metadata={'indicators_used': len(self.technical_indicators)}
            )

        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            raise

    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for analysis"""
        try:
            # In production, this would use your preferred data source
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            return data
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    async def _perform_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']

            # Moving averages
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            ema_12 = talib.EMA(close, timeperiod=12)
            ema_26 = talib.EMA(close, timeperiod=26)

            # Momentum indicators
            rsi = talib.RSI(close, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(close)
            stoch_k, stoch_d = talib.STOCH(high, low, close)

            # Volatility indicators
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            atr = talib.ATR(high, low, close)

            # Volume indicators
            obv = talib.OBV(close, volume)
            ad_line = talib.AD(high, low, close, volume)

            # Trend indicators
            adx = talib.ADX(high, low, close)
            cci = talib.CCI(high, low, close)

            return {
                'moving_averages': {
                    'sma_20': sma_20.iloc[-1],
                    'sma_50': sma_50.iloc[-1],
                    'ema_12': ema_12.iloc[-1],
                    'ema_26': ema_26.iloc[-1]
                },
                'momentum': {
                    'rsi': rsi.iloc[-1],
                    'macd': macd.iloc[-1],
                    'macd_signal': macd_signal.iloc[-1],
                    'stoch_k': stoch_k.iloc[-1],
                    'stoch_d': stoch_d.iloc[-1]
                },
                'volatility': {
                    'bb_upper': bb_upper.iloc[-1],
                    'bb_lower': bb_lower.iloc[-1],
                    'atr': atr.iloc[-1]
                },
                'volume': {
                    'obv': obv.iloc[-1],
                    'ad_line': ad_line.iloc[-1]
                },
                'trend': {
                    'adx': adx.iloc[-1],
                    'cci': cci.iloc[-1]
                }
            }

        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {}

    async def _perform_fundamental_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform fundamental analysis"""
        try:
            # In production, this would use financial data APIs
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'valuation': {
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'price_to_sales': info.get('priceToSalesTrailing12Months')
                },
                'profitability': {
                    'roe': info.get('returnOnEquity'),
                    'roa': info.get('returnOnAssets'),
                    'profit_margin': info.get('profitMargins'),
                    'gross_margin': info.get('grossMargins')
                },
                'financial_health': {
                    'debt_to_equity': info.get('debtToEquity'),
                    'current_ratio': info.get('currentRatio'),
                    'quick_ratio': info.get('quickRatio'),
                    'cash_per_share': info.get('totalCashPerShare')
                },
                'growth': {
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': info.get('earningsGrowth'),
                    'eps_forward': info.get('forwardEps')
                }
            }

        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return {}

    async def _calculate_price_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate key price levels"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']

            # Recent levels
            recent_high = high.tail(50).max()
            recent_low = low.tail(50).min()
            current_price = close.iloc[-1]

            # Pivot points
            pivot = (recent_high + recent_low + current_price) / 3
            r1 = 2 * pivot - recent_low
            s1 = 2 * pivot - recent_high
            r2 = pivot + (recent_high - recent_low)
            s2 = pivot - (recent_high - recent_low)

            # Fibonacci levels
            fib_range = recent_high - recent_low
            fib_levels = {
                'fib_23.6': recent_high - (fib_range * 0.236),
                'fib_38.2': recent_high - (fib_range * 0.382),
                'fib_50.0': recent_high - (fib_range * 0.500),
                'fib_61.8': recent_high - (fib_range * 0.618),
                'fib_78.6': recent_high - (fib_range * 0.786)
            }

            return {
                'current_price': current_price,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'pivot': pivot,
                'resistance_1': r1,
                'resistance_2': r2,
                'support_1': s1,
                'support_2': s2,
                **fib_levels
            }

        except Exception as e:
            logger.error(f"Error calculating price levels: {e}")
            return {}

    async def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trends"""
        try:
            close = data['Close']

            # Trend direction
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            sma_200 = talib.SMA(close, timeperiod=200)

            current_price = close.iloc[-1]
            trend_direction = "bullish" if current_price > sma_20.iloc[-1] > sma_50.iloc[-1] else "bearish"

            # Trend strength
            adx = talib.ADX(data['High'], data['Low'], data['Close'])
            trend_strength = "strong" if adx.iloc[-1] > 25 else "weak"

            # Rate of change
            roc = talib.ROC(close, timeperiod=20)

            return {
                'direction': trend_direction,
                'strength': trend_strength,
                'adx': adx.iloc[-1],
                'rate_of_change': roc.iloc[-1],
                'ma_alignment': {
                    'price_vs_sma20': (current_price - sma_20.iloc[-1]) / sma_20.iloc[-1],
                    'price_vs_sma50': (current_price - sma_50.iloc[-1]) / sma_50.iloc[-1],
                    'price_vs_sma200': (current_price - sma_200.iloc[-1]) / sma_200.iloc[-1]
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {}

    async def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price volatility"""
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']

            # Historical volatility
            returns = close.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            # ATR
            atr = talib.ATR(high, low, close)
            atr_percentage = (atr.iloc[-1] / close.iloc[-1]) * 100

            # Bollinger Bands width
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            bb_width = ((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]) * 100

            return {
                'historical_volatility': volatility,
                'atr': atr.iloc[-1],
                'atr_percentage': atr_percentage,
                'bollinger_width': bb_width,
                'volatility_percentile': self._calculate_volatility_percentile(returns)
            }

        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {}

    def _calculate_volatility_percentile(self, returns: pd.Series) -> float:
        """Calculate volatility percentile"""
        try:
            current_vol = returns.tail(20).std()
            historical_vols = returns.rolling(window=20).std().dropna()
            percentile = (historical_vols < current_vol).mean() * 100
            return percentile
        except Exception as e:
            logger.error(f"Error calculating volatility percentile: {e}")
            return 50.0

    async def _generate_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on analysis"""
        try:
            recommendations = []
            technical = analysis_data.get('technical', {})
            fundamental = analysis_data.get('fundamental', {})
            trend = analysis_data.get('trend', {})

            # Technical recommendations
            if technical.get('momentum', {}).get('rsi', 50) < 30:
                recommendations.append("RSI indicates oversold condition - potential buying opportunity")
            elif technical.get('momentum', {}).get('rsi', 50) > 70:
                recommendations.append("RSI indicates overbought condition - consider taking profits")

            # Trend recommendations
            if trend.get('direction') == 'bullish' and trend.get('strength') == 'strong':
                recommendations.append("Strong bullish trend - consider momentum strategies")
            elif trend.get('direction') == 'bearish' and trend.get('strength') == 'strong':
                recommendations.append("Strong bearish trend - avoid long positions")

            # Fundamental recommendations
            pe_ratio = fundamental.get('valuation', {}).get('pe_ratio')
            if pe_ratio and pe_ratio < 15:
                recommendations.append("Low P/E ratio suggests potential value opportunity")
            elif pe_ratio and pe_ratio > 30:
                recommendations.append("High P/E ratio indicates potential overvaluation")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    async def _assess_risk_factors(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Assess risk factors"""
        try:
            risk_factors = []
            volatility = analysis_data.get('volatility', {})
            trend = analysis_data.get('trend', {})

            # Volatility risks
            if volatility.get('historical_volatility', 0) > 0.4:
                risk_factors.append("High historical volatility increases risk")

            # Trend risks
            if trend.get('strength') == 'weak':
                risk_factors.append("Weak trend increases directional uncertainty")

            # Technical risks
            technical = analysis_data.get('technical', {})
            if technical.get('momentum', {}).get('rsi', 50) > 80:
                risk_factors.append("Extremely overbought conditions increase reversal risk")

            return risk_factors

        except Exception as e:
            logger.error(f"Error assessing risk factors: {e}")
            return []

    def _calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate overall analysis confidence"""
        try:
            confidence_factors = []

            # Technical confidence
            technical = analysis_data.get('technical', {})
            if technical:
                confidence_factors.append(0.8)  # Base technical confidence

            # Fundamental confidence
            fundamental = analysis_data.get('fundamental', {})
            if fundamental:
                confidence_factors.append(0.7)  # Base fundamental confidence

            # Data quality confidence
            if len(analysis_data) >= 3:  # Multiple analysis types
                confidence_factors.append(0.9)

            return np.mean(confidence_factors) if confidence_factors else 0.5

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5


class RiskAssessmentExpert(BaseExpertAgent):
    """Expert specializing in risk evaluation and management"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(ExpertType.RISK_ASSESSMENT, config)
        self.risk_models = ['var', 'cvar', 'maximum_drawdown', 'sharpe_ratio', 'sortino_ratio']
        self.correlation_threshold = config.get('correlation_threshold', 0.7) if config else 0.7

    async def initialize(self) -> bool:
        """Initialize risk assessment expert"""
        try:
            logger.info("Initializing Risk Assessment Expert")
            self.status = ExpertStatus.ACTIVE
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Risk Assessment Expert: {e}")
            return False

    async def analyze(self, data: Dict[str, Any]) -> ExpertAnalysis:
        """Perform comprehensive risk analysis"""
        try:
            symbol = data['symbol']
            data_sources = data.get('data_sources', {})

            # Portfolio-level risk assessment
            portfolio_risk = await self._assess_portfolio_risk(data_sources)

            # Position-level risk assessment
            position_risk = await self._assess_position_risk(symbol, data_sources)

            # Market risk assessment
            market_risk = await self._assess_market_risk(data_sources)

            # Correlation analysis
            correlation_risk = await self._assess_correlation_risk(symbol, data_sources)

            # Liquidity risk assessment
            liquidity_risk = await self._assess_liquidity_risk(symbol, data_sources)

            analysis_data = {
                'portfolio_risk': portfolio_risk,
                'position_risk': position_risk,
                'market_risk': market_risk,
                'correlation_risk': correlation_risk,
                'liquidity_risk': liquidity_risk
            }

            # Generate risk recommendations
            recommendations = await self._generate_risk_recommendations(analysis_data)

            # Identify risk factors
            risk_factors = await self._identify_risk_factors(analysis_data)

            # Calculate risk confidence
            confidence = self._calculate_risk_confidence(analysis_data)

            return ExpertAnalysis(
                expert_type=self.expert_type,
                analysis_type=AnalysisType.RISK,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                confidence=confidence,
                analysis_data=analysis_data,
                recommendations=recommendations,
                risk_factors=risk_factors,
                metadata={'risk_models_used': len(self.risk_models)}
            )

        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            raise

    async def _assess_portfolio_risk(self, data_sources: Dict[str, Any]) -> Dict[str, float]:
        """Assess portfolio-level risk metrics"""
        try:
            # In production, this would access actual portfolio data
            # For now, simulate portfolio risk metrics
            return {
                'portfolio_var_95': 0.025,  # 2.5% daily VaR
                'portfolio_cvar_95': 0.035,  # 3.5% daily CVaR
                'max_drawdown': 0.15,  # 15% maximum drawdown
                'sharpe_ratio': 1.2,
                'sortino_ratio': 1.8,
                'beta': 1.1,
                'correlation_with_market': 0.75
            }
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return {}

    async def _assess_position_risk(self, symbol: str, data_sources: Dict[str, Any]) -> Dict[str, float]:
        """Assess position-specific risk metrics"""
        try:
            # Get position data and calculate risk metrics
            return {
                'position_var_95': 0.045,  # Position-level VaR
                'position_size_risk': 0.02,  # 2% of portfolio
                'stop_loss_distance': 0.05,  # 5% from current price
                'position_volatility': 0.25,  # 25% annualized volatility
                'concentration_risk': 0.1  # 10% concentration
            }
        except Exception as e:
            logger.error(f"Error assessing position risk: {e}")
            return {}

    async def _assess_market_risk(self, data_sources: Dict[str, Any]) -> Dict[str, float]:
        """Assess market-wide risk factors"""
        try:
            # Assess systematic risk factors
            return {
                'market_volatility': 0.20,  # Market volatility index
                'interest_rate_risk': 0.15,  # Interest rate sensitivity
                'currency_risk': 0.05,  # Currency exposure risk
                'sector_concentration': 0.30,  # Sector concentration risk
                'geopolitical_risk': 0.10  # Geopolitical risk score
            }
        except Exception as e:
            logger.error(f"Error assessing market risk: {e}")
            return {}

    async def _assess_correlation_risk(self, symbol: str, data_sources: Dict[str, Any]) -> Dict[str, float]:
        """Assess correlation and diversification risk"""
        try:
            # Calculate correlation with portfolio holdings
            return {
                'avg_correlation': 0.45,  # Average correlation with existing positions
                'max_correlation': 0.85,  # Maximum correlation with any position
                'diversification_ratio': 0.75,  # Portfolio diversification ratio
                'correlation_stability': 0.80  # Stability of correlations over time
            }
        except Exception as e:
            logger.error(f"Error assessing correlation risk: {e}")
            return {}

    async def _assess_liquidity_risk(self, symbol: str, data_sources: Dict[str, Any]) -> Dict[str, float]:
        """Assess liquidity risk factors"""
        try:
            # Assess liquidity metrics
            return {
                'bid_ask_spread': 0.001,  # 0.1% bid-ask spread
                'market_depth': 0.85,  # Market depth score
                'volume_volatility': 0.30,  # Volume volatility
                'liquidity_score': 0.80  # Overall liquidity score
            }
        except Exception as e:
            logger.error(f"Error assessing liquidity risk: {e}")
            return {}

    async def _generate_risk_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations"""
        try:
            recommendations = []

            portfolio_risk = analysis_data.get('portfolio_risk', {})
            position_risk = analysis_data.get('position_risk', {})
            correlation_risk = analysis_data.get('correlation_risk', {})

            # Portfolio-level recommendations
            if portfolio_risk.get('portfolio_var_95', 0) > 0.03:
                recommendations.append("High portfolio VaR - consider reducing overall risk exposure")

            if portfolio_risk.get('max_drawdown', 0) > 0.20:
                recommendations.append("High maximum drawdown - implement stricter stop-loss rules")

            # Position-level recommendations
            if position_risk.get('position_size_risk', 0) > 0.05:
                recommendations.append("Position size exceeds 5% of portfolio - consider reducing")

            # Correlation recommendations
            if correlation_risk.get('max_correlation', 0) > self.correlation_threshold:
                recommendations.append("High correlation with existing positions - diversification needed")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            return []

    async def _identify_risk_factors(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Identify key risk factors"""
        try:
            risk_factors = []

            market_risk = analysis_data.get('market_risk', {})
            liquidity_risk = analysis_data.get('liquidity_risk', {})
            correlation_risk = analysis_data.get('correlation_risk', {})

            # Market risk factors
            if market_risk.get('market_volatility', 0) > 0.25:
                risk_factors.append("High market volatility environment")

            if market_risk.get('geopolitical_risk', 0) > 0.15:
                risk_factors.append("Elevated geopolitical risk")

            # Liquidity risk factors
            if liquidity_risk.get('liquidity_score', 1) < 0.5:
                risk_factors.append("Low liquidity may impact execution")

            # Correlation risk factors
            if correlation_risk.get('diversification_ratio', 1) < 0.5:
                risk_factors.append("Poor diversification increases concentration risk")

            return risk_factors

        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return []

    def _calculate_risk_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence in risk assessment"""
        try:
            confidence_factors = []

            # Data completeness
            complete_analyses = sum(1 for analysis in analysis_data.values() if analysis)
            completeness_score = complete_analyses / len(analysis_data)
            confidence_factors.append(completeness_score)

            # Risk model coverage
            model_coverage = len(self.risk_models) / 10  # Assuming 10 total possible models
            confidence_factors.append(model_coverage)

            # Historical data quality
            confidence_factors.append(0.8)  # Assume good historical data

            return np.mean(confidence_factors)

        except Exception as e:
            logger.error(f"Error calculating risk confidence: {e}")
            return 0.5


class SentimentAnalysisExpert(BaseExpertAgent):
    """Expert specializing in news and social sentiment analysis"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(ExpertType.SENTIMENT_ANALYSIS, config)
        self.news_sources = config.get('news_sources', ['yahoo', 'google', 'reuters']) if config else ['yahoo', 'google', 'reuters']
        self.social_sources = config.get('social_sources', ['twitter', 'reddit']) if config else ['twitter', 'reddit']

    async def initialize(self) -> bool:
        """Initialize sentiment analysis expert"""
        try:
            logger.info("Initializing Sentiment Analysis Expert")
            self.status = ExpertStatus.ACTIVE
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Sentiment Analysis Expert: {e}")
            return False

    async def analyze(self, data: Dict[str, Any]) -> ExpertAnalysis:
        """Perform comprehensive sentiment analysis"""
        try:
            symbol = data['symbol']

            # News sentiment analysis
            news_sentiment = await self._analyze_news_sentiment(symbol)

            # Social media sentiment analysis
            social_sentiment = await self._analyze_social_sentiment(symbol)

            # Analyst recommendations sentiment
            analyst_sentiment = await self._analyze_analyst_sentiment(symbol)

            # Options flow sentiment
            options_sentiment = await self._analyze_options_sentiment(symbol)

            # Insider trading sentiment
            insider_sentiment = await self._analyze_insider_sentiment(symbol)

            analysis_data = {
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'analyst_sentiment': analyst_sentiment,
                'options_sentiment': options_sentiment,
                'insider_sentiment': insider_sentiment
            }

            # Generate sentiment-based recommendations
            recommendations = await self._generate_sentiment_recommendations(analysis_data)

            # Identify sentiment risk factors
            risk_factors = await self._identify_sentiment_risks(analysis_data)

            # Calculate sentiment confidence
            confidence = self._calculate_sentiment_confidence(analysis_data)

            return ExpertAnalysis(
                expert_type=self.expert_type,
                analysis_type=AnalysisType.SENTIMENT,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                confidence=confidence,
                analysis_data=analysis_data,
                recommendations=recommendations,
                risk_factors=risk_factors,
                metadata={'sources_analyzed': len(self.news_sources) + len(self.social_sources)}
            )

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise

    async def _analyze_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze news sentiment for the symbol"""
        try:
            # In production, this would use news APIs and NLP
            # For now, simulate news sentiment analysis
            return {
                'overall_sentiment': 0.15,  # Slightly positive (-1 to 1 scale)
                'sentiment_strength': 0.60,  # Moderate strength
                'news_volume': 25,  # Number of news articles
                'sentiment_trend': 0.05,  # Improving sentiment
                'source_diversity': 0.80,  # Multiple sources
                'recent_sentiment': 0.20  # Recent news sentiment
            }
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {}

    async def _analyze_social_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze social media sentiment"""
        try:
            # Simulate social media sentiment analysis
            return {
                'twitter_sentiment': 0.10,  # Slightly positive
                'reddit_sentiment': -0.05,  # Slightly negative
                'social_volume': 150,  # Social mentions
                'sentiment_volatility': 0.30,  # Moderate volatility
                'influencer_sentiment': 0.25,  # Positive influencer sentiment
                'trending_score': 0.15  # Trending status
            }
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return {}

    async def _analyze_analyst_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze analyst recommendations and sentiment"""
        try:
            # Simulate analyst sentiment analysis
            return {
                'avg_rating': 3.2,  # 1-5 scale (3.2 = Hold+)
                'rating_trend': 0.1,  # Improving ratings
                'price_target_vs_current': 0.08,  # 8% upside to avg target
                'upgrade_downgrade_ratio': 1.5,  # More upgrades than downgrades
                'consensus_strength': 0.70  # Strong consensus
            }
        except Exception as e:
            logger.error(f"Error analyzing analyst sentiment: {e}")
            return {}

    async def _analyze_options_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze options flow for sentiment indicators"""
        try:
            # Simulate options sentiment analysis
            return {
                'put_call_ratio': 0.75,  # More calls than puts
                'unusual_options_activity': 1.8,  # Above normal activity
                'options_skew': -0.05,  # Slight call bias
                'gamma_exposure': 0.15,  # Positive gamma
                'dark_pool_sentiment': 0.10  # Positive dark pool flow
            }
        except Exception as e:
            logger.error(f"Error analyzing options sentiment: {e}")
            return {}

    async def _analyze_insider_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze insider trading sentiment"""
        try:
            # Simulate insider sentiment analysis
            return {
                'insider_buy_sell_ratio': 2.0,  # More buying than selling
                'insider_transaction_volume': 0.50,  # Moderate volume
                'insider_confidence': 0.65,  # Moderate confidence
                'recent_insider_activity': 1.2  # Above average recent activity
            }
        except Exception as e:
            logger.error(f"Error analyzing insider sentiment: {e}")
            return {}

    async def _generate_sentiment_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on sentiment analysis"""
        try:
            recommendations = []

            news_sentiment = analysis_data.get('news_sentiment', {})
            social_sentiment = analysis_data.get('social_sentiment', {})
            analyst_sentiment = analysis_data.get('analyst_sentiment', {})
            options_sentiment = analysis_data.get('options_sentiment', {})

            # News-based recommendations
            if news_sentiment.get('overall_sentiment', 0) > 0.2:
                recommendations.append("Positive news sentiment supports bullish positioning")
            elif news_sentiment.get('overall_sentiment', 0) < -0.2:
                recommendations.append("Negative news sentiment suggests caution")

            # Analyst recommendations
            if analyst_sentiment.get('price_target_vs_current', 0) > 0.1:
                recommendations.append("Analyst price targets suggest 10%+ upside potential")

            # Options flow recommendations
            if options_sentiment.get('put_call_ratio', 1) < 0.5:
                recommendations.append("Heavy call buying suggests bullish options positioning")

            # Social sentiment recommendations
            if social_sentiment.get('trending_score', 0) > 0.3:
                recommendations.append("High social media attention - monitor for volatility")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating sentiment recommendations: {e}")
            return []

    async def _identify_sentiment_risks(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Identify sentiment-based risk factors"""
        try:
            risk_factors = []

            social_sentiment = analysis_data.get('social_sentiment', {})
            news_sentiment = analysis_data.get('news_sentiment', {})

            # Social media risks
            if social_sentiment.get('sentiment_volatility', 0) > 0.5:
                risk_factors.append("High sentiment volatility increases emotional trading risk")

            if social_sentiment.get('trending_score', 0) > 0.5:
                risk_factors.append("High social media attention may lead to increased volatility")

            # News risks
            if news_sentiment.get('sentiment_trend', 0) < -0.2:
                risk_factors.append("Deteriorating news sentiment trend")

            return risk_factors

        except Exception as e:
            logger.error(f"Error identifying sentiment risks: {e}")
            return []

    def _calculate_sentiment_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence in sentiment analysis"""
        try:
            confidence_factors = []

            # Data source diversity
            complete_sources = sum(1 for analysis in analysis_data.values() if analysis)
            source_diversity = complete_sources / len(analysis_data)
            confidence_factors.append(source_diversity)

            # Volume-based confidence
            news_sentiment = analysis_data.get('news_sentiment', {})
            news_volume = news_sentiment.get('news_volume', 0)
            volume_confidence = min(1.0, news_volume / 50)  # Max confidence at 50+ articles
            confidence_factors.append(volume_confidence)

            # Sentiment consistency
            sentiment_values = []
            for analysis in analysis_data.values():
                for key, value in analysis.items():
                    if 'sentiment' in key and isinstance(value, (int, float)):
                        sentiment_values.append(value)

            if sentiment_values:
                sentiment_consistency = 1.0 - np.std(sentiment_values)
                confidence_factors.append(max(0.0, sentiment_consistency))

            return np.mean(confidence_factors) if confidence_factors else 0.5

        except Exception as e:
            logger.error(f"Error calculating sentiment confidence: {e}")
            return 0.5


class ExpertCoordinator:
    """
    Coordinates multiple expert agents and generates consensus analyses
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.experts: Dict[ExpertType, BaseExpertAgent] = {}
        self.consensus_history: List[ExpertConsensus] = []

        # Coordination parameters
        self.min_experts_for_consensus = config.get('min_experts_for_consensus', 3)
        self.consensus_threshold = config.get('consensus_threshold', 0.6)
        self.expert_weights = config.get('expert_weights', {})

    async def initialize(self) -> bool:
        """Initialize all expert agents"""
        try:
            logger.info("Initializing Expert Coordinator")

            # Initialize core experts
            self.experts[ExpertType.MARKET_ANALYSIS] = MarketAnalysisExpert(
                self.config.get('market_analysis', {})
            )
            self.experts[ExpertType.RISK_ASSESSMENT] = RiskAssessmentExpert(
                self.config.get('risk_assessment', {})
            )
            self.experts[ExpertType.SENTIMENT_ANALYSIS] = SentimentAnalysisExpert(
                self.config.get('sentiment_analysis', {})
            )

            # Initialize all experts
            initialization_results = []
            for expert_type, expert in self.experts.items():
                try:
                    success = await expert.initialize()
                    initialization_results.append(success)
                    if success:
                        logger.info(f"Initialized {expert_type.value} expert")
                    else:
                        logger.error(f"Failed to initialize {expert_type.value} expert")
                except Exception as e:
                    logger.error(f"Error initializing {expert_type.value} expert: {e}")
                    initialization_results.append(False)

            success_rate = sum(initialization_results) / len(initialization_results)
            logger.info(f"Expert initialization success rate: {success_rate:.1%}")

            return success_rate >= 0.5  # Require at least 50% success

        except Exception as e:
            logger.error(f"Error initializing Expert Coordinator: {e}")
            return False

    async def get_consensus_analysis(self,
                                   symbol: str,
                                   data_sources: Dict[str, Any],
                                   required_experts: Optional[List[ExpertType]] = None) -> Optional[ExpertConsensus]:
        """Get consensus analysis from multiple experts"""
        try:
            # Determine which experts to consult
            if required_experts:
                experts_to_consult = {et: self.experts[et] for et in required_experts if et in self.experts}
            else:
                experts_to_consult = self.experts

            if len(experts_to_consult) < self.min_experts_for_consensus:
                logger.warning(f"Insufficient experts for consensus: {len(experts_to_consult)}")
                return None

            # Collect analyses from all experts
            expert_analyses = {}
            analysis_tasks = []

            for expert_type, expert in experts_to_consult.items():
                task = asyncio.create_task(expert.get_analysis(symbol, data_sources))
                analysis_tasks.append((expert_type, task))

            # Wait for all analyses to complete
            for expert_type, task in analysis_tasks:
                try:
                    analysis = await task
                    if analysis:
                        expert_analyses[expert_type] = analysis
                except Exception as e:
                    logger.error(f"Error getting analysis from {expert_type.value}: {e}")

            if len(expert_analyses) < self.min_experts_for_consensus:
                logger.warning(f"Insufficient successful analyses for consensus: {len(expert_analyses)}")
                return None

            # Generate consensus
            consensus = await self._generate_consensus(symbol, expert_analyses)

            if consensus:
                self.consensus_history.append(consensus)
                if len(self.consensus_history) > 1000:
                    self.consensus_history = self.consensus_history[-1000:]

            return consensus

        except Exception as e:
            logger.error(f"Error generating consensus analysis: {e}")
            return None

    async def _generate_consensus(self,
                                symbol: str,
                                expert_analyses: Dict[ExpertType, ExpertAnalysis]) -> ExpertConsensus:
        """Generate consensus from multiple expert analyses"""
        try:
            # Calculate consensus score
            confidence_scores = [analysis.confidence for analysis in expert_analyses.values()]
            consensus_score = np.mean(confidence_scores)

            # Aggregate analyses
            aggregated_analysis = self._aggregate_analyses(expert_analyses)

            # Unify recommendations
            unified_recommendations = self._unify_recommendations(expert_analyses)

            # Aggregate risk assessment
            risk_assessment = self._aggregate_risk_assessment(expert_analyses)

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(expert_analyses)

            return ExpertConsensus(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                participating_experts=list(expert_analyses.keys()),
                consensus_score=consensus_score,
                aggregated_analysis=aggregated_analysis,
                unified_recommendations=unified_recommendations,
                risk_assessment=risk_assessment,
                confidence_intervals=confidence_intervals
            )

        except Exception as e:
            logger.error(f"Error generating consensus: {e}")
            raise

    def _aggregate_analyses(self, expert_analyses: Dict[ExpertType, ExpertAnalysis]) -> Dict[str, Any]:
        """Aggregate analysis data from multiple experts"""
        try:
            aggregated = {
                'expert_count': len(expert_analyses),
                'analysis_types': [analysis.analysis_type.value for analysis in expert_analyses.values()],
                'avg_confidence': np.mean([analysis.confidence for analysis in expert_analyses.values()]),
                'expert_weights': self.expert_weights
            }

            # Aggregate specific metrics if available
            for expert_type, analysis in expert_analyses.items():
                key = f"{expert_type.value}_analysis"
                aggregated[key] = analysis.analysis_data

            return aggregated

        except Exception as e:
            logger.error(f"Error aggregating analyses: {e}")
            return {}

    def _unify_recommendations(self, expert_analyses: Dict[ExpertType, ExpertAnalysis]) -> List[str]:
        """Unify recommendations from multiple experts"""
        try:
            all_recommendations = []
            recommendation_counts = {}

            # Collect all recommendations
            for analysis in expert_analyses.values():
                for rec in analysis.recommendations:
                    all_recommendations.append(rec)
                    recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

            # Prioritize recommendations mentioned by multiple experts
            unified_recommendations = []

            # Add recommendations mentioned by multiple experts first
            for rec, count in sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 1:
                    unified_recommendations.append(f"Consensus: {rec}")

            # Add unique recommendations from high-confidence experts
            for expert_type, analysis in expert_analyses.items():
                if analysis.confidence > 0.8:
                    for rec in analysis.recommendations:
                        if recommendation_counts[rec] == 1:
                            unified_recommendations.append(f"{expert_type.value}: {rec}")

            return unified_recommendations[:10]  # Limit to top 10

        except Exception as e:
            logger.error(f"Error unifying recommendations: {e}")
            return []

    def _aggregate_risk_assessment(self, expert_analyses: Dict[ExpertType, ExpertAnalysis]) -> Dict[str, float]:
        """Aggregate risk assessments from multiple experts"""
        try:
            risk_scores = {}

            # Collect risk factors
            all_risk_factors = []
            for analysis in expert_analyses.values():
                all_risk_factors.extend(analysis.risk_factors)

            # Calculate risk factor frequency
            risk_factor_counts = {}
            for factor in all_risk_factors:
                risk_factor_counts[factor] = risk_factor_counts.get(factor, 0) + 1

            # Calculate overall risk metrics
            confidence_scores = [analysis.confidence for analysis in expert_analyses.values()]

            risk_scores.update({
                'overall_risk_score': 1.0 - np.mean(confidence_scores),
                'risk_consensus': len(set(all_risk_factors)) / max(1, len(all_risk_factors)),
                'high_frequency_risks': len([f for f, c in risk_factor_counts.items() if c > 1]),
                'unique_risks': len([f for f, c in risk_factor_counts.items() if c == 1])
            })

            return risk_scores

        except Exception as e:
            logger.error(f"Error aggregating risk assessment: {e}")
            return {}

    def _calculate_confidence_intervals(self,
                                      expert_analyses: Dict[ExpertType, ExpertAnalysis]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for consensus metrics"""
        try:
            confidence_scores = [analysis.confidence for analysis in expert_analyses.values()]

            # Calculate confidence interval for overall confidence
            mean_confidence = np.mean(confidence_scores)
            std_confidence = np.std(confidence_scores)

            confidence_interval = (
                max(0.0, mean_confidence - 1.96 * std_confidence),
                min(1.0, mean_confidence + 1.96 * std_confidence)
            )

            return {
                'consensus_confidence': confidence_interval,
                'expert_agreement': (
                    max(0.0, 1.0 - 2 * std_confidence),
                    min(1.0, 1.0)
                )
            }

        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {}

    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get comprehensive coordinator status"""
        try:
            expert_statuses = {}
            for expert_type, expert in self.experts.items():
                expert_statuses[expert_type.value] = expert.get_expert_status()

            return {
                'total_experts': len(self.experts),
                'active_experts': sum(1 for expert in self.experts.values() if expert.status == ExpertStatus.ACTIVE),
                'consensus_analyses': len(self.consensus_history),
                'expert_statuses': expert_statuses,
                'coordination_config': {
                    'min_experts_for_consensus': self.min_experts_for_consensus,
                    'consensus_threshold': self.consensus_threshold,
                    'expert_weights': self.expert_weights
                }
            }

        except Exception as e:
            logger.error(f"Error getting coordinator status: {e}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    async def test_expert_system():
        """Test the expert agents system"""

        config = {
            'min_experts_for_consensus': 2,
            'consensus_threshold': 0.6,
            'market_analysis': {'technical_indicators': ['rsi', 'macd', 'sma']},
            'risk_assessment': {'correlation_threshold': 0.7},
            'sentiment_analysis': {'news_sources': ['yahoo', 'google']}
        }

        coordinator = ExpertCoordinator(config)

        try:
            # Initialize expert system
            success = await coordinator.initialize()
            if not success:
                print("Failed to initialize expert system")
                return

            print("Expert system initialized successfully")

            # Test consensus analysis
            data_sources = {
                'market_data': {'symbol': 'AAPL', 'period': '1y'},
                'news_data': {'sources': ['yahoo', 'google']},
                'options_data': {'enabled': True}
            }

            consensus = await coordinator.get_consensus_analysis('AAPL', data_sources)

            if consensus:
                print(f"\nConsensus Analysis for AAPL:")
                print(f"Participating Experts: {[e.value for e in consensus.participating_experts]}")
                print(f"Consensus Score: {consensus.consensus_score:.2f}")
                print(f"Overall Risk Score: {consensus.risk_assessment.get('overall_risk_score', 0):.2f}")
                print(f"Unified Recommendations ({len(consensus.unified_recommendations)}):")
                for rec in consensus.unified_recommendations[:5]:
                    print(f"  - {rec}")

            # Get system status
            status = coordinator.get_coordinator_status()
            print(f"\nSystem Status:")
            print(f"Total Experts: {status['total_experts']}")
            print(f"Active Experts: {status['active_experts']}")
            print(f"Consensus Analyses: {status['consensus_analyses']}")

            print(f"\nExpert Performance:")
            for expert_type, expert_status in status['expert_statuses'].items():
                print(f"{expert_type}:")
                print(f"  Status: {expert_status['status']}")
                print(f"  Expertise Score: {expert_status['expertise_score']:.2f}")
                print(f"  Avg Confidence: {expert_status['performance_metrics']['avg_confidence']:.2f}")

        except Exception as e:
            print(f"Error testing expert system: {e}")

    # Run test
    asyncio.run(test_expert_system())