\"\"\"
Short Selling Agent for the LangGraph Adaptive Multi-Strategy AI Trading System.

This agent implements sophisticated short selling strategies that identify overvalued securities,
analyze borrowing costs and availability, and manage short squeeze risks. It integrates with 
sentiment analysis, fundamental data, and technical indicators to generate high-conviction 
short selling signals.

Key Features:
- Overvaluation detection using multiple valuation metrics
- Borrow cost and availability analysis
- Short squeeze risk management
- Integration with sentiment and fundamental analysis
- Dynamic position sizing based on conviction levels
\"\"\"

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
 from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from config.settings import SHORT_SELLING_SETTINGS
from data.market_data import get_historical_data, get_real_time_data
from strategies.technical_indicators import TechnicalIndicators
from strategies.fibonacci_analysis import FibonacciAnalyzer
from .news_sentiment_agent import NewsSentimentAgent
from .risk_manager_agent import RiskManagerAgent

logger = logging.getLogger(__name__)

class ShortSignalType(str, Enum):
    \"\"\"Types of short selling signals\"\"\"
    OVERVALUATION = "overvaluation"
    TECHNICAL_DIVERGENCE = "technical_divergence"
    SENTIMENT_DIVERGENCE = "sentiment_divergence"
    FUNDAMENTAL_DETERIORATION = "fundamental_deterioration"
    SHORT_SQUEEZE_RECOVERY = "short_squeeze_recovery"

class ShortPositionData(BaseModel):
    \"\"\"Data model for short position information\"\"\"
    symbol: str
    price: float
    quantity: float
    timestamp: datetime
    conviction_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    borrow_cost: float
    borrow_availability: bool
    signal_type: ShortSignalType
    explanation: str
    technical_factors: List[str]
    fundamental_factors: List[str]
    sentiment_factors: List[str]

class ShortSellingMetrics(BaseModel):
    \"\"\"Metrics for evaluating short selling opportunities\"\"\"
    symbol: str
    valuation_score: float = Field(ge=0.0, le=1.0)  # Higher = more overvalued
    technical_score: float = Field(ge=0.0, le=1.0)  # Higher = more bearish technicals
    sentiment_score: float = Field(ge=0.0, le=1.0)  # Higher = more negative sentiment
    fundamental_score: float = Field(ge=0.0, le=1.0)  # Higher = deteriorating fundamentals
    short_interest_score: float = Field(ge=0.0, le=1.0)  # Higher = more crowded short
    borrow_cost: float  # Annualized borrowing cost percentage
    borrow_available: bool  # Whether shares are available to borrow
    short_squeeze_risk: float = Field(ge=0.0, le=1.0)  # Higher = more short squeeze risk
    overall_score: float = Field(ge=0.0, le=1.0)  # Composite score

class ShortSellingAgent:
    \"\"\"LangGraph agent for sophisticated short selling strategies\"\"\"
    
    def __init__(self):
        self.name = \"short_selling_agent\"
        self.settings = SHORT_SELLING_SETTINGS
        self.technical_indicators = TechnicalIndicators()
        self.fibonacci_analyzer = FibonacciAnalyzer()
        self.news_sentiment_agent = NewsSentimentAgent()
        self.risk_manager = RiskManagerAgent()
        self.valuation_metrics = {
            'pe_ratio': 25.0,  # Industry average P/E
            'pb_ratio': 3.0,   # Industry average P/B
            'ps_ratio': 5.0,   # Industry average P/S
        }
        
    async def analyze_short_opportunities(self, symbols: List[str]) -> List[ShortSellingMetrics]:
        \"\"\"Analyze multiple symbols for short selling opportunities\"\"\"
        short_metrics = []
        
        for symbol in symbols:
            try:
                metrics = await self._analyze_single_symbol(symbol)
                short_metrics.append(metrics)
            except Exception as e:
                logger.error(f\"Error analyzing {symbol} for short opportunities: {e}\")
                continue
                
        # Sort by overall score (highest first)
        short_metrics.sort(key=lambda x: x.overall_score, reverse=True)
        return short_metrics
        
    async def _analyze_single_symbol(self, symbol: str) -> ShortSellingMetrics:
        \"\"\"Analyze a single symbol for short selling opportunity\"\"\"
        # Get historical and real-time data
        hist_data = await get_historical_data(symbol, days=90)
        real_time_data = await get_real_time_data(symbol)
        
        # Calculate various scores
        valuation_score = await self._calculate_valuation_score(symbol, hist_data, real_time_data)
        technical_score = await self._calculate_technical_score(symbol, hist_data)
        sentiment_score = await self._calculate_sentiment_score(symbol)
        fundamental_score = await self._calculate_fundamental_score(symbol)
        short_interest_data = await self._get_short_interest_data(symbol)
        short_interest_score = short_interest_data['score']
        borrow_cost = short_interest_data['borrow_cost']
        borrow_available = short_interest_data['borrow_available']
        
        # Calculate short squeeze risk
        short_squeeze_risk = await self._calculate_short_squeeze_risk(
            short_interest_score, technical_score, sentiment_score
        )
        
        # Calculate overall score with weighted components
        overall_score = self._calculate_overall_score(
            valuation_score, technical_score, sentiment_score, 
            fundamental_score, short_interest_score, borrow_cost, borrow_available
        )
        
        return ShortSellingMetrics(
            symbol=symbol,
            valuation_score=valuation_score,
            technical_score=technical_score,
            sentiment_score=sentiment_score,
            fundamental_score=fundamental_score,
            short_interest_score=short_interest_score,
            borrow_cost=borrow_cost,
            borrow_available=borrow_available,
            short_squeeze_risk=short_squeeze_risk,
            overall_score=overall_score
        )
        
    async def _calculate_valuation_score(self, symbol: str, hist_data: pd.DataFrame, real_time_data: Dict) -> float:
        \"\"\"Calculate valuation-based short score\"\"\"
        try:
            # Get current metrics
            current_pe = real_time_data.get('pe_ratio', self.valuation_metrics['pe_ratio'])
            current_pb = real_time_data.get('pb_ratio', self.valuation_metrics['pb_ratio'])
            current_ps = real_time_data.get('ps_ratio', self.valuation_metrics['ps_ratio'])
            
            # Calculate percentile ranks vs historical data
            hist_pe = hist_data['pe_ratio'].dropna()
            hist_pb = hist_data['pb_ratio'].dropna()
            hist_ps = hist_data['ps_ratio'].dropna()
            
            if len(hist_pe) > 10:
                pe_percentile = np.percentile(hist_pe, 80)  # 80th percentile as overvalued threshold
                pe_score = min(1.0, max(0.0, (current_pe - pe_percentile) / (pe_percentile * 0.5)))
            else:
                pe_score = 0.5 if current_pe > self.valuation_metrics['pe_ratio'] else 0.0
                
            if len(hist_pb) > 10:
                pb_percentile = np.percentile(hist_pb, 80)
                pb_score = min(1.0, max(0.0, (current_pb - pb_percentile) / (pb_percentile * 0.5)))
            else:
                pb_score = 0.5 if current_pb > self.valuation_metrics['pb_ratio'] else 0.0
                
            if len(hist_ps) > 10:
                ps_percentile = np.percentile(hist_ps, 80)
                ps_score = min(1.0, max(0.0, (current_ps - ps_percentile) / (ps_percentile * 0.5)))
            else:
                ps_score = 0.5 if current_ps > self.valuation_metrics['ps_ratio'] else 0.0
            
            # Weighted average of valuation metrics
            valuation_score = (pe_score * 0.4 + pb_score * 0.3 + ps_score * 0.3)
            return min(1.0, valuation_score)
        except Exception as e:
            logger.warning(f\"Error calculating valuation score for {symbol}: {e}\")
            return 0.0
            
    async def _calculate_technical_score(self, symbol: str, hist_data: pd.DataFrame) -> float:
        \"\"\"Calculate technical-based short score\"\"\"
        try:
            if len(hist_data) < 20:
                return 0.0
                
            # Calculate technical indicators
            rsi = self.technical_indicators.calculate_rsi(hist_data['close'], period=14)
            macd_line, signal_line, _ = self.technical_indicators.calculate_macd(hist_data['close'])
            upper_band, middle_band, lower_band = self.technical_indicators.calculate_bollinger_bands(hist_data['close'])
            
            # Get latest values
            latest_rsi = rsi.iloc[-1] if not rsi.empty else 50
            latest_macd = macd_line.iloc[-1] if not macd_line.empty else 0
            latest_signal = signal_line.iloc[-1] if not signal_line.empty else 0
            latest_price = hist_data['close'].iloc[-1]
            latest_upper = upper_band.iloc[-1] if not upper_band.empty else latest_price
            latest_middle = middle_band.iloc[-1] if not middle_band.empty else latest_price
            
            # Technical bearish signals (higher score = more bearish)
            rsi_score = max(0.0, (latest_rsi - 70) / 30) if latest_rsi > 70 else 0.0  # Overbought
            macd_score = 1.0 if latest_macd < latest_signal else 0.0  # Bearish crossover
            bollinger_score = max(0.0, (latest_price - latest_upper) / (latest_upper * 0.05)) if latest_price > latest_upper else 0.0  # Above upper band
            
            # Weighted average
            technical_score = (rsi_score * 0.4 + macd_score * 0.3 + bollinger_score * 0.3)
            return min(1.0, technical_score)
        except Exception as e:
            logger.warning(f\"Error calculating technical score for {symbol}: {e}\")
            return 0.0
            
    async def _calculate_sentiment_score(self, symbol: str) -> float:
        \"\"\"Calculate sentiment-based short score\"\"\"
        try:
            # Get sentiment analysis for the symbol
            sentiment_data = await self.news_sentiment_agent.analyze_sentiment_for_symbol(symbol)
            
            if not sentiment_data:
                return 0.0
                
            # Extract sentiment scores
            overall_sentiment = sentiment_data.get('overall_sentiment', 0.0)
            confidence = sentiment_data.get('confidence', 0.5)
            
            # Convert to short score (negative sentiment = positive for shorting)
            # Range: -1 (very negative) to 1 (very positive)
            # For shorting, we want negative sentiment, so we invert and normalize
            sentiment_score = max(0.0, min(1.0, (-overall_sentiment + 1) / 2))
            
            # Weight by confidence
            return sentiment_score * confidence
        except Exception as e:
            logger.warning(f\"Error calculating sentiment score for {symbol}: {e}\")
            return 0.0
            
    async def _calculate_fundamental_score(self, symbol: str) -> float:
        \"\"\"Calculate fundamental-based short score\"\"\"
        try:
            # In a real implementation, this would fetch actual fundamental data
            # For now, we'll simulate with some basic logic
            
            # Simulate fundamental deterioration detection
            # This could include declining earnings, increasing debt, decreasing margins, etc.
            fundamental_deterioration_signals = 0
            total_signals = 5  # Number of signals we're checking
            
            # In a real implementation, we would check:
            # 1. Earnings surprises (negative)
            # 2. Revenue growth deceleration
            # 3. Margin compression
            # 4. Debt-to-equity increasing
            # 5. Insider selling
            
            # For simulation, we'll return a random score weighted toward lower values
            # since fundamental deterioration is relatively rare
            fundamental_score = np.random.beta(2, 5)  # Skewed toward lower values
            return fundamental_score
        except Exception as e:
            logger.warning(f\"Error calculating fundamental score for {symbol}: {e}\")
            return 0.0
            
    async def _get_short_interest_data(self, symbol: str) -> Dict:
        \"\"\"Get short interest and borrowing data\"\"\"
        try:
            # In a real implementation, this would connect to a data provider
            # that provides short interest and borrow data
            
            # Simulate short interest data
            short_interest = np.random.uniform(0.01, 0.15)  # 1-15% short interest
            days_to_cover = short_interest * 252 / 12  # Rough approximation
            
            # Short interest score (higher = more crowded short)
            short_interest_score = min(1.0, short_interest / 0.10)  # Normalize to 10% threshold
            
            # Borrow cost simulation (higher for hard-to-borrow stocks)
            base_borrow_cost = 0.5  # 0.5% base rate
            borrow_cost_multiplier = 1 + (short_interest_score * 2)  # Up to 3x for high short interest
            borrow_cost = base_borrow_cost * borrow_cost_multiplier
            
            # Borrow availability (lower for hard-to-borrow stocks)
            borrow_available = short_interest < 0.20  # Generally available below 20% short interest
            
            return {
                'score': short_interest_score,
                'borrow_cost': borrow_cost,
                'borrow_available': borrow_available
            }
        except Exception as e:
            logger.warning(f\"Error getting short interest data for {symbol}: {e}\")
            return {
                'score': 0.0,
                'borrow_cost': 1.0,  # Default higher cost
                'borrow_available': False
            }
            
    async def _calculate_short_squeeze_risk(self, short_interest_score: float, technical_score: float, sentiment_score: float) -> float:
        \"\"\"Calculate short squeeze risk\"\"\"
        # High short interest + bullish technicals + positive sentiment = high short squeeze risk
        squeeze_risk = (
            short_interest_score * 0.5 +  # 50% weight to short interest
            (1 - technical_score) * 0.3 +  # 30% weight to technicals (inverted)
            (1 - sentiment_score) * 0.2    # 20% weight to sentiment (inverted)
        )
        return min(1.0, squeeze_risk)
        
    def _calculate_overall_score(self, valuation_score: float, technical_score: float, 
                                sentiment_score: float, fundamental_score: float,
                                short_interest_score: float, borrow_cost: float, 
                                borrow_available: bool) -> float:
        \"\"\"Calculate overall short selling score\"\"\"
        # Weight components
        weighted_valuation = valuation_score * 0.25
        weighted_technical = technical_score * 0.20
        weighted_sentiment = sentiment_score * 0.15
        weighted_fundamental = fundamental_score * 0.15
        weighted_short_interest = short_interest_score * 0.10
        
        # Borrow cost penalty (higher costs reduce score)
        borrow_cost_penalty = max(0.0, (borrow_cost - 5.0) / 10.0)  # Start penalizing above 5%
        borrow_cost_factor = max(0.0, 1.0 - borrow_cost_penalty)
        
        # Borrow availability factor (unavailable = 0 score)
        availability_factor = 1.0 if borrow_available else 0.0
        
        # Combine all factors
        overall_score = (
            weighted_valuation + 
            weighted_technical + 
            weighted_sentiment + 
            weighted_fundamental + 
            weighted_short_interest
        ) * borrow_cost_factor * availability_factor
        
        return min(1.0, max(0.0, overall_score))
        
    async def generate_short_signals(self, symbols: List[str], max_positions: int = 5) -> List[ShortPositionData]:
        \"\"\"Generate short selling signals for top opportunities\"\"\"
        # Analyze all symbols
        short_metrics = await self.analyze_short_opportunities(symbols)
        
        # Filter for qualified opportunities (score > threshold)
        qualified_opportunities = [
            metric for metric in short_metrics 
            if metric.overall_score > self.settings.min_short_score_threshold
            and metric.borrow_available
            and metric.borrow_cost < self.settings.max_borrow_cost
        ]
        
        # Sort by overall score and take top N
        qualified_opportunities.sort(key=lambda x: x.overall_score, reverse=True)
        top_opportunities = qualified_opportunities[:max_positions]
        
        # Generate signals for top opportunities
        signals = []
        for opportunity in top_opportunities:
            try:
                signal = await self._generate_signal_for_opportunity(opportunity)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f\"Error generating signal for {opportunity.symbol}: {e}\")
                continue
                
        return signals
        
    async def _generate_signal_for_opportunity(self, opportunity: ShortSellingMetrics) -> Optional[ShortPositionData]:
        \"\"\"Generate detailed short position signal for an opportunity\"\"\"
        try:
            # Get current price
            real_time_data = await get_real_time_data(opportunity.symbol)
            current_price = real_time_data.get('price', 0.0)
            
            if current_price <= 0:
                return None
                
            # Determine signal type based on highest contributing factor
            score_components = {
                'valuation': opportunity.valuation_score,
                'technical': opportunity.technical_score,
                'sentiment': opportunity.sentiment_score,
                'fundamental': opportunity.fundamental_score
            }
            
            signal_type = ShortSignalType(max(score_components, key=score_components.get))
            
            # Create explanation
            explanation = self._create_explanation(opportunity, signal_type)
            
            # Determine technical, fundamental, and sentiment factors
            technical_factors = self._extract_technical_factors(opportunity)
            fundamental_factors = self._extract_fundamental_factors(opportunity)
            sentiment_factors = self._extract_sentiment_factors(opportunity)
            
            # Calculate conviction score (based on overall score and risk)
            conviction_score = max(0.1, opportunity.overall_score * (1 - opportunity.short_squeeze_risk))
            
            # Calculate risk score
            risk_score = opportunity.short_squeeze_risk
            
            # Calculate borrow cost
            borrow_cost = opportunity.borrow_cost
            
            # Check borrow availability
            borrow_availability = opportunity.borrow_available
            
            return ShortPositionData(
                symbol=opportunity.symbol,
                price=current_price,
                quantity=0.0,  # Quantity will be determined by portfolio allocator
                timestamp=datetime.now(),
                conviction_score=conviction_score,
                risk_score=risk_score,
                borrow_cost=borrow_cost,
                borrow_availability=borrow_availability,
                signal_type=signal_type,
                explanation=explanation,
                technical_factors=technical_factors,
                fundamental_factors=fundamental_factors,
                sentiment_factors=sentiment_factors
            )
        except Exception as e:
            logger.error(f\"Error generating signal for {opportunity.symbol}: {e}\")
            return None
            
    def _create_explanation(self, opportunity: ShortSellingMetrics, signal_type: ShortSignalType) -> str:
        \"\"\"Create human-readable explanation for the short signal\"\"\"
        explanations = {
            ShortSignalType.OVERVALUATION: (
                f\"{opportunity.symbol} appears overvalued based on multiple valuation metrics. \" +
                f\"Valuation score: {opportunity.valuation_score:.2f}. \" +
                f\"Consider shorting as the stock may revert to fair value.\"
            ),
            ShortSignalType.TECHNICAL_DIVERGENCE: (
                f\"{opportunity.symbol} showing bearish technical patterns with score {opportunity.technical_score:.2f}. \" +
                f\"Technical indicators suggest downside momentum.\"
            ),
            ShortSignalType.SENTIMENT_DIVERGENCE: (
                f\"Negative sentiment detected for {opportunity.symbol} with sentiment score {opportunity.sentiment_score:.2f}. \" +
                f\"Market perception is turning negative.\"
            ),
            ShortSignalType.FUNDAMENTAL_DETERIORATION: (
                f\"Fundamental deterioration detected for {opportunity.symbol} with score {opportunity.fundamental_score:.2f}. \" +
                f\"Company fundamentals appear to be weakening.\"
            ),
            ShortSignalType.SHORT_SQUEEZE_RECOVERY: (
                f\"{opportunity.symbol} showing signs of short squeeze recovery potential. \" +
                f\"Short interest score: {opportunity.short_interest_score:.2f}. \" +
                f\"May be an opportunity to cover shorts or avoid new short positions.\"
            )
        }
        return explanations.get(signal_type, f\"Short signal generated for {opportunity.symbol} with overall score {opportunity.overall_score:.2f}\")
        
    def _extract_technical_factors(self, opportunity: ShortSellingMetrics) -> List[str]:
        \"\"\"Extract key technical factors\"\"\"
        factors = []
        if opportunity.technical_score > 0.7:
            factors.append(\"Strong bearish technicals\")
        elif opportunity.technical_score > 0.5:
            factors.append(\"Moderate bearish technicals\")
        else:
            factors.append(\"Neutral technicals\")
            
        if opportunity.valuation_score > 0.7:
            factors.append(\"Overvalued technically\")
            
        return factors
        
    def _extract_fundamental_factors(self, opportunity: ShortSellingMetrics) -> List[str]:
        \"\"\"Extract key fundamental factors\"\"\"
        factors = []
        if opportunity.fundamental_score > 0.7:
            factors.append(\"Deteriorating fundamentals\")
        elif opportunity.fundamental_score > 0.5:
            factors.append(\"Some fundamental concerns\")
        else:
            factors.append(\"Stable fundamentals\")
            
        return factors
        
    def _extract_sentiment_factors(self, opportunity: ShortSellingMetrics) -> List[str]:
        \"\"\"Extract key sentiment factors\"\"\"
        factors = []
        if opportunity.sentiment_score > 0.7:
            factors.append(\"Strongly negative sentiment\")
        elif opportunity.sentiment_score > 0.5:
            factors.append(\"Moderately negative sentiment\")
        else:
            factors.append(\"Neutral sentiment\")
            
        return factors

# Create global instance
short_selling_agent = ShortSellingAgent()

# Convenience function for LangGraph integration
async def short_selling_agent_node(state):
    \"\"\"LangGraph node function for short selling agent\"\"\"
    try:
        agent = ShortSellingAgent()
        symbols = state.get('symbols', [])
        
        if not symbols:
            logger.warning(\"No symbols provided for short selling analysis\")
            return {\"short_signals\": []}
            
        # Generate short signals
        signals = await agent.generate_short_signals(symbols)
        
        return {
            \"short_signals\": signals,
            \"analysis_timestamp\": datetime.now()
        }
    except Exception as e:
        logger.error(f\"Error in short selling agent node: {e}\")
        return {
            \"short_signals\": [],
            \"error\": str(e)
        }