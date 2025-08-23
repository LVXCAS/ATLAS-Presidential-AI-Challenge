"""
Sentiment Analysis Agent for Bloomberg Terminal
Advanced sentiment analysis using news, social media, and market indicators.
"""

import asyncio
import logging
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, TradingSignal, SignalType, AgentStatus

# Optional imports - gracefully handle if not available
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    """
    Advanced sentiment analysis agent using:
    - News headline analysis
    - Social media sentiment (Twitter, Reddit)
    - Market-based sentiment indicators
    - Options flow sentiment
    - Institutional sentiment proxies
    - Real-time sentiment momentum
    """
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        default_config = {
            'news_sources': ['polygon', 'alpaca', 'yahoo'],
            'sentiment_threshold': 0.3,
            'news_lookback_hours': 24,
            'min_news_count': 3,
            'social_weight': 0.3,
            'news_weight': 0.4,
            'market_weight': 0.3,
            'min_confidence': 0.55,
            'signal_interval': 300,  # 5 minutes
            'sentiment_momentum_period': 6,  # hours
            'relevance_threshold': 0.6,
            'enable_social_media': False,  # Disabled by default
            'enable_options_sentiment': False  # Disabled by default
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(
            name="SentimentAgent",
            symbols=symbols,
            config=default_config
        )
        
        # Agent-specific state
        self.sentiment_cache: Dict[str, Dict] = {}
        self.news_cache: Dict[str, List] = {}
        self.sentiment_history: Dict[str, List] = {}
        
        # API endpoints and keys
        self.news_apis = {
            'polygon': 'https://api.polygon.io/v2/reference/news',
            'alpha_vantage': 'https://www.alphavantage.co/query'
        }
        
    async def initialize(self) -> None:
        """Initialize sentiment-specific components."""
        logger.info(f"Initializing {self.name} for sentiment analysis")
        
        # Check dependencies
        if not TEXTBLOB_AVAILABLE:
            logger.warning("TextBlob not available - basic sentiment analysis only")
        
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests not available - news fetching disabled")
        
        # Initialize caches for all symbols
        for symbol in self.symbols:
            self.sentiment_cache[symbol] = {}
            self.news_cache[symbol] = []
            self.sentiment_history[symbol] = []
        
        logger.info(f"{self.name} initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup sentiment-specific resources."""
        self.sentiment_cache.clear()
        self.news_cache.clear()
        self.sentiment_history.clear()
        logger.info(f"{self.name} cleanup completed")
    
    async def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate sentiment-based trading signal.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None if no signal
        """
        try:
            # Calculate sentiment features
            features = await self.calculate_features(symbol)
            if not features:
                return None
            
            # Analyze sentiment patterns
            sentiment_analysis = await self._analyze_sentiment(symbol, features)
            if not sentiment_analysis:
                return None
            
            # Generate signal based on analysis
            signal = await self._generate_sentiment_signal(symbol, sentiment_analysis, features)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating sentiment signal for {symbol}: {e}")
            return None
    
    async def calculate_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate sentiment-specific features.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of feature names to values
        """
        try:
            # Check cache first
            cache_key = f"sentiment_features:{symbol}"
            cached_features = await self.get_cached_feature(cache_key, ttl=300)
            if cached_features:
                return cached_features
            
            features = {}
            
            # Get news sentiment
            news_sentiment = await self._get_news_sentiment(symbol)
            features.update(news_sentiment)
            
            # Get social media sentiment (if enabled)
            if self.config['enable_social_media']:
                social_sentiment = await self._get_social_sentiment(symbol)
                features.update(social_sentiment)
            
            # Get market-based sentiment indicators
            market_sentiment = await self._get_market_sentiment_indicators(symbol)
            features.update(market_sentiment)
            
            # Calculate sentiment momentum
            sentiment_momentum = await self._calculate_sentiment_momentum(symbol, features)
            features.update(sentiment_momentum)
            
            # Get options sentiment (if enabled)
            if self.config['enable_options_sentiment']:
                options_sentiment = await self._get_options_sentiment(symbol)
                features.update(options_sentiment)
            
            # Calculate composite sentiment scores
            composite_scores = self._calculate_composite_sentiment(features)
            features.update(composite_scores)
            
            # Cache the features
            await self.cache_feature(cache_key, features, ttl=300)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating sentiment features for {symbol}: {e}")
            return {}
    
    async def _get_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get sentiment from news headlines and articles."""
        features = {}
        
        try:
            # Get recent news
            news_items = await self._fetch_recent_news(symbol)
            
            if not news_items:
                features.update({
                    'news_sentiment_score': 0.0,
                    'news_sentiment_strength': 0.0,
                    'news_volume': 0,
                    'news_relevance': 0.0
                })
                return features
            
            # Analyze sentiment for each news item
            sentiment_scores = []
            relevance_scores = []
            
            for news_item in news_items:
                headline = news_item.get('headline', '')
                summary = news_item.get('summary', '')
                
                # Calculate sentiment
                headline_sentiment = self._analyze_text_sentiment(headline)
                summary_sentiment = self._analyze_text_sentiment(summary) if summary else 0.0
                
                # Weighted average (headline has more weight)
                combined_sentiment = (headline_sentiment * 0.7) + (summary_sentiment * 0.3)
                sentiment_scores.append(combined_sentiment)
                
                # Calculate relevance
                relevance = self._calculate_news_relevance(headline + " " + summary, symbol)
                relevance_scores.append(relevance)
            
            # Calculate aggregate metrics
            valid_scores = [s for s, r in zip(sentiment_scores, relevance_scores) 
                          if r >= self.config['relevance_threshold']]
            
            if valid_scores:
                features['news_sentiment_score'] = np.mean(valid_scores)
                features['news_sentiment_strength'] = abs(features['news_sentiment_score'])
                features['news_volume'] = len(valid_scores)
                features['news_relevance'] = np.mean([r for r in relevance_scores 
                                                    if r >= self.config['relevance_threshold']])
                
                # Sentiment distribution
                positive_news = len([s for s in valid_scores if s > 0.1])
                negative_news = len([s for s in valid_scores if s < -0.1])
                features['news_sentiment_ratio'] = (positive_news - negative_news) / len(valid_scores)
            else:
                features.update({
                    'news_sentiment_score': 0.0,
                    'news_sentiment_strength': 0.0,
                    'news_volume': 0,
                    'news_relevance': 0.0,
                    'news_sentiment_ratio': 0.0
                })
            
        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            features = {
                'news_sentiment_score': 0.0,
                'news_sentiment_strength': 0.0,
                'news_volume': 0,
                'news_relevance': 0.0,
                'news_sentiment_ratio': 0.0
            }
        
        return features
    
    async def _fetch_recent_news(self, symbol: str) -> List[Dict]:
        """Fetch recent news for a symbol."""
        if not REQUESTS_AVAILABLE:
            return []
        
        try:
            # Check cache first
            cache_key = f"news:{symbol}"
            cached_news = await self.get_cached_feature(cache_key, ttl=1800)  # 30 minutes
            if cached_news:
                return cached_news
            
            news_items = []
            
            # Mock news data for demonstration (replace with actual API calls)
            mock_news = [
                {
                    'headline': f'{symbol} reports strong quarterly earnings beating expectations',
                    'summary': f'{symbol} exceeded analyst expectations with strong revenue growth',
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'source': 'mock'
                },
                {
                    'headline': f'Analysts raise price target for {symbol} citing market expansion',
                    'summary': f'Investment firm raises {symbol} target price due to positive outlook',
                    'timestamp': datetime.now() - timedelta(hours=5),
                    'source': 'mock'
                },
                {
                    'headline': f'{symbol} announces new product line expected to drive growth',
                    'summary': f'Company unveils innovative products in key market segments',
                    'timestamp': datetime.now() - timedelta(hours=8),
                    'source': 'mock'
                }
            ]
            
            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=self.config['news_lookback_hours'])
            news_items = [item for item in mock_news if item['timestamp'] > cutoff_time]
            
            # Cache the results
            await self.cache_feature(cache_key, news_items, ttl=1800)
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using available tools."""
        if not text or not text.strip():
            return 0.0
        
        try:
            if TEXTBLOB_AVAILABLE:
                # Use TextBlob for sentiment analysis
                blob = TextBlob(text)
                return blob.sentiment.polarity  # Returns -1 to 1
            else:
                # Simple keyword-based sentiment analysis
                return self._simple_keyword_sentiment(text)
                
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return 0.0
    
    def _simple_keyword_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment analysis."""
        text = text.lower()
        
        positive_keywords = [
            'strong', 'growth', 'increase', 'beat', 'exceed', 'positive', 'bull',
            'rally', 'gain', 'rise', 'up', 'buy', 'upgrade', 'outperform',
            'bullish', 'optimistic', 'promising', 'good', 'excellent', 'success'
        ]
        
        negative_keywords = [
            'weak', 'decline', 'decrease', 'miss', 'below', 'negative', 'bear',
            'fall', 'drop', 'down', 'sell', 'downgrade', 'underperform',
            'bearish', 'pessimistic', 'concerning', 'bad', 'poor', 'failure'
        ]
        
        positive_count = sum(1 for word in positive_keywords if word in text)
        negative_count = sum(1 for word in negative_keywords if word in text)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0
        
        total_count = positive_count + negative_count
        sentiment_score = (positive_count - negative_count) / total_count
        
        return sentiment_score
    
    def _calculate_news_relevance(self, text: str, symbol: str) -> float:
        """Calculate relevance of news text to the symbol."""
        text = text.lower()
        symbol_lower = symbol.lower()
        
        # Basic relevance scoring
        relevance_score = 0.0
        
        # Symbol mention
        if symbol_lower in text:
            relevance_score += 0.5
        
        # Company-specific keywords (this would be enhanced with actual company data)
        company_keywords = [symbol_lower, 'company', 'stock', 'shares', 'earnings']
        keyword_matches = sum(1 for keyword in company_keywords if keyword in text)
        relevance_score += min(keyword_matches * 0.1, 0.3)
        
        # Financial keywords
        financial_keywords = ['revenue', 'profit', 'earnings', 'guidance', 'outlook']
        financial_matches = sum(1 for keyword in financial_keywords if keyword in text)
        relevance_score += min(financial_matches * 0.05, 0.2)
        
        return min(relevance_score, 1.0)
    
    async def _get_social_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get sentiment from social media sources."""
        # Placeholder implementation - would integrate with Twitter API, Reddit API, etc.
        features = {
            'social_sentiment_score': 0.0,
            'social_volume': 0,
            'social_engagement': 0.0,
            'social_momentum': 0.0
        }
        
        try:
            # Mock social sentiment data
            import random
            features['social_sentiment_score'] = random.uniform(-0.5, 0.5)
            features['social_volume'] = random.randint(50, 500)
            features['social_engagement'] = random.uniform(0, 1)
            features['social_momentum'] = random.uniform(-0.3, 0.3)
            
        except Exception as e:
            logger.error(f"Error getting social sentiment: {e}")
        
        return features
    
    async def _get_market_sentiment_indicators(self, symbol: str) -> Dict[str, float]:
        """Get market-based sentiment indicators."""
        features = {}
        
        try:
            # Get market data for sentiment analysis
            df = await self.get_market_data(symbol, 50)
            if df is None or df.empty:
                return {
                    'price_momentum_sentiment': 0.0,
                    'volume_sentiment': 0.0,
                    'volatility_sentiment': 0.0
                }
            
            # Price momentum sentiment
            if len(df) >= 10:
                recent_returns = df['price'].pct_change().tail(10)
                price_momentum = recent_returns.mean()
                features['price_momentum_sentiment'] = np.tanh(price_momentum * 100)  # Normalize
            
            # Volume sentiment
            if 'volume' in df.columns and len(df) >= 20:
                recent_volume = df['volume'].tail(5).mean()
                avg_volume = df['volume'].tail(20).mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                features['volume_sentiment'] = np.tanh((volume_ratio - 1) * 2)  # Normalize
            
            # Volatility sentiment (high vol = negative sentiment)
            if len(df) >= 10:
                volatility = df['price'].pct_change().tail(10).std()
                features['volatility_sentiment'] = -np.tanh(volatility * 50)  # High vol = negative
            
            # Put/Call ratio sentiment (placeholder)
            features['put_call_sentiment'] = 0.0
            
        except Exception as e:
            logger.error(f"Error getting market sentiment indicators: {e}")
            features = {
                'price_momentum_sentiment': 0.0,
                'volume_sentiment': 0.0,
                'volatility_sentiment': 0.0,
                'put_call_sentiment': 0.0
            }
        
        return features
    
    async def _calculate_sentiment_momentum(self, symbol: str, current_features: Dict[str, float]) -> Dict[str, float]:
        """Calculate sentiment momentum over time."""
        features = {}
        
        try:
            # Store current sentiment in history
            current_sentiment = current_features.get('news_sentiment_score', 0.0)
            
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            
            self.sentiment_history[symbol].append({
                'timestamp': datetime.now(),
                'sentiment': current_sentiment
            })
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(hours=self.config['sentiment_momentum_period'])
            self.sentiment_history[symbol] = [
                item for item in self.sentiment_history[symbol]
                if item['timestamp'] > cutoff_time
            ]
            
            # Calculate momentum
            if len(self.sentiment_history[symbol]) >= 3:
                sentiment_values = [item['sentiment'] for item in self.sentiment_history[symbol]]
                
                # Simple momentum calculation
                recent_sentiment = np.mean(sentiment_values[-3:])
                older_sentiment = np.mean(sentiment_values[:-3]) if len(sentiment_values) > 3 else sentiment_values[0]
                
                features['sentiment_momentum'] = recent_sentiment - older_sentiment
                features['sentiment_trend'] = 1 if features['sentiment_momentum'] > 0.1 else (-1 if features['sentiment_momentum'] < -0.1 else 0)
                features['sentiment_consistency'] = 1.0 - np.std(sentiment_values) if sentiment_values else 0.0
            else:
                features['sentiment_momentum'] = 0.0
                features['sentiment_trend'] = 0
                features['sentiment_consistency'] = 0.0
                
        except Exception as e:
            logger.error(f"Error calculating sentiment momentum: {e}")
            features = {
                'sentiment_momentum': 0.0,
                'sentiment_trend': 0,
                'sentiment_consistency': 0.0
            }
        
        return features
    
    async def _get_options_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get sentiment from options flow and positioning."""
        # Placeholder implementation
        features = {
            'options_put_call_ratio': 0.0,
            'options_volume_sentiment': 0.0,
            'options_skew_sentiment': 0.0
        }
        
        try:
            # Mock options sentiment data
            import random
            features['options_put_call_ratio'] = random.uniform(0.5, 2.0)
            features['options_volume_sentiment'] = random.uniform(-0.5, 0.5)
            features['options_skew_sentiment'] = random.uniform(-0.3, 0.3)
            
        except Exception as e:
            logger.error(f"Error getting options sentiment: {e}")
        
        return features
    
    def _calculate_composite_sentiment(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite sentiment scores from all sources."""
        composite_features = {}
        
        try:
            # Extract individual sentiment components
            news_sentiment = features.get('news_sentiment_score', 0.0)
            social_sentiment = features.get('social_sentiment_score', 0.0)
            market_sentiment = (
                features.get('price_momentum_sentiment', 0.0) +
                features.get('volume_sentiment', 0.0) +
                features.get('volatility_sentiment', 0.0)
            ) / 3.0
            
            # Weighted composite score
            weights = {
                'news': self.config['news_weight'],
                'social': self.config['social_weight'],
                'market': self.config['market_weight']
            }
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
            
            composite_features['composite_sentiment_score'] = (
                news_sentiment * weights['news'] +
                social_sentiment * weights['social'] +
                market_sentiment * weights['market']
            )
            
            composite_features['composite_sentiment_strength'] = abs(composite_features['composite_sentiment_score'])
            
            # Sentiment agreement (how aligned different sources are)
            sentiment_scores = [news_sentiment, social_sentiment, market_sentiment]
            sentiment_scores = [s for s in sentiment_scores if abs(s) > 0.01]  # Filter near-zero
            
            if len(sentiment_scores) >= 2:
                # Calculate agreement as inverse of standard deviation
                sentiment_std = np.std(sentiment_scores)
                composite_features['sentiment_agreement'] = max(0, 1 - sentiment_std)
            else:
                composite_features['sentiment_agreement'] = 0.5
            
            # Sentiment conviction (combines strength and agreement)
            composite_features['sentiment_conviction'] = (
                composite_features['composite_sentiment_strength'] * 0.7 +
                composite_features['sentiment_agreement'] * 0.3
            )
            
        except Exception as e:
            logger.error(f"Error calculating composite sentiment: {e}")
            composite_features = {
                'composite_sentiment_score': 0.0,
                'composite_sentiment_strength': 0.0,
                'sentiment_agreement': 0.0,
                'sentiment_conviction': 0.0
            }
        
        return composite_features
    
    async def _analyze_sentiment(self, symbol: str, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Comprehensive sentiment analysis.
        
        Args:
            symbol: Trading symbol
            features: Calculated features
            
        Returns:
            Sentiment analysis results
        """
        try:
            analysis = {
                'overall_sentiment': 'NEUTRAL',
                'sentiment_strength': 0.0,
                'confidence_factors': [],
                'risk_factors': [],
                'sentiment_sources': {},
                'momentum_factor': 0.0
            }
            
            composite_sentiment = features.get('composite_sentiment_score', 0.0)
            sentiment_strength = features.get('composite_sentiment_strength', 0.0)
            sentiment_conviction = features.get('sentiment_conviction', 0.0)
            
            # Determine overall sentiment
            if composite_sentiment > self.config['sentiment_threshold']:
                analysis['overall_sentiment'] = 'BULLISH'
            elif composite_sentiment < -self.config['sentiment_threshold']:
                analysis['overall_sentiment'] = 'BEARISH'
            else:
                analysis['overall_sentiment'] = 'NEUTRAL'
            
            analysis['sentiment_strength'] = sentiment_strength
            
            # Analyze individual sources
            news_sentiment = features.get('news_sentiment_score', 0.0)
            news_volume = features.get('news_volume', 0)
            
            if abs(news_sentiment) > 0.2 and news_volume >= self.config['min_news_count']:
                direction = "positive" if news_sentiment > 0 else "negative"
                analysis['confidence_factors'].append(f"Strong {direction} news sentiment ({news_volume} articles)")
                analysis['sentiment_sources']['news'] = news_sentiment
            
            # Social sentiment analysis
            social_sentiment = features.get('social_sentiment_score', 0.0)
            if abs(social_sentiment) > 0.2:
                direction = "positive" if social_sentiment > 0 else "negative"
                analysis['confidence_factors'].append(f"Social media sentiment {direction}")
                analysis['sentiment_sources']['social'] = social_sentiment
            
            # Market sentiment analysis
            market_momentum = features.get('price_momentum_sentiment', 0.0)
            if abs(market_momentum) > 0.2:
                direction = "positive" if market_momentum > 0 else "negative"
                analysis['confidence_factors'].append(f"Market momentum {direction}")
                analysis['sentiment_sources']['market'] = market_momentum
            
            # Sentiment momentum analysis
            sentiment_momentum = features.get('sentiment_momentum', 0.0)
            analysis['momentum_factor'] = sentiment_momentum
            
            if abs(sentiment_momentum) > 0.1:
                direction = "improving" if sentiment_momentum > 0 else "deteriorating"
                analysis['confidence_factors'].append(f"Sentiment {direction}")
            
            # Risk factors
            sentiment_agreement = features.get('sentiment_agreement', 0.0)
            if sentiment_agreement < 0.3:
                analysis['risk_factors'].append("Low agreement between sentiment sources")
            
            if news_volume < self.config['min_news_count']:
                analysis['risk_factors'].append("Insufficient news coverage for reliable sentiment")
            
            news_relevance = features.get('news_relevance', 0.0)
            if news_relevance < self.config['relevance_threshold']:
                analysis['risk_factors'].append("Low news relevance to symbol")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return None
    
    async def _generate_sentiment_signal(
        self, 
        symbol: str, 
        analysis: Dict[str, Any], 
        features: Dict[str, float]
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal based on sentiment analysis.
        
        Args:
            symbol: Trading symbol
            analysis: Sentiment analysis results
            features: Calculated features
            
        Returns:
            TradingSignal or None
        """
        try:
            overall_sentiment = analysis['overall_sentiment']
            sentiment_strength = analysis['sentiment_strength']
            sentiment_conviction = features.get('sentiment_conviction', 0.0)
            
            # Require minimum conviction
            if sentiment_conviction < 0.4 or sentiment_strength < self.config['sentiment_threshold']:
                return None
            
            # Determine signal type
            momentum_factor = analysis['momentum_factor']
            
            if overall_sentiment == 'BULLISH':
                if sentiment_strength > 0.6 and momentum_factor > 0.1:
                    signal_type = SignalType.STRONG_BUY
                else:
                    signal_type = SignalType.BUY
            elif overall_sentiment == 'BEARISH':
                if sentiment_strength > 0.6 and momentum_factor < -0.1:
                    signal_type = SignalType.STRONG_SELL
                else:
                    signal_type = SignalType.SELL
            else:
                return None  # No signal for neutral sentiment
            
            # Calculate confidence
            base_confidence = min(sentiment_conviction * 0.8, 0.85)
            
            # Boost confidence for multiple confirming sources
            source_count = len(analysis['sentiment_sources'])
            if source_count >= 2:
                confidence_boost = min(source_count * 0.05, 0.1)
                base_confidence = min(base_confidence + confidence_boost, 0.85)
            
            # Momentum boost
            if abs(momentum_factor) > 0.15:
                base_confidence = min(base_confidence + 0.05, 0.85)
            
            # Risk factor penalty
            risk_penalty = len(analysis['risk_factors']) * 0.08
            confidence = max(base_confidence - risk_penalty, 0.3)
            
            # Skip if confidence too low
            if confidence < self.config['min_confidence']:
                return None
            
            # Calculate targets (sentiment signals typically have longer horizons)
            price_target_pct = sentiment_strength * 0.04  # Up to 4% target
            stop_loss_pct = 0.025  # 2.5% stop loss
            
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return None
            
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
                strength=sentiment_strength,
                reasoning={
                    'analysis': analysis,
                    'sentiment_breakdown': {
                        'composite_score': features.get('composite_sentiment_score', 0.0),
                        'news_sentiment': features.get('news_sentiment_score', 0.0),
                        'social_sentiment': features.get('social_sentiment_score', 0.0),
                        'market_sentiment': features.get('price_momentum_sentiment', 0.0)
                    },
                    'conviction_factors': {
                        'strength': sentiment_strength,
                        'agreement': features.get('sentiment_agreement', 0.0),
                        'momentum': momentum_factor
                    }
                },
                features_used=features,
                prediction_horizon=60,  # 1 hour for sentiment signals
                target_price=target_price,
                stop_loss=stop_loss,
                risk_score=len(analysis['risk_factors']) / 5.0,
                expected_return=price_target_pct
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating sentiment signal: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            if self.market_data_service:
                return await self.market_data_service.get_latest_price(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None


# Convenience function for creating sentiment agent
def create_sentiment_agent(symbols: List[str], **kwargs) -> SentimentAgent:
    """Create a sentiment analysis agent with default configuration."""
    return SentimentAgent(symbols, kwargs)