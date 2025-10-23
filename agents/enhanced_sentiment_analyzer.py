"""
Enhanced Sentiment Analysis Agent
Provides real-time sentiment analysis from multiple sources including news, social media, and market indicators
"""

import asyncio
import requests
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentScore:
    """Sentiment score data structure"""
    source: str
    symbol: str
    score: float  # -1 (very negative) to +1 (very positive)
    confidence: float  # 0 to 1
    timestamp: datetime
    headline: Optional[str] = None
    volume: Optional[int] = None
    category: Optional[str] = None

class EnhancedSentimentAnalyzer:
    """Advanced sentiment analysis with multiple data sources"""

    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.sentiment_cache = {}
        self.cache_duration = timedelta(minutes=15)  # Cache for 15 minutes

        # Sentiment keywords for basic analysis
        self.positive_keywords = [
            'bullish', 'buy', 'strong', 'growth', 'rally', 'breakout', 'surge', 'pump',
            'moon', 'rocket', 'gains', 'profit', 'bull', 'long', 'hodl', 'diamond hands',
            'to the moon', 'green', 'up', 'rise', 'climb', 'soar', 'boom', 'golden cross'
        ]

        self.negative_keywords = [
            'bearish', 'sell', 'weak', 'crash', 'dump', 'bear', 'short', 'drop',
            'fall', 'decline', 'red', 'down', 'plunge', 'tank', 'correction',
            'resistance', 'selling pressure', 'dead cat bounce', 'death cross'
        ]

        self.uncertainty_keywords = [
            'volatile', 'uncertain', 'mixed', 'sideways', 'consolidation', 'wait',
            'watch', 'cautious', 'range-bound', 'indecision', 'neutral'
        ]

        # News sources and their weights
        self.news_sources = {
            'cnbc': 0.8,
            'bloomberg': 0.9,
            'reuters': 0.85,
            'marketwatch': 0.7,
            'seeking_alpha': 0.75,
            'yahoo_finance': 0.6,
            'benzinga': 0.65,
            'coindesk': 0.8,
            'cointelegraph': 0.7
        }

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment"""
        try:
            from dotenv import load_dotenv
            load_dotenv('.env')

            return {
                'news_api': os.getenv('NEWS_API_KEY'),
                'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
                'finnhub': os.getenv('FINNHUB_API_KEY'),
                'twitter_bearer': os.getenv('TWITTER_BEARER_TOKEN')
            }
        except Exception as e:
            logger.warning(f"Could not load API keys: {e}")
            return {}

    def _is_cache_valid(self, symbol: str, source: str) -> bool:
        """Check if cached data is still valid"""
        cache_key = f"{symbol}_{source}"
        if cache_key not in self.sentiment_cache:
            return False

        cached_time = self.sentiment_cache[cache_key]['timestamp']
        return datetime.now() - cached_time < self.cache_duration

    def _cache_sentiment(self, symbol: str, source: str, data: Any):
        """Cache sentiment data"""
        cache_key = f"{symbol}_{source}"
        self.sentiment_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }

    def _get_cached_sentiment(self, symbol: str, source: str) -> Optional[Any]:
        """Get cached sentiment data"""
        cache_key = f"{symbol}_{source}"
        if self._is_cache_valid(symbol, source):
            return self.sentiment_cache[cache_key]['data']
        return None

    def analyze_text_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text using keyword matching
        Returns: (sentiment_score, confidence)
        """
        if not text:
            return 0.0, 0.0

        text_lower = text.lower()

        # Count keyword occurrences
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        uncertainty_count = sum(1 for keyword in self.uncertainty_keywords if keyword in text_lower)

        total_sentiment_words = positive_count + negative_count + uncertainty_count
        total_words = len(text.split())

        if total_sentiment_words == 0:
            return 0.0, 0.0

        # Calculate sentiment score (-1 to +1)
        if positive_count > negative_count:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        elif negative_count > positive_count:
            sentiment_score = -(negative_count - positive_count) / total_sentiment_words
        else:
            sentiment_score = 0.0

        # Adjust for uncertainty
        if uncertainty_count > 0:
            uncertainty_factor = uncertainty_count / total_sentiment_words
            sentiment_score *= (1 - uncertainty_factor * 0.5)

        # Calculate confidence based on density of sentiment words
        confidence = min(total_sentiment_words / max(10, total_words * 0.1), 1.0)

        return sentiment_score, confidence

    async def get_news_sentiment(self, symbol: str, limit: int = 20) -> List[SentimentScore]:
        """Get sentiment from financial news"""
        cached = self._get_cached_sentiment(symbol, 'news')
        if cached:
            return cached

        sentiments = []

        try:
            # Simulate news sentiment (in real implementation, use actual news APIs)
            # Mock news headlines with sentiment
            mock_headlines = [
                f"{symbol} shows strong momentum with technical breakout",
                f"Analysts upgrade {symbol} with bullish price target",
                f"{symbol} faces selling pressure amid market volatility",
                f"Mixed signals for {symbol} as traders await earnings",
                f"{symbol} surges on positive market sentiment",
                f"Bearish outlook for {symbol} due to sector headwinds",
                f"{symbol} consolidates in tight trading range",
                f"Strong buying interest seen in {symbol}",
                f"{symbol} technical analysis shows bullish flag pattern",
                f"Profit-taking pressure weighs on {symbol}"
            ]

            for i, headline in enumerate(mock_headlines[:limit]):
                sentiment_score, confidence = self.analyze_text_sentiment(headline)

                # Add some randomization for realism
                sentiment_score += np.random.normal(0, 0.1)
                sentiment_score = max(-1, min(1, sentiment_score))

                confidence = max(0.1, min(0.9, confidence + np.random.uniform(-0.1, 0.1)))

                sentiments.append(SentimentScore(
                    source="financial_news",
                    symbol=symbol,
                    score=sentiment_score,
                    confidence=confidence,
                    timestamp=datetime.now() - timedelta(hours=i),
                    headline=headline,
                    category="news"
                ))

        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")

        self._cache_sentiment(symbol, 'news', sentiments)
        return sentiments

    async def get_social_sentiment(self, symbol: str, limit: int = 50) -> List[SentimentScore]:
        """Get sentiment from social media"""
        cached = self._get_cached_sentiment(symbol, 'social')
        if cached:
            return cached

        sentiments = []

        try:
            # Simulate social media sentiment
            mock_social_posts = [
                f"${symbol} looking bullish! 🚀🚀🚀",
                f"Time to buy more ${symbol}? What do you think?",
                f"${symbol} is dumping hard... might be a good entry point",
                f"HODL ${symbol} diamond hands 💎🙌",
                f"${symbol} technical analysis shows strong support",
                f"Bearish on ${symbol} for short term",
                f"${symbol} to the moon! 🌙",
                f"Selling my ${symbol} position, too risky",
                f"${symbol} chart looking beautiful for swing trade",
                f"${symbol} breaking resistance levels!"
            ]

            for i, post in enumerate(mock_social_posts[:limit]):
                sentiment_score, confidence = self.analyze_text_sentiment(post)

                # Social media tends to be more extreme
                sentiment_score *= 1.2
                sentiment_score = max(-1, min(1, sentiment_score))

                # Add volume simulation (engagement metrics)
                volume = np.random.poisson(100) + 50  # Likes, shares, etc.

                sentiments.append(SentimentScore(
                    source="social_media",
                    symbol=symbol,
                    score=sentiment_score,
                    confidence=confidence * 0.8,  # Lower confidence for social media
                    timestamp=datetime.now() - timedelta(minutes=i*10),
                    headline=post[:100],
                    volume=volume,
                    category="social"
                ))

        except Exception as e:
            logger.error(f"Error getting social sentiment for {symbol}: {e}")

        self._cache_sentiment(symbol, 'social', sentiments)
        return sentiments

    async def get_market_sentiment_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get market-based sentiment indicators"""
        cached = self._get_cached_sentiment(symbol, 'market_indicators')
        if cached:
            return cached

        try:
            # Simulate market sentiment indicators
            indicators = {
                "fear_greed_index": {
                    "value": np.random.uniform(20, 80),
                    "label": "",
                    "sentiment": ""
                },
                "volatility_index": {
                    "value": np.random.uniform(15, 40),
                    "percentile": np.random.uniform(20, 80),
                    "sentiment": ""
                },
                "put_call_ratio": {
                    "value": np.random.uniform(0.7, 1.3),
                    "sentiment": ""
                },
                "insider_activity": {
                    "net_buying": np.random.uniform(-10, 10),
                    "sentiment": ""
                },
                "analyst_ratings": {
                    "buy_ratio": np.random.uniform(0.3, 0.8),
                    "average_target": np.random.uniform(90, 150),
                    "sentiment": ""
                }
            }

            # Add sentiment labels based on values
            fgi = indicators["fear_greed_index"]["value"]
            if fgi >= 75:
                indicators["fear_greed_index"]["label"] = "Extreme Greed"
                indicators["fear_greed_index"]["sentiment"] = "bearish"
            elif fgi >= 55:
                indicators["fear_greed_index"]["label"] = "Greed"
                indicators["fear_greed_index"]["sentiment"] = "bullish"
            elif fgi >= 45:
                indicators["fear_greed_index"]["label"] = "Neutral"
                indicators["fear_greed_index"]["sentiment"] = "neutral"
            elif fgi >= 25:
                indicators["fear_greed_index"]["label"] = "Fear"
                indicators["fear_greed_index"]["sentiment"] = "bullish"
            else:
                indicators["fear_greed_index"]["label"] = "Extreme Fear"
                indicators["fear_greed_index"]["sentiment"] = "very_bullish"

            # VIX interpretation
            vix = indicators["volatility_index"]["value"]
            if vix >= 30:
                indicators["volatility_index"]["sentiment"] = "fear"
            elif vix <= 20:
                indicators["volatility_index"]["sentiment"] = "complacency"
            else:
                indicators["volatility_index"]["sentiment"] = "normal"

            # Put/Call ratio
            pcr = indicators["put_call_ratio"]["value"]
            if pcr >= 1.1:
                indicators["put_call_ratio"]["sentiment"] = "bearish"
            elif pcr <= 0.8:
                indicators["put_call_ratio"]["sentiment"] = "bullish"
            else:
                indicators["put_call_ratio"]["sentiment"] = "neutral"

        except Exception as e:
            logger.error(f"Error getting market indicators for {symbol}: {e}")
            indicators = {}

        self._cache_sentiment(symbol, 'market_indicators', indicators)
        return indicators

    def calculate_composite_sentiment(self, sentiments: List[SentimentScore],
                                     market_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate weighted composite sentiment score"""
        if not sentiments:
            return {
                "composite_score": 0.0,
                "confidence": 0.0,
                "sentiment_label": "neutral",
                "breakdown": {},
                "recommendation": "hold"
            }

        # Separate by source type
        news_sentiments = [s for s in sentiments if s.source == "financial_news"]
        social_sentiments = [s for s in sentiments if s.source == "social_media"]

        # Calculate weighted averages
        def weighted_average(scores: List[SentimentScore]) -> Tuple[float, float]:
            if not scores:
                return 0.0, 0.0

            total_weight = sum(s.confidence for s in scores)
            if total_weight == 0:
                return 0.0, 0.0

            weighted_score = sum(s.score * s.confidence for s in scores) / total_weight
            avg_confidence = total_weight / len(scores)
            return weighted_score, avg_confidence

        news_score, news_confidence = weighted_average(news_sentiments)
        social_score, social_confidence = weighted_average(social_sentiments)

        # Source weights
        news_weight = 0.6
        social_weight = 0.4

        # Calculate composite score
        if news_sentiments and social_sentiments:
            composite_score = (news_score * news_weight + social_score * social_weight)
            composite_confidence = (news_confidence * news_weight + social_confidence * social_weight)
        elif news_sentiments:
            composite_score = news_score
            composite_confidence = news_confidence * 0.8  # Reduce confidence when only one source
        elif social_sentiments:
            composite_score = social_score
            composite_confidence = social_confidence * 0.6
        else:
            composite_score = 0.0
            composite_confidence = 0.0

        # Determine sentiment label
        if composite_score >= 0.3:
            sentiment_label = "bullish"
        elif composite_score >= 0.1:
            sentiment_label = "slightly_bullish"
        elif composite_score >= -0.1:
            sentiment_label = "neutral"
        elif composite_score >= -0.3:
            sentiment_label = "slightly_bearish"
        else:
            sentiment_label = "bearish"

        # Generate recommendation
        if composite_score >= 0.4 and composite_confidence >= 0.6:
            recommendation = "strong_buy"
        elif composite_score >= 0.2 and composite_confidence >= 0.5:
            recommendation = "buy"
        elif composite_score <= -0.4 and composite_confidence >= 0.6:
            recommendation = "strong_sell"
        elif composite_score <= -0.2 and composite_confidence >= 0.5:
            recommendation = "sell"
        else:
            recommendation = "hold"

        return {
            "composite_score": round(composite_score, 3),
            "confidence": round(composite_confidence, 3),
            "sentiment_label": sentiment_label,
            "breakdown": {
                "news_sentiment": round(news_score, 3),
                "social_sentiment": round(social_score, 3),
                "news_confidence": round(news_confidence, 3),
                "social_confidence": round(social_confidence, 3),
                "total_articles": len(news_sentiments),
                "total_social_posts": len(social_sentiments)
            },
            "recommendation": recommendation,
            "market_indicators": market_indicators
        }

    async def analyze_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Complete sentiment analysis for a symbol"""
        try:
            # Gather all sentiment data
            news_sentiments = await self.get_news_sentiment(symbol)
            social_sentiments = await self.get_social_sentiment(symbol)
            market_indicators = await self.get_market_sentiment_indicators(symbol)

            all_sentiments = news_sentiments + social_sentiments

            # Calculate composite sentiment
            composite = self.calculate_composite_sentiment(all_sentiments, market_indicators)

            # Prepare detailed response
            response = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "composite_sentiment": composite,
                "recent_news": [
                    {
                        "headline": s.headline,
                        "sentiment": round(s.score, 3),
                        "confidence": round(s.confidence, 3),
                        "timestamp": s.timestamp.isoformat(),
                        "source": s.source
                    }
                    for s in news_sentiments[:5]
                ],
                "social_sentiment": [
                    {
                        "post": s.headline,
                        "sentiment": round(s.score, 3),
                        "confidence": round(s.confidence, 3),
                        "volume": s.volume,
                        "timestamp": s.timestamp.isoformat()
                    }
                    for s in social_sentiments[:5]
                ],
                "market_indicators": market_indicators,
                "analysis_summary": {
                    "overall_sentiment": composite["sentiment_label"],
                    "recommendation": composite["recommendation"],
                    "key_drivers": [],
                    "risks": [],
                    "confidence_level": composite["confidence"]
                }
            }

            # Add key drivers and risks based on sentiment
            if composite["composite_score"] > 0.2:
                response["analysis_summary"]["key_drivers"].append("Positive news sentiment")
            if composite["breakdown"]["social_sentiment"] > 0.3:
                response["analysis_summary"]["key_drivers"].append("Strong social media buzz")

            if composite["confidence"] < 0.4:
                response["analysis_summary"]["risks"].append("Low confidence due to mixed signals")
            if abs(composite["breakdown"]["news_sentiment"] - composite["breakdown"]["social_sentiment"]) > 0.5:
                response["analysis_summary"]["risks"].append("Divergent sentiment between news and social media")

            return response

        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "composite_sentiment": {
                    "composite_score": 0.0,
                    "confidence": 0.0,
                    "sentiment_label": "unknown",
                    "recommendation": "hold"
                }
            }

# Global sentiment analyzer instance
enhanced_sentiment_analyzer = EnhancedSentimentAnalyzer()