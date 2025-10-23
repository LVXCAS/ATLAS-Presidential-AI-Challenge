"""
Sentiment Analysis API Endpoints
Provides real-time sentiment analysis from news, social media, and market indicators
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import asyncio
import sys
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import sentiment analyzer
try:
    from agents.enhanced_sentiment_analyzer import enhanced_sentiment_analyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    enhanced_sentiment_analyzer = None

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])

# Pydantic models
class SentimentRequest(BaseModel):
    symbol: str
    sources: Optional[List[str]] = ["news", "social", "market"]
    limit: Optional[int] = 20

class BulkSentimentRequest(BaseModel):
    symbols: List[str]
    sources: Optional[List[str]] = ["news", "social"]

@router.get("/analyze/{symbol}")
async def analyze_symbol_sentiment(symbol: str):
    """Get comprehensive sentiment analysis for a symbol"""
    if not SENTIMENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sentiment analysis not available")

    try:
        analysis = await enhanced_sentiment_analyzer.analyze_symbol_sentiment(symbol.upper())
        return {
            "status": "success",
            "data": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {e}")

@router.post("/analyze")
async def analyze_sentiment_custom(request: SentimentRequest):
    """Get customized sentiment analysis for a symbol"""
    if not SENTIMENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sentiment analysis not available")

    try:
        symbol = request.symbol.upper()

        # Get data based on requested sources
        news_sentiments = []
        social_sentiments = []
        market_indicators = {}

        if "news" in request.sources:
            news_sentiments = await enhanced_sentiment_analyzer.get_news_sentiment(symbol, request.limit)

        if "social" in request.sources:
            social_sentiments = await enhanced_sentiment_analyzer.get_social_sentiment(symbol, request.limit)

        if "market" in request.sources:
            market_indicators = await enhanced_sentiment_analyzer.get_market_sentiment_indicators(symbol)

        # Calculate composite sentiment
        all_sentiments = news_sentiments + social_sentiments
        composite = enhanced_sentiment_analyzer.calculate_composite_sentiment(all_sentiments, market_indicators)

        return {
            "status": "success",
            "symbol": symbol,
            "composite_sentiment": composite,
            "news_count": len(news_sentiments),
            "social_count": len(social_sentiments),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {e}")

@router.post("/bulk-analyze")
async def bulk_sentiment_analysis(request: BulkSentimentRequest):
    """Analyze sentiment for multiple symbols"""
    if not SENTIMENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sentiment analysis not available")

    if len(request.symbols) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")

    try:
        results = {}

        for symbol in request.symbols:
            symbol_upper = symbol.upper()
            analysis = await enhanced_sentiment_analyzer.analyze_symbol_sentiment(symbol_upper)
            results[symbol_upper] = {
                "composite_score": analysis["composite_sentiment"]["composite_score"],
                "sentiment_label": analysis["composite_sentiment"]["sentiment_label"],
                "confidence": analysis["composite_sentiment"]["confidence"],
                "recommendation": analysis["composite_sentiment"]["recommendation"]
            }

        return {
            "status": "success",
            "results": results,
            "symbols_analyzed": len(results),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk sentiment analysis error: {e}")

@router.get("/news/{symbol}")
async def get_news_sentiment(symbol: str, limit: int = 20):
    """Get news sentiment for a specific symbol"""
    if not SENTIMENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sentiment analysis not available")

    try:
        news_sentiments = await enhanced_sentiment_analyzer.get_news_sentiment(symbol.upper(), limit)

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "news_sentiment": [
                {
                    "headline": s.headline,
                    "sentiment_score": round(s.score, 3),
                    "confidence": round(s.confidence, 3),
                    "timestamp": s.timestamp.isoformat(),
                    "source": s.source,
                    "category": s.category
                }
                for s in news_sentiments
            ],
            "count": len(news_sentiments),
            "average_sentiment": round(sum(s.score for s in news_sentiments) / len(news_sentiments), 3) if news_sentiments else 0,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News sentiment error: {e}")

@router.get("/social/{symbol}")
async def get_social_sentiment(symbol: str, limit: int = 50):
    """Get social media sentiment for a specific symbol"""
    if not SENTIMENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sentiment analysis not available")

    try:
        social_sentiments = await enhanced_sentiment_analyzer.get_social_sentiment(symbol.upper(), limit)

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "social_sentiment": [
                {
                    "post": s.headline,
                    "sentiment_score": round(s.score, 3),
                    "confidence": round(s.confidence, 3),
                    "volume": s.volume,
                    "timestamp": s.timestamp.isoformat()
                }
                for s in social_sentiments
            ],
            "count": len(social_sentiments),
            "average_sentiment": round(sum(s.score for s in social_sentiments) / len(social_sentiments), 3) if social_sentiments else 0,
            "total_engagement": sum(s.volume for s in social_sentiments if s.volume),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Social sentiment error: {e}")

@router.get("/market-indicators/{symbol}")
async def get_market_sentiment_indicators(symbol: str):
    """Get market-based sentiment indicators"""
    if not SENTIMENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sentiment analysis not available")

    try:
        indicators = await enhanced_sentiment_analyzer.get_market_sentiment_indicators(symbol.upper())

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "market_indicators": indicators,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market indicators error: {e}")

@router.get("/trending")
async def get_trending_sentiment():
    """Get trending sentiment across popular symbols"""
    if not SENTIMENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sentiment analysis not available")

    try:
        # Popular trading symbols
        trending_symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "NVDA"]

        trending_data = []

        for symbol in trending_symbols[:5]:  # Limit to 5 for performance
            try:
                analysis = await enhanced_sentiment_analyzer.analyze_symbol_sentiment(symbol)
                trending_data.append({
                    "symbol": symbol,
                    "sentiment_score": analysis["composite_sentiment"]["composite_score"],
                    "sentiment_label": analysis["composite_sentiment"]["sentiment_label"],
                    "confidence": analysis["composite_sentiment"]["confidence"],
                    "recommendation": analysis["composite_sentiment"]["recommendation"],
                    "news_count": analysis["composite_sentiment"]["breakdown"]["total_articles"],
                    "social_count": analysis["composite_sentiment"]["breakdown"]["total_social_posts"]
                })
            except:
                continue

        # Sort by sentiment score
        trending_data.sort(key=lambda x: x["sentiment_score"], reverse=True)

        return {
            "status": "success",
            "trending_sentiment": trending_data,
            "most_bullish": trending_data[0] if trending_data else None,
            "most_bearish": trending_data[-1] if trending_data else None,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trending sentiment error: {e}")

@router.get("/sentiment-heatmap")
async def get_sentiment_heatmap():
    """Get sentiment heatmap data for visualization"""
    if not SENTIMENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sentiment analysis not available")

    try:
        # Major sectors and representative stocks
        sectors = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA"],
            "Finance": ["JPM", "BAC", "GS", "WFC"],
            "Healthcare": ["JNJ", "PFE", "MRK", "ABBV"],
            "Energy": ["XOM", "CVX", "COP", "SLB"],
            "Consumer": ["AMZN", "TSLA", "NKE", "SBUX"]
        }

        heatmap_data = {}

        for sector, symbols in sectors.items():
            sector_sentiments = []

            for symbol in symbols[:2]:  # Limit symbols per sector for performance
                try:
                    analysis = await enhanced_sentiment_analyzer.analyze_symbol_sentiment(symbol)
                    sector_sentiments.append({
                        "symbol": symbol,
                        "sentiment": analysis["composite_sentiment"]["composite_score"],
                        "confidence": analysis["composite_sentiment"]["confidence"]
                    })
                except:
                    continue

            if sector_sentiments:
                avg_sentiment = sum(s["sentiment"] for s in sector_sentiments) / len(sector_sentiments)
                avg_confidence = sum(s["confidence"] for s in sector_sentiments) / len(sector_sentiments)

                heatmap_data[sector] = {
                    "average_sentiment": round(avg_sentiment, 3),
                    "average_confidence": round(avg_confidence, 3),
                    "stocks": sector_sentiments,
                    "trend": "bullish" if avg_sentiment > 0.1 else "bearish" if avg_sentiment < -0.1 else "neutral"
                }

        return {
            "status": "success",
            "heatmap_data": heatmap_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment heatmap error: {e}")

@router.get("/health")
async def sentiment_health_check():
    """Health check for sentiment analysis system"""
    return {
        "status": "healthy" if SENTIMENT_AVAILABLE else "unavailable",
        "sentiment_analyzer_loaded": SENTIMENT_AVAILABLE,
        "cache_enabled": True,
        "supported_sources": ["news", "social_media", "market_indicators"],
        "max_symbols_bulk": 10,
        "cache_duration_minutes": 15,
        "timestamp": datetime.now().isoformat()
    }