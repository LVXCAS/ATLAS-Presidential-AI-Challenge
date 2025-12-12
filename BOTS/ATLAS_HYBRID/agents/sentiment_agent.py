"""
Sentiment Analysis Agent

Analyzes news headlines and market sentiment using transformers NLP.

Uses FinBERT model fine-tuned on financial text for accurate sentiment classification.
"""

from typing import Dict, Tuple, List
from .base_agent import BaseAgent
import numpy as np
from datetime import datetime, timedelta
import requests
import os


class SentimentAgent(BaseAgent):
    """
    NLP sentiment analysis agent using transformers.

    Analyzes recent news sentiment for currency pairs.
    Positive sentiment → BUY bias
    Negative sentiment → SELL bias
    """

    def __init__(self, initial_weight: float = 1.5):
        super().__init__(name="SentimentAgent", initial_weight=initial_weight)

        # Lazy load transformers (heavy import)
        self.sentiment_pipeline = None
        self.model_loaded = False

        # Sentiment cache (avoid re-analyzing same news)
        self.sentiment_cache = {}
        self.cache_duration = timedelta(hours=1)

        # Store Alpha Vantage sentiment data (pre-computed)
        self.av_sentiment_data = []

    def _load_model(self):
        """Lazy load FinBERT model (only when first needed)."""
        if self.model_loaded:
            return

        try:
            from transformers import pipeline

            # Use FinBERT for financial sentiment (PyTorch backend)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # CPU
            )
            self.model_loaded = True
            print("[SentimentAgent] FinBERT model loaded successfully")

        except Exception as e:
            print(f"[SentimentAgent] Failed to load model: {e}")
            print("[SentimentAgent] Falling back to simple sentiment")
            self.model_loaded = False

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Analyze market sentiment from news.

        Returns:
            (vote, confidence, reasoning)
        """
        pair = market_data.get("pair", "UNKNOWN")

        # Check cache first
        cache_key = f"{pair}_{datetime.now().strftime('%Y%m%d_%H')}"
        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            if datetime.now() - cached['timestamp'] < self.cache_duration:
                return (cached['vote'], cached['confidence'], cached['reasoning'])

        # Get news headlines for pair
        headlines = self._get_recent_headlines(pair, market_data)

        if not headlines:
            # No news available - return neutral
            return ("NEUTRAL", 0.3, {
                "agent": self.name,
                "status": "no_news",
                "pair": pair
            })

        # Use Alpha Vantage pre-computed sentiment if available
        if hasattr(self, 'av_sentiment_data') and len(self.av_sentiment_data) > 0:
            # Use Alpha Vantage's professional sentiment analysis
            sentiment_scores = []
            for item in self.av_sentiment_data:
                score = item['sentiment_score']  # Already -1 to +1 scale
                label = item['sentiment_label'].lower()

                # Map Alpha Vantage labels to our format
                if 'bullish' in label or 'positive' in label:
                    label = 'positive'
                elif 'bearish' in label or 'negative' in label:
                    label = 'negative'
                else:
                    label = 'neutral'

                sentiment_scores.append({
                    'label': label,
                    'score': score,
                    'confidence': abs(score)
                })

            print(f"[SentimentAgent] Using Alpha Vantage sentiment for {len(sentiment_scores)} articles")

        else:
            # Fallback to FinBERT or simple analysis
            self._load_model()

            if self.model_loaded and self.sentiment_pipeline:
                sentiment_scores = self._analyze_with_finbert(headlines)
            else:
                sentiment_scores = self._analyze_simple(headlines)

        # Aggregate sentiment
        avg_score = np.mean([s['score'] for s in sentiment_scores])
        positive_count = sum(1 for s in sentiment_scores if s['label'] == 'positive')
        negative_count = sum(1 for s in sentiment_scores if s['label'] == 'negative')

        # Make decision
        if avg_score > 0.3:
            vote = "BUY"
            confidence = min(avg_score * 0.8, 0.85)
        elif avg_score < -0.3:
            vote = "SELL"
            confidence = min(abs(avg_score) * 0.8, 0.85)
        else:
            vote = "NEUTRAL"
            confidence = 0.5

        reasoning = {
            "agent": self.name,
            "vote": vote,
            "avg_sentiment": round(avg_score, 3),
            "positive_news": positive_count,
            "negative_news": negative_count,
            "neutral_news": len(headlines) - positive_count - negative_count,
            "headlines_analyzed": len(headlines)
        }

        # Cache result
        self.sentiment_cache[cache_key] = {
            'vote': vote,
            'confidence': confidence,
            'reasoning': reasoning,
            'timestamp': datetime.now()
        }

        return (vote, confidence, reasoning)

    def _get_recent_headlines(self, pair: str, market_data: Dict) -> List[str]:
        """
        Get recent news headlines for currency pair from multiple sources.

        Sources:
        - Alpha Vantage News API (forex news)
        - Polygon.io News API (market news)
        - Fallback to synthetic headlines if APIs fail
        """
        # Extract base currencies from pair
        if "_" in pair:
            base, quote = pair.split("_")
        else:
            return []

        headlines = []

        # Try Alpha Vantage first (best for forex)
        try:
            av_headlines = self._fetch_alpha_vantage_news(base, quote)
            headlines.extend(av_headlines)
        except Exception as e:
            print(f"[SentimentAgent] Alpha Vantage failed: {e}")

        # Try Polygon.io (good for general market news)
        try:
            polygon_headlines = self._fetch_polygon_news(base, quote)
            headlines.extend(polygon_headlines)
        except Exception as e:
            print(f"[SentimentAgent] Polygon failed: {e}")

        # If both APIs fail, use synthetic headlines as fallback
        if len(headlines) == 0:
            print("[SentimentAgent] All news APIs failed, using synthetic headlines")
            headlines = self._generate_synthetic_headlines(pair, market_data)

        # Limit to 10 most recent headlines
        return headlines[:10]

    def _fetch_alpha_vantage_news(self, base: str, quote: str) -> List[str]:
        """
        Fetch forex news from Alpha Vantage NEWS_SENTIMENT endpoint.

        API: https://www.alphavantage.co/documentation/#news-sentiment
        Returns headlines with embedded sentiment scores for better analysis.
        """
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            # Try loading from .env file if not in environment
            try:
                from pathlib import Path
                env_file = Path(__file__).parent.parent / ".env"
                if env_file.exists():
                    with open(env_file) as f:
                        for line in f:
                            if line.startswith("ALPHA_VANTAGE_API_KEY="):
                                api_key = line.split("=", 1)[1].strip()
                                break
            except:
                pass

        if not api_key:
            return []

        # Use multiple targeted topics for better forex coverage
        # Topics: economy_monetary (Fed/ECB), forex (direct), economy_macro (GDP/employment)
        topics = "economy_monetary,forex,economy_macro"

        # Get news from last 24 hours
        from datetime import datetime, timedelta
        time_from = (datetime.now() - timedelta(days=1)).strftime("%Y%m%dT0000")

        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics={topics}&time_from={time_from}&limit=20&apikey={api_key}"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return []

            data = response.json()

            # Check for API limit errors
            if "Note" in data or "Information" in data:
                print(f"[SentimentAgent] Alpha Vantage API limit reached")
                return []

            headlines = []

            # Extract headlines from feed with sentiment scores
            for item in data.get("feed", [])[:10]:  # Max 10 recent articles
                title = item.get("title", "")

                # Get Alpha Vantage's pre-computed sentiment score
                sentiment_score = item.get("overall_sentiment_score", 0)
                sentiment_label = item.get("overall_sentiment_label", "Neutral")

                # Filter for relevance to our currency pair
                relevance_score = 0

                # Check if base or quote currency mentioned in title
                title_lower = title.lower()
                if base.lower() in title_lower or quote.lower() in title_lower:
                    relevance_score = 0.9
                elif any(word in title_lower for word in ['fed', 'ecb', 'boe', 'central bank', 'interest rate', 'inflation']):
                    relevance_score = 0.7
                elif any(word in title_lower for word in ['dollar', 'euro', 'pound', 'yen', 'forex', 'currency']):
                    relevance_score = 0.6

                # Only include relevant news (relevance > 0.5)
                if relevance_score > 0.5:
                    # Store sentiment with headline for later use
                    # Format: [SENTIMENT_SCORE] headline
                    # This lets us use Alpha Vantage's sentiment instead of FinBERT
                    headlines.append({
                        'text': title,
                        'sentiment_score': sentiment_score,
                        'sentiment_label': sentiment_label,
                        'relevance': relevance_score
                    })

            # Sort by relevance and return top 5
            headlines.sort(key=lambda x: x['relevance'], reverse=True)

            # Store sentiment data for later use
            self.av_sentiment_data = headlines[:5]

            # Return just the text for FinBERT analysis
            return [h['text'] for h in headlines[:5]]

        except Exception as e:
            print(f"[SentimentAgent] Alpha Vantage API error: {e}")
            return []

    def _fetch_polygon_news(self, base: str, quote: str) -> List[str]:
        """
        Fetch market news from Polygon.io ticker news endpoint.

        API: https://polygon.io/docs/stocks/get_v2_reference_news
        """
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            # Try loading from .env file if not in environment
            try:
                from pathlib import Path
                env_file = Path(__file__).parent.parent / ".env"
                if env_file.exists():
                    with open(env_file) as f:
                        for line in f:
                            if line.startswith("POLYGON_API_KEY="):
                                api_key = line.split("=", 1)[1].strip()
                                break
            except:
                pass

        if not api_key:
            return []

        # Polygon uses ticker format (C:EURUSD for forex)
        ticker = f"C:{base}{quote}"
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=5&apiKey={api_key}"

        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []

        data = response.json()
        headlines = []

        # Extract headlines
        for item in data.get("results", [])[:5]:  # Max 5 from Polygon
            title = item.get("title", "")
            description = item.get("description", "")

            # Combine title + description
            if description:
                headlines.append(f"{title}. {description[:100]}")
            else:
                headlines.append(title)

        return headlines

    def _generate_synthetic_headlines(self, pair: str, market_data: Dict) -> List[str]:
        """
        Generate synthetic headlines based on technical indicators.

        Fallback when news APIs are unavailable.
        """
        base, quote = pair.split("_")
        indicators = market_data.get("indicators", {})
        rsi = indicators.get("rsi", 50)
        macd_hist = indicators.get("macd_hist", 0)

        headlines = []

        if rsi > 70:
            headlines.append(f"{base} strength continues as demand surges")
        elif rsi < 30:
            headlines.append(f"{base} weakness persists amid selling pressure")

        if macd_hist > 0:
            headlines.append(f"{base}/{quote} shows bullish momentum")
        elif macd_hist < 0:
            headlines.append(f"{base}/{quote} faces bearish pressure")

        headlines.append(f"{base} central bank maintains policy stance")

        return headlines

    def _analyze_with_finbert(self, headlines: list) -> list:
        """Analyze headlines using FinBERT transformer model."""
        if not self.sentiment_pipeline:
            return []

        try:
            results = self.sentiment_pipeline(headlines)

            # Convert FinBERT output to normalized scores
            sentiment_scores = []
            for result in results:
                label = result['label'].lower()
                score = result['score']

                # Convert to -1 to +1 scale
                if label == 'positive':
                    normalized_score = score
                elif label == 'negative':
                    normalized_score = -score
                else:  # neutral
                    normalized_score = 0

                sentiment_scores.append({
                    'label': label,
                    'score': normalized_score,
                    'confidence': score
                })

            return sentiment_scores

        except Exception as e:
            print(f"[SentimentAgent] FinBERT analysis failed: {e}")
            return self._analyze_simple(headlines)

    def _analyze_simple(self, headlines: list) -> list:
        """
        Simple keyword-based sentiment analysis (fallback).

        Uses positive/negative keyword matching.
        """
        positive_keywords = ['surge', 'strength', 'bullish', 'rally', 'gain', 'rise', 'up', 'positive', 'momentum']
        negative_keywords = ['weakness', 'bearish', 'fall', 'drop', 'decline', 'down', 'negative', 'pressure', 'selling']

        sentiment_scores = []

        for headline in headlines:
            headline_lower = headline.lower()

            pos_count = sum(1 for word in positive_keywords if word in headline_lower)
            neg_count = sum(1 for word in negative_keywords if word in headline_lower)

            if pos_count > neg_count:
                sentiment_scores.append({'label': 'positive', 'score': 0.6})
            elif neg_count > pos_count:
                sentiment_scores.append({'label': 'negative', 'score': -0.6})
            else:
                sentiment_scores.append({'label': 'neutral', 'score': 0.0})

        return sentiment_scores
