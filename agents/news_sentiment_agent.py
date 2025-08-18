"""
News and Sentiment Analysis Agent - LangGraph Implementation

This agent handles autonomous news ingestion, sentiment analysis using FinBERT and Gemini/DeepSeek,
event detection, and impact prediction with PostgreSQL storage.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
from decimal import Decimal

# LangGraph imports
from langgraph.graph import StateGraph, END

# NLP and sentiment analysis imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    pipeline = None
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

# News ingestion imports
import feedparser
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    Article = None

# Database imports
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Configuration
from config.settings import settings
from config.secure_config import get_api_keys

logger = logging.getLogger(__name__)

class SentimentScore(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

class NewsSource(Enum):
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    CNBC = "cnbc"
    MARKETWATCH = "marketwatch"
    YAHOO_FINANCE = "yahoo_finance"
    SEEKING_ALPHA = "seeking_alpha"
    RSS_FEED = "rss_feed"

class EventType(Enum):
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    ECONOMIC_DATA = "economic_data"
    COMPANY_NEWS = "company_news"
    MARKET_MOVING = "market_moving"
    SECTOR_NEWS = "sector_news"

@dataclass
class NewsArticle:
    """Core news article structure"""
    title: str
    content: str
    url: str
    source: NewsSource
    published_at: datetime
    symbols: List[str]
    author: Optional[str] = None
    summary: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'source': self.source.value,
            'published_at': self.published_at,
            'symbols': json.dumps(self.symbols),
            'author': self.author,
            'summary': self.summary
        }

@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    article_id: int
    symbol: str
    finbert_score: float
    finbert_label: str
    finbert_confidence: float
    gemini_score: float
    gemini_reasoning: str
    composite_score: float
    composite_label: str
    confidence_level: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'article_id': self.article_id,
            'symbol': self.symbol,
            'finbert_score': self.finbert_score,
            'finbert_label': self.finbert_label,
            'finbert_confidence': self.finbert_confidence,
            'gemini_score': self.gemini_score,
            'gemini_reasoning': self.gemini_reasoning,
            'composite_score': self.composite_score,
            'composite_label': self.composite_label,
            'confidence_level': self.confidence_level,
            'timestamp': self.timestamp
        }

@dataclass
class MarketEvent:
    """Detected market event"""
    event_type: EventType
    title: str
    description: str
    symbols: List[str]
    impact_score: float
    confidence: float
    predicted_direction: str  # 'bullish', 'bearish', 'neutral'
    time_horizon: str  # 'immediate', 'short_term', 'long_term'
    source_articles: List[int]
    detected_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'event_type': self.event_type.value,
            'title': self.title,
            'description': self.description,
            'symbols': json.dumps(self.symbols),
            'impact_score': self.impact_score,
            'confidence': self.confidence,
            'predicted_direction': self.predicted_direction,
            'time_horizon': self.time_horizon,
            'source_articles': json.dumps(self.source_articles),
            'detected_at': self.detected_at
        }

@dataclass
class SentimentState:
    """LangGraph state for sentiment analysis"""
    symbols: List[str]
    time_range: timedelta
    news_sources: List[NewsSource]
    raw_articles: List[Dict[str, Any]]
    processed_articles: List[NewsArticle]
    sentiment_analyses: List[SentimentAnalysis]
    detected_events: List[MarketEvent]
    processing_stats: Dict[str, Any]
    errors: List[str]

class NewsIngestor:
    """News ingestion from multiple sources"""
    
    def __init__(self):
        self.settings = settings
        self.api_keys = get_api_keys()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def fetch_news_articles(self, 
                                symbols: List[str], 
                                sources: List[NewsSource],
                                hours_back: int = 24) -> List[Dict[str, Any]]:
        """Fetch news articles from multiple sources"""
        articles = []
        
        for source in sources:
            try:
                if source == NewsSource.RSS_FEED:
                    source_articles = await self._fetch_rss_feeds(symbols, hours_back)
                elif source == NewsSource.YAHOO_FINANCE:
                    source_articles = await self._fetch_yahoo_finance(symbols, hours_back)
                elif source == NewsSource.MARKETWATCH:
                    source_articles = await self._fetch_marketwatch(symbols, hours_back)
                else:
                    # Generic RSS/API approach for other sources
                    source_articles = await self._fetch_generic_source(source, symbols, hours_back)
                
                articles.extend(source_articles)
                logger.info(f"Fetched {len(source_articles)} articles from {source.value}")
                
            except Exception as e:
                logger.error(f"Failed to fetch from {source.value}: {e}")
                continue
        
        return articles
    
    async def _fetch_rss_feeds(self, symbols: List[str], hours_back: int) -> List[Dict[str, Any]]:
        """Fetch from RSS feeds"""
        articles = []
        
        # Major financial RSS feeds
        rss_feeds = [
            'https://feeds.reuters.com/reuters/businessNews',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://feeds.marketwatch.com/marketwatch/marketpulse/',
        ]
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Parse publication date
                    pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.utcnow()
                    
                    if pub_date < cutoff_time:
                        continue
                    
                    # Extract relevant symbols from title and summary
                    relevant_symbols = self._extract_symbols_from_text(
                        f"{entry.title} {getattr(entry, 'summary', '')}", symbols
                    )
                    
                    if relevant_symbols:
                        articles.append({
                            'title': entry.title,
                            'content': getattr(entry, 'summary', ''),
                            'url': entry.link,
                            'source': NewsSource.RSS_FEED,
                            'published_at': pub_date,
                            'symbols': relevant_symbols,
                            'author': getattr(entry, 'author', None)
                        })
                
            except Exception as e:
                logger.error(f"Failed to parse RSS feed {feed_url}: {e}")
                continue
        
        return articles
    
    async def _fetch_yahoo_finance(self, symbols: List[str], hours_back: int) -> List[Dict[str, Any]]:
        """Fetch from Yahoo Finance"""
        articles = []
        
        for symbol in symbols:
            try:
                # Yahoo Finance news API (unofficial)
                url = f"https://query1.finance.yahoo.com/v1/finance/search"
                params = {
                    'q': symbol,
                    'lang': 'en-US',
                    'region': 'US',
                    'quotesCount': 1,
                    'newsCount': 10
                }
                
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                news_items = data.get('news', [])
                
                cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
                
                for item in news_items:
                    pub_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                    
                    if pub_date < cutoff_time:
                        continue
                    
                    articles.append({
                        'title': item.get('title', ''),
                        'content': item.get('summary', ''),
                        'url': item.get('link', ''),
                        'source': NewsSource.YAHOO_FINANCE,
                        'published_at': pub_date,
                        'symbols': [symbol],
                        'author': item.get('publisher', '')
                    })
                
            except Exception as e:
                logger.error(f"Failed to fetch Yahoo Finance news for {symbol}: {e}")
                continue
        
        return articles
    
    async def _fetch_marketwatch(self, symbols: List[str], hours_back: int) -> List[Dict[str, Any]]:
        """Fetch from MarketWatch"""
        articles = []
        
        try:
            # MarketWatch RSS feed
            feed_url = "https://feeds.marketwatch.com/marketwatch/marketpulse/"
            feed = feedparser.parse(feed_url)
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            for entry in feed.entries:
                pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.utcnow()
                
                if pub_date < cutoff_time:
                    continue
                
                # Extract relevant symbols
                relevant_symbols = self._extract_symbols_from_text(
                    f"{entry.title} {getattr(entry, 'summary', '')}", symbols
                )
                
                if relevant_symbols:
                    articles.append({
                        'title': entry.title,
                        'content': getattr(entry, 'summary', ''),
                        'url': entry.link,
                        'source': NewsSource.MARKETWATCH,
                        'published_at': pub_date,
                        'symbols': relevant_symbols,
                        'author': getattr(entry, 'author', None)
                    })
            
        except Exception as e:
            logger.error(f"Failed to fetch MarketWatch news: {e}")
        
        return articles
    
    async def _fetch_generic_source(self, source: NewsSource, symbols: List[str], hours_back: int) -> List[Dict[str, Any]]:
        """Generic source fetching (placeholder for future sources)"""
        # This would be implemented for specific sources like Bloomberg API, etc.
        return []
    
    def _extract_symbols_from_text(self, text: str, target_symbols: List[str]) -> List[str]:
        """Extract relevant stock symbols from text"""
        found_symbols = []
        text_upper = text.upper()
        
        for symbol in target_symbols:
            # Look for symbol mentions in various formats
            patterns = [
                rf'\b{symbol}\b',  # Exact match
                rf'\({symbol}\)',  # In parentheses
                rf'{symbol}:',     # With colon
                rf'\${symbol}\b'   # With dollar sign
            ]
            
            for pattern in patterns:
                if re.search(pattern, text_upper):
                    found_symbols.append(symbol)
                    break
        
        return found_symbols
    
    async def extract_full_article_content(self, url: str) -> Optional[str]:
        """Extract full article content using newspaper3k"""
        if not NEWSPAPER_AVAILABLE:
            logger.warning("newspaper3k not available, skipping full content extraction")
            return None
            
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            return None

class SentimentAnalyzer:
    """Advanced sentiment analysis using FinBERT and Gemini/DeepSeek"""
    
    def __init__(self):
        self.api_keys = get_api_keys()
        self._init_finbert()
        self._init_gemini()
    
    def _init_finbert(self):
        """Initialize FinBERT model"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, FinBERT will be disabled")
            self.finbert_pipeline = None
            return
            
        try:
            model_name = "ProsusAI/finbert"
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("FinBERT model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FinBERT: {e}")
            self.finbert_pipeline = None
    
    def _init_gemini(self):
        """Initialize Gemini API"""
        if not GEMINI_AVAILABLE:
            logger.warning("Google Generative AI not available, Gemini will be disabled")
            self.gemini_model = None
            return
            
        try:
            gemini_api_key = self.api_keys.get('GEMINI_API_KEY')
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                logger.info("Gemini API initialized successfully")
            else:
                logger.warning("Gemini API key not found")
                self.gemini_model = None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.gemini_model = None
    
    async def analyze_sentiment(self, article: NewsArticle, symbol: str) -> SentimentAnalysis:
        """Perform comprehensive sentiment analysis"""
        
        # Combine title and content for analysis
        text_to_analyze = f"{article.title}. {article.content}"
        
        # FinBERT analysis
        finbert_result = await self._analyze_with_finbert(text_to_analyze)
        
        # Gemini analysis
        gemini_result = await self._analyze_with_gemini(text_to_analyze, symbol)
        
        # Composite scoring
        composite_score, composite_label, confidence = self._calculate_composite_sentiment(
            finbert_result, gemini_result
        )
        
        return SentimentAnalysis(
            article_id=0,  # Will be set after article is stored
            symbol=symbol,
            finbert_score=finbert_result['score'],
            finbert_label=finbert_result['label'],
            finbert_confidence=finbert_result['confidence'],
            gemini_score=gemini_result['score'],
            gemini_reasoning=gemini_result['reasoning'],
            composite_score=composite_score,
            composite_label=composite_label,
            confidence_level=confidence,
            timestamp=datetime.utcnow()
        )
    
    async def _analyze_with_finbert(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using FinBERT"""
        if not self.finbert_pipeline:
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
        
        try:
            # Truncate text to model's max length
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.finbert_pipeline(text)[0]
            
            # Convert to numerical score
            label_to_score = {
                'positive': 1.0,
                'negative': -1.0,
                'neutral': 0.0
            }
            
            score = label_to_score.get(result['label'].lower(), 0.0)
            
            return {
                'score': score,
                'label': result['label'],
                'confidence': result['score']
            }
            
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
    
    async def _analyze_with_gemini(self, text: str, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment using Gemini with advanced reasoning"""
        if not self.gemini_model:
            return {'score': 0.0, 'reasoning': 'Gemini not available'}
        
        try:
            prompt = f"""
            Analyze the sentiment of this financial news article about {symbol} and provide:
            1. A sentiment score from -2 (very negative) to +2 (very positive)
            2. Detailed reasoning for your assessment
            3. Key factors that influenced your decision
            
            Article: {text}
            
            Please respond in JSON format:
            {{
                "sentiment_score": <number between -2 and 2>,
                "reasoning": "<detailed explanation>",
                "key_factors": ["<factor1>", "<factor2>", "<factor3>"]
            }}
            """
            
            response = await asyncio.to_thread(
                self.gemini_model.generate_content, prompt
            )
            
            # Parse JSON response
            try:
                result = json.loads(response.text)
                return {
                    'score': float(result.get('sentiment_score', 0.0)),
                    'reasoning': result.get('reasoning', 'No reasoning provided'),
                    'key_factors': result.get('key_factors', [])
                }
            except json.JSONDecodeError:
                # Fallback: extract score from text
                score = self._extract_score_from_text(response.text)
                return {
                    'score': score,
                    'reasoning': response.text,
                    'key_factors': []
                }
                
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return {'score': 0.0, 'reasoning': f'Analysis failed: {str(e)}'}
    
    def _extract_score_from_text(self, text: str) -> float:
        """Extract numerical score from text response"""
        # Look for patterns like "score: 1.5" or "sentiment: -0.8"
        patterns = [
            r'score[:\s]+(-?\d+\.?\d*)',
            r'sentiment[:\s]+(-?\d+\.?\d*)',
            r'(-?\d+\.?\d*)\s*(?:out of|/)\s*[25]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return 0.0
    
    def _calculate_composite_sentiment(self, finbert_result: Dict, gemini_result: Dict) -> tuple:
        """Calculate composite sentiment score"""
        finbert_score = finbert_result['score']
        finbert_confidence = finbert_result['confidence']
        gemini_score = gemini_result['score']
        
        # Weight scores by confidence
        finbert_weight = finbert_confidence
        gemini_weight = 0.8  # Fixed weight for Gemini
        
        total_weight = finbert_weight + gemini_weight
        
        if total_weight > 0:
            composite_score = (
                finbert_score * finbert_weight + 
                gemini_score * gemini_weight
            ) / total_weight
        else:
            composite_score = 0.0
        
        # Determine label
        if composite_score > 0.5:
            composite_label = 'positive'
        elif composite_score < -0.5:
            composite_label = 'negative'
        else:
            composite_label = 'neutral'
        
        # Calculate confidence
        confidence = min(finbert_confidence + 0.2, 1.0)  # Boost confidence with Gemini
        
        return composite_score, composite_label, confidence

class EventDetector:
    """Market event detection and impact prediction"""
    
    def __init__(self):
        self.event_patterns = {
            EventType.EARNINGS: [
                r'earnings', r'quarterly results', r'q[1-4] results',
                r'revenue', r'profit', r'eps', r'guidance'
            ],
            EventType.MERGER_ACQUISITION: [
                r'merger', r'acquisition', r'takeover', r'buyout',
                r'deal', r'acquire', r'purchase'
            ],
            EventType.REGULATORY: [
                r'fda approval', r'regulation', r'compliance',
                r'investigation', r'lawsuit', r'settlement'
            ],
            EventType.ECONOMIC_DATA: [
                r'gdp', r'inflation', r'unemployment', r'fed',
                r'interest rate', r'monetary policy'
            ]
        }
    
    async def detect_events(self, articles: List[NewsArticle]) -> List[MarketEvent]:
        """Detect market events from news articles"""
        events = []
        
        for article in articles:
            detected_events = await self._analyze_article_for_events(article)
            events.extend(detected_events)
        
        # Deduplicate and merge similar events
        events = self._deduplicate_events(events)
        
        return events
    
    async def _analyze_article_for_events(self, article: NewsArticle) -> List[MarketEvent]:
        """Analyze single article for events"""
        events = []
        text = f"{article.title} {article.content}".lower()
        
        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Calculate impact score based on article characteristics
                    impact_score = self._calculate_impact_score(article, event_type)
                    
                    if impact_score > 0.3:  # Threshold for significance
                        event = MarketEvent(
                            event_type=event_type,
                            title=article.title,
                            description=article.content[:500],
                            symbols=article.symbols,
                            impact_score=impact_score,
                            confidence=0.7,  # Base confidence
                            predicted_direction=self._predict_direction(text, event_type),
                            time_horizon=self._predict_time_horizon(event_type),
                            source_articles=[0],  # Will be updated with actual IDs
                            detected_at=datetime.utcnow()
                        )
                        events.append(event)
                    break
        
        return events
    
    def _calculate_impact_score(self, article: NewsArticle, event_type: EventType) -> float:
        """Calculate potential market impact score"""
        score = 0.5  # Base score
        
        # Adjust based on event type
        event_weights = {
            EventType.EARNINGS: 0.8,
            EventType.MERGER_ACQUISITION: 0.9,
            EventType.REGULATORY: 0.7,
            EventType.ECONOMIC_DATA: 0.6,
            EventType.COMPANY_NEWS: 0.5
        }
        
        score *= event_weights.get(event_type, 0.5)
        
        # Adjust based on source credibility
        source_weights = {
            NewsSource.REUTERS: 1.0,
            NewsSource.BLOOMBERG: 1.0,
            NewsSource.CNBC: 0.9,
            NewsSource.MARKETWATCH: 0.8,
            NewsSource.YAHOO_FINANCE: 0.7
        }
        
        score *= source_weights.get(article.source, 0.6)
        
        # Adjust based on recency
        hours_old = (datetime.utcnow() - article.published_at).total_seconds() / 3600
        if hours_old < 1:
            score *= 1.2
        elif hours_old < 6:
            score *= 1.0
        else:
            score *= 0.8
        
        return min(score, 1.0)
    
    def _predict_direction(self, text: str, event_type: EventType) -> str:
        """Predict market direction impact"""
        positive_words = ['beat', 'exceed', 'strong', 'growth', 'positive', 'up', 'gain']
        negative_words = ['miss', 'weak', 'decline', 'loss', 'negative', 'down', 'fall']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 'bullish'
        elif negative_count > positive_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def _predict_time_horizon(self, event_type: EventType) -> str:
        """Predict impact time horizon"""
        horizon_map = {
            EventType.EARNINGS: 'immediate',
            EventType.MERGER_ACQUISITION: 'short_term',
            EventType.REGULATORY: 'long_term',
            EventType.ECONOMIC_DATA: 'short_term',
            EventType.COMPANY_NEWS: 'immediate'
        }
        
        return horizon_map.get(event_type, 'short_term')
    
    def _deduplicate_events(self, events: List[MarketEvent]) -> List[MarketEvent]:
        """Remove duplicate events"""
        # Simple deduplication based on title similarity
        unique_events = []
        
        for event in events:
            is_duplicate = False
            for existing_event in unique_events:
                if (event.event_type == existing_event.event_type and
                    set(event.symbols) & set(existing_event.symbols) and
                    self._calculate_text_similarity(event.title, existing_event.title) > 0.8):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_events.append(event)
        
        return unique_events
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple implementation)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)

class DatabaseManager:
    """PostgreSQL database operations for news and sentiment data"""
    
    def __init__(self):
        self.settings = settings
        self.async_engine = None
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            database_url = f"postgresql+asyncpg://{self.settings.database.username}:{self.settings.database.password}@{self.settings.database.host}:{self.settings.database.port}/{self.settings.database.database}"
            self.async_engine = create_async_engine(database_url, echo=False)
            logger.info("Database connections initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def create_tables(self):
        """Create news and sentiment tables"""
        create_tables_sql = """
        -- News articles table
        CREATE TABLE IF NOT EXISTS news_articles (
            id BIGSERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            url TEXT UNIQUE NOT NULL,
            source VARCHAR(50) NOT NULL,
            published_at TIMESTAMPTZ NOT NULL,
            symbols JSONB NOT NULL,
            author VARCHAR(255),
            summary TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Sentiment analysis table
        CREATE TABLE IF NOT EXISTS sentiment_analysis (
            id BIGSERIAL PRIMARY KEY,
            article_id BIGINT REFERENCES news_articles(id),
            symbol VARCHAR(20) NOT NULL,
            finbert_score DECIMAL(5,4) NOT NULL,
            finbert_label VARCHAR(20) NOT NULL,
            finbert_confidence DECIMAL(5,4) NOT NULL,
            gemini_score DECIMAL(5,4) NOT NULL,
            gemini_reasoning TEXT,
            composite_score DECIMAL(5,4) NOT NULL,
            composite_label VARCHAR(20) NOT NULL,
            confidence_level DECIMAL(5,4) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Market events table
        CREATE TABLE IF NOT EXISTS market_events (
            id BIGSERIAL PRIMARY KEY,
            event_type VARCHAR(50) NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            symbols JSONB NOT NULL,
            impact_score DECIMAL(5,4) NOT NULL,
            confidence DECIMAL(5,4) NOT NULL,
            predicted_direction VARCHAR(20) NOT NULL,
            time_horizon VARCHAR(20) NOT NULL,
            source_articles JSONB NOT NULL,
            detected_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_news_articles_published_at ON news_articles(published_at DESC);
        CREATE INDEX IF NOT EXISTS idx_news_articles_symbols ON news_articles USING GIN(symbols);
        CREATE INDEX IF NOT EXISTS idx_sentiment_analysis_symbol_timestamp ON sentiment_analysis(symbol, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_market_events_symbols ON market_events USING GIN(symbols);
        CREATE INDEX IF NOT EXISTS idx_market_events_detected_at ON market_events(detected_at DESC);
        """
        
        try:
            async with self.async_engine.begin() as conn:
                await conn.execute(text(create_tables_sql))
            logger.info("News and sentiment tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def insert_articles(self, articles: List[NewsArticle]) -> List[int]:
        """Insert news articles and return IDs"""
        if not articles:
            return []
        
        insert_sql = """
        INSERT INTO news_articles (
            title, content, url, source, published_at, symbols, author, summary
        ) VALUES (
            :title, :content, :url, :source, :published_at, :symbols, :author, :summary
        ) ON CONFLICT (url) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            published_at = EXCLUDED.published_at,
            symbols = EXCLUDED.symbols
        RETURNING id
        """
        
        try:
            article_dicts = [article.to_dict() for article in articles]
            
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text(insert_sql), article_dicts)
                return [row.id for row in result.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to insert articles: {e}")
            raise
    
    async def insert_sentiment_analyses(self, analyses: List[SentimentAnalysis]) -> int:
        """Insert sentiment analyses"""
        if not analyses:
            return 0
        
        insert_sql = """
        INSERT INTO sentiment_analysis (
            article_id, symbol, finbert_score, finbert_label, finbert_confidence,
            gemini_score, gemini_reasoning, composite_score, composite_label,
            confidence_level, timestamp
        ) VALUES (
            :article_id, :symbol, :finbert_score, :finbert_label, :finbert_confidence,
            :gemini_score, :gemini_reasoning, :composite_score, :composite_label,
            :confidence_level, :timestamp
        )
        """
        
        try:
            analysis_dicts = [analysis.to_dict() for analysis in analyses]
            
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text(insert_sql), analysis_dicts)
                return result.rowcount
                
        except Exception as e:
            logger.error(f"Failed to insert sentiment analyses: {e}")
            raise
    
    async def insert_events(self, events: List[MarketEvent]) -> int:
        """Insert market events"""
        if not events:
            return 0
        
        insert_sql = """
        INSERT INTO market_events (
            event_type, title, description, symbols, impact_score, confidence,
            predicted_direction, time_horizon, source_articles, detected_at
        ) VALUES (
            :event_type, :title, :description, :symbols, :impact_score, :confidence,
            :predicted_direction, :time_horizon, :source_articles, :detected_at
        )
        """
        
        try:
            event_dicts = [event.to_dict() for event in events]
            
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text(insert_sql), event_dicts)
                return result.rowcount
                
        except Exception as e:
            logger.error(f"Failed to insert events: {e}")
            raise

class NewsSentimentAgent:
    """LangGraph-based News and Sentiment Analysis Agent"""
    
    def __init__(self):
        self.news_ingestor = NewsIngestor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.event_detector = EventDetector()
        self.db_manager = DatabaseManager()
    
    async def initialize(self):
        """Initialize the agent"""
        await self.db_manager.initialize()
        await self.db_manager.create_tables()
        logger.info("News and Sentiment Analysis Agent initialized")
    
    def create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for news and sentiment analysis"""
        
        workflow = StateGraph(SentimentState)
        
        # Add nodes
        workflow.add_node("fetch_news", self.fetch_news_node)
        workflow.add_node("process_articles", self.process_articles_node)
        workflow.add_node("analyze_sentiment", self.analyze_sentiment_node)
        workflow.add_node("detect_events", self.detect_events_node)
        workflow.add_node("store_data", self.store_data_node)
        workflow.add_node("generate_stats", self.generate_stats_node)
        
        # Define workflow edges
        workflow.set_entry_point("fetch_news")
        workflow.add_edge("fetch_news", "process_articles")
        workflow.add_edge("process_articles", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "detect_events")
        workflow.add_edge("detect_events", "store_data")
        workflow.add_edge("store_data", "generate_stats")
        workflow.add_edge("generate_stats", END)
        
        return workflow
    
    async def fetch_news_node(self, state: SentimentState) -> SentimentState:
        """Fetch news articles from multiple sources"""
        logger.info(f"Fetching news for {len(state.symbols)} symbols from {len(state.news_sources)} sources")
        
        try:
            hours_back = int(state.time_range.total_seconds() / 3600)
            raw_articles = await self.news_ingestor.fetch_news_articles(
                symbols=state.symbols,
                sources=state.news_sources,
                hours_back=hours_back
            )
            
            state.raw_articles = raw_articles
            state.processing_stats['articles_fetched'] = len(raw_articles)
            
            logger.info(f"Successfully fetched {len(raw_articles)} articles")
            
        except Exception as e:
            error_msg = f"Failed to fetch news articles: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def process_articles_node(self, state: SentimentState) -> SentimentState:
        """Process raw articles into structured format"""
        logger.info("Processing raw articles")
        
        processed_articles = []
        
        for article_data in state.raw_articles:
            try:
                # Extract full content if needed
                if len(article_data.get('content', '')) < 100:
                    full_content = await self.news_ingestor.extract_full_article_content(
                        article_data['url']
                    )
                    if full_content:
                        article_data['content'] = full_content
                
                # Create NewsArticle object
                article = NewsArticle(
                    title=article_data['title'],
                    content=article_data['content'],
                    url=article_data['url'],
                    source=article_data['source'],
                    published_at=article_data['published_at'],
                    symbols=article_data['symbols'],
                    author=article_data.get('author'),
                    summary=article_data.get('summary')
                )
                
                processed_articles.append(article)
                
            except Exception as e:
                logger.error(f"Failed to process article {article_data.get('url', 'unknown')}: {e}")
                continue
        
        state.processed_articles = processed_articles
        state.processing_stats['articles_processed'] = len(processed_articles)
        
        logger.info(f"Successfully processed {len(processed_articles)} articles")
        
        return state
    
    async def analyze_sentiment_node(self, state: SentimentState) -> SentimentState:
        """Analyze sentiment for all articles and symbols"""
        logger.info("Analyzing sentiment")
        
        sentiment_analyses = []
        
        for article in state.processed_articles:
            for symbol in article.symbols:
                try:
                    analysis = await self.sentiment_analyzer.analyze_sentiment(article, symbol)
                    sentiment_analyses.append(analysis)
                    
                except Exception as e:
                    logger.error(f"Failed to analyze sentiment for {symbol} in article {article.url}: {e}")
                    continue
        
        state.sentiment_analyses = sentiment_analyses
        state.processing_stats['sentiment_analyses'] = len(sentiment_analyses)
        
        logger.info(f"Completed {len(sentiment_analyses)} sentiment analyses")
        
        return state
    
    async def detect_events_node(self, state: SentimentState) -> SentimentState:
        """Detect market events from articles"""
        logger.info("Detecting market events")
        
        try:
            detected_events = await self.event_detector.detect_events(state.processed_articles)
            state.detected_events = detected_events
            state.processing_stats['events_detected'] = len(detected_events)
            
            logger.info(f"Detected {len(detected_events)} market events")
            
        except Exception as e:
            error_msg = f"Failed to detect events: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def store_data_node(self, state: SentimentState) -> SentimentState:
        """Store all processed data in database"""
        logger.info("Storing data in database")
        
        try:
            # Insert articles and get IDs
            article_ids = await self.db_manager.insert_articles(state.processed_articles)
            
            # Update sentiment analyses with article IDs
            for i, analysis in enumerate(state.sentiment_analyses):
                if i < len(article_ids):
                    analysis.article_id = article_ids[i // len(state.symbols)]
            
            # Insert sentiment analyses
            sentiment_count = await self.db_manager.insert_sentiment_analyses(state.sentiment_analyses)
            
            # Update events with article IDs and insert
            for event in state.detected_events:
                event.source_articles = article_ids[:len(state.processed_articles)]
            
            event_count = await self.db_manager.insert_events(state.detected_events)
            
            state.processing_stats['articles_stored'] = len(article_ids)
            state.processing_stats['sentiment_stored'] = sentiment_count
            state.processing_stats['events_stored'] = event_count
            
            logger.info(f"Stored {len(article_ids)} articles, {sentiment_count} sentiment analyses, {event_count} events")
            
        except Exception as e:
            error_msg = f"Failed to store data: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def generate_stats_node(self, state: SentimentState) -> SentimentState:
        """Generate final processing statistics"""
        logger.info("Generating processing statistics")
        
        try:
            # Calculate sentiment distribution
            if state.sentiment_analyses:
                positive_count = sum(1 for s in state.sentiment_analyses if s.composite_score > 0.5)
                negative_count = sum(1 for s in state.sentiment_analyses if s.composite_score < -0.5)
                neutral_count = len(state.sentiment_analyses) - positive_count - negative_count
                
                state.processing_stats['sentiment_distribution'] = {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                }
                
                # Calculate average confidence
                avg_confidence = sum(s.confidence_level for s in state.sentiment_analyses) / len(state.sentiment_analyses)
                state.processing_stats['average_confidence'] = avg_confidence
            
            # Calculate event distribution
            if state.detected_events:
                event_types = {}
                for event in state.detected_events:
                    event_type = event.event_type.value
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                
                state.processing_stats['event_distribution'] = event_types
            
            state.processing_stats['success'] = len(state.errors) == 0
            
            logger.info(f"Processing complete: {state.processing_stats}")
            
        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
        
        return state
    
    async def analyze_news_sentiment(self, 
                                   symbols: List[str],
                                   hours_back: int = 24,
                                   sources: Optional[List[NewsSource]] = None) -> Dict[str, Any]:
        """Main method to analyze news sentiment"""
        
        if sources is None:
            sources = [NewsSource.RSS_FEED, NewsSource.YAHOO_FINANCE, NewsSource.MARKETWATCH]
        
        # Create initial state
        initial_state = SentimentState(
            symbols=symbols,
            time_range=timedelta(hours=hours_back),
            news_sources=sources,
            raw_articles=[],
            processed_articles=[],
            sentiment_analyses=[],
            detected_events=[],
            processing_stats={
                'start_time': datetime.utcnow(),
                'symbols_requested': len(symbols),
                'sources_requested': len(sources)
            },
            errors=[]
        )
        
        # Create and run workflow
        workflow = self.create_workflow()
        app = workflow.compile()
        
        try:
            # Execute the workflow
            final_state = await app.ainvoke(initial_state)
            
            # Add completion time
            final_state.processing_stats['end_time'] = datetime.utcnow()
            final_state.processing_stats['duration'] = (
                final_state.processing_stats['end_time'] - 
                final_state.processing_stats['start_time']
            ).total_seconds()
            
            return {
                'success': len(final_state.errors) == 0,
                'statistics': final_state.processing_stats,
                'sentiment_analyses': [s.to_dict() for s in final_state.sentiment_analyses],
                'detected_events': [e.to_dict() for e in final_state.detected_events],
                'errors': final_state.errors
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                'success': False,
                'statistics': initial_state.processing_stats,
                'sentiment_analyses': [],
                'detected_events': [],
                'errors': [str(e)]
            }

# Factory function for creating the agent
async def create_news_sentiment_agent() -> NewsSentimentAgent:
    """Factory function to create and initialize the News and Sentiment Analysis Agent"""
    agent = NewsSentimentAgent()
    await agent.initialize()
    return agent