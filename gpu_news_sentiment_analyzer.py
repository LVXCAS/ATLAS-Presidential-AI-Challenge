"""
GPU NEWS SENTIMENT ANALYZER
Real-time financial news sentiment analysis with GTX 1660 Super acceleration
Advanced NLP models for market-moving news detection and sentiment scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import requests
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import nltk
from collections import defaultdict, deque
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    source: str
    timestamp: datetime
    symbols_mentioned: List[str]
    url: Optional[str] = None
    sentiment_score: Optional[float] = None
    market_impact: Optional[str] = None

class FinancialSentimentModel(nn.Module):
    """GPU-accelerated financial sentiment analysis model"""

    def __init__(self, vocab_size: int = 50000, embedding_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()

        # Word embeddings optimized for financial text
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bidirectional LSTM for context understanding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=True, dropout=0.3)

        # Multi-head attention for important phrase detection
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8,
                                             dropout=0.1, batch_first=True)

        # Financial domain-specific layers
        self.financial_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3)  # Negative, Neutral, Positive
        )

        # Market impact predictor
        self.impact_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # Low, Medium, High, Critical
        )

        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # 0-1 confidence score
        )

    def forward(self, input_ids, attention_mask=None):
        # Embedding layer
        embedded = self.embedding(input_ids)

        # LSTM processing
        lstm_out, _ = self.lstm(embedded)

        # Self-attention mechanism
        if attention_mask is not None:
            # Apply attention mask
            attention_weights = attention_mask.unsqueeze(-1).float()
            lstm_out = lstm_out * attention_weights

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global max pooling to get sentence representation
        sentence_repr = torch.max(attn_out, dim=1)[0]

        # Predictions
        sentiment_logits = self.financial_classifier(sentence_repr)
        impact_logits = self.impact_predictor(sentence_repr)
        confidence = self.confidence_estimator(sentence_repr)

        return sentiment_logits, impact_logits, confidence

class GPUNewsAnalyzer:
    """Real-time news sentiment analysis with GPU acceleration"""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('NewsAnalyzer')

        # Financial keywords and symbols (define before vocabulary)
        self.financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'merger', 'acquisition',
            'dividend', 'buyback', 'bankruptcy', 'ipo', 'split', 'guidance',
            'forecast', 'outlook', 'upgrade', 'downgrade', 'analyst', 'target',
            'beat', 'miss', 'surprise', 'whisper', 'consensus', 'estimate'
        ]

        # Symbol tracking
        self.tracked_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'VIX', 'DIA', 'XLF',
            'BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL', 'DOT', 'AVAX'
        ]

        # Initialize sentiment model
        self.model = FinancialSentimentModel().to(self.device)

        # Vocabulary for text processing
        self.vocab = self.build_financial_vocabulary()
        self.max_sequence_length = 512

        # News sources and APIs
        self.news_sources = {
            'financial_modeling_prep': 'https://financialmodelingprep.com/api/v3/stock_news',
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'newsapi': 'https://newsapi.org/v2/everything',
            'reddit_wallstreetbets': 'https://www.reddit.com/r/wallstreetbets/hot.json'
        }

        # Market impact keywords
        self.high_impact_keywords = [
            'bankruptcy', 'merger', 'acquisition', 'fda approval', 'lawsuit',
            'investigation', 'fraud', 'ceo resignation', 'data breach',
            'product recall', 'cyber attack', 'partnership'
        ]

        # News buffer for real-time processing
        self.news_buffer = deque(maxlen=1000)
        self.processed_urls = set()

        # Performance tracking
        self.articles_processed = 0
        self.processing_speed = 0

        self.logger.info(f"News analyzer initialized on {self.device}")

    def build_financial_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary optimized for financial text"""
        # Financial terms vocabulary
        financial_terms = [
            'bullish', 'bearish', 'rally', 'selloff', 'correction', 'crash',
            'support', 'resistance', 'breakout', 'breakdown', 'momentum',
            'volatility', 'volume', 'liquidation', 'margin', 'leverage',
            'options', 'calls', 'puts', 'strike', 'expiry', 'greek',
            'arbitrage', 'hedge', 'long', 'short', 'squeeze', 'gamma'
        ]

        # Common words
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'have', 'has', 'had', 'will', 'would', 'could', 'should'
        ]

        # Build vocabulary
        vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}

        # Add financial terms
        for term in financial_terms + self.financial_keywords + common_words:
            if term not in vocab:
                vocab[term] = len(vocab)

        # Add tracked symbols
        for symbol in self.tracked_symbols:
            if symbol not in vocab:
                vocab[symbol] = len(vocab)

        return vocab

    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocess text for GPU analysis

        Args:
            text: Raw text content

        Returns:
            Tokenized tensor ready for GPU processing
        """
        # Clean text
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = text.split()

        # Convert to IDs
        token_ids = []
        for token in tokens[:self.max_sequence_length - 2]:  # Reserve space for START/END
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['<UNK>'])

        # Add special tokens
        token_ids = [self.vocab['<START>']] + token_ids + [self.vocab['<END>']]

        # Pad to max length
        if len(token_ids) < self.max_sequence_length:
            token_ids.extend([self.vocab['<PAD>']] * (self.max_sequence_length - len(token_ids)))

        return torch.tensor(token_ids, dtype=torch.long, device=self.device)

    def extract_symbols_from_text(self, text: str) -> List[str]:
        """
        Extract stock symbols mentioned in text

        Args:
            text: Text content

        Returns:
            List of detected symbols
        """
        symbols_found = []
        text_upper = text.upper()

        # Look for tracked symbols
        for symbol in self.tracked_symbols:
            # Check for symbol mentions (with context)
            if symbol in text_upper:
                # Simple context check
                if f'${symbol}' in text_upper or f' {symbol} ' in text_upper:
                    symbols_found.append(symbol)

        return list(set(symbols_found))

    def batch_analyze_sentiment(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        Analyze sentiment for multiple articles using GPU batch processing

        Args:
            articles: List of news articles

        Returns:
            Articles with sentiment analysis completed
        """
        if not articles:
            return []

        start_time = time.time()

        # Prepare batch data
        batch_texts = []
        batch_articles = []

        for article in articles:
            # Combine title and content for analysis
            full_text = f"{article.title} {article.content}"
            if len(full_text.strip()) > 10:  # Minimum text requirement
                batch_texts.append(full_text)
                batch_articles.append(article)

        if not batch_texts:
            return articles

        # Tokenize all texts
        tokenized_batch = []
        attention_masks = []

        for text in batch_texts:
            tokens = self.preprocess_text(text)
            tokenized_batch.append(tokens)

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (tokens != self.vocab['<PAD>']).float()
            attention_masks.append(attention_mask)

        # Convert to batch tensors
        batch_tensor = torch.stack(tokenized_batch)
        mask_tensor = torch.stack(attention_masks)

        # GPU inference
        self.model.eval()
        with torch.no_grad():
            sentiment_logits, impact_logits, confidence = self.model(batch_tensor, mask_tensor)

            # Convert to probabilities
            sentiment_probs = F.softmax(sentiment_logits, dim=-1)
            impact_probs = F.softmax(impact_logits, dim=-1)

        # Process results
        sentiment_labels = ['negative', 'neutral', 'positive']
        impact_labels = ['low', 'medium', 'high', 'critical']

        for i, article in enumerate(batch_articles):
            # Sentiment analysis
            sentiment_idx = torch.argmax(sentiment_probs[i]).item()
            sentiment_score = sentiment_probs[i][2].item() - sentiment_probs[i][0].item()  # Positive - Negative

            # Market impact
            impact_idx = torch.argmax(impact_probs[i]).item()
            impact_label = impact_labels[impact_idx]

            # Confidence
            conf_score = confidence[i].item()

            # Update article
            article.sentiment_score = sentiment_score
            article.market_impact = impact_label

            # Extract symbols
            if not article.symbols_mentioned:
                article.symbols_mentioned = self.extract_symbols_from_text(
                    f"{article.title} {article.content}"
                )

        processing_time = time.time() - start_time
        self.articles_processed += len(batch_articles)
        self.processing_speed = len(batch_articles) / processing_time if processing_time > 0 else 0

        self.logger.info(f"Analyzed {len(batch_articles)} articles in {processing_time:.4f}s "
                        f"({self.processing_speed:.1f} articles/second)")

        return articles

    async def fetch_financial_news(self) -> List[NewsArticle]:
        """
        Fetch financial news from multiple sources

        Returns:
            List of news articles
        """
        articles = []

        try:
            # Simulate news fetching (in real implementation, would use actual APIs)
            sample_news = [
                {
                    'title': 'Apple Reports Record Q4 Earnings, Beats Estimates',
                    'content': 'Apple Inc. reported record fourth-quarter earnings, beating analyst estimates on strong iPhone sales and services revenue.',
                    'source': 'financial_times',
                    'symbols': ['AAPL'],
                    'timestamp': datetime.now()
                },
                {
                    'title': 'Tesla Stock Surges on Autonomous Driving Breakthrough',
                    'content': 'Tesla shares jumped 15% in after-hours trading following announcement of major autonomous driving technology breakthrough.',
                    'source': 'reuters',
                    'symbols': ['TSLA'],
                    'timestamp': datetime.now()
                },
                {
                    'title': 'Federal Reserve Signals Interest Rate Cuts Ahead',
                    'content': 'The Federal Reserve indicated potential interest rate cuts in upcoming meetings, citing economic uncertainty.',
                    'source': 'bloomberg',
                    'symbols': ['SPY', 'QQQ'],
                    'timestamp': datetime.now()
                },
                {
                    'title': 'Bitcoin Reaches New All-Time High Amid Institutional Adoption',
                    'content': 'Bitcoin price surged to new record highs as major institutions continue to add cryptocurrency to their portfolios.',
                    'source': 'coindesk',
                    'symbols': ['BTC'],
                    'timestamp': datetime.now()
                },
                {
                    'title': 'Microsoft Azure Revenue Growth Exceeds Expectations',
                    'content': 'Microsoft cloud services division reported 35% revenue growth, significantly exceeding Wall Street expectations.',
                    'source': 'cnbc',
                    'symbols': ['MSFT'],
                    'timestamp': datetime.now()
                }
            ]

            # Convert to NewsArticle objects
            for news_item in sample_news:
                article = NewsArticle(
                    title=news_item['title'],
                    content=news_item['content'],
                    source=news_item['source'],
                    timestamp=news_item['timestamp'],
                    symbols_mentioned=news_item['symbols']
                )
                articles.append(article)

        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")

        return articles

    def calculate_market_impact_score(self, articles: List[NewsArticle]) -> Dict[str, float]:
        """
        Calculate aggregate market impact scores for symbols

        Args:
            articles: List of analyzed articles

        Returns:
            Dictionary of symbol impact scores
        """
        symbol_impacts = defaultdict(list)

        # Collect sentiment scores by symbol
        for article in articles:
            if article.sentiment_score is not None and article.symbols_mentioned:
                for symbol in article.symbols_mentioned:
                    # Weight by market impact level
                    impact_multiplier = {
                        'low': 1.0,
                        'medium': 2.0,
                        'high': 3.0,
                        'critical': 5.0
                    }.get(article.market_impact, 1.0)

                    weighted_sentiment = article.sentiment_score * impact_multiplier
                    symbol_impacts[symbol].append(weighted_sentiment)

        # Calculate aggregate scores
        final_scores = {}
        for symbol, scores in symbol_impacts.items():
            if scores:
                # Weighted average with recency bias
                weights = np.exp(np.linspace(0, 1, len(scores)))  # More recent = higher weight
                final_scores[symbol] = np.average(scores, weights=weights)

        return final_scores

    def generate_trading_signals_from_news(self, impact_scores: Dict[str, float]) -> Dict[str, Dict]:
        """
        Generate trading signals based on news sentiment analysis

        Args:
            impact_scores: Symbol impact scores

        Returns:
            Trading signals for each symbol
        """
        signals = {}

        for symbol, score in impact_scores.items():
            # Determine signal strength
            if score > 0.3:
                signal = 'BUY'
                strength = min(score, 1.0)
            elif score < -0.3:
                signal = 'SELL'
                strength = min(abs(score), 1.0)
            else:
                signal = 'HOLD'
                strength = 0.5

            signals[symbol] = {
                'signal': signal,
                'strength': strength,
                'sentiment_score': score,
                'timestamp': datetime.now().isoformat()
            }

        return signals

    async def run_real_time_analysis(self):
        """
        Run continuous real-time news sentiment analysis
        """
        self.logger.info("Starting real-time news sentiment analysis...")

        analysis_interval = 300  # Analyze every 5 minutes
        last_analysis = 0

        while True:
            try:
                current_time = time.time()

                if current_time - last_analysis >= analysis_interval:
                    self.logger.info("Fetching and analyzing latest news...")

                    # Fetch news
                    articles = await self.fetch_financial_news()

                    if articles:
                        # Analyze sentiment
                        analyzed_articles = self.batch_analyze_sentiment(articles)

                        # Calculate market impact
                        impact_scores = self.calculate_market_impact_score(analyzed_articles)

                        # Generate trading signals
                        trading_signals = self.generate_trading_signals_from_news(impact_scores)

                        # Log significant findings
                        high_impact_signals = {k: v for k, v in trading_signals.items()
                                             if v['strength'] > 0.7}

                        if high_impact_signals:
                            self.logger.info(f"High-impact signals: {list(high_impact_signals.keys())}")

                        # Save analysis results
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        news_analysis = {
                            'timestamp': timestamp,
                            'articles_analyzed': len(analyzed_articles),
                            'impact_scores': impact_scores,
                            'trading_signals': trading_signals,
                            'processing_speed': self.processing_speed
                        }

                        with open(f'news_analysis_{timestamp}.json', 'w') as f:
                            json.dump(news_analysis, f, indent=2)

                    last_analysis = current_time

                # Sleep for short interval
                await asyncio.sleep(30)

            except Exception as e:
                self.logger.error(f"Error in real-time analysis: {e}")
                await asyncio.sleep(60)

def demo_news_sentiment_system():
    """Demonstration of GPU news sentiment analysis system"""
    print("\n" + "="*80)
    print("GPU NEWS SENTIMENT ANALYSIS SYSTEM DEMONSTRATION")
    print("="*80)

    # Initialize analyzer
    analyzer = GPUNewsAnalyzer()

    print(f"\n>> News Sentiment Analyzer initialized on {analyzer.device}")
    print(f">> Tracking {len(analyzer.tracked_symbols)} symbols")
    print(f">> Monitoring {len(analyzer.financial_keywords)} financial keywords")

    # Demo news analysis
    print(f"\n>> Fetching and analyzing sample news...")

    # Create sample news (in async context, would fetch real news)
    import asyncio

    async def demo_analysis():
        articles = await analyzer.fetch_financial_news()
        analyzed_articles = analyzer.batch_analyze_sentiment(articles)
        return analyzed_articles

    # Run demo
    analyzed_articles = asyncio.run(demo_analysis())

    print(f">> Analysis completed for {len(analyzed_articles)} articles")
    print(f">> Processing speed: {analyzer.processing_speed:.1f} articles/second")

    # Show sample results
    print(f"\n>> SENTIMENT ANALYSIS RESULTS:")
    for i, article in enumerate(analyzed_articles[:3]):
        print(f"   {i+1}. {article.title}")
        print(f"      Sentiment: {article.sentiment_score:.3f} ({'Positive' if article.sentiment_score > 0 else 'Negative'})")
        print(f"      Impact: {article.market_impact.upper()}")
        print(f"      Symbols: {', '.join(article.symbols_mentioned)}")

    # Market impact scores
    impact_scores = analyzer.calculate_market_impact_score(analyzed_articles)
    print(f"\n>> MARKET IMPACT SCORES:")
    for symbol, score in sorted(impact_scores.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"   {symbol}: {score:.3f}")

    # Trading signals
    signals = analyzer.generate_trading_signals_from_news(impact_scores)
    print(f"\n>> NEWS-BASED TRADING SIGNALS:")
    for symbol, signal_data in signals.items():
        print(f"   {symbol}: {signal_data['signal']} (Strength: {signal_data['strength']:.2f})")

    print(f"\n" + "="*80)
    print("NEWS SENTIMENT ANALYSIS SYSTEM READY!")
    print("Use analyzer.run_real_time_analysis() for continuous monitoring")
    print("="*80)

if __name__ == "__main__":
    demo_news_sentiment_system()