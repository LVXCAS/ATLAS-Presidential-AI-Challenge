"""
Test suite for News and Sentiment Analysis Agent
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from agents.news_sentiment_agent import (
    NewsSentimentAgent,
    NewsIngestor,
    SentimentAnalyzer,
    EventDetector,
    NewsArticle,
    SentimentAnalysis,
    MarketEvent,
    NewsSource,
    EventType,
    SentimentState,
    create_news_sentiment_agent
)

class TestNewsIngestor:
    """Test news ingestion functionality"""
    
    @pytest.fixture
    def news_ingestor(self):
        return NewsIngestor()
    
    @pytest.mark.asyncio
    async def test_fetch_news_articles(self, news_ingestor):
        """Test fetching news articles from multiple sources"""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        sources = [NewsSource.RSS_FEED, NewsSource.YAHOO_FINANCE]
        
        with patch.object(news_ingestor, '_fetch_rss_feeds', return_value=[
            {
                'title': 'Apple Reports Strong Q4 Earnings',
                'content': 'Apple Inc. reported better than expected earnings...',
                'url': 'https://example.com/apple-earnings',
                'source': NewsSource.RSS_FEED,
                'published_at': datetime.utcnow(),
                'symbols': ['AAPL'],
                'author': 'Financial Reporter'
            }
        ]), patch.object(news_ingestor, '_fetch_yahoo_finance', return_value=[
            {
                'title': 'Google Stock Surges on AI News',
                'content': 'Alphabet Inc. shares rose after announcing new AI features...',
                'url': 'https://example.com/google-ai',
                'source': NewsSource.YAHOO_FINANCE,
                'published_at': datetime.utcnow(),
                'symbols': ['GOOGL'],
                'author': 'Tech Reporter'
            }
        ]):
            articles = await news_ingestor.fetch_news_articles(symbols, sources, 24)
            
            assert len(articles) == 2
            assert any('AAPL' in article['symbols'] for article in articles)
            assert any('GOOGL' in article['symbols'] for article in articles)
    
    def test_extract_symbols_from_text(self, news_ingestor):
        """Test symbol extraction from text"""
        text = "Apple (AAPL) and Microsoft (MSFT) reported earnings. Tesla stock is up."
        symbols = ['AAPL', 'MSFT', 'TSLA']
        
        found_symbols = news_ingestor._extract_symbols_from_text(text, symbols)
        
        assert 'AAPL' in found_symbols
        assert 'MSFT' in found_symbols
        # TSLA should not be found as it's not in the expected format
    
    @pytest.mark.asyncio
    async def test_extract_full_article_content(self, news_ingestor):
        """Test full article content extraction"""
        with patch('newspaper.Article') as mock_article:
            mock_instance = Mock()
            mock_instance.text = "Full article content here..."
            mock_article.return_value = mock_instance
            
            content = await news_ingestor.extract_full_article_content('https://example.com/article')
            
            assert content == "Full article content here..."
            mock_instance.download.assert_called_once()
            mock_instance.parse.assert_called_once()

class TestSentimentAnalyzer:
    """Test sentiment analysis functionality"""
    
    @pytest.fixture
    def sentiment_analyzer(self):
        analyzer = SentimentAnalyzer()
        # Mock the models to avoid loading actual models in tests
        analyzer.finbert_pipeline = Mock()
        analyzer.gemini_model = Mock()
        return analyzer
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, sentiment_analyzer):
        """Test comprehensive sentiment analysis"""
        article = NewsArticle(
            title="Apple Reports Record Earnings",
            content="Apple Inc. exceeded expectations with strong quarterly results...",
            url="https://example.com/apple-earnings",
            source=NewsSource.REUTERS,
            published_at=datetime.utcnow(),
            symbols=['AAPL']
        )
        
        # Mock FinBERT response
        sentiment_analyzer.finbert_pipeline.return_value = [{
            'label': 'positive',
            'score': 0.85
        }]
        
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = '{"sentiment_score": 1.5, "reasoning": "Strong earnings beat expectations"}'
        sentiment_analyzer.gemini_model.generate_content.return_value = mock_response
        
        with patch('asyncio.to_thread', return_value=mock_response):
            analysis = await sentiment_analyzer.analyze_sentiment(article, 'AAPL')
            
            assert analysis.symbol == 'AAPL'
            assert analysis.finbert_score == 1.0  # positive
            assert analysis.finbert_label == 'positive'
            assert analysis.finbert_confidence == 0.85
            assert analysis.gemini_score == 1.5
            assert analysis.composite_score > 0  # Should be positive
            assert analysis.confidence_level > 0.8
    
    @pytest.mark.asyncio
    async def test_finbert_analysis(self, sentiment_analyzer):
        """Test FinBERT sentiment analysis"""
        sentiment_analyzer.finbert_pipeline.return_value = [{
            'label': 'negative',
            'score': 0.92
        }]
        
        result = await sentiment_analyzer._analyze_with_finbert("Company reports major losses")
        
        assert result['score'] == -1.0  # negative
        assert result['label'] == 'negative'
        assert result['confidence'] == 0.92
    
    @pytest.mark.asyncio
    async def test_gemini_analysis(self, sentiment_analyzer):
        """Test Gemini sentiment analysis"""
        mock_response = Mock()
        mock_response.text = '{"sentiment_score": -1.2, "reasoning": "Significant revenue decline"}'
        sentiment_analyzer.gemini_model.generate_content.return_value = mock_response
        
        with patch('asyncio.to_thread', return_value=mock_response):
            result = await sentiment_analyzer._analyze_with_gemini("Revenue declined significantly", "AAPL")
            
            assert result['score'] == -1.2
            assert "revenue decline" in result['reasoning'].lower()
    
    def test_composite_sentiment_calculation(self, sentiment_analyzer):
        """Test composite sentiment score calculation"""
        finbert_result = {'score': 1.0, 'confidence': 0.9}
        gemini_result = {'score': 0.8}
        
        composite_score, composite_label, confidence = sentiment_analyzer._calculate_composite_sentiment(
            finbert_result, gemini_result
        )
        
        assert composite_score > 0  # Should be positive
        assert composite_label == 'positive'
        assert confidence > 0.9

class TestEventDetector:
    """Test event detection functionality"""
    
    @pytest.fixture
    def event_detector(self):
        return EventDetector()
    
    @pytest.mark.asyncio
    async def test_detect_events(self, event_detector):
        """Test market event detection"""
        articles = [
            NewsArticle(
                title="Apple Reports Q4 Earnings Beat",
                content="Apple Inc. reported quarterly earnings that exceeded analyst expectations...",
                url="https://example.com/apple-earnings",
                source=NewsSource.REUTERS,
                published_at=datetime.utcnow(),
                symbols=['AAPL']
            ),
            NewsArticle(
                title="Microsoft Announces Major Acquisition",
                content="Microsoft Corp. announced plans to acquire a major AI company...",
                url="https://example.com/msft-acquisition",
                source=NewsSource.BLOOMBERG,
                published_at=datetime.utcnow(),
                symbols=['MSFT']
            )
        ]
        
        events = await event_detector.detect_events(articles)
        
        assert len(events) >= 1
        
        # Check for earnings event
        earnings_events = [e for e in events if e.event_type == EventType.EARNINGS]
        assert len(earnings_events) >= 1
        assert 'AAPL' in earnings_events[0].symbols
        
        # Check for M&A event
        ma_events = [e for e in events if e.event_type == EventType.MERGER_ACQUISITION]
        if ma_events:  # May not be detected depending on impact score
            assert 'MSFT' in ma_events[0].symbols
    
    def test_calculate_impact_score(self, event_detector):
        """Test impact score calculation"""
        article = NewsArticle(
            title="Major Earnings Beat",
            content="Company reports record earnings...",
            url="https://example.com/earnings",
            source=NewsSource.REUTERS,
            published_at=datetime.utcnow(),
            symbols=['AAPL']
        )
        
        score = event_detector._calculate_impact_score(article, EventType.EARNINGS)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should be significant for Reuters + earnings
    
    def test_predict_direction(self, event_detector):
        """Test market direction prediction"""
        positive_text = "company beats earnings expectations with strong growth"
        negative_text = "company misses earnings with weak performance and decline"
        neutral_text = "company reports earnings in line with expectations"
        
        assert event_detector._predict_direction(positive_text, EventType.EARNINGS) == 'bullish'
        assert event_detector._predict_direction(negative_text, EventType.EARNINGS) == 'bearish'
        assert event_detector._predict_direction(neutral_text, EventType.EARNINGS) == 'neutral'
    
    def test_predict_time_horizon(self, event_detector):
        """Test time horizon prediction"""
        assert event_detector._predict_time_horizon(EventType.EARNINGS) == 'immediate'
        assert event_detector._predict_time_horizon(EventType.MERGER_ACQUISITION) == 'short_term'
        assert event_detector._predict_time_horizon(EventType.REGULATORY) == 'long_term'

class TestNewsSentimentAgent:
    """Test the main News and Sentiment Analysis Agent"""
    
    @pytest.fixture
    async def agent(self):
        agent = NewsSentimentAgent()
        # Mock database initialization
        agent.db_manager.initialize = AsyncMock()
        agent.db_manager.create_tables = AsyncMock()
        await agent.initialize()
        return agent
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self, agent):
        """Test LangGraph workflow creation"""
        workflow = agent.create_workflow()
        
        assert workflow is not None
        # Verify nodes are added
        nodes = workflow.nodes
        expected_nodes = [
            'fetch_news', 'process_articles', 'analyze_sentiment',
            'detect_events', 'store_data', 'generate_stats'
        ]
        
        for node in expected_nodes:
            assert node in nodes
    
    @pytest.mark.asyncio
    async def test_fetch_news_node(self, agent):
        """Test news fetching node"""
        state = SentimentState(
            symbols=['AAPL', 'GOOGL'],
            time_range=timedelta(hours=24),
            news_sources=[NewsSource.RSS_FEED],
            raw_articles=[],
            processed_articles=[],
            sentiment_analyses=[],
            detected_events=[],
            processing_stats={},
            errors=[]
        )
        
        # Mock news ingestor
        agent.news_ingestor.fetch_news_articles = AsyncMock(return_value=[
            {
                'title': 'Test Article',
                'content': 'Test content',
                'url': 'https://example.com/test',
                'source': NewsSource.RSS_FEED,
                'published_at': datetime.utcnow(),
                'symbols': ['AAPL'],
                'author': 'Test Author'
            }
        ])
        
        result_state = await agent.fetch_news_node(state)
        
        assert len(result_state.raw_articles) == 1
        assert result_state.processing_stats['articles_fetched'] == 1
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_node(self, agent):
        """Test sentiment analysis node"""
        article = NewsArticle(
            title="Test Article",
            content="Test content about positive earnings",
            url="https://example.com/test",
            source=NewsSource.REUTERS,
            published_at=datetime.utcnow(),
            symbols=['AAPL']
        )
        
        state = SentimentState(
            symbols=['AAPL'],
            time_range=timedelta(hours=24),
            news_sources=[NewsSource.RSS_FEED],
            raw_articles=[],
            processed_articles=[article],
            sentiment_analyses=[],
            detected_events=[],
            processing_stats={},
            errors=[]
        )
        
        # Mock sentiment analyzer
        mock_analysis = SentimentAnalysis(
            article_id=1,
            symbol='AAPL',
            finbert_score=1.0,
            finbert_label='positive',
            finbert_confidence=0.85,
            gemini_score=1.2,
            gemini_reasoning='Positive earnings news',
            composite_score=1.1,
            composite_label='positive',
            confidence_level=0.9,
            timestamp=datetime.utcnow()
        )
        
        agent.sentiment_analyzer.analyze_sentiment = AsyncMock(return_value=mock_analysis)
        
        result_state = await agent.analyze_sentiment_node(state)
        
        assert len(result_state.sentiment_analyses) == 1
        assert result_state.sentiment_analyses[0].symbol == 'AAPL'
        assert result_state.sentiment_analyses[0].composite_score > 0
    
    @pytest.mark.asyncio
    async def test_full_workflow_execution(self, agent):
        """Test complete workflow execution"""
        symbols = ['AAPL', 'GOOGL']
        
        # Mock all components
        agent.news_ingestor.fetch_news_articles = AsyncMock(return_value=[
            {
                'title': 'Apple Earnings Beat',
                'content': 'Apple reported strong quarterly earnings...',
                'url': 'https://example.com/apple-earnings',
                'source': NewsSource.REUTERS,
                'published_at': datetime.utcnow(),
                'symbols': ['AAPL'],
                'author': 'Financial Reporter'
            }
        ])
        
        agent.news_ingestor.extract_full_article_content = AsyncMock(return_value=None)
        
        mock_analysis = SentimentAnalysis(
            article_id=1,
            symbol='AAPL',
            finbert_score=1.0,
            finbert_label='positive',
            finbert_confidence=0.85,
            gemini_score=1.2,
            gemini_reasoning='Strong earnings performance',
            composite_score=1.1,
            composite_label='positive',
            confidence_level=0.9,
            timestamp=datetime.utcnow()
        )
        
        agent.sentiment_analyzer.analyze_sentiment = AsyncMock(return_value=mock_analysis)
        
        mock_event = MarketEvent(
            event_type=EventType.EARNINGS,
            title='Apple Earnings Beat',
            description='Apple reported strong quarterly earnings...',
            symbols=['AAPL'],
            impact_score=0.8,
            confidence=0.7,
            predicted_direction='bullish',
            time_horizon='immediate',
            source_articles=[1],
            detected_at=datetime.utcnow()
        )
        
        agent.event_detector.detect_events = AsyncMock(return_value=[mock_event])
        
        # Mock database operations
        agent.db_manager.insert_articles = AsyncMock(return_value=[1])
        agent.db_manager.insert_sentiment_analyses = AsyncMock(return_value=1)
        agent.db_manager.insert_events = AsyncMock(return_value=1)
        
        result = await agent.analyze_news_sentiment(symbols, hours_back=24)
        
        assert result['success'] is True
        assert len(result['sentiment_analyses']) == 1
        assert len(result['detected_events']) == 1
        assert result['statistics']['articles_fetched'] == 1
        assert result['statistics']['sentiment_analyses'] == 1
        assert result['statistics']['events_detected'] == 1

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_create_agent_factory(self):
        """Test agent factory function"""
        with patch('agents.news_sentiment_agent.DatabaseManager') as mock_db:
            mock_db_instance = Mock()
            mock_db_instance.initialize = AsyncMock()
            mock_db_instance.create_tables = AsyncMock()
            mock_db.return_value = mock_db_instance
            
            agent = await create_news_sentiment_agent()
            
            assert isinstance(agent, NewsSentimentAgent)
            mock_db_instance.initialize.assert_called_once()
            mock_db_instance.create_tables.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in workflow"""
        agent = NewsSentimentAgent()
        agent.db_manager.initialize = AsyncMock()
        agent.db_manager.create_tables = AsyncMock()
        await agent.initialize()
        
        # Mock failure in news fetching
        agent.news_ingestor.fetch_news_articles = AsyncMock(side_effect=Exception("API Error"))
        
        result = await agent.analyze_news_sentiment(['AAPL'], hours_back=24)
        
        assert result['success'] is False
        assert len(result['errors']) > 0
        assert 'API Error' in str(result['errors'])

# Performance and load tests
class TestPerformance:
    """Performance tests for the sentiment analysis agent"""
    
    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing large batches of articles"""
        agent = NewsSentimentAgent()
        agent.db_manager.initialize = AsyncMock()
        agent.db_manager.create_tables = AsyncMock()
        await agent.initialize()
        
        # Create 100 mock articles
        mock_articles = []
        for i in range(100):
            mock_articles.append({
                'title': f'Test Article {i}',
                'content': f'Test content for article {i} about positive earnings',
                'url': f'https://example.com/article-{i}',
                'source': NewsSource.RSS_FEED,
                'published_at': datetime.utcnow(),
                'symbols': ['AAPL'],
                'author': f'Author {i}'
            })
        
        agent.news_ingestor.fetch_news_articles = AsyncMock(return_value=mock_articles)
        agent.news_ingestor.extract_full_article_content = AsyncMock(return_value=None)
        
        # Mock sentiment analysis to return quickly
        mock_analysis = SentimentAnalysis(
            article_id=1,
            symbol='AAPL',
            finbert_score=1.0,
            finbert_label='positive',
            finbert_confidence=0.85,
            gemini_score=1.2,
            gemini_reasoning='Positive sentiment',
            composite_score=1.1,
            composite_label='positive',
            confidence_level=0.9,
            timestamp=datetime.utcnow()
        )
        
        agent.sentiment_analyzer.analyze_sentiment = AsyncMock(return_value=mock_analysis)
        agent.event_detector.detect_events = AsyncMock(return_value=[])
        
        # Mock database operations
        agent.db_manager.insert_articles = AsyncMock(return_value=list(range(1, 101)))
        agent.db_manager.insert_sentiment_analyses = AsyncMock(return_value=100)
        agent.db_manager.insert_events = AsyncMock(return_value=0)
        
        start_time = datetime.utcnow()
        result = await agent.analyze_news_sentiment(['AAPL'], hours_back=24)
        end_time = datetime.utcnow()
        
        processing_time = (end_time - start_time).total_seconds()
        
        assert result['success'] is True
        assert len(result['sentiment_analyses']) == 100
        assert processing_time < 30  # Should complete within 30 seconds
        
        # Verify performance metrics
        assert result['statistics']['articles_fetched'] == 100
        assert result['statistics']['sentiment_analyses'] == 100

if __name__ == '__main__':
    pytest.main([__file__])