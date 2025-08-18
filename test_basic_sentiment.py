#!/usr/bin/env python3
"""
Basic test for News and Sentiment Analysis Agent
"""

import asyncio
import sys
sys.path.append('.')

from agents.news_sentiment_agent import NewsIngestor, NewsSource

async def test_basic_functionality():
    """Test basic news ingestion"""
    print("Testing basic news ingestion...")
    
    try:
        ingestor = NewsIngestor()
        symbols = ['AAPL', 'GOOGL']
        
        print(f"Fetching news for symbols: {symbols}")
        articles = await ingestor.fetch_news_articles(symbols, [NewsSource.RSS_FEED], 24)
        
        print(f"Fetched {len(articles)} articles")
        
        if articles:
            print(f"Sample article title: {articles[0]['title'][:50]}...")
            print(f"Sample article symbols: {articles[0]['symbols']}")
        
        print("✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    sys.exit(0 if success else 1)