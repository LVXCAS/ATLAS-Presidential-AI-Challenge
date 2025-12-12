#!/usr/bin/env python3
"""
Test Massive.com (formerly Polygon.io) News API
Check what news/sentiment data is available for forex trading
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"

def test_ticker_news():
    """Test ticker-specific news (forex tickers)"""
    print("\n[1/4] Testing Forex Ticker News...")

    # Forex tickers on Polygon use format C:EURUSD, C:GBPUSD, etc.
    tickers = ["C:EURUSD", "C:GBPUSD", "C:USDJPY"]

    for ticker in tickers:
        url = f"{BASE_URL}/v2/reference/news"
        params = {
            "ticker": ticker,
            "limit": 3,
            "apiKey": API_KEY
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print(f"\n  {ticker}: {len(results)} articles found")

            for article in results[:2]:
                print(f"    - {article.get('title')[:80]}...")
                print(f"      Published: {article.get('published_utc')}")
                if article.get('insights'):
                    print(f"      Insights: {article.get('insights')}")
        else:
            print(f"  {ticker}: ERROR {response.status_code} - {response.text}")

def test_general_market_news():
    """Test general market/economic news"""
    print("\n\n[2/4] Testing General Market News...")

    url = f"{BASE_URL}/v2/reference/news"
    params = {
        "limit": 5,
        "order": "desc",
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        print(f"  Found {len(results)} recent articles")

        for article in results:
            print(f"\n  Title: {article.get('title')}")
            print(f"  Publisher: {article.get('publisher', {}).get('name')}")
            print(f"  Published: {article.get('published_utc')}")
            if article.get('tickers'):
                print(f"  Tickers: {', '.join(article.get('tickers', [])[:5])}")
    else:
        print(f"  ERROR {response.status_code}: {response.text}")

def test_sentiment_analysis():
    """Check if sentiment data is available"""
    print("\n\n[3/4] Testing Sentiment Analysis...")

    url = f"{BASE_URL}/v2/reference/news"
    params = {
        "ticker": "C:EURUSD",
        "limit": 5,
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])

        sentiment_found = False
        for article in results:
            # Check for sentiment fields
            insights = article.get('insights', [])
            sentiment = article.get('sentiment')

            if insights or sentiment:
                sentiment_found = True
                print(f"\n  Article: {article.get('title')[:60]}...")
                if sentiment:
                    print(f"    Sentiment: {sentiment}")
                if insights:
                    print(f"    Insights: {insights}")

        if not sentiment_found:
            print("  No sentiment data found in articles")
            print("  (May require higher-tier API plan)")
    else:
        print(f"  ERROR {response.status_code}: {response.text}")

def test_real_time_updates():
    """Check latest news update times"""
    print("\n\n[4/4] Testing Real-Time News Updates...")

    url = f"{BASE_URL}/v2/reference/news"
    params = {
        "limit": 1,
        "order": "desc",
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])

        if results:
            latest = results[0]
            pub_time = datetime.fromisoformat(latest['published_utc'].replace('Z', '+00:00'))
            now = datetime.now(pub_time.tzinfo)
            delay = now - pub_time

            print(f"  Latest article: {latest.get('title')[:70]}...")
            print(f"  Published: {pub_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  Delay: {delay.total_seconds() / 60:.1f} minutes ago")

            if delay.total_seconds() < 3600:
                print("  ✓ Real-time updates available (< 1 hour delay)")
            else:
                print(f"  ⚠ Delay of {delay.total_seconds() / 3600:.1f} hours")
    else:
        print(f"  ERROR {response.status_code}: {response.text}")

def main():
    print("=" * 70)
    print("MASSIVE.COM (Polygon.io) NEWS API TEST")
    print("=" * 70)
    print(f"API Key: {API_KEY[:20]}...")

    test_ticker_news()
    test_general_market_news()
    test_sentiment_analysis()
    test_real_time_updates()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
