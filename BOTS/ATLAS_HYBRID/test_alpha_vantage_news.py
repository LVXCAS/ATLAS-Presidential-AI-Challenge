#!/usr/bin/env python3
"""
Test Alpha Vantage News Sentiment API
Free tier includes sentiment scores for forex-related news
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def test_forex_news_sentiment():
    """Test news sentiment for forex topics"""
    print("\n[1/3] Testing Forex News Sentiment...")

    topics = ["forex", "federal_reserve", "interest_rates"]

    for topic in topics:
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": topic,
            "apikey": API_KEY,
            "limit": 5
        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            data = response.json()

            if "feed" in data:
                feed = data["feed"]
                print(f"\n  Topic '{topic}': {len(feed)} articles")

                for article in feed[:3]:
                    print(f"\n    Title: {article.get('title')[:70]}...")
                    print(f"    Published: {article.get('time_published')}")

                    # Sentiment scores
                    sentiment = article.get('overall_sentiment_score', 0)
                    label = article.get('overall_sentiment_label', 'Unknown')
                    print(f"    Sentiment: {label} (score: {sentiment:.3f})")

                    # Ticker relevance
                    ticker_sentiment = article.get('ticker_sentiment', [])
                    if ticker_sentiment:
                        print(f"    Tickers: {', '.join([t['ticker'] for t in ticker_sentiment[:3]])}")
            else:
                print(f"\n  Topic '{topic}': {data}")
        else:
            print(f"\n  Topic '{topic}': ERROR {response.status_code}")

def test_ticker_news():
    """Test ticker-specific news (may not work for forex)"""
    print("\n\n[2/3] Testing Ticker-Specific News...")

    # Try forex majors (Alpha Vantage may not support forex tickers)
    tickers = ["EURUSD", "GBPUSD", "EUR", "USD"]

    for ticker in tickers[:2]:  # Limit to avoid rate limit
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": API_KEY,
            "limit": 3
        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            data = response.json()

            if "feed" in data:
                feed = data["feed"]
                print(f"\n  {ticker}: {len(feed)} articles")

                if feed:
                    article = feed[0]
                    print(f"    {article.get('title')[:60]}...")
                    print(f"    Sentiment: {article.get('overall_sentiment_label')} ({article.get('overall_sentiment_score', 0):.3f})")
            else:
                print(f"\n  {ticker}: No feed data")
        else:
            print(f"\n  {ticker}: ERROR {response.status_code}")

def test_time_range():
    """Test recent news (last 24 hours)"""
    print("\n\n[3/3] Testing Time Range Filtering...")

    # Format: YYYYMMDDTHHMM
    time_from = "20251203T0000"

    params = {
        "function": "NEWS_SENTIMENT",
        "topics": "economy_monetary",
        "time_from": time_from,
        "apikey": API_KEY,
        "limit": 5
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json()

        if "feed" in data:
            feed = data["feed"]
            print(f"\n  Found {len(feed)} articles since Dec 3")

            for article in feed[:3]:
                pub_time = article.get('time_published', '')
                formatted_time = f"{pub_time[:4]}-{pub_time[4:6]}-{pub_time[6:8]} {pub_time[9:11]}:{pub_time[11:13]}"

                print(f"\n    {article.get('title')[:60]}...")
                print(f"    Published: {formatted_time}")
                print(f"    Sentiment: {article.get('overall_sentiment_label')} ({article.get('overall_sentiment_score', 0):.3f})")
        else:
            print(f"\n  No feed: {data}")
    else:
        print(f"\n  ERROR {response.status_code}")

def main():
    print("=" * 70)
    print("ALPHA VANTAGE NEWS SENTIMENT API TEST")
    print("=" * 70)
    print(f"API Key: {API_KEY}")

    test_forex_news_sentiment()
    # test_ticker_news()  # Commented to avoid rate limit
    test_time_range()

    print("\n" + "=" * 70)
    print("NOTE: Alpha Vantage free tier = 25 API calls per day")
    print("=" * 70)

if __name__ == "__main__":
    main()
