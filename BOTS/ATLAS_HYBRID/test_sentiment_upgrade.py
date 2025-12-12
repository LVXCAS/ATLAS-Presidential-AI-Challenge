#!/usr/bin/env python3
"""
Test upgraded SentimentAgent with Alpha Vantage real news
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agents.sentiment_agent import SentimentAgent

def test_sentiment_agent():
    print("=" * 70)
    print("TESTING UPGRADED SENTIMENT AGENT")
    print("=" * 70)

    agent = SentimentAgent(initial_weight=1.5)

    # Test EUR/USD sentiment
    market_data = {
        "pair": "EUR_USD",
        "price": 1.16624,
        "indicators": {
            "rsi": 55,
            "macd": 0.001
        }
    }

    print("\n[Test 1] EUR_USD Sentiment Analysis")
    print("-" * 70)

    vote, confidence, reasoning = agent.analyze(market_data)

    print(f"\nVote: {vote}")
    print(f"Confidence: {confidence:.2f}")
    print(f"\nReasoning:")
    for key, value in reasoning.items():
        print(f"  {key}: {value}")

    # Check if Alpha Vantage data was used
    if hasattr(agent, 'av_sentiment_data') and len(agent.av_sentiment_data) > 0:
        print(f"\n[OK] Using Alpha Vantage real news ({len(agent.av_sentiment_data)} articles)")
        print("\nTop Headlines:")
        for i, item in enumerate(agent.av_sentiment_data[:3], 1):
            print(f"\n  {i}. {item['text'][:80]}...")
            print(f"     Sentiment: {item['sentiment_label']} (score: {item['sentiment_score']:.3f})")
            print(f"     Relevance: {item['relevance']:.2f}")
    else:
        print("\n[WARN] No Alpha Vantage data - using fallback sentiment")

    # Test GBP/USD
    print("\n" + "=" * 70)
    print("[Test 2] GBP_USD Sentiment Analysis")
    print("-" * 70)

    market_data['pair'] = "GBP_USD"
    vote2, confidence2, reasoning2 = agent.analyze(market_data)

    print(f"\nVote: {vote2}")
    print(f"Confidence: {confidence2:.2f}")
    print(f"Positive news: {reasoning2.get('positive_news', 0)}")
    print(f"Negative news: {reasoning2.get('negative_news', 0)}")
    print(f"Avg sentiment: {reasoning2.get('avg_sentiment', 0):.3f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    # Summary
    print("\n[SUMMARY]")
    if hasattr(agent, 'av_sentiment_data') and len(agent.av_sentiment_data) > 0:
        print("[OK] Alpha Vantage integration working")
        print("[OK] Real news sentiment analysis active")
        print("[OK] No more synthetic headlines!")
    else:
        print("[FAIL] Alpha Vantage not working - check API key")

if __name__ == "__main__":
    test_sentiment_agent()
