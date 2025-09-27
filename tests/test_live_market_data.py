"""
TEST LIVE MARKET DATA ENGINE
Simple test without Unicode characters for Windows compatibility
"""

import asyncio
import sys
import os
sys.path.append(os.getcwd())

from live_market_data_engine import LiveMarketDataEngine, MarketTick

async def test_market_data():
    """Test the live market data engine"""
    print("="*80)
    print("LIVE MARKET DATA ENGINE TEST")
    print("Real-time data feeds for autonomous trading")
    print("="*80)

    # Initialize engine
    engine = LiveMarketDataEngine()

    # Simple tick handler
    async def handle_tick(tick: MarketTick):
        print(f"TICK: {tick.symbol}: ${tick.price:.2f} (Vol: {tick.volume})")

    # Simple news handler
    async def handle_news(news_items):
        for news in news_items[-1:]:  # Show latest news
            print(f"NEWS: {news['headline']} (Sentiment: {news['sentiment']:.2f})")

    # Subscribe to data
    engine.subscribe_to_ticks(handle_tick)
    engine.subscribe_to_news(handle_news)

    print("\nStarting live data streams...")
    print("This is DEMO mode - configure API keys for real data")

    # Start streaming for demo
    try:
        await asyncio.wait_for(engine.start_streaming(), timeout=10)
    except asyncio.TimeoutError:
        print("\nDemo completed")
    finally:
        engine.stop_streaming()

        # Show final status
        status = engine.get_market_status()
        print("\nFINAL STATUS:")
        print(f"   Quotes received: {status['latest_quotes_count']}")
        print(f"   Tick buffer: {status['tick_buffer_size']}")
        print(f"   News items: {status['news_buffer_size']}")

    print("\nLive market data engine ready for autonomous trading!")

if __name__ == "__main__":
    asyncio.run(test_market_data())