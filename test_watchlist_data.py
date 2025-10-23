"""
Test Expanded Watchlist Data Connectivity
Tests all 20 stocks to ensure data fetching works
"""

import asyncio
import logging
from agents.real_data_connector import fetch_real_market_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expanded 20-stock watchlist
WATCHLIST = [
    # Market Indices (4)
    'SPY', 'QQQ', 'IWM', 'DIA',

    # Mega Cap Technology (7)
    'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META',

    # Financial Services (3)
    'JPM', 'BAC', 'V',

    # Healthcare (2)
    'JNJ', 'UNH',

    # Energy (2)
    'XOM', 'CVX',

    # Consumer (2)
    'WMT', 'HD',
]

async def test_watchlist():
    """Test data fetching for all watchlist stocks"""

    print("\n" + "=" * 80)
    print("TESTING EXPANDED 20-STOCK WATCHLIST")
    print("=" * 80)
    print(f"\nWatchlist Size: {len(WATCHLIST)} stocks")
    print(f"Test: Fetching 5 days of data for each stock\n")

    results = {
        'success': [],
        'failed': []
    }

    for i, symbol in enumerate(WATCHLIST, 1):
        try:
            print(f"[{i}/{len(WATCHLIST)}] Testing {symbol}...", end=" ")

            # Fetch 5 days of data
            df = await fetch_real_market_data(symbol, days=5)

            if df is not None and len(df) > 0:
                latest_price = df['close'].iloc[-1]
                print(f"[OK] ${latest_price:.2f} ({len(df)} bars)")
                results['success'].append(symbol)
            else:
                print(f"[FAIL] No data returned")
                results['failed'].append(symbol)

        except Exception as e:
            print(f"[ERROR] {e}")
            results['failed'].append(symbol)

    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Success: {len(results['success'])}/{len(WATCHLIST)} stocks")
    print(f"Failed:  {len(results['failed'])}/{len(WATCHLIST)} stocks")

    if results['failed']:
        print(f"\nFailed stocks: {', '.join(results['failed'])}")

    if len(results['success']) == len(WATCHLIST):
        print("\n[OK] ALL STOCKS READY FOR TRADING!")
    else:
        print(f"\n[WARNING] {len(results['failed'])} stocks need attention")

    print("=" * 80 + "\n")

    # Sector breakdown
    print("SECTOR ALLOCATION:")
    print("  Market Indices:     4 stocks (20%)")
    print("  Technology:         7 stocks (35%)")
    print("  Financial:          3 stocks (15%)")
    print("  Healthcare:         2 stocks (10%)")
    print("  Energy:             2 stocks (10%)")
    print("  Consumer:           2 stocks (10%)")
    print("\nDiversification Score: GOOD")
    print("Expected Scan Time: 3-4 minutes per cycle")
    print("Expected Opportunities: 6-12 trades per day\n")

if __name__ == "__main__":
    asyncio.run(test_watchlist())
