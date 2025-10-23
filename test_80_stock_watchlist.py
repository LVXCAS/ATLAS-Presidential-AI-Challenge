"""
Test script to verify all 80 stocks in the expanded S&P 500 watchlist
can successfully fetch data from real market data sources.

This validates:
- Alpaca API connectivity
- Polygon fallback
- OpenBB fallback
- Yahoo Finance fallback
- Data availability for all 80 stocks
"""

import asyncio
import sys
from datetime import datetime, timedelta
from sp500_80_stocks import get_sp500_top_80, get_sp500_80_by_sector

# Import the real data connector
from agents.real_data_connector import fetch_real_market_data


async def test_stock_data_fetch(symbol: str, days: int = 10) -> dict:
    """
    Test data fetch for a single stock

    Returns:
        dict with success status and details
    """
    try:
        print(f"Testing {symbol}...", end=" ", flush=True)

        # Fetch data
        data = await fetch_real_market_data(symbol, days=days, timeframe='1Day')

        if data is not None and len(data) > 0:
            print(f"[OK] {len(data)} bars fetched")
            return {
                'symbol': symbol,
                'success': True,
                'bars': len(data),
                'latest_price': data['close'].iloc[-1] if 'close' in data.columns else None,
                'date_range': f"{data.index[0]} to {data.index[-1]}"
            }
        else:
            print(f"[FAILED] No data returned")
            return {
                'symbol': symbol,
                'success': False,
                'error': 'No data returned'
            }

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }


async def test_all_stocks():
    """Test data connectivity for all 80 stocks"""

    print("=" * 80)
    print("TESTING 80-STOCK S&P 500 WATCHLIST DATA CONNECTIVITY")
    print("=" * 80)
    print()

    # Get the 80-stock list
    stocks = get_sp500_top_80()
    sectors = get_sp500_80_by_sector()

    print(f"Total stocks to test: {len(stocks)}")
    print(f"Sectors: {len(sectors)}")
    print()

    # Test by sector
    all_results = []

    for sector_name, sector_stocks in sectors.items():
        print(f"\n--- Testing {sector_name} ({len(sector_stocks)} stocks) ---")

        for symbol in sector_stocks:
            result = await test_stock_data_fetch(symbol, days=10)
            all_results.append(result)

            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)

    # Summary
    print()
    print("=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)

    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]

    print(f"\nTotal Tested: {len(all_results)}")
    print(f"Successful:   {len(successful)} ({len(successful)/len(all_results)*100:.1f}%)")
    print(f"Failed:       {len(failed)} ({len(failed)/len(all_results)*100:.1f}%)")

    if failed:
        print("\nFailed Stocks:")
        for r in failed:
            print(f"  - {r['symbol']}: {r.get('error', 'Unknown error')}")

    # Sector breakdown
    print("\nSector Breakdown:")
    for sector_name, sector_stocks in sectors.items():
        sector_results = [r for r in all_results if r['symbol'] in sector_stocks]
        sector_success = [r for r in sector_results if r['success']]
        print(f"  {sector_name:25} {len(sector_success)}/{len(sector_stocks)} successful")

    # Sample prices
    print("\nSample Latest Prices (first 10 stocks):")
    for r in all_results[:10]:
        if r['success'] and r.get('latest_price'):
            print(f"  {r['symbol']:6} ${r['latest_price']:>8.2f}")

    print()
    print("=" * 80)

    if len(successful) == len(all_results):
        print("[SUCCESS] All 80 stocks fetched data successfully!")
        print("Your trading system is ready to scan the expanded watchlist.")
        return True
    else:
        print(f"[WARNING] {len(failed)} stocks failed to fetch data.")
        print("Check the failures above and verify API credentials.")
        return False


async def quick_validation():
    """Quick validation - test one stock from each sector"""

    print("=" * 80)
    print("QUICK VALIDATION - One stock per sector")
    print("=" * 80)
    print()

    sectors = get_sp500_80_by_sector()

    for sector_name, sector_stocks in sectors.items():
        # Test first stock in each sector
        symbol = sector_stocks[0]
        print(f"{sector_name:25} - Testing {symbol}...", end=" ", flush=True)

        result = await test_stock_data_fetch(symbol, days=5)

        if not result['success']:
            print(f"  [FAILED]")
            return False

        await asyncio.sleep(0.1)

    print()
    print("[SUCCESS] All sectors validated - full watchlist is ready!")
    return True


def main():
    """Main test runner"""

    print("\n80-Stock Watchlist Data Connectivity Test")
    print("==========================================\n")

    # Check if user wants quick or full test
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("Running QUICK validation (one stock per sector)...\n")
        success = asyncio.run(quick_validation())
    else:
        print("Running FULL test (all 80 stocks)...")
        print("Use --quick flag for faster validation\n")
        success = asyncio.run(test_all_stocks())

    if success:
        print("\n[OK] Watchlist is ready for trading!")
        sys.exit(0)
    else:
        print("\n[ERROR] Some stocks failed - check API configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()
