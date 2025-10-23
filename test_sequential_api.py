#!/usr/bin/env python3
"""
Test Sequential API Manager
"""

from agents.sequential_api_manager import get_api_manager

def test_api_manager():
    """Test the sequential API manager"""

    print("\n" + "="*70)
    print("TESTING SEQUENTIAL API MANAGER")
    print("="*70)

    # Get API manager
    api_manager = get_api_manager()

    print(f"\nCurrent API: {api_manager.get_current_api()}")
    print(f"Available APIs: {api_manager.available_apis}")

    # Test fetching data for a few symbols
    test_symbols = ['AAPL', 'SPY', 'MSFT', 'QQQ', 'TSLA']

    print(f"\nTesting data fetch for: {', '.join(test_symbols)}")
    print("-" * 70)

    for symbol in test_symbols:
        data = api_manager.get_market_data(symbol)

        if data:
            print(f"[OK] {symbol}: ${data['current_price']:.2f} from {data['source'].upper()}")
        else:
            print(f"[FAIL] {symbol}: Failed to fetch data")

    # Show stats
    print("\n" + "="*70)
    print("API USAGE STATISTICS")
    print("="*70)

    stats = api_manager.get_stats()
    print(f"Current API: {stats['current_api'].upper()}")
    print(f"\nSuccess counts:")
    for api, count in stats['api_success'].items():
        if count > 0:
            print(f"  {api}: {count} calls")

    print(f"\nError counts:")
    for api, count in stats['api_errors'].items():
        if count > 0:
            print(f"  {api}: {count} errors")

    print("="*70)

if __name__ == "__main__":
    test_api_manager()
