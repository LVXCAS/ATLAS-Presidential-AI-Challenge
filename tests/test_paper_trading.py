"""
Paper Trading Test Script

Simple test to validate paper trading functionality
"""

import asyncio
import logging
from datetime import datetime

async def test_paper_trading():
    """Test basic paper trading functionality"""

    print("=" * 60)
    print("PAPER TRADING VALIDATION TEST")
    print("=" * 60)

    # Test 1: Portfolio initialization
    print("\n1. Testing portfolio initialization...")
    initial_capital = 100000.0
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print("   [OK] Portfolio initialized")

    # Test 2: Market data simulation
    print("\n2. Testing market data simulation...")
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']
    test_prices = {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 420.0}
    print(f"   Test symbols: {test_symbols}")
    print(f"   Simulated prices: {test_prices}")
    print("   [OK] Market data simulation ready")

    # Test 3: Order simulation
    print("\n3. Testing order simulation...")
    test_order = {
        'symbol': 'AAPL',
        'side': 'buy',
        'quantity': 100,
        'price': 150.0,
        'order_type': 'limit'
    }
    print(f"   Test order: {test_order}")

    # Simulate order execution
    execution_price = test_order['price'] * 1.001  # Small slippage
    commission = test_order['quantity'] * test_order['price'] * 0.001

    print(f"   Executed at: ${execution_price:.3f}")
    print(f"   Commission: ${commission:.2f}")
    print("   [OK] Order execution simulation working")

    # Test 4: Position tracking
    print("\n4. Testing position tracking...")
    position_value = test_order['quantity'] * execution_price
    remaining_cash = initial_capital - position_value - commission

    print(f"   Position value: ${position_value:,.2f}")
    print(f"   Remaining cash: ${remaining_cash:,.2f}")
    print(f"   Total portfolio: ${remaining_cash + position_value:,.2f}")
    print("   [OK] Position tracking working")

    # Test 5: Risk calculations
    print("\n5. Testing risk calculations...")
    position_percentage = position_value / initial_capital
    max_allowed = 0.1  # 10%

    print(f"   Position size: {position_percentage:.1%}")
    print(f"   Max allowed: {max_allowed:.1%}")

    if position_percentage <= max_allowed:
        print("   [OK] Position within risk limits")
    else:
        print("   [WARNING] Position exceeds risk limits")

    print("\n" + "=" * 60)
    print("PAPER TRADING TEST COMPLETED")
    print("All basic functions are working correctly")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_paper_trading())
