"""
TEST LIVE BROKER EXECUTION ENGINE
Simple test without Unicode characters for Windows compatibility
"""

import asyncio
import sys
import os
sys.path.append(os.getcwd())

from live_broker_execution_engine import LiveBrokerExecutionEngine

async def test_broker_execution():
    """Test the live broker execution engine"""
    print("="*80)
    print("LIVE BROKER EXECUTION ENGINE TEST")
    print("Real-time order execution for autonomous trading")
    print("="*80)

    # Initialize execution engine
    engine = LiveBrokerExecutionEngine()

    print("\nBroker connections status:")
    status = engine.get_system_status()
    for broker, connected in status['broker_connections'].items():
        connection_status = "CONNECTED" if connected else "DEMO MODE"
        print(f"   {broker}: {connection_status}")

    print(f"\nCurrent portfolio:")
    portfolio = engine.get_portfolio_summary()
    print(f"   Cash: ${portfolio['total_cash']:,.2f}")
    print(f"   Equity Value: ${portfolio['total_equity']:,.2f}")
    print(f"   Total Value: ${portfolio['total_value']:,.2f}")
    print(f"   Open Positions: {portfolio['position_count']}")

    # Test order validation
    print(f"\nTesting order validation...")

    test_orders = [
        {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 10,
            'order_type': 'MARKET',
            'strategy_id': 'GPU_TEST_001'
        },
        {
            'symbol': 'MSFT',
            'action': 'BUY',
            'quantity': 5,
            'order_type': 'LIMIT',
            'limit_price': 400.00,
            'strategy_id': 'GPU_TEST_002'
        }
    ]

    for order in test_orders:
        validation = engine.validate_order(order)
        status_text = "VALID" if validation['is_valid'] else "INVALID"
        print(f"   Order {order['symbol']} {order['action']}: {status_text}")
        if validation['risk_warnings']:
            for warning in validation['risk_warnings']:
                print(f"     WARNING: {warning}")

    # Test demo execution
    print(f"\nTesting demo order execution...")
    for order in test_orders:
        if engine.validate_order(order)['is_valid']:
            try:
                result = await engine.execute_order(order)
                print(f"   {order['symbol']}: Order ID {result['order_id']} - {result['status']}")
            except Exception as e:
                print(f"   {order['symbol']}: Execution failed - {e}")

    # Show final status
    print(f"\nFinal system status:")
    final_status = engine.get_system_status()
    print(f"   Total Orders: {final_status['total_orders']}")
    print(f"   Successful Orders: {final_status['successful_orders']}")
    print(f"   Failed Orders: {final_status['failed_orders']}")
    print(f"   Real-time monitoring: {final_status['monitoring_active']}")

    print(f"\nLive broker execution engine ready for autonomous trading!")

if __name__ == "__main__":
    asyncio.run(test_broker_execution())