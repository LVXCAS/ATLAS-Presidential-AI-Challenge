#!/usr/bin/env python3
"""
Test placing an actual order right now
"""
import asyncio
import sys
sys.path.append('.')

async def test_real_order():
    """Test placing a real order to Alpaca"""
    try:
        from agents.broker_integration import AlpacaBrokerIntegration, OrderRequest, OrderSide, OrderType
        
        print("Testing real order placement...")
        
        # Initialize broker
        broker = AlpacaBrokerIntegration(paper_trading=True)
        
        # Get account info first
        account = await broker.get_account_info()
        if account:
            print(f"Account: {account.get('account_number', 'N/A')}")
            print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
        
        # Create a small test order (1 share of SPY)
        test_order = OrderRequest(
            symbol='SPY',
            qty=1,
            side=OrderSide.BUY,
            type=OrderType.MARKET
        )
        
        print("Placing test order: BUY 1 SPY...")
        order_response = await broker.submit_order(test_order)
        
        if order_response and hasattr(order_response, 'id'):
            print(f"SUCCESS! Order placed with ID: {order_response.id}")
            print(f"Check your Alpaca dashboard: https://app.alpaca.markets/paper/dashboard/orders")
            return True
        else:
            print("Order submitted but no response received")
            return False
            
    except Exception as e:
        print(f"Error placing test order: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing live order placement...")
    success = asyncio.run(test_real_order())
    if success:
        print("SUCCESS: Bot can place orders on your Alpaca account!")
    else:
        print("FAILED: Order placement not working")
    
    input("Press Enter to close...")