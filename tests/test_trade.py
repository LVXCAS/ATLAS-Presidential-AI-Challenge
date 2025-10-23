#!/usr/bin/env python3
"""
Test trade execution (paper trading only)
"""
import alpaca_trade_api as tradeapi

def test_trade_execution():
    """Test placing a small paper trade"""
    
    # Your credentials
    api_key = "PKMC9M6ZF34LF4A3ZW56"
    secret_key = "A8ihqlDX8up3Sqc9GIjmHqfdLa77CWbI0dEp3yCI"
    base_url = "https://paper-api.alpaca.markets"
    
    print("Testing Trade Execution...")
    print("=" * 50)
    
    try:
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url
        )
        
        # Check account first
        account = api.get_account()
        print(f"Account Status: {account.status}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        
        # Test a small paper trade (1 share of SPY)
        print("\nAttempting to place test order...")
        print("Symbol: SPY (1 share)")
        print("Order Type: Market Buy")
        print("This is PAPER TRADING - no real money!")
        
        order = api.submit_order(
            symbol='SPY',
            qty=1,
            side='buy',
            type='market',
            time_in_force='day'
        )
        
        print(f"\n[SUCCESS] Test order placed!")
        print(f"Order ID: {order.id}")
        print(f"Symbol: {order.symbol}")
        print(f"Quantity: {order.qty}")
        print(f"Side: {order.side}")
        print(f"Status: {order.status}")
        
        print(f"\n[OK] TRADE EXECUTION CONFIRMED!")
        print(f"[INFO] Check your order at: https://app.alpaca.markets/paper/dashboard/orders")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Trade test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_trade_execution()
    print("\n" + "=" * 50)
    if success:
        print("[OK] Your bot CAN make trades on your Alpaca account!")
    else:
        print("[X] Trade execution test failed")
    
    input("Press Enter to close...")