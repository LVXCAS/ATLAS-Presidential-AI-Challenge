#!/usr/bin/env python3
"""
Direct Alpaca API test with hardcoded credentials
"""
import alpaca_trade_api as tradeapi

def test_direct_connection():
    """Test direct connection to Alpaca with provided credentials"""
    
    # Your provided credentials  
    api_key = "PKQ5FPEZS2ZY13C0Q9QX"
    secret_key = "UdTRNXTgGYBxjF1NDcVeZZo6mZb2S9UdDJ6fN4JD"
    base_url = "https://paper-api.alpaca.markets"
    
    print("Testing Direct Alpaca Connection...")
    print("=" * 50)
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Base URL: {base_url}")
    print()
    
    try:
        # Initialize API client
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url
        )
        
        # Get account information
        account = api.get_account()
        
        print("[SUCCESS] Connected to your Alpaca account!")
        print("=" * 50)
        print("ACCOUNT DETAILS:")
        print(f"Account ID: {account.id}")
        print(f"Account Number: {account.account_number}")
        print(f"Status: {account.status}")
        print(f"Currency: {account.currency}")
        print()
        print("FINANCIAL DETAILS:")
        print(f"Account Equity: ${float(account.equity):,.2f}")
        print(f"Cash: ${float(account.cash):,.2f}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Day Trading Buying Power: ${float(account.daytrading_buying_power):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print()
        
        # Get current positions
        positions = api.list_positions()
        print(f"CURRENT POSITIONS: {len(positions)}")
        if positions:
            for pos in positions:
                print(f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_cost):.2f}")
        else:
            print("  No open positions")
        print()
        
        # Get recent orders
        orders = api.list_orders(status='all', limit=5)
        print(f"RECENT ORDERS: {len(orders)}")
        if orders:
            for order in orders:
                print(f"  {order.symbol} {order.side} {order.qty} @ {order.order_type} - Status: {order.status}")
        else:
            print("  No recent orders")
        print()
        
        print("TRADING CONFIRMATION:")
        print(f"[SUCCESS] Your bot IS connected to your Alpaca account!")
        print(f"[SUCCESS] All trades will appear on Alpaca website:")
        print(f"         https://app.alpaca.markets/paper/dashboard/overview")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_direct_connection()
    print("\n" + "=" * 50)
    if success:
        print("[SUCCESS] Your account is ready for bot trading!")
    else:
        print("[ERROR] Connection failed - check credentials")