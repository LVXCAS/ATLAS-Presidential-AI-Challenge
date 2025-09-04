#!/usr/bin/env python3
"""
Check current P&L and trading performance
"""
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

def check_trading_performance():
    """Check current account performance and positions"""
    try:
        # Connect to Alpaca
        api = tradeapi.REST(
            key_id='PKQ5FPEZS2ZY13C0Q9QX',
            secret_key='UdTRNXTgGYBxjF1NDcVeZZo6mZb2S9UdDJ6fN4JD',
            base_url='https://paper-api.alpaca.markets'
        )
        
        print("TRADING PERFORMANCE REPORT")
        print("=" * 50)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Get account info
        account = api.get_account()
        print("ACCOUNT SUMMARY:")
        print(f"Account ID: {account.id}")
        print(f"Account Number: {account.account_number}")
        print(f"Total Equity: ${float(account.equity):,.2f}")
        print(f"Cash: ${float(account.cash):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print()
        
        # Calculate P&L (assuming started with $100,000)
        starting_capital = 100000.0
        current_equity = float(account.equity)
        total_pnl = current_equity - starting_capital
        total_pnl_pct = (total_pnl / starting_capital) * 100
        
        print("PROFIT & LOSS:")
        print(f"Starting Capital: ${starting_capital:,.2f}")
        print(f"Current Equity: ${current_equity:,.2f}")
        print(f"Total P&L: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
        print()
        
        # Get current positions
        positions = api.list_positions()
        print(f"CURRENT POSITIONS ({len(positions)}):")
        if positions:
            total_position_value = 0
            for pos in positions:
                qty = float(pos.qty)
                market_value = float(pos.market_value)
                unrealized_pnl = float(pos.unrealized_pl)
                unrealized_pnl_pct = float(pos.unrealized_plpc) * 100
                avg_cost = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                
                total_position_value += market_value
                
                print(f"  {pos.symbol}:")
                print(f"    Quantity: {qty:,.0f} shares")
                print(f"    Avg Cost: ${avg_cost:.2f}")
                print(f"    Current Price: ${current_price:.2f}")
                print(f"    Market Value: ${market_value:,.2f}")
                print(f"    Unrealized P&L: ${unrealized_pnl:+,.2f} ({unrealized_pnl_pct:+.2f}%)")
                print()
            
            print(f"Total Position Value: ${total_position_value:,.2f}")
        else:
            print("  No open positions")
        print()
        
        # Get recent orders
        orders = api.list_orders(status='all', limit=10)
        print(f"RECENT ORDERS ({len(orders)}):")
        if orders:
            for i, order in enumerate(orders, 1):
                side = order.side.upper()
                status = order.status.upper()
                filled_qty = float(order.filled_qty) if order.filled_qty else 0
                qty = float(order.qty)
                
                # Get filled price if available
                filled_price = None
                if hasattr(order, 'filled_avg_price') and order.filled_avg_price:
                    filled_price = float(order.filled_avg_price)
                
                print(f"  {i}. {order.symbol} {side} {qty:.0f} shares - {status}")
                print(f"     Order ID: {order.id}")
                print(f"     Submitted: {order.submitted_at}")
                if filled_qty > 0:
                    print(f"     Filled: {filled_qty:.0f} @ ${filled_price:.2f}" if filled_price else f"     Filled: {filled_qty:.0f}")
                print()
        else:
            print("  No recent orders")
        
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"Error checking performance: {e}")
        return False

if __name__ == "__main__":
    print("Checking trading performance...")
    success = check_trading_performance()
    if not success:
        print("Failed to retrieve performance data")
    
    input("Press Enter to close...")