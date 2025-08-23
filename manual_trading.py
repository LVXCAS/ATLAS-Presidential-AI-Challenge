#!/usr/bin/env python3
"""
Hive Trade - Manual Trading Interface
Execute manual trades through command line
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import sys

class ManualTrader:
    def __init__(self):
        load_dotenv()
        
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            api_version='v2'
        )
        
        print("HIVE TRADE - MANUAL TRADING INTERFACE")
        print("=" * 50)

    def show_account_info(self):
        """Display account information"""
        account = self.api.get_account()
        print(f"Account Status: {account.status}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Day Trade Count: {account.daytrade_count}")

    def show_positions(self):
        """Display current positions"""
        positions = self.api.list_positions()
        if positions:
            print("\nCURRENT POSITIONS:")
            print("-" * 80)
            print(f"{'SYMBOL':<8} {'QTY':<8} {'AVG COST':<10} {'MKT VAL':<12} {'P&L':<12} {'P&L%':<8}")
            print("-" * 80)
            
            total_value = 0
            total_pnl = 0
            
            for pos in positions:
                market_value = float(pos.market_value)
                unrealized_pnl = float(pos.unrealized_pl)
                unrealized_pnl_pct = float(pos.unrealized_plpc) * 100
                
                total_value += market_value
                total_pnl += unrealized_pnl
                
                print(f"{pos.symbol:<8} {pos.qty:<8} ${float(pos.avg_cost):<9.2f} ${market_value:<11.2f} ${unrealized_pnl:<11.2f} {unrealized_pnl_pct:<7.2f}%")
            
            print("-" * 80)
            print(f"TOTAL: ${total_value:,.2f} | P&L: ${total_pnl:,.2f}")
        else:
            print("No current positions")

    def get_quote(self, symbol):
        """Get current quote for a symbol"""
        try:
            latest_trade = self.api.get_latest_trade(symbol)
            latest_quote = self.api.get_latest_quote(symbol)
            
            print(f"\n{symbol} QUOTE:")
            print(f"  Last Trade: ${latest_trade.price} (Size: {latest_trade.size})")
            print(f"  Bid: ${latest_quote.bid_price} x {latest_quote.bid_size}")
            print(f"  Ask: ${latest_quote.ask_price} x {latest_quote.ask_size}")
            print(f"  Spread: ${float(latest_quote.ask_price) - float(latest_quote.bid_price):.3f}")
            
            return float(latest_trade.price)
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return None

    def submit_order(self, symbol, qty, side, order_type='market'):
        """Submit a trading order"""
        try:
            print(f"\nSubmitting {side.upper()} order:")
            print(f"  Symbol: {symbol}")
            print(f"  Quantity: {qty}")
            print(f"  Side: {side.upper()}")
            print(f"  Type: {order_type.upper()}")
            
            # Get current quote for confirmation
            current_price = self.get_quote(symbol)
            if not current_price:
                print("Cannot get current price, order cancelled")
                return None
            
            estimated_value = qty * current_price
            print(f"  Estimated Value: ${estimated_value:,.2f}")
            
            # Confirm order
            confirm = input("\nConfirm order? (y/N): ").lower()
            if confirm != 'y':
                print("Order cancelled")
                return None
            
            # Submit order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='day'
            )
            
            print(f"\n[ORDER SUBMITTED]")
            print(f"  Order ID: {order.id}")
            print(f"  Status: {order.status}")
            print(f"  Created: {order.created_at}")
            
            return order
            
        except Exception as e:
            print(f"Error submitting order: {e}")
            return None

    def show_recent_orders(self):
        """Show recent orders"""
        orders = self.api.list_orders(status='all', limit=10)
        
        if orders:
            print("\nRECENT ORDERS:")
            print("-" * 100)
            print(f"{'TIME':<20} {'SYMBOL':<8} {'SIDE':<5} {'QTY':<8} {'TYPE':<8} {'STATUS':<10} {'FILLED QTY':<10}")
            print("-" * 100)
            
            for order in orders:
                created_time = order.created_at[:19].replace('T', ' ')
                filled_qty = order.filled_qty if order.filled_qty else '0'
                
                print(f"{created_time:<20} {order.symbol:<8} {order.side:<5} {order.qty:<8} {order.type:<8} {order.status:<10} {filled_qty:<10}")
        else:
            print("No recent orders")

    def interactive_mode(self):
        """Interactive trading mode"""
        print("\n" + "=" * 50)
        print("INTERACTIVE TRADING MODE")
        print("Commands: account, positions, quote, buy, sell, orders, quit")
        print("=" * 50)
        
        while True:
            try:
                command = input("\nHIVE> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'account':
                    self.show_account_info()
                elif command == 'positions':
                    self.show_positions()
                elif command.startswith('quote'):
                    parts = command.split()
                    if len(parts) == 2:
                        self.get_quote(parts[1].upper())
                    else:
                        symbol = input("Enter symbol: ").upper()
                        self.get_quote(symbol)
                elif command.startswith('buy'):
                    self.handle_trade_command('buy', command)
                elif command.startswith('sell'):
                    self.handle_trade_command('sell', command)
                elif command == 'orders':
                    self.show_recent_orders()
                elif command == 'help':
                    print("\nAvailable commands:")
                    print("  account     - Show account information")
                    print("  positions   - Show current positions")  
                    print("  quote SYMBOL - Get quote for symbol")
                    print("  buy SYMBOL QTY - Buy shares")
                    print("  sell SYMBOL QTY - Sell shares")
                    print("  orders      - Show recent orders")
                    print("  quit        - Exit")
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nTrading session ended.")

    def handle_trade_command(self, side, command):
        """Handle buy/sell commands"""
        parts = command.split()
        
        if len(parts) == 3:
            # Command format: buy SYMBOL QTY
            symbol = parts[1].upper()
            try:
                qty = int(parts[2])
                self.submit_order(symbol, qty, side)
            except ValueError:
                print("Invalid quantity")
        else:
            # Interactive input
            symbol = input("Enter symbol: ").upper()
            try:
                qty = int(input("Enter quantity: "))
                self.submit_order(symbol, qty, side)
            except ValueError:
                print("Invalid quantity")

def main():
    if len(sys.argv) > 1:
        # Command line mode
        trader = ManualTrader()
        command = sys.argv[1].lower()
        
        if command == 'account':
            trader.show_account_info()
        elif command == 'positions':
            trader.show_positions()
        elif command == 'orders':
            trader.show_recent_orders()
        elif command == 'quote' and len(sys.argv) > 2:
            trader.get_quote(sys.argv[2].upper())
        elif command in ['buy', 'sell'] and len(sys.argv) > 3:
            symbol = sys.argv[2].upper()
            qty = int(sys.argv[3])
            trader.submit_order(symbol, qty, command)
        else:
            print("Usage: python manual_trading.py [account|positions|orders|quote SYMBOL|buy SYMBOL QTY|sell SYMBOL QTY]")
    else:
        # Interactive mode
        trader = ManualTrader()
        trader.interactive_mode()

if __name__ == "__main__":
    main()