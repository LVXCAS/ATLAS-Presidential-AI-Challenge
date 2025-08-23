#!/usr/bin/env python3
"""
Hive Trade - Crypto Trading System (24/7)
"""

import alpaca_trade_api as tradeapi
import os
import sys
from dotenv import load_dotenv
import time

class CryptoTrader:
    def __init__(self):
        load_dotenv()
        
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )
        
        self.crypto_assets = ['BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD', 'SOLUSD']
        
    def get_crypto_quote(self, symbol):
        """Get crypto quote using regular quote method"""
        try:
            # Use regular latest trade method for crypto
            trade = self.api.get_latest_trade(symbol)
            return {
                'symbol': symbol,
                'price': float(trade.price),
                'size': float(trade.size),
                'timestamp': trade.timestamp
            }
        except Exception as e:
            print(f"Error getting {symbol} quote: {e}")
            return None
    
    def place_crypto_order(self, symbol, side, amount_usd):
        """Place crypto order by dollar amount"""
        try:
            print(f"\n[CRYPTO ORDER] {side.upper()} ${amount_usd} of {symbol}")
            
            # Get current price for confirmation
            quote = self.get_crypto_quote(symbol)
            if quote:
                print(f"  Current Price: ${quote['price']}")
                estimated_qty = amount_usd / quote['price']
                print(f"  Estimated Quantity: {estimated_qty:.6f}")
            
            # Place order using notional (dollar) amount for crypto
            order = self.api.submit_order(
                symbol=symbol,
                notional=amount_usd,  # Dollar amount instead of qty
                side=side,
                type='market',
                time_in_force='gtc'  # Good til cancelled for crypto
            )
            
            print(f"  [ORDER PLACED] ID: {order.id}")
            print(f"  Status: {order.status}")
            print(f"  Time: {order.created_at}")
            return order
            
        except Exception as e:
            print(f"  [CRYPTO ORDER ERROR] {e}")
            return None
    
    def show_crypto_prices(self):
        """Show live crypto prices"""
        print("\n[LIVE CRYPTO PRICES - 24/7]")
        print("-" * 40)
        
        for symbol in self.crypto_assets:
            quote = self.get_crypto_quote(symbol)
            if quote:
                crypto_name = symbol.replace('USD', '')
                print(f"{crypto_name:>6}: ${quote['price']:>10,.2f}")
    
    def start_crypto_trading(self, symbol, amount):
        """Start crypto trading with specified amount"""
        print(f"\n*** STARTING CRYPTO TRADING: {symbol} ***")
        
        # Place initial buy order
        order = self.place_crypto_order(symbol, 'buy', amount)
        
        if order:
            print(f"\nCrypto order placed! Trading {symbol} 24/7")
            print("This will execute even when stock markets are closed!")
            return True
        return False
    
    def monitor_crypto(self):
        """Monitor crypto positions and prices"""
        print("\n[CRYPTO MONITORING - 24/7]")
        
        # Check positions
        positions = self.api.list_positions()
        crypto_positions = [p for p in positions if p.symbol in self.crypto_assets]
        
        if crypto_positions:
            print("Current Crypto Positions:")
            for pos in crypto_positions:
                pnl = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100
                pnl_color = "PROFIT" if pnl >= 0 else "LOSS"
                
                print(f"  {pos.symbol}: {pos.qty} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) - {pnl_color}")
        else:
            print("No crypto positions currently held")
        
        # Show live prices
        self.show_crypto_prices()

def main():
    if len(sys.argv) < 2:
        trader = CryptoTrader()
        print("HIVE TRADE - CRYPTO TRADING SYSTEM (24/7)")
        print("=" * 50)
        trader.show_crypto_prices()
        trader.monitor_crypto()
        
        print("\n[CRYPTO TRADING COMMANDS]")
        print("  python crypto_trader.py buy BTCUSD 100   # Buy $100 BTC")
        print("  python crypto_trader.py buy ETHUSD 50    # Buy $50 ETH") 
        print("  python crypto_trader.py sell BTCUSD 25   # Sell $25 BTC")
        print("  python crypto_trader.py monitor          # Monitor positions")
        
    else:
        trader = CryptoTrader()
        command = sys.argv[1].lower()
        
        if command == 'monitor':
            trader.monitor_crypto()
            
        elif command in ['buy', 'sell'] and len(sys.argv) >= 4:
            symbol = sys.argv[2].upper()
            amount = float(sys.argv[3])
            
            if symbol in trader.crypto_assets:
                trader.place_crypto_order(symbol, command, amount)
            else:
                print(f"Error: {symbol} not available. Available: {trader.crypto_assets}")
        else:
            print("Usage: python crypto_trader.py [buy|sell] [SYMBOL] [AMOUNT_USD]")
            print("       python crypto_trader.py monitor")

if __name__ == "__main__":
    main()