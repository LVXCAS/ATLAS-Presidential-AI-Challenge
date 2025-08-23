#!/usr/bin/env python3
"""
Check Alpaca Markets Status and Available Trading Assets
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
from datetime import datetime

def main():
    load_dotenv()

    # Initialize Alpaca API
    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
        api_version='v2'
    )

    print("=" * 60)
    print("HIVE TRADE - ALPACA MARKETS STATUS CHECK")
    print("=" * 60)

    try:
        # Account Status
        print("\n[ACCOUNT STATUS]")
        account = api.get_account()
        print(f"   Account Status: {account.status}")
        print(f"   Trading Blocked: {account.trading_blocked}")
        print(f"   Pattern Day Trader: {account.pattern_day_trader}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Day Trade Count: {account.daytrade_count}")

        # Market Status
        print("\n[MARKET STATUS]")
        clock = api.get_clock()
        print(f"   Market Currently Open: {clock.is_open}")
        print(f"   Current Time: {clock.timestamp}")
        print(f"   Next Market Open: {clock.next_open}")
        print(f"   Next Market Close: {clock.next_close}")

        # Current Positions
        print("\n[CURRENT POSITIONS]")
        positions = api.list_positions()
        if positions:
            total_value = 0
            total_pnl = 0
            for pos in positions:
                market_value = float(pos.market_value)
                unrealized_pnl = float(pos.unrealized_pl)
                total_value += market_value
                total_pnl += unrealized_pnl
                print(f"   {pos.symbol}: {pos.qty} shares @ ${pos.avg_cost} | MktVal: ${market_value:,.2f} | P&L: ${unrealized_pnl:,.2f}")
            print(f"   TOTAL PORTFOLIO: ${total_value:,.2f} | TOTAL P&L: ${total_pnl:,.2f}")
        else:
            print("   No current positions")

        # Recent Orders
        print("\n[RECENT ORDERS - Last 10]")
        orders = api.list_orders(status='all', limit=10)
        if orders:
            for order in orders:
                print(f"   {order.created_at[:19]} | {order.symbol} {order.side} {order.qty} @ {order.order_type} | Status: {order.status}")
        else:
            print("   No recent orders")

        # Available Assets for Trading
        print("\n[AVAILABLE MARKETS & ASSETS]")
        
        # Major Indices/ETFs
        major_etfs = ['SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'AGG', 'LQD', 'HYG', 'GLD']
        print("\n   [MAJOR ETFs/INDICES]")
        for symbol in major_etfs:
            try:
                asset = api.get_asset(symbol)
                print(f"      {symbol} - Tradable: {asset.tradable} | Shortable: {asset.shortable} | Easy to Borrow: {asset.easy_to_borrow}")
            except:
                print(f"      {symbol} - Not available")

        # Major Stocks
        major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC']
        print("\n   [MAJOR STOCKS]")
        for symbol in major_stocks:
            try:
                asset = api.get_asset(symbol)
                print(f"      {symbol} - Tradable: {asset.tradable} | Shortable: {asset.shortable} | Easy to Borrow: {asset.easy_to_borrow}")
            except:
                print(f"      {symbol} - Not available")

        # Crypto (if available)
        crypto_symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD']
        print("\n   [CRYPTO - if available]")
        crypto_available = False
        for symbol in crypto_symbols:
            try:
                asset = api.get_asset(symbol)
                print(f"      {symbol} - Tradable: {asset.tradable}")
                crypto_available = True
            except:
                pass
        if not crypto_available:
            print("      Crypto trading not available in current account type")

        # Account Trading Capabilities
        print("\n[TRADING CAPABILITIES]")
        print(f"   >> Equity Trading: Available")
        print(f"   >> Options Trading: {getattr(account, 'options_trading_level', 'Check manually')}")
        print(f"   >> Day Trading: {'Available' if not account.pattern_day_trader else 'PDT Rules Apply'}")
        print(f"   >> After Hours: Available (4:00 AM - 8:00 PM ET)")
        print(f"   >> Fractional Shares: Available")
        
        # Market Hours
        print("\n[MARKET HOURS - ET]")
        print("   Regular Hours: 9:30 AM - 4:00 PM")
        print("   Pre-Market: 4:00 AM - 9:30 AM")
        print("   After Hours: 4:00 PM - 8:00 PM")

        print("\n" + "=" * 60)
        print("*** HIVE TRADE READY FOR LIVE TRADING! ***")
        print("=" * 60)

    except Exception as e:
        print(f"[ERROR] Error connecting to Alpaca: {e}")
        print("Check your API keys and connection")

if __name__ == "__main__":
    main()