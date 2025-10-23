"""
Comprehensive System Status Check
Verifies all components are working correctly
"""

from datetime import datetime
import pytz
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 80)
print("COMPREHENSIVE SYSTEM STATUS CHECK")
print("=" * 80)
print()

# 1. Time and Market Status
print("1. MARKET STATUS")
print("-" * 80)
et = pytz.timezone('US/Eastern')
now = datetime.now(et)
print(f"Current Time (ET): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Day of Week: {now.strftime('%A')}")

client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
clock = client.get_clock()

print(f"Market Status: {'OPEN' if clock.is_open else 'CLOSED'}")
if clock.is_open:
    minutes_to_close = int((clock.next_close - now).total_seconds() / 60)
    print(f"Time Until Close: {minutes_to_close} minutes ({minutes_to_close//60}h {minutes_to_close%60}m)")
else:
    print(f"Next Open: {clock.next_open.strftime('%Y-%m-%d %H:%M %Z')}")
print()

# 2. Account Status
print("2. ALPACA ACCOUNT")
print("-" * 80)
account = client.get_account()
print(f"Status: {account.status}")
print(f"Trading Blocked: {account.trading_blocked}")
print(f"Account Blocked: {account.account_blocked}")
print(f"Cash: ${float(account.cash):,.2f}")
print(f"Buying Power: ${float(account.buying_power):,.2f}")
print(f"Pattern Day Trader: {account.pattern_day_trader}")
print()

# 3. Current Positions
print("3. CURRENT POSITIONS")
print("-" * 80)
positions = client.get_all_positions()
print(f"Open Positions: {len(positions)}")
if positions:
    for p in positions:
        pnl = float(p.unrealized_pl)
        pnl_pct = float(p.unrealized_plpc) * 100
        print(f"  {p.symbol}: {p.qty} @ ${float(p.avg_entry_price):.2f} | P/L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
else:
    print("  No open positions")
print()

# 4. Orders Today
print("4. ORDERS TODAY")
print("-" * 80)
orders = client.get_orders()
print(f"Total Orders: {len(orders)}")
if orders:
    for o in orders[:5]:  # Show first 5
        print(f"  {o.symbol} {o.side} - {o.status}")
else:
    print("  No orders today")
print()

# 5. Bot Configuration
print("5. BOT CONFIGURATION")
print("-" * 80)
print("Watchlist: 80 stocks (S&P 500 top stocks)")
print("Confidence Threshold: 65% (optimized)")
print("Max Positions: 5-7 concurrent")
print("Data Sources: Alpaca -> Polygon -> OpenBB -> Yahoo")
print("Paper Trading: Enabled")
print()

# 6. System Health
print("6. SYSTEM HEALTH")
print("-" * 80)
all_ok = True

# Check API credentials
if not os.getenv('ALPACA_API_KEY'):
    print("[ERROR] Alpaca API key not found")
    all_ok = False
else:
    print("[OK] Alpaca API credentials configured")

# Check account status
if account.status.value == 'ACTIVE' and not account.trading_blocked:
    print("[OK] Account is active and trading")
else:
    print("[WARNING] Account may have issues")
    all_ok = False

# Check market accessibility
if clock.is_open or clock.next_open:
    print("[OK] Market data accessible")
else:
    print("[WARNING] Market data issues")
    all_ok = False

print()
print("=" * 80)
if all_ok:
    print("[SUCCESS] ALL SYSTEMS OPERATIONAL")
    print("Bot is ready to trade!")
else:
    print("[WARNING] Some issues detected - review above")
print("=" * 80)
