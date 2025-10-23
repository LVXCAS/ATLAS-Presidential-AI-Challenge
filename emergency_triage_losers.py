#!/usr/bin/env python3
"""
EMERGENCY POSITION TRIAGE - Close Losing Positions
Aggressive cleanup to stop the bleeding
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv('.env.paper')

# Initialize Alpaca
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')
api = TradingClient(api_key, secret_key, paper=True)

print("\n" + "=" * 80)
print("EMERGENCY POSITION TRIAGE - CLOSE LOSING POSITIONS")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%I:%M:%S %p')}")
print()

# Get account before
account_before = api.get_account()
portfolio_before = float(account_before.portfolio_value)
equity_before = float(account_before.equity)
cash_before = float(account_before.cash)

print(f"BEFORE:")
print(f"  Portfolio Value: ${portfolio_before:,.2f}")
print(f"  Equity: ${equity_before:,.2f}")
print(f"  Cash: ${cash_before:,.2f}")
print()

# Get all positions
positions = api.get_all_positions()

print(f"Total positions: {len(positions)}")
print()

# Analyze positions
losing_positions = []
winning_positions = []
massive_positions = []  # Stock positions > $50k
worthless_positions = []  # Options worth $0
critical_losses = []  # Losses > 30%

for pos in positions:
    try:
        symbol = pos.symbol
        qty = float(pos.qty)
        market_value = float(pos.market_value) if pos.market_value else 0
        unrealized_pl = float(pos.unrealized_pl) if pos.unrealized_pl else 0
        unrealized_plpc = float(pos.unrealized_plpc) * 100 if pos.unrealized_plpc else 0
        current_price = float(pos.current_price) if pos.current_price else 0

        position_data = {
            'symbol': symbol,
            'qty': qty,
            'market_value': market_value,
            'unrealized_pl': unrealized_pl,
            'unrealized_plpc': unrealized_plpc,
            'current_price': current_price,
            'side': pos.side,
            'asset_class': pos.asset_class
        }

        # Categorize positions
        if unrealized_pl < 0:
            losing_positions.append(position_data)

        if unrealized_pl >= 0:
            winning_positions.append(position_data)

        # Check for massive stock positions (> $50k)
        if len(symbol) < 10 and abs(market_value) > 50000:  # Not an options contract
            massive_positions.append(position_data)

        # Check for worthless options
        if len(symbol) > 10 and current_price == 0:  # Options contract with $0 value
            worthless_positions.append(position_data)

        # Check for critical losses (> 30%)
        if unrealized_plpc < -30:
            critical_losses.append(position_data)

    except Exception as e:
        print(f"[WARNING] Error analyzing {pos.symbol}: {e}")

# Sort losing positions by dollar loss (worst first)
losing_positions.sort(key=lambda x: x['unrealized_pl'])
massive_positions.sort(key=lambda x: abs(x['market_value']), reverse=True)
critical_losses.sort(key=lambda x: x['unrealized_pl'])

print("=" * 80)
print("POSITION ANALYSIS")
print("=" * 80)
print(f"Losing positions: {len(losing_positions)}")
print(f"Winning positions: {len(winning_positions)}")
print(f"Massive positions (>$50k): {len(massive_positions)}")
print(f"Worthless options ($0): {len(worthless_positions)}")
print(f"Critical losses (>30%): {len(critical_losses)}")
print()

# Show top 10 losers
print("=" * 80)
print("TOP 10 WORST LOSERS")
print("=" * 80)
for i, pos in enumerate(losing_positions[:10], 1):
    print(f"{i}. {pos['symbol']}")
    print(f"   Market Value: ${pos['market_value']:,.2f}")
    print(f"   Unrealized P&L: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:.1f}%)")
    print()

# Show massive positions
if massive_positions:
    print("=" * 80)
    print("MASSIVE POSITIONS (>$50k)")
    print("=" * 80)
    for pos in massive_positions:
        print(f"Symbol: {pos['symbol']}")
        print(f"  Qty: {pos['qty']:,.0f}")
        print(f"  Market Value: ${pos['market_value']:,.2f}")
        print(f"  Unrealized P&L: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:.1f}%)")
        print()

# Decision logic - what to close?
positions_to_close = []

# 1. Close all worthless options (100% loss)
for pos in worthless_positions:
    if pos not in positions_to_close:
        positions_to_close.append(pos)

# 2. Close all critical losses (>30% down)
for pos in critical_losses:
    if pos not in positions_to_close:
        positions_to_close.append(pos)

# 3. Close massive stock positions if they're losing or barely winning
for pos in massive_positions:
    # Close if losing OR winning less than 2%
    if pos['unrealized_plpc'] < 2:
        if pos not in positions_to_close:
            positions_to_close.append(pos)

# 4. Close all losses > $500
for pos in losing_positions:
    if pos['unrealized_pl'] < -500:
        if pos not in positions_to_close:
            positions_to_close.append(pos)

# Remove duplicates
positions_to_close = list({pos['symbol']: pos for pos in positions_to_close}.values())

# Sort by loss severity
positions_to_close.sort(key=lambda x: x['unrealized_pl'])

print("=" * 80)
print(f"POSITIONS TO CLOSE: {len(positions_to_close)}")
print("=" * 80)

total_loss_to_realize = sum(pos['unrealized_pl'] for pos in positions_to_close)
print(f"Total loss to realize: ${total_loss_to_realize:,.2f}")
print()

# Confirm before proceeding
print("Positions marked for closure:")
for i, pos in enumerate(positions_to_close, 1):
    print(f"{i}. {pos['symbol']}: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:.1f}%)")

print()
response = input("Proceed with closing these positions? (YES to confirm): ")

if response != "YES":
    print("Aborted.")
    exit(0)

print()
print("=" * 80)
print("CLOSING POSITIONS...")
print("=" * 80)

# Close positions
closed_positions = []
failed_positions = []
total_realized_pl = 0

for pos in positions_to_close:
    try:
        symbol = pos['symbol']
        print(f"\nClosing {symbol}...")

        # Use close_position API for simplicity
        api.close_position(symbol)

        print(f"  [OK] Closed {symbol}")
        print(f"       Realized P&L: ${pos['unrealized_pl']:,.2f}")

        closed_positions.append(pos)
        total_realized_pl += pos['unrealized_pl']

    except Exception as e:
        print(f"  [FAILED] {symbol}: {e}")
        failed_positions.append({**pos, 'error': str(e)})

# Wait for orders to settle
import time
print("\nWaiting for orders to settle...")
time.sleep(5)

# Get account after
account_after = api.get_account()
portfolio_after = float(account_after.portfolio_value)
equity_after = float(account_after.equity)
cash_after = float(account_after.cash)

# Get remaining positions
remaining_positions = api.get_all_positions()
remaining_unrealized_pl = sum(
    float(pos.unrealized_pl) if pos.unrealized_pl else 0
    for pos in remaining_positions
)

print()
print("=" * 80)
print("EMERGENCY TRIAGE COMPLETE")
print("=" * 80)
print()
print("RESULTS:")
print(f"  Positions closed: {len(closed_positions)}/{len(positions_to_close)}")
print(f"  Failed closures: {len(failed_positions)}")
print(f"  Total realized P&L: ${total_realized_pl:,.2f}")
print()
print("BEFORE vs AFTER:")
print(f"  Portfolio Value: ${portfolio_before:,.2f} -> ${portfolio_after:,.2f} ({portfolio_after - portfolio_before:+,.2f})")
print(f"  Cash: ${cash_before:,.2f} -> ${cash_after:,.2f} ({cash_after - cash_before:+,.2f})")
print(f"  Positions: {len(positions)} -> {len(remaining_positions)} ({len(remaining_positions) - len(positions):+d})")
print(f"  Unrealized P&L: -> ${remaining_unrealized_pl:,.2f}")
print()

# Calculate metrics
stopped_bleeding = abs(total_loss_to_realize) - abs(remaining_unrealized_pl)
print(f"STOPPED BLEEDING: ${stopped_bleeding:,.2f}")
print()

# Generate report
report = {
    'timestamp': datetime.now().isoformat(),
    'before': {
        'portfolio_value': portfolio_before,
        'equity': equity_before,
        'cash': cash_before,
        'positions_count': len(positions),
        'unrealized_pl': sum(pos['unrealized_pl'] for pos in losing_positions + winning_positions)
    },
    'after': {
        'portfolio_value': portfolio_after,
        'equity': equity_after,
        'cash': cash_after,
        'positions_count': len(remaining_positions),
        'unrealized_pl': remaining_unrealized_pl
    },
    'actions': {
        'positions_closed': len(closed_positions),
        'positions_failed': len(failed_positions),
        'total_realized_pl': total_realized_pl,
        'stopped_bleeding': stopped_bleeding
    },
    'closed_positions': [
        {
            'symbol': pos['symbol'],
            'realized_pl': pos['unrealized_pl'],
            'reason': (
                'worthless' if pos['current_price'] == 0 else
                'critical_loss' if pos['unrealized_plpc'] < -30 else
                'massive_position' if abs(pos['market_value']) > 50000 else
                'large_loss'
            )
        }
        for pos in closed_positions
    ],
    'failed_positions': failed_positions
}

# Save report
report_file = f"emergency_triage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(report_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"Report saved to: {report_file}")
print()

# Show remaining losing positions
remaining_losers = [
    pos for pos in remaining_positions
    if pos.unrealized_pl and float(pos.unrealized_pl) < 0
]

if remaining_losers:
    print("=" * 80)
    print(f"REMAINING LOSING POSITIONS: {len(remaining_losers)}")
    print("=" * 80)
    remaining_losers.sort(key=lambda x: float(x.unrealized_pl))
    for i, pos in enumerate(remaining_losers[:10], 1):
        pl = float(pos.unrealized_pl)
        plpc = float(pos.unrealized_plpc) * 100
        print(f"{i}. {pos.symbol}: ${pl:,.2f} ({plpc:.1f}%)")
    print()

print("=" * 80)
print("TRIAGE COMPLETE - BLEEDING STOPPED!")
print("=" * 80)
