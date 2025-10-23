#!/usr/bin/env python3
"""
AUTO CLEANUP AT MARKET OPEN
Runs automatically when market opens to close losing positions
"""

import os
import json
from datetime import datetime, time
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
import time as time_module

load_dotenv('.env.paper')

# Initialize Alpaca
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')
api = TradingClient(api_key, secret_key, paper=True)

def is_market_open():
    """Check if market is currently open"""
    try:
        clock = api.get_clock()
        return clock.is_open
    except:
        return False

def wait_for_market_open():
    """Wait until market opens"""
    print("Waiting for market to open...")
    while not is_market_open():
        time_module.sleep(60)  # Check every minute
    print("Market is open! Starting cleanup...")

def close_position_safe(symbol):
    """Close a position with error handling"""
    try:
        api.close_position(symbol)
        print(f"  [OK] Closed {symbol}")
        return True
    except Exception as e:
        print(f"  [FAILED] {symbol}: {e}")
        return False

def main():
    print("\n" + "=" * 80)
    print("AUTO CLEANUP - MARKET OPEN")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
    print()

    # Check if market is open
    if not is_market_open():
        print("Market is currently CLOSED")
        print("Waiting for market open to execute cleanup...")
        wait_for_market_open()
    else:
        print("Market is OPEN - proceeding with cleanup")

    print()
    print("=" * 80)
    print("PHASE 1: CLOSE WORTHLESS OPTIONS")
    print("=" * 80)

    # Get current positions
    positions = api.get_all_positions()

    # Find worthless options (current price = $0)
    worthless_options = []
    for pos in positions:
        try:
            if len(pos.symbol) > 10:  # Options contract
                current_price = float(pos.current_price) if pos.current_price else 0
                if current_price == 0 or current_price < 0.01:
                    worthless_options.append({
                        'symbol': pos.symbol,
                        'qty': float(pos.qty),
                        'unrealized_pl': float(pos.unrealized_pl) if pos.unrealized_pl else 0
                    })
        except Exception as e:
            print(f"[WARNING] Error analyzing {pos.symbol}: {e}")

    print(f"\nFound {len(worthless_options)} worthless options to close")

    closed_worthless = 0
    realized_pl_worthless = 0

    for opt in worthless_options:
        print(f"\nClosing {opt['symbol']}...")
        if close_position_safe(opt['symbol']):
            closed_worthless += 1
            realized_pl_worthless += opt['unrealized_pl']
            time_module.sleep(1)  # Rate limiting

    print(f"\n[PHASE 1 COMPLETE]")
    print(f"  Closed: {closed_worthless}/{len(worthless_options)}")
    print(f"  Realized P&L: ${realized_pl_worthless:,.2f}")

    print()
    print("=" * 80)
    print("PHASE 2: CLOSE CRITICAL LOSSES (>30%)")
    print("=" * 80)

    # Refresh positions
    positions = api.get_all_positions()

    critical_losses = []
    for pos in positions:
        try:
            unrealized_plpc = float(pos.unrealized_plpc) * 100 if pos.unrealized_plpc else 0
            unrealized_pl = float(pos.unrealized_pl) if pos.unrealized_pl else 0

            if unrealized_plpc < -30:
                critical_losses.append({
                    'symbol': pos.symbol,
                    'unrealized_pl': unrealized_pl,
                    'unrealized_plpc': unrealized_plpc
                })
        except Exception as e:
            print(f"[WARNING] Error analyzing {pos.symbol}: {e}")

    print(f"\nFound {len(critical_losses)} positions with >30% loss")

    closed_critical = 0
    realized_pl_critical = 0

    for pos in critical_losses:
        print(f"\nClosing {pos['symbol']} ({pos['unrealized_plpc']:.1f}% loss)...")
        if close_position_safe(pos['symbol']):
            closed_critical += 1
            realized_pl_critical += pos['unrealized_pl']
            time_module.sleep(1)

    print(f"\n[PHASE 2 COMPLETE]")
    print(f"  Closed: {closed_critical}/{len(critical_losses)}")
    print(f"  Realized P&L: ${realized_pl_critical:,.2f}")

    print()
    print("=" * 80)
    print("PHASE 3: CLOSE LARGE LOSSES (>$500)")
    print("=" * 80)

    # Refresh positions
    positions = api.get_all_positions()

    large_losses = []
    for pos in positions:
        try:
            unrealized_pl = float(pos.unrealized_pl) if pos.unrealized_pl else 0

            if unrealized_pl < -500:
                large_losses.append({
                    'symbol': pos.symbol,
                    'unrealized_pl': unrealized_pl
                })
        except Exception as e:
            print(f"[WARNING] Error analyzing {pos.symbol}: {e}")

    # Sort by loss amount
    large_losses.sort(key=lambda x: x['unrealized_pl'])

    print(f"\nFound {len(large_losses)} positions with >$500 loss")

    closed_large = 0
    realized_pl_large = 0

    for pos in large_losses:
        print(f"\nClosing {pos['symbol']} (${pos['unrealized_pl']:.2f} loss)...")
        if close_position_safe(pos['symbol']):
            closed_large += 1
            realized_pl_large += pos['unrealized_pl']
            time_module.sleep(1)

    print(f"\n[PHASE 3 COMPLETE]")
    print(f"  Closed: {closed_large}/{len(large_losses)}")
    print(f"  Realized P&L: ${realized_pl_large:,.2f}")

    # Wait for orders to settle
    print("\nWaiting for orders to settle...")
    time_module.sleep(10)

    print()
    print("=" * 80)
    print("CLEANUP COMPLETE - FINAL STATUS")
    print("=" * 80)

    # Get final account status
    account = api.get_account()
    positions = api.get_all_positions()

    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)
    buying_power = float(account.buying_power)

    print(f"\nAccount Status:")
    print(f"  Portfolio Value: ${portfolio_value:,.2f}")
    print(f"  Cash: ${cash:,.2f}")
    print(f"  Buying Power: ${buying_power:,.2f}")
    print(f"  Open Positions: {len(positions)}")

    # Calculate remaining P&L
    total_unrealized_pl = sum(
        float(pos.unrealized_pl) if pos.unrealized_pl else 0
        for pos in positions
    )

    losing_positions = sum(
        1 for pos in positions
        if pos.unrealized_pl and float(pos.unrealized_pl) < 0
    )

    winning_positions = sum(
        1 for pos in positions
        if pos.unrealized_pl and float(pos.unrealized_pl) > 0
    )

    print(f"  Unrealized P&L: ${total_unrealized_pl:,.2f}")
    print(f"  Losing Positions: {losing_positions}")
    print(f"  Winning Positions: {winning_positions}")
    if len(positions) > 0:
        win_rate = (winning_positions / len(positions)) * 100
        print(f"  Win Rate: {win_rate:.1f}%")

    print()
    print("Summary:")
    print(f"  Total Positions Closed: {closed_worthless + closed_critical + closed_large}")
    print(f"  Total Realized P&L: ${realized_pl_worthless + realized_pl_critical + realized_pl_large:,.2f}")
    print()

    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'cleanup_phases': {
            'worthless_options': {
                'found': len(worthless_options),
                'closed': closed_worthless,
                'realized_pl': realized_pl_worthless
            },
            'critical_losses': {
                'found': len(critical_losses),
                'closed': closed_critical,
                'realized_pl': realized_pl_critical
            },
            'large_losses': {
                'found': len(large_losses),
                'closed': closed_large,
                'realized_pl': realized_pl_large
            }
        },
        'final_status': {
            'portfolio_value': portfolio_value,
            'cash': cash,
            'buying_power': buying_power,
            'positions_count': len(positions),
            'unrealized_pl': total_unrealized_pl,
            'losing_positions': losing_positions,
            'winning_positions': winning_positions,
            'win_rate': (winning_positions / len(positions) * 100) if len(positions) > 0 else 0
        }
    }

    # Save report
    report_file = f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {report_file}")
    print()
    print("=" * 80)
    print("CLEANUP COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
