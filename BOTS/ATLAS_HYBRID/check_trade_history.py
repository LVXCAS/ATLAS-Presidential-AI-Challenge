"""
Check trade history and timing for profit analysis
"""
import os, sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(Path(__file__).parent.parent.parent / '.env')

sys.path.append(str(Path(__file__).parent))
from adapters.oanda_adapter import OandaAdapter

oanda = OandaAdapter()

print("\n" + "="*80)
print("TRADE HISTORY & TIMING ANALYSIS")
print("="*80 + "\n")

# Get recent transactions
response = oanda._request('GET', f'/v3/accounts/{oanda.account_id}/transactions',
                         params={'count': 100})

if response and 'transactions' in response:
    # Filter for order fills
    fills = [tx for tx in response['transactions'] if tx.get('type') == 'ORDER_FILL']

    if fills:
        print(f"Found {len(fills)} filled orders\n")

        # Group by instrument
        trades_by_pair = {}
        for fill in fills:
            instrument = fill.get('instrument')
            if instrument not in trades_by_pair:
                trades_by_pair[instrument] = []
            trades_by_pair[instrument].append(fill)

        # Analyze each pair
        for instrument, pair_trades in trades_by_pair.items():
            print(f"\n{instrument}:")
            print("-" * 60)

            total_pl = 0
            earliest = None
            latest = None

            for trade in pair_trades[-10:]:  # Last 10 trades per pair
                trade_id = trade.get('id')
                units = float(trade.get('units', 0))
                price = float(trade.get('price', 0))
                pl = float(trade.get('pl', 0))
                time_str = trade.get('time', '')

                # Parse timestamp
                try:
                    time_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    if earliest is None or time_obj < earliest:
                        earliest = time_obj
                    if latest is None or time_obj > latest:
                        latest = time_obj
                except:
                    time_obj = None

                total_pl += pl

                direction = "BUY" if units > 0 else "SELL"
                lots = abs(units) / 100000

                print(f"  #{trade_id} | {time_str[:19]} | {direction:4s} {lots:5.1f} lots @ {price:.5f} | P/L: ${pl:+8.2f}")

            print(f"\n  Total P/L for {instrument}: ${total_pl:+,.2f}")

            if earliest and latest:
                duration = latest - earliest
                hours = duration.total_seconds() / 3600
                days = hours / 24

                print(f"  First trade: {earliest.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"  Last trade:  {latest.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"  Duration: {days:.2f} days ({hours:.1f} hours)")

                if total_pl != 0:
                    hourly_rate = total_pl / max(hours, 0.01)
                    daily_rate = total_pl / max(days, 0.01)
                    print(f"  Hourly Rate: ${hourly_rate:+.2f}/hr")
                    print(f"  Daily Rate: ${daily_rate:+.2f}/day")

        # Overall stats
        print("\n" + "="*80)
        print("OVERALL STATISTICS")
        print("="*80)

        total_pl_all = sum(float(f.get('pl', 0)) for f in fills)
        all_times = [datetime.fromisoformat(f.get('time', '').replace('Z', '+00:00'))
                    for f in fills if f.get('time')]

        if all_times:
            first = min(all_times)
            last = max(all_times)
            total_duration = last - first
            total_hours = total_duration.total_seconds() / 3600
            total_days = total_hours / 24

            print(f"Total Trades: {len(fills)}")
            print(f"Total P/L: ${total_pl_all:+,.2f}")
            print(f"First Trade: {first.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"Last Trade: {last.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"Total Duration: {total_days:.2f} days ({total_hours:.1f} hours)")

            if total_pl_all != 0:
                print(f"\nPerformance Metrics:")
                print(f"  Hourly Rate: ${total_pl_all/max(total_hours, 0.01):+.2f}/hr")
                print(f"  Daily Rate: ${total_pl_all/max(total_days, 0.01):+.2f}/day")
                print(f"  Average per Trade: ${total_pl_all/len(fills):+.2f}")

                # Calculate ROI
                starting_balance = 200000  # E8 starting balance
                current_balance = starting_balance + total_pl_all
                roi = (total_pl_all / starting_balance) * 100

                print(f"\n  Starting Balance: ${starting_balance:,.2f}")
                print(f"  Current Balance: ${current_balance:,.2f}")
                print(f"  ROI: {roi:+.3f}%")

                if total_days > 0:
                    monthly_roi = (roi / total_days) * 30
                    print(f"  Projected Monthly ROI: {monthly_roi:+.2f}%")
    else:
        print("No filled orders found in recent history")
else:
    print("Could not fetch transaction history")

print("\n" + "="*80 + "\n")
