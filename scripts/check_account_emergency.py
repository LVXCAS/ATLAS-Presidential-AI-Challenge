"""Emergency account check"""
import sys
sys.path.insert(0, 'BOTS')

from E8_TRADELOCKER_ADAPTER import E8TradeLockerAdapter

print("=" * 70)
print("EMERGENCY ACCOUNT STATUS CHECK")
print("=" * 70)

try:
    tl = E8TradeLockerAdapter()
    summary = tl.get_account_summary()

    balance = summary['balance']
    equity = summary['NAV']
    unrealized = summary['unrealizedPL']

    print(f"\nBalance: ${balance:,.2f}")
    print(f"Equity: ${equity:,.2f}")
    print(f"Unrealized P/L: ${unrealized:,.2f}")

    # Check DD
    peak = 208163.0
    current_dd = (peak - equity) / peak
    dd_percent = current_dd * 100
    max_dd = 6.0

    print(f"\nPeak Balance: ${peak:,.2f}")
    print(f"Current DD: {dd_percent:.2f}%")
    print(f"Max DD Allowed: {max_dd:.2f}%")

    if dd_percent >= max_dd:
        print("\n" + "!" * 70)
        print("CHALLENGE FAILED - DD EXCEEDED 6%")
        print("!" * 70)
    elif dd_percent >= 5.5:
        print("\n" + "!" * 70)
        print("CRITICAL - VERY CLOSE TO DD VIOLATION")
        print("!" * 70)
    elif dd_percent >= 4.5:
        print("\nWARNING - Approaching DD limit")
    else:
        print("\nStatus: Within DD limits")

    # Get positions
    print("\n" + "=" * 70)
    print("OPEN POSITIONS:")
    print("=" * 70)

    positions = tl.get_open_positions()
    if positions:
        for pos in positions:
            symbol = pos.get('instrument', pos.get('symbol', 'UNKNOWN'))
            side = pos.get('side', 'UNKNOWN')
            unrealized_pl = float(pos.get('unrealizedPL', 0))
            print(f"\n{symbol} {side}: ${unrealized_pl:,.2f}")
    else:
        print("\nNo open positions")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
