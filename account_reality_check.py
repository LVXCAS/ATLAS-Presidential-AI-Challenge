#!/usr/bin/env python3
"""
ACCOUNT REALITY CHECK
Check what's really blocking execution
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

def check_account_reality():
    load_dotenv()

    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL')
    )

    try:
        account = api.get_account()

        print("=== ACCOUNT REALITY CHECK ===")
        print(f"Equity: ${float(account.equity):,.2f}")
        print(f"Cash: ${float(account.cash):,.2f}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Trading Blocked: {account.trading_blocked}")
        print()

        # Check positions
        positions = api.list_positions()
        print(f"Current Positions: {len(positions)}")
        if positions:
            total_value = sum(float(pos.market_value) for pos in positions)
            print(f"Total Position Value: ${total_value:,.2f}")
        print()

        # Reality check
        equity = float(account.equity)
        buying_power = float(account.buying_power)

        print("=== TRADING READINESS ANALYSIS ===")
        if equity <= 0:
            print("BLOCKED: Negative equity prevents new positions")
            print(f"   Need ${abs(equity):,.2f} to reach positive equity")
            print("   This is the PRIMARY BLOCKER for trading")
        else:
            print("READY: Equity is positive - no equity restriction")

        if buying_power < 1000:
            print("BLOCKED: Insufficient buying power")
        else:
            print(f"AVAILABLE: Buying power ${buying_power:,.2f}")
            if equity <= 0:
                print("   BUT equity restriction overrides buying power")

        # Test small order
        if equity > 0 and buying_power > 100:
            print("\n=== ATTEMPTING TEST ORDER ===")
            try:
                # Try to submit a very small test order
                test_order = api.submit_order(
                    symbol='SPY',
                    qty=1,
                    side='buy',
                    type='market',
                    time_in_force='day',
                    extended_hours=False
                )
                print("✅ TEST ORDER SUBMITTED SUCCESSFULLY!")
                print(f"Order ID: {test_order.id}")

                # Cancel it immediately
                api.cancel_order(test_order.id)
                print("✅ Test order canceled - account can trade!")

            except Exception as e:
                print(f"❌ TEST ORDER FAILED: {e}")
                if "insufficient" in str(e).lower():
                    print("   Issue: Insufficient funds")
                elif "restricted" in str(e).lower():
                    print("   Issue: Account restrictions")
                elif "equity" in str(e).lower():
                    print("   Issue: Equity requirements")
        else:
            print("\n❌ Cannot test order - fundamental restrictions present")

    except Exception as e:
        print(f"Error checking account: {e}")

if __name__ == "__main__":
    check_account_reality()