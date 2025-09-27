#!/usr/bin/env python3
"""
COMPLETE ACCOUNT RESET
Nuclear option - liquidate everything to restore trading capability
This will reset your account to a clean state for Intel-puts-style trading
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import time
from datetime import datetime

class CompleteAccountReset:
    """Complete account reset system"""

    def __init__(self):
        load_dotenv()

        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL')
        )

    def check_current_status(self):
        """Check current account status"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            orders = self.api.list_orders(status='open')

            print("=== CURRENT ACCOUNT STATUS ===")
            print(f"Equity: ${float(account.equity):,.2f}")
            print(f"Cash: ${float(account.cash):,.2f}")
            print(f"Buying Power: ${float(account.buying_power):,.2f}")
            print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"Open Positions: {len(positions)}")
            print(f"Open Orders: {len(orders)}")

            # Calculate total position value
            total_position_value = sum(float(pos.market_value) for pos in positions)
            print(f"Total Position Value: ${total_position_value:,.2f}")
            print()

            return {
                'account': account,
                'positions': positions,
                'orders': orders,
                'total_position_value': total_position_value
            }

        except Exception as e:
            print(f"Error checking status: {e}")
            return None

    def cancel_all_orders(self):
        """Cancel all pending orders"""
        print("=== CANCELING ALL ORDERS ===")
        try:
            orders = self.api.list_orders(status='open')

            if not orders:
                print("No open orders to cancel")
                return True

            print(f"Canceling {len(orders)} orders...")

            for order in orders:
                try:
                    self.api.cancel_order(order.id)
                    print(f"Canceled: {order.symbol} {order.side} {order.qty}")
                except Exception as e:
                    print(f"Failed to cancel {order.id}: {e}")

            time.sleep(3)
            print("All orders canceled")
            return True

        except Exception as e:
            print(f"Error canceling orders: {e}")
            return False

    def liquidate_all_positions(self):
        """Liquidate every single position"""
        print("=== LIQUIDATING ALL POSITIONS ===")
        try:
            positions = self.api.list_positions()

            if not positions:
                print("No positions to liquidate")
                return True

            print(f"Liquidating {len(positions)} positions...")
            liquidation_orders = []

            for pos in positions:
                symbol = pos.symbol
                qty = int(float(pos.qty))
                market_value = float(pos.market_value)

                print(f"\nLiquidating: {symbol}")
                print(f"  Quantity: {qty}")
                print(f"  Market Value: ${market_value:,.2f}")

                # Skip worthless positions
                if abs(market_value) < 1:
                    print(f"  [SKIP] Position worth less than $1")
                    continue

                try:
                    if qty > 0:
                        # Long position - sell all
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=abs(qty),
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                    elif qty < 0:
                        # Short position - buy to cover
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=abs(qty),
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                    else:
                        continue

                    liquidation_orders.append({
                        'symbol': symbol,
                        'order_id': order.id,
                        'expected_cash': abs(market_value)
                    })

                    print(f"  [SUCCESS] Liquidation order: {order.id}")

                except Exception as e:
                    print(f"  [ERROR] Failed to liquidate {symbol}: {e}")

                    # Try with reduced quantity if insufficient qty error
                    if "insufficient" in str(e).lower():
                        try:
                            reduced_qty = int(abs(qty) * 0.95)
                            if reduced_qty > 0:
                                order = self.api.submit_order(
                                    symbol=symbol,
                                    qty=reduced_qty,
                                    side='sell' if qty > 0 else 'buy',
                                    type='market',
                                    time_in_force='day'
                                )
                                print(f"  [RETRY SUCCESS] Reduced quantity: {order.id}")
                        except Exception as retry_e:
                            print(f"  [RETRY FAILED] {retry_e}")

                # Brief pause between orders
                time.sleep(1)

            print(f"\nSubmitted {len(liquidation_orders)} liquidation orders")
            expected_total_cash = sum(order['expected_cash'] for order in liquidation_orders)
            print(f"Expected cash recovery: ${expected_total_cash:,.2f}")

            return liquidation_orders

        except Exception as e:
            print(f"Error liquidating positions: {e}")
            return []

    def wait_for_settlement(self, wait_minutes=5):
        """Wait for liquidation orders to settle"""
        print(f"\n=== WAITING FOR SETTLEMENT ({wait_minutes} minutes) ===")

        for minute in range(wait_minutes):
            time.sleep(60)

            try:
                account = self.api.get_account()
                positions = self.api.list_positions()

                print(f"Minute {minute + 1}: Equity ${float(account.equity):,.2f}, Positions: {len(positions)}")

                if float(account.equity) > 0:
                    print("POSITIVE EQUITY ACHIEVED!")
                    break

            except Exception as e:
                print(f"Error checking settlement: {e}")

    def verify_reset_success(self):
        """Verify account has been successfully reset"""
        print("\n=== VERIFYING RESET SUCCESS ===")

        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            orders = self.api.list_orders(status='open')

            equity = float(account.equity)
            cash = float(account.cash)
            buying_power = float(account.buying_power)

            print("FINAL ACCOUNT STATUS:")
            print(f"Equity: ${equity:,.2f}")
            print(f"Cash: ${cash:,.2f}")
            print(f"Buying Power: ${buying_power:,.2f}")
            print(f"Remaining Positions: {len(positions)}")
            print(f"Open Orders: {len(orders)}")

            # Determine reset success
            reset_successful = equity > 0 and len(positions) == 0

            if reset_successful:
                print("\n🎉 ACCOUNT RESET SUCCESSFUL!")
                print("✅ Positive equity achieved")
                print("✅ All positions liquidated")
                print("✅ Ready for Intel-puts-style trading")

                # Test trading capability
                can_trade = equity > 0 and buying_power > 1000
                if can_trade:
                    print("✅ TRADING CAPABILITY RESTORED")
                    print("✅ Your autonomous systems can now execute trades")
                else:
                    print("⚠️  Trading capability needs verification")

            else:
                print("\n⚠️  RESET INCOMPLETE")
                if equity <= 0:
                    print(f"❌ Equity still negative: ${equity:,.2f}")
                if len(positions) > 0:
                    print(f"❌ {len(positions)} positions remain:")
                    for pos in positions:
                        print(f"   {pos.symbol}: ${float(pos.market_value):,.2f}")

            return reset_successful

        except Exception as e:
            print(f"Error verifying reset: {e}")
            return False

    def execute_complete_reset(self):
        """Execute complete account reset sequence"""
        print("COMPLETE ACCOUNT RESET")
        print("=" * 60)
        print("WARNING: This will liquidate ALL positions")
        print("Goal: Restore positive equity for Intel-puts-style trading")
        print("=" * 60)

        # Step 1: Check current status
        status = self.check_current_status()
        if not status:
            print("Failed to check account status")
            return False

        # Step 2: Cancel all orders
        if not self.cancel_all_orders():
            print("Failed to cancel orders")
            return False

        # Step 3: Liquidate all positions
        liquidation_orders = self.liquidate_all_positions()
        if not liquidation_orders:
            print("No liquidation orders submitted")
            # Could still be success if no positions

        # Step 4: Wait for settlement
        self.wait_for_settlement(wait_minutes=3)

        # Step 5: Verify reset success
        success = self.verify_reset_success()

        if success:
            print("\n" + "=" * 60)
            print("ACCOUNT RESET COMPLETE")
            print("Your autonomous trading systems can now execute trades!")
            print("Ready to deploy Intel-puts-style strategies")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("RESET PARTIALLY COMPLETE")
            print("Some manual intervention may be required")
            print("=" * 60)

        return success

def main():
    """Execute complete account reset"""
    resetter = CompleteAccountReset()
    resetter.execute_complete_reset()

if __name__ == "__main__":
    main()