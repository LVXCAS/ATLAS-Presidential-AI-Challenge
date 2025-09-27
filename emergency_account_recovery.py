#!/usr/bin/env python3
"""
EMERGENCY ACCOUNT RECOVERY SYSTEM
Immediately liquidate large positions to restore positive cash balance
Priority: Get account functional for Intel-puts-style trades
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import logging

class EmergencyAccountRecovery:
    """Emergency system to restore account to positive cash balance"""

    def __init__(self):
        load_dotenv()

        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL')
        )

        # PRESERVE these winning options at all costs
        self.preserve_winners = {
            'LYFT250926P00021000',   # +47.1% winner
            'RIVN250926P00014000',   # +93.2% winner
            'SNAP250926C00009000',   # Options position
            'SNAP250926P00008000',   # +44.7% winner
        }

        # Target: Get cash balance positive (need ~$1.6M)
        self.target_cash_recovery = 2000000  # $2M to be safe

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - EMERGENCY - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_emergency_status(self):
        """Get critical account metrics"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()

            return {
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'portfolio_value': float(account.portfolio_value),
                'positions': positions,
                'critical': float(account.cash) < 0
            }
        except Exception as e:
            self.logger.error(f"Error getting emergency status: {e}")
            return None

    def identify_emergency_liquidations(self, positions):
        """Identify large positions to liquidate for immediate cash recovery"""
        emergency_list = []

        print("EMERGENCY LIQUIDATION ANALYSIS")
        print("=" * 60)

        for pos in positions:
            symbol = pos.symbol
            qty = int(pos.qty)
            market_value = abs(float(pos.market_value))
            unrealized_pl = float(pos.unrealized_pl)

            # Skip our winning options - NEVER liquidate these
            if symbol in self.preserve_winners:
                print(f"[PRESERVE] {symbol} - Winning options position (keeping)")
                continue

            # Emergency liquidation criteria - focus on large positions
            should_liquidate = False
            reasons = []
            priority = 0

            # Priority 1: Largest positions for immediate cash
            if market_value > 100000:  # >$100K positions
                should_liquidate = True
                reasons.append(f"Large position: ${market_value:,.0f}")
                priority += 100

            # Priority 2: Any position if we need emergency cash
            elif market_value > 10000:  # >$10K positions
                should_liquidate = True
                reasons.append(f"Medium position: ${market_value:,.0f}")
                priority += 50

            # Priority 3: Even small positions if desperate
            elif market_value > 1000:  # >$1K positions
                should_liquidate = True
                reasons.append(f"Small position: ${market_value:,.0f}")
                priority += 10

            if should_liquidate:
                emergency_list.append({
                    'symbol': symbol,
                    'qty': qty,
                    'market_value': market_value,
                    'unrealized_pl': unrealized_pl,
                    'reasons': reasons,
                    'priority': priority
                })

        # Sort by priority (liquidate largest first)
        emergency_list.sort(key=lambda x: x['priority'], reverse=True)
        return emergency_list

    def execute_emergency_liquidations(self, emergency_candidates):
        """Execute emergency liquidations to restore cash balance"""
        liquidated_positions = []
        total_cash_recovered = 0

        print(f"\nEMERGENCY LIQUIDATION STARTING")
        print(f"CRITICAL: Need ${abs(self.target_cash_recovery):,.0f} cash recovery")
        print(f"Emergency candidates: {len(emergency_candidates)}")
        print("=" * 80)

        for candidate in emergency_candidates:
            if total_cash_recovered >= self.target_cash_recovery:
                print(f"CASH RECOVERY TARGET REACHED: ${total_cash_recovered:,.0f}")
                break

            symbol = candidate['symbol']
            qty = candidate['qty']
            market_value = candidate['market_value']
            reasons = candidate['reasons']

            try:
                print(f"\n[EMERGENCY] LIQUIDATING {symbol}:")
                print(f"  Quantity: {qty}")
                print(f"  Market Value: ${market_value:,.2f}")
                print(f"  Reasons: {', '.join(reasons)}")

                # Execute emergency liquidation
                if qty > 0:
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=abs(qty),
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                elif qty < 0:  # Short position - buy to cover
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=abs(qty),
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )

                print(f"  [ORDER SUBMITTED] {order.id}")
                liquidated_positions.append({
                    'symbol': symbol,
                    'qty': qty,
                    'market_value': market_value,
                    'order_id': order.id,
                    'reasons': reasons
                })

                total_cash_recovered += market_value
                print(f"  [CASH RECOVERED] ${total_cash_recovered:,.2f}")

                # Brief pause between emergency orders
                time.sleep(1)

            except Exception as e:
                self.logger.error(f"EMERGENCY LIQUIDATION FAILED {symbol}: {e}")
                print(f"  [ERROR] Failed to liquidate {symbol}: {e}")

        print(f"\nEMERGENCY LIQUIDATION COMPLETE")
        print(f"Total positions liquidated: {len(liquidated_positions)}")
        print(f"Total cash recovered: ${total_cash_recovered:,.2f}")
        print("=" * 80)

        return liquidated_positions, total_cash_recovered

    def run_emergency_recovery(self):
        """Execute complete emergency account recovery"""
        print("EMERGENCY ACCOUNT RECOVERY SYSTEM")
        print("=" * 80)
        print("CRITICAL MISSION: Restore positive cash balance immediately")
        print("PRESERVE: All winning options positions")
        print("=" * 80)

        # Get current emergency status
        status = self.get_emergency_status()
        if not status:
            print("ERROR: Cannot access account status")
            return

        cash = status['cash']
        buying_power = status['buying_power']
        positions = status['positions']

        print(f"CURRENT EMERGENCY STATUS:")
        print(f"Cash: ${cash:,.2f}")
        print(f"Buying Power: ${buying_power:,.2f}")
        print(f"Positions: {len(positions)}")
        print(f"Critical Status: {'YES' if status['critical'] else 'NO'}")

        if not status['critical']:
            print("GOOD NEWS: Cash balance is positive - no emergency action needed")
            return

        print(f"\nCASH RECOVERY NEEDED: ${abs(cash):,.2f}")

        # Identify emergency liquidations
        emergency_candidates = self.identify_emergency_liquidations(positions)

        if not emergency_candidates:
            print("NO EMERGENCY LIQUIDATION CANDIDATES FOUND")
            return

        print(f"\nEMERGENCY LIQUIDATION PLAN:")
        print("Priority | Symbol   | Market Value | P/L | Reasons")
        print("-" * 70)
        for candidate in emergency_candidates[:10]:  # Show top 10
            print(f"{candidate['priority']:>8} | {candidate['symbol']:8} | "
                  f"${candidate['market_value']:>10,.0f} | "
                  f"${candidate['unrealized_pl']:>7,.0f} | "
                  f"{', '.join(candidate['reasons'][:2])}")

        # Execute emergency liquidations
        liquidated_positions, cash_recovered = self.execute_emergency_liquidations(
            emergency_candidates
        )

        # Wait for orders to fill
        print(f"\nWaiting for emergency orders to fill...")
        time.sleep(15)

        # Check recovery status
        new_status = self.get_emergency_status()
        if new_status:
            print(f"\nEMERGENCY RECOVERY STATUS:")
            print(f"New Cash: ${new_status['cash']:,.2f}")
            print(f"New Buying Power: ${new_status['buying_power']:,.2f}")
            print(f"Cash Improvement: ${new_status['cash'] - cash:,.2f}")

            if new_status['cash'] > 0:
                print("[SUCCESS] ACCOUNT RECOVERED - Cash balance is now positive!")
                print("Ready for Intel-puts-style trades!")
            elif new_status['cash'] > cash:
                print("[PROGRESS] Cash balance improved - may need additional recovery")
            else:
                print("[WARNING] Cash balance unchanged - orders may still be processing")

        return liquidated_positions

def main():
    """Run emergency account recovery"""
    recovery = EmergencyAccountRecovery()
    recovery.run_emergency_recovery()

if __name__ == "__main__":
    main()