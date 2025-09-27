#!/usr/bin/env python3
"""
AUTONOMOUS PORTFOLIO CLEANUP ENGINE
Automatically liquidates losing positions to free capital for Intel-puts-style trades
Runs continuously to maintain optimal portfolio state
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import logging

class AutonomousPortfolioCleanup:
    """Autonomous engine to cleanup portfolio and free capital"""

    def __init__(self):
        load_dotenv()

        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL')
        )

        # Target capital to free up for Intel-puts-style trades
        self.target_free_capital = 1000000  # $1M for new trades

        # Protected positions (keep these)
        self.protected_symbols = {
            'INTC',    # Your 70.6% winner
            'SNAP',    # Keep core position
            'TSLA',    # Quality stock
            'LYFT250926P00021000',   # Winning put +63%
            'RIVN250926P00014000',   # Winning put +93%
            'SNAP250926C00009000',   # Options position
            'SNAP250926P00008000'    # Options position
        }

        # Liquidation criteria
        self.liquidation_rules = {
            'loss_threshold': -0.05,  # Liquidate if down >5%
            'penny_stock_threshold': 1.0,  # Liquidate penny stocks under $1
            'low_volume_threshold': 1000,   # Liquidate low volume positions
            'worthless_threshold': -0.90    # Liquidate if down >90%
        }

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - CLEANUP - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_account_status(self):
        """Get current account status"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()

            return {
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'positions': positions,
                'positions_count': len(positions)
            }
        except Exception as e:
            self.logger.error(f"Error getting account status: {e}")
            return None

    def analyze_liquidation_candidates(self, positions):
        """Identify positions to liquidate automatically"""
        liquidation_list = []

        for pos in positions:
            symbol = pos.symbol
            qty = int(pos.qty)
            current_price = float(pos.market_value) / abs(qty) if qty != 0 else 0
            unrealized_pl = float(pos.unrealized_pl)
            unrealized_pl_pct = float(pos.unrealized_plpc)
            market_value = abs(float(pos.market_value))

            # Skip protected positions
            if symbol in self.protected_symbols:
                continue

            liquidation_reasons = []

            # Rule 1: Heavy losers (>5% down)
            if unrealized_pl_pct < self.liquidation_rules['loss_threshold']:
                liquidation_reasons.append(f"Heavy loss: {unrealized_pl_pct:.1%}")

            # Rule 2: Penny stocks
            if current_price < self.liquidation_rules['penny_stock_threshold']:
                liquidation_reasons.append(f"Penny stock: ${current_price:.3f}")

            # Rule 3: Worthless positions (>90% down)
            if unrealized_pl_pct < self.liquidation_rules['worthless_threshold']:
                liquidation_reasons.append(f"Worthless: {unrealized_pl_pct:.1%}")

            # Rule 4: Large positions bleeding money
            if market_value > 100000 and unrealized_pl < -10000:  # >$100K position losing >$10K
                liquidation_reasons.append(f"Large bleeder: -${abs(unrealized_pl):,.0f}")

            if liquidation_reasons:
                liquidation_list.append({
                    'symbol': symbol,
                    'qty': qty,
                    'market_value': market_value,
                    'unrealized_pl': unrealized_pl,
                    'unrealized_pl_pct': unrealized_pl_pct,
                    'current_price': current_price,
                    'reasons': liquidation_reasons,
                    'priority': self._calculate_liquidation_priority(
                        market_value, unrealized_pl, unrealized_pl_pct
                    )
                })

        # Sort by liquidation priority (highest priority first)
        liquidation_list.sort(key=lambda x: x['priority'], reverse=True)
        return liquidation_list

    def _calculate_liquidation_priority(self, market_value, unrealized_pl, unrealized_pl_pct):
        """Calculate priority score for liquidation (higher = liquidate first)"""
        priority = 0

        # Big positions losing money = high priority
        if market_value > 500000 and unrealized_pl < -50000:
            priority += 100
        elif market_value > 100000 and unrealized_pl < -10000:
            priority += 50

        # Heavy percentage losses = high priority
        if unrealized_pl_pct < -0.50:  # >50% loss
            priority += 75
        elif unrealized_pl_pct < -0.20:  # >20% loss
            priority += 40

        # Worthless positions = medium priority (preserve capital)
        if unrealized_pl_pct < -0.90:
            priority += 30

        return priority

    def execute_liquidations(self, liquidation_candidates, target_capital_to_free):
        """Execute liquidation orders autonomously"""
        liquidated_positions = []
        total_capital_freed = 0

        print(f"\nAUTONOMOUS LIQUIDATION STARTING")
        print(f"Target capital to free: ${target_capital_to_free:,.0f}")
        print(f"Liquidation candidates: {len(liquidation_candidates)}")
        print("=" * 60)

        for candidate in liquidation_candidates:
            if total_capital_freed >= target_capital_to_free:
                print(f"Target capital reached: ${total_capital_freed:,.0f}")
                break

            symbol = candidate['symbol']
            qty = candidate['qty']
            market_value = candidate['market_value']
            reasons = candidate['reasons']

            try:
                print(f"\nLIQUIDATING {symbol}:")
                print(f"  Quantity: {qty}")
                print(f"  Market Value: ${market_value:,.2f}")
                print(f"  Reasons: {', '.join(reasons)}")

                # Execute market sell order with batch liquidation for large positions
                orders_submitted = []
                remaining_qty = abs(qty)
                batch_size = 1000  # Alpaca options limit

                # Check if this is an options position (contains expiration date)
                is_options = any(char.isdigit() for char in symbol) and ('C' in symbol[-9:] or 'P' in symbol[-9:])

                if is_options and remaining_qty > batch_size:
                    # Batch liquidation for large options positions
                    print(f"  BATCH LIQUIDATION: {remaining_qty} contracts in batches of {batch_size}")

                    while remaining_qty > 0:
                        current_batch = min(remaining_qty, batch_size)

                        if qty > 0:
                            order = self.api.submit_order(
                                symbol=symbol,
                                qty=current_batch,
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                        else:  # Short position - buy to cover
                            order = self.api.submit_order(
                                symbol=symbol,
                                qty=current_batch,
                                side='buy',
                                type='market',
                                time_in_force='day'
                            )

                        orders_submitted.append(order)
                        remaining_qty -= current_batch
                        print(f"    Batch order submitted: {current_batch} contracts (Order: {order.id})")

                        # Brief pause between batch orders
                        time.sleep(1)

                    # Use the first order for tracking
                    order = orders_submitted[0]
                    print(f"  BATCH COMPLETE: {len(orders_submitted)} orders submitted")

                else:
                    # Standard liquidation for smaller positions
                    if qty > 0:
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=remaining_qty,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                    else:  # Short position - buy to cover
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=remaining_qty,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )

                print(f"  ORDER SUBMITTED: {order.id}")
                liquidated_positions.append({
                    'symbol': symbol,
                    'qty': qty,
                    'market_value': market_value,
                    'order_id': order.id,
                    'reasons': reasons
                })

                total_capital_freed += market_value

                # Brief pause between orders
                time.sleep(2)

            except Exception as e:
                self.logger.error(f"Failed to liquidate {symbol}: {e}")
                print(f"  ERROR: Failed to liquidate {symbol}: {e}")

        print(f"\nLIQUIDATION COMPLETE")
        print(f"Total positions liquidated: {len(liquidated_positions)}")
        print(f"Total capital freed: ${total_capital_freed:,.2f}")
        print("=" * 60)

        return liquidated_positions, total_capital_freed

    def check_and_cleanup_portfolio(self):
        """Main autonomous cleanup cycle"""
        print(f"\nAUTONOMOUS PORTFOLIO CLEANUP - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 70)

        # Get current account status
        account_status = self.get_account_status()
        if not account_status:
            return

        cash = account_status['cash']
        buying_power = account_status['buying_power']
        positions = account_status['positions']

        print(f"Current Cash: ${cash:,.2f}")
        print(f"Buying Power: ${buying_power:,.2f}")
        print(f"Positions: {len(positions)}")

        # If we have sufficient buying power, no cleanup needed
        if buying_power >= self.target_free_capital:
            print(f"SUFFICIENT CAPITAL AVAILABLE - No cleanup needed")
            return

        # Calculate how much capital we need to free
        capital_needed = self.target_free_capital - max(0, buying_power)
        print(f"Capital needed: ${capital_needed:,.2f}")

        # Analyze liquidation candidates
        liquidation_candidates = self.analyze_liquidation_candidates(positions)

        if not liquidation_candidates:
            print("NO LIQUIDATION CANDIDATES FOUND")
            return

        print(f"\nLIQUIDATION CANDIDATES ({len(liquidation_candidates)}):")
        print("Priority | Symbol   | Market Value | Loss | Loss% | Reasons")
        print("-" * 70)
        for candidate in liquidation_candidates[:10]:  # Show top 10
            print(f"{candidate['priority']:>8} | {candidate['symbol']:8} | "
                  f"${candidate['market_value']:>10,.0f} | "
                  f"${candidate['unrealized_pl']:>7,.0f} | "
                  f"{candidate['unrealized_pl_pct']:>5.1%} | "
                  f"{', '.join(candidate['reasons'][:2])}")

        # Execute liquidations
        liquidated_positions, capital_freed = self.execute_liquidations(
            liquidation_candidates, capital_needed
        )

        # Wait a moment for orders to fill
        time.sleep(10)

        # Check new account status
        new_status = self.get_account_status()
        if new_status:
            print(f"\nUPDATED ACCOUNT STATUS:")
            print(f"New Buying Power: ${new_status['buying_power']:,.2f}")
            print(f"Capital increase: ${new_status['buying_power'] - buying_power:,.2f}")

            if new_status['buying_power'] >= self.target_free_capital * 0.8:  # 80% of target
                print("SUCCESS: Sufficient capital now available for Intel-puts-style trades!")
            else:
                print("PARTIAL SUCCESS: More cleanup may be needed")

        return liquidated_positions

    def run_continuous_cleanup(self):
        """Run autonomous cleanup continuously"""
        print("AUTONOMOUS PORTFOLIO CLEANUP ENGINE")
        print("=" * 70)
        print("Continuously monitoring and cleaning portfolio for Intel-puts-style trades")
        print("Protected positions: " + ", ".join(self.protected_symbols))
        print("=" * 70)

        while True:
            try:
                self.check_and_cleanup_portfolio()

                # Wait 5 minutes before next check
                print(f"\nNext cleanup check in 5 minutes...")
                time.sleep(300)

            except KeyboardInterrupt:
                print("\nAutonomous cleanup stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Cleanup cycle error: {e}")
                time.sleep(60)  # Wait 1 minute on error

def main():
    """Run autonomous portfolio cleanup"""
    cleanup_engine = AutonomousPortfolioCleanup()
    cleanup_engine.run_continuous_cleanup()

if __name__ == "__main__":
    main()