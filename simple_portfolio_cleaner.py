#!/usr/bin/env python3
"""
SIMPLE PORTFOLIO CLEANER
Clean up account positions and prepare for concentrated strategy
Focus on clearing losing positions to free capital
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

class SimplePortfolioCleaner:
    """Simple portfolio cleaner for account reset"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

    def analyze_account_status(self):
        """Get current account status"""

        print("SIMPLE PORTFOLIO CLEANUP")
        print("=" * 50)
        print("Analyzing account for cleanup opportunities")
        print("=" * 50)

        try:
            # Get account info
            account = self.alpaca.get_account()

            print(f"Account Status: {account.status}")
            print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"Cash: ${float(account.cash):,.2f}")
            print(f"Buying Power: ${float(account.buying_power):,.2f}")

            # Get positions
            positions = self.alpaca.list_positions()

            print(f"\nCurrent Positions: {len(positions)}")

            if positions:
                print("\nPosition Summary:")
                print("Symbol | Qty | Market Value | Unrealized P&L")
                print("-" * 50)

                total_value = 0
                total_pl = 0

                for pos in positions:
                    market_value = float(pos.market_value)
                    unrealized_pl = float(pos.unrealized_pl)

                    total_value += market_value
                    total_pl += unrealized_pl

                    print(f"{pos.symbol:>6} | {pos.qty:>3} | ${market_value:>8,.0f} | ${unrealized_pl:>8,.0f}")

                print("-" * 50)
                print(f"TOTAL  |     | ${total_value:>8,.0f} | ${total_pl:>8,.0f}")

                portfolio_pl_pct = (total_pl / total_value) * 100 if total_value > 0 else 0
                print(f"Portfolio P&L: {portfolio_pl_pct:+.1f}%")

                # Categorize positions
                losing_positions = [p for p in positions if float(p.unrealized_pl) < -500]  # $500+ loss
                small_positions = [p for p in positions if abs(float(p.market_value)) < 1000]  # Under $1K

                print(f"\nPositions losing $500+: {len(losing_positions)}")
                print(f"Positions under $1K: {len(small_positions)}")

                if losing_positions or small_positions:
                    print("\nRecommended Cleanup Actions:")

                    cleanup_positions = set()

                    for pos in losing_positions:
                        cleanup_positions.add(pos.symbol)
                        print(f"  LIQUIDATE {pos.symbol} (${float(pos.unrealized_pl):,.0f} loss)")

                    for pos in small_positions:
                        if pos.symbol not in cleanup_positions:
                            cleanup_positions.add(pos.symbol)
                            print(f"  LIQUIDATE {pos.symbol} (small position ${float(pos.market_value):,.0f})")

                    # Calculate cleanup impact
                    cleanup_value = sum(float(p.market_value) for p in positions if p.symbol in cleanup_positions)
                    cleanup_pl = sum(float(p.unrealized_pl) for p in positions if p.symbol in cleanup_positions)

                    print(f"\nCleanup Impact:")
                    print(f"  Positions to close: {len(cleanup_positions)}")
                    print(f"  Current value: ${cleanup_value:,.0f}")
                    print(f"  Realized P&L: ${cleanup_pl:,.0f}")
                    print(f"  Cash freed: ${cleanup_value * 0.95:,.0f} (est. after fees)")

                    return {
                        'cleanup_positions': list(cleanup_positions),
                        'cleanup_value': cleanup_value,
                        'cleanup_pl': cleanup_pl,
                        'positions_data': positions
                    }

            return {'cleanup_positions': [], 'positions_data': positions}

        except Exception as e:
            print(f"Error analyzing account: {e}")
            return {}

    def close_all_positions(self, execute=False):
        """Close all positions to reset account"""

        print(f"\n=== {'EXECUTING' if execute else 'SIMULATING'} POSITION CLOSURE ===")

        try:
            positions = self.alpaca.list_positions()

            if not positions:
                print("No positions to close")
                return []

            closed_orders = []

            for position in positions:
                symbol = position.symbol
                qty = int(position.qty)
                side = "sell" if qty > 0 else "buy"
                qty = abs(qty)

                print(f"Closing {symbol}: {side.upper()} {qty} shares")

                if execute:
                    try:
                        order = self.alpaca.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side=side,
                            type='market',
                            time_in_force='day'
                        )
                        closed_orders.append({
                            'symbol': symbol,
                            'qty': qty,
                            'side': side,
                            'order_id': order.id,
                            'status': 'submitted'
                        })
                        print(f"  SUCCESS Order submitted: {order.id}")
                    except Exception as e:
                        print(f"  ERROR closing {symbol}: {e}")
                else:
                    closed_orders.append({
                        'symbol': symbol,
                        'qty': qty,
                        'side': side,
                        'order_id': 'SIMULATED',
                        'status': 'simulated'
                    })

            print(f"\n{'Executed' if execute else 'Simulated'} {len(closed_orders)} position closures")

            # Save report
            report = {
                'timestamp': datetime.now().isoformat(),
                'action': 'close_all_positions',
                'executed': execute,
                'closed_orders': closed_orders,
                'original_positions': len(positions)
            }

            filename = f'position_cleanup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"Report saved: {filename}")

            if execute:
                print("\nACCOUNT CLEANUP COMPLETE!")
                print("All positions closed - ready for fresh concentrated strategy deployment")
            else:
                print(f"\nDRY RUN COMPLETE")
                print("Run with execute=True to actually close positions")

            return closed_orders

        except Exception as e:
            print(f"Error closing positions: {e}")
            return []

def main():
    cleaner = SimplePortfolioCleaner()

    # First analyze
    analysis = cleaner.analyze_account_status()

    if analysis.get('cleanup_positions'):
        print(f"\n{'='*50}")
        print("CLEANUP RECOMMENDATION:")
        print("Run cleaner.close_all_positions(execute=True) to clean account")
        print("This will free up capital for concentrated strategy deployment")
        print(f"{'='*50}")

    return cleaner, analysis

if __name__ == "__main__":
    cleaner, analysis = main()