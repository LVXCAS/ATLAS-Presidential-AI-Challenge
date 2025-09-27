#!/usr/bin/env python3
"""
TOMORROW'S PROFIT-TAKING ORDERS
Prepare automated orders for market open to lock in massive gains
Execute at 6:30 AM PDT market open for maximum efficiency
"""

import os
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

class TomorrowProfitTaking:
    """Automated profit-taking orders for market open"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

    def prepare_profit_taking_plan(self):
        """Prepare tomorrow's profit-taking strategy"""

        print("TOMORROW'S PROFIT-TAKING PLAN")
        print("=" * 60)
        print("Market Open: 6:30 AM PDT - Execute immediately")
        print("=" * 60)

        # Get current positions to verify
        try:
            positions = self.alpaca.list_positions()
            winning_options = []

            for pos in positions:
                symbol = pos.symbol
                unrealized_plpc = float(pos.unrealized_plpc) * 100

                if any(opt in symbol for opt in ['C00', 'P00']) and unrealized_plpc > 20:
                    winning_options.append({
                        'symbol': symbol,
                        'qty': int(pos.qty),
                        'gain_pct': unrealized_plpc,
                        'market_value': float(pos.market_value),
                        'profit_dollars': float(pos.unrealized_pl)
                    })

            # Show profit-taking plan
            print(f"\nWINNING OPTIONS TO CLOSE (50% EACH):")
            print("Symbol | Qty | Gain | Profit | Action Tomorrow")
            print("-" * 60)

            total_profit_to_lock = 0

            for opt in winning_options:
                qty_to_close = abs(opt['qty']) // 2  # Close 50%
                profit_to_lock = opt['profit_dollars'] * 0.5

                total_profit_to_lock += profit_to_lock

                side = "buy" if opt['qty'] < 0 else "sell"  # Reverse for options

                print(f"{opt['symbol']:>25} | {opt['qty']:>3} | {opt['gain_pct']:>+5.1f}% | ${opt['profit_dollars']:>6,.0f} | {side.upper()} {qty_to_close}")

            print("-" * 60)
            print(f"TOTAL PROFIT TO LOCK: ${total_profit_to_lock:,.0f}")

            # Calculate deployment capital
            current_buying_power = 1972000  # From last check
            freed_profit = total_profit_to_lock
            total_deployment_capital = current_buying_power + freed_profit

            print(f"\nCAPITAL AVAILABLE FOR DEPLOYMENT:")
            print(f"  Current Buying Power: ${current_buying_power:,.0f}")
            print(f"  Freed Profit: ${freed_profit:,.0f}")
            print(f"  TOTAL AVAILABLE: ${total_deployment_capital:,.0f}")

            print(f"\nSUGGESTED ALLOCATIONS:")
            print(f"  High Conviction Play: ${total_deployment_capital * 0.15:,.0f} (15%)")
            print(f"  Medium Plays (2x): ${total_deployment_capital * 0.10:,.0f} each (10%)")
            print(f"  Momentum Plays (3x): ${total_deployment_capital * 0.05:,.0f} each (5%)")

            return winning_options, total_profit_to_lock, total_deployment_capital

        except Exception as e:
            print(f"Error preparing plan: {e}")
            return [], 0, 0

    def create_market_open_strategy(self):
        """Create complete market open execution strategy"""

        winning_options, profit_to_lock, deployment_capital = self.prepare_profit_taking_plan()

        print(f"\n" + "=" * 60)
        print("TOMORROW'S EXECUTION STRATEGY")
        print("=" * 60)
        print("EXACT TIMING: 6:30 AM PDT Market Open")

        print(f"\nSTEP 1 (6:30-6:35 AM): LOCK IN PROFITS")
        for opt in winning_options:
            qty_to_close = abs(opt['qty']) // 2
            side = "buy" if opt['qty'] < 0 else "sell"
            print(f"  Execute: {side.upper()} {qty_to_close} {opt['symbol']}")

        print(f"\nSTEP 2 (6:35-7:00 AM): DEPLOY CAPITAL")
        print(f"  Available: ${deployment_capital:,.0f}")
        print(f"  Strategy: Intel-puts-style concentrated positions")
        print(f"  Target: 25-50% monthly returns")

        print(f"\nSTEP 3 (7:00+ AM): MONITOR & OPTIMIZE")
        print(f"  Let AI systems continue finding opportunities")
        print(f"  Adjust positions based on market momentum")

        print(f"\n" + "=" * 60)
        print("PREPARATION COMPLETE")
        print("Systems ready for tomorrow's profit maximization")
        print("=" * 60)

        return {
            'profit_orders': winning_options,
            'deployment_capital': deployment_capital,
            'execution_time': '6:30 AM PDT'
        }

def main():
    """Prepare tomorrow's profit-taking strategy"""
    profit_taker = TomorrowProfitTaking()
    strategy = profit_taker.create_market_open_strategy()
    return strategy

if __name__ == "__main__":
    main()