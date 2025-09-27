#!/usr/bin/env python3
"""
DEPLOY MOMENTUM TRADES
Execute the momentum positions for RIVN, SNAP, INTC that user spotted moving up
Use available buying power to capture the explosive moves
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

class MomentumTradeDeployer:
    """Deploy momentum trades for user-spotted explosive moves"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

    def deploy_momentum_positions(self):
        """Deploy the momentum positions for RIVN, SNAP, INTC moves you spotted"""

        print("DEPLOYING MOMENTUM TRADES")
        print("=" * 50)
        print("Executing positions for RIVN, SNAP, INTC explosive moves")
        print("Intel-puts-style concentrated allocation")
        print("=" * 50)

        # Available buying power: $1.97M
        buying_power = 1972000

        # Get current prices and deploy positions
        momentum_trades = [
            {
                'symbol': 'RIVN',
                'allocation': 0.15,  # 15% of buying power
                'strategy': 'stock',  # Direct stock position for momentum
                'rationale': 'User spotted explosive upward momentum'
            },
            {
                'symbol': 'SNAP',
                'allocation': 0.10,  # 10% of buying power
                'strategy': 'stock',
                'rationale': 'User spotted explosive upward momentum'
            },
            {
                'symbol': 'INTC',
                'allocation': 0.08,  # 8% of buying power
                'strategy': 'stock',
                'rationale': 'Following successful Intel puts - momentum continuation'
            }
        ]

        executed_orders = []

        for trade in momentum_trades:
            symbol = trade['symbol']
            allocation = trade['allocation']
            position_value = buying_power * allocation

            try:
                # Get current price
                latest_trade = self.alpaca.get_latest_trade(symbol)
                current_price = float(latest_trade.price)

                # Calculate shares to buy
                shares = int(position_value / current_price)
                actual_cost = shares * current_price

                print(f"\n{symbol} MOMENTUM POSITION:")
                print(f"  Current Price: ${current_price:.2f}")
                print(f"  Allocation: {allocation:.1%} (${position_value:,.0f})")
                print(f"  Shares: {shares:,}")
                print(f"  Actual Cost: ${actual_cost:,.0f}")
                print(f"  Rationale: {trade['rationale']}")

                # Execute the trade
                order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )

                executed_orders.append({
                    'symbol': symbol,
                    'shares': shares,
                    'price': current_price,
                    'cost': actual_cost,
                    'allocation': allocation,
                    'order_id': order.id,
                    'rationale': trade['rationale']
                })

                print(f"  SUCCESS: Order {order.id} submitted")

            except Exception as e:
                print(f"  ERROR deploying {symbol}: {e}")

        # Summary
        total_deployed = sum(order['cost'] for order in executed_orders)
        total_allocation = sum(order['allocation'] for order in executed_orders)

        print("\n" + "=" * 50)
        print("MOMENTUM DEPLOYMENT COMPLETE")
        print("=" * 50)
        print(f"Positions Deployed: {len(executed_orders)}")
        print(f"Total Capital: ${total_deployed:,.0f}")
        print(f"Total Allocation: {total_allocation:.1%} of buying power")

        for order in executed_orders:
            print(f"  {order['symbol']:>4}: {order['shares']:>6,} shares @ ${order['price']:>6.2f} = ${order['cost']:>10,.0f}")

        print("\nReady to capture the explosive momentum you spotted!")
        print("Positions sized for maximum profit with controlled risk")

        return executed_orders

def main():
    """Deploy momentum positions"""
    deployer = MomentumTradeDeployer()
    orders = deployer.deploy_momentum_positions()
    return orders

if __name__ == "__main__":
    main()