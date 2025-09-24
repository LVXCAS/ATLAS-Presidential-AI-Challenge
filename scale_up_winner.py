"""
SCALE UP THE WINNING SYSTEM
Your +4.03% performance today proves the approach works
Let's intelligently increase position sizes while managing risk
"""

import asyncio
import alpaca_trade_api as tradeapi
import yfinance as yf
import numpy as np
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class ScaleUpWinner:
    """Scale up the proven profitable system"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        logging.basicConfig(level=logging.INFO)

    async def analyze_current_winners(self):
        """Identify which positions are driving profits"""

        print("ANALYZING TODAY'S WINNERS")
        print("=" * 40)

        try:
            positions = self.alpaca.list_positions()
            winners = []

            for pos in positions:
                unrealized_pl = float(pos.unrealized_pl)
                market_value = float(pos.market_value)
                pnl_pct = (unrealized_pl / market_value) * 100

                if unrealized_pl > 0:
                    winners.append({
                        'symbol': pos.symbol,
                        'shares': int(pos.qty),
                        'value': market_value,
                        'profit': unrealized_pl,
                        'profit_pct': pnl_pct
                    })

            # Sort by profit percentage
            winners.sort(key=lambda x: x['profit_pct'], reverse=True)

            print("TOP PERFORMERS TODAY:")
            for winner in winners:
                print(f"  {winner['symbol']}: +${winner['profit']:,.0f} ({winner['profit_pct']:+.1f}%)")

            return winners

        except Exception as e:
            logging.error(f"Error analyzing winners: {e}")
            return []

    async def calculate_scale_up_plan(self, winners):
        """Calculate intelligent scale-up plan"""

        print("\nSCALE-UP PLAN CALCULATION")
        print("-" * 40)

        account = self.alpaca.get_account()
        buying_power = float(account.buying_power)
        portfolio_value = float(account.portfolio_value)

        print(f"Available Buying Power: ${buying_power:,.0f}")

        if buying_power < 5000:
            print("Insufficient buying power for scaling")
            return []

        scale_up_plan = []

        # Focus on top 2 performers with >2% gains
        top_performers = [w for w in winners if w['profit_pct'] > 2.0][:2]

        for performer in top_performers:
            symbol = performer['symbol']
            current_value = performer['value']

            # Scale up by 10-20% of current position
            scale_factor = 0.15 if performer['profit_pct'] > 5.0 else 0.10
            additional_value = min(current_value * scale_factor, buying_power * 0.4)

            if additional_value > 1000:  # Minimum $1000 trade
                scale_up_plan.append({
                    'symbol': symbol,
                    'additional_value': additional_value,
                    'scale_factor': scale_factor,
                    'reason': f"Top performer (+{performer['profit_pct']:.1f}%)"
                })

        return scale_up_plan

    async def execute_scale_up(self, scale_plan):
        """Execute the scale-up trades"""

        print("\nEXECUTING SCALE-UP TRADES")
        print("-" * 40)

        executed_trades = []

        for trade in scale_plan:
            symbol = trade['symbol']
            target_value = trade['additional_value']

            try:
                # Get current price
                ticker = yf.Ticker(symbol)
                current_price = ticker.history(period='1d', interval='1m').iloc[-1]['Close']

                # Calculate shares to buy
                shares_to_buy = int(target_value / current_price)

                if shares_to_buy > 0:
                    print(f"Scaling up {symbol}: +{shares_to_buy} shares (${target_value:,.0f})")

                    # Execute trade
                    order = self.alpaca.submit_order(
                        symbol=symbol,
                        qty=shares_to_buy,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )

                    executed_trades.append({
                        'symbol': symbol,
                        'shares': shares_to_buy,
                        'value': target_value,
                        'order_id': order.id
                    })

                    print(f"✅ Order executed: {order.id}")

            except Exception as e:
                print(f"❌ Failed to scale up {symbol}: {e}")

        return executed_trades

async def main():
    """Execute scale-up plan"""

    print("SCALING UP THE WINNING SYSTEM")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%H:%M:%S PT')}")
    print("Current performance: +4.03% (+$20,909)")
    print("Strategy: Scale up proven winners")

    scaler = ScaleUpWinner()

    # Step 1: Analyze winners
    winners = await scaler.analyze_current_winners()

    if not winners:
        print("No profitable positions to scale")
        return

    # Step 2: Calculate scale-up plan
    scale_plan = await scaler.calculate_scale_up_plan(winners)

    if not scale_plan:
        print("No suitable scaling opportunities")
        return

    print(f"\nPROPOSED SCALE-UP:")
    total_investment = sum(trade['additional_value'] for trade in scale_plan)
    print(f"Total additional investment: ${total_investment:,.0f}")

    for trade in scale_plan:
        print(f"  {trade['symbol']}: +${trade['additional_value']:,.0f} ({trade['reason']})")

    # Ask for confirmation
    response = input("\nExecute scale-up trades? (y/n): ")

    if response.lower() == 'y':
        executed = await scaler.execute_scale_up(scale_plan)

        print(f"\n✅ SCALE-UP COMPLETED")
        print(f"Executed {len(executed)} trades")

        total_invested = sum(trade['value'] for trade in executed)
        print(f"Total invested: ${total_invested:,.0f}")

    else:
        print("Scale-up cancelled")

if __name__ == "__main__":
    asyncio.run(main())