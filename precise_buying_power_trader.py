#!/usr/bin/env python3
"""
PRECISE BUYING POWER TRADER
Uses the exact buying power shown in account to execute trades
No more guessing - uses actual available capital
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - PRECISE - %(message)s')

class PreciseBuyingPowerTrader:
    """Trade using exact available buying power"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        logging.info("PRECISE BUYING POWER TRADER INITIALIZED")
        logging.info("Will use exact available buying power for trades")

    async def get_precise_buying_power(self):
        """Get the exact buying power available for trading"""

        try:
            account = self.alpaca.get_account()

            # Get all the different buying power metrics
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
            buying_power = float(account.buying_power)

            # Try to get day trading buying power if available
            try:
                day_trade_bp = float(account.daytrading_buying_power)
            except:
                day_trade_bp = buying_power

            print("=== PRECISE BUYING POWER ANALYSIS ===")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")
            print(f"Cash: ${cash:,.2f}")
            print(f"Buying Power: ${buying_power:,.2f}")
            print(f"Day Trading BP: ${day_trade_bp:,.2f}")

            # Use the most restrictive (realistic) buying power
            effective_bp = min(buying_power, day_trade_bp)

            # But also check what we can actually use
            positions = self.alpaca.list_positions()
            print(f"Current Positions: {len(positions)}")

            print(f"\nEFFECTIVE BUYING POWER: ${effective_bp:,.2f}")

            return {
                'effective_buying_power': effective_bp,
                'cash': cash,
                'portfolio_value': portfolio_value,
                'position_count': len(positions)
            }

        except Exception as e:
            logging.error(f"Buying power check error: {e}")
            return None

    async def get_executable_opportunities(self):
        """Get opportunities that we can actually execute"""

        # High conviction opportunities from your running systems
        opportunities = [
            {
                'symbol': 'EDIT',
                'conviction': 'HIGH',
                'reason': 'Top scorer from profit engine (6.50)',
                'target_return': 0.15
            },
            {
                'symbol': 'NTLA',
                'conviction': 'HIGH',
                'reason': 'Current winner +7.9%, biotech momentum',
                'target_return': 0.20
            },
            {
                'symbol': 'LCID',
                'conviction': 'MEDIUM',
                'reason': 'Current winner +6.3%, EV sector',
                'target_return': 0.12
            },
            {
                'symbol': 'RIVN',
                'conviction': 'MEDIUM',
                'reason': 'Current winner +6.7%, momentum',
                'target_return': 0.15
            },
            {
                'symbol': 'VTVT',
                'conviction': 'HIGH',
                'reason': 'Market scanner +4.60%, strong signal',
                'target_return': 0.18
            }
        ]

        return opportunities

    async def calculate_precise_positions(self, opportunities, bp_info):
        """Calculate exact positions based on available buying power"""

        effective_bp = bp_info['effective_buying_power']

        if effective_bp < 1000:
            print(f"Insufficient buying power: ${effective_bp:.2f}")
            return []

        # Reserve some buying power for safety
        usable_bp = effective_bp * 0.90  # Use 90% to be safe

        print(f"\n=== PRECISE POSITION CALCULATION ===")
        print(f"Available: ${effective_bp:,.2f}")
        print(f"Using: ${usable_bp:,.2f} (90%)")
        print("-" * 50)

        positions = []

        # Focus on top 2-3 opportunities to concentrate buying power
        top_opps = sorted(opportunities, key=lambda x: (x['conviction'] == 'HIGH', x['target_return']), reverse=True)[:3]

        position_value = usable_bp / len(top_opps)

        for opp in top_opps:
            try:
                # Get current price
                quote = self.alpaca.get_latest_quote(opp['symbol'])
                current_price = float(quote.ask_price) if quote.ask_price else float(quote.bid_price)

                if not current_price:
                    continue

                # Calculate shares we can afford
                shares = int(position_value / current_price)
                actual_value = shares * current_price

                if shares > 0 and actual_value > 100:  # Minimum $100 position
                    positions.append({
                        'symbol': opp['symbol'],
                        'shares': shares,
                        'price': current_price,
                        'value': actual_value,
                        'conviction': opp['conviction'],
                        'reason': opp['reason'],
                        'target_return': opp['target_return']
                    })

                    print(f"{opp['symbol']:>6} | {shares:>6} shares | ${current_price:>8.2f} | ${actual_value:>10,.0f} | {opp['conviction']}")

            except Exception as e:
                logging.error(f"Price lookup error for {opp['symbol']}: {e}")

        total_value = sum(p['value'] for p in positions)
        print("-" * 50)
        print(f"TOTAL: ${total_value:,.2f} ({(total_value/usable_bp)*100:.1f}% of available)")

        return positions

    async def execute_precise_trades(self, positions):
        """Execute trades with precise buying power management"""

        if not positions:
            print("No positions calculated")
            return False

        print(f"\n=== PRECISE EXECUTION ===")

        successful = 0
        total_deployed = 0

        for pos in positions:
            try:
                print(f"\nExecuting: {pos['symbol']} - {pos['shares']} shares")
                print(f"  Value: ${pos['value']:,.2f}")
                print(f"  Reason: {pos['reason']}")

                order = self.alpaca.submit_order(
                    symbol=pos['symbol'],
                    qty=pos['shares'],
                    side='buy',
                    type='market',
                    time_in_force='day'
                )

                print(f"  SUCCESS: Order {order.id}")
                print(f"  Expected return: {pos['target_return']*100:.0f}%")

                successful += 1
                total_deployed += pos['value']

                await asyncio.sleep(1)

            except Exception as e:
                print(f"  FAILED: {e}")

                # If it's buying power error, try smaller size
                if "insufficient" in str(e).lower():
                    try:
                        smaller_shares = max(1, pos['shares'] // 2)
                        print(f"  RETRY: Trying {smaller_shares} shares")

                        retry_order = self.alpaca.submit_order(
                            symbol=pos['symbol'],
                            qty=smaller_shares,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )

                        print(f"  RETRY SUCCESS: Order {retry_order.id}")
                        successful += 1
                        total_deployed += smaller_shares * pos['price']

                    except Exception as retry_error:
                        print(f"  RETRY FAILED: {retry_error}")

        print(f"\n=== EXECUTION RESULTS ===")
        print(f"Successful: {successful}/{len(positions)}")
        print(f"Deployed: ${total_deployed:,.2f}")

        if successful > 0:
            print("PRECISE BUYING POWER TRADING SUCCESS!")
            return True

        return False

    async def run_precise_trading(self):
        """Run precise buying power trading"""

        print("PRECISE BUYING POWER TRADER")
        print("="*50)
        print("Using exact available buying power for trades")
        print("="*50)

        # Get precise buying power
        bp_info = await self.get_precise_buying_power()
        if not bp_info:
            return

        # Get opportunities
        opportunities = await self.get_executable_opportunities()

        # Calculate positions
        positions = await self.calculate_precise_positions(opportunities, bp_info)

        # Execute trades
        success = await self.execute_precise_trades(positions)

        if success:
            print("\nPRECISE TRADING COMPLETE!")
            print("Capital deployed using exact buying power limits!")

async def main():
    """Run precise buying power trading"""
    trader = PreciseBuyingPowerTrader()
    await trader.run_precise_trading()

if __name__ == "__main__":
    asyncio.run(main())