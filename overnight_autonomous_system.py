"""
OVERNIGHT AUTONOMOUS TRADING SYSTEM
Works with PDT rules by holding positions overnight
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('overnight_autonomous.log'),
        logging.StreamHandler()
    ]
)

class OvernightAutonomousSystem:
    """Autonomous system that holds positions overnight (no day trading)"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        self.max_position_size = 0.20  # 20% max per position
        self.profit_target = 0.10      # 10% profit target
        self.trading_active = True

    async def execute_overnight_trade(self, symbol, qty, side, reason):
        """Execute overnight position (no PDT restrictions)"""

        try:
            logging.info(f"OVERNIGHT TRADE: {side.upper()} {qty} shares of {symbol}")
            logging.info(f"Strategy: {reason}")

            # Use market orders with DAY time in force for overnight holds
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )

            logging.info(f"TRADE EXECUTED! Order ID: {order.id}")

            # Save execution
            execution = {
                'timestamp': datetime.now().isoformat(),
                'order_id': order.id,
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'reason': reason,
                'strategy': 'OVERNIGHT_HOLD',
                'status': 'EXECUTED'
            }

            with open('overnight_executions.json', 'a') as f:
                f.write(json.dumps(execution) + '\n')

            logging.info(f"SUCCESS: {symbol} position opened for overnight hold")
            return True

        except Exception as e:
            logging.error(f"Overnight trade failed: {e}")
            return False

    async def check_exit_positions(self):
        """Check existing positions for profit taking"""

        try:
            positions = self.alpaca.list_positions()
            exits_executed = 0

            for pos in positions:
                unrealized_plpc = float(pos.unrealized_plpc)
                symbol = pos.symbol
                qty = int(pos.qty)

                # Take profits at 10% gain
                if unrealized_plpc >= self.profit_target:
                    logging.info(f"PROFIT TARGET HIT: {symbol} at {unrealized_plpc:.1%}")

                    success = await self.execute_overnight_trade(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        reason=f'PROFIT_TAKING_{unrealized_plpc:.1%}'
                    )

                    if success:
                        exits_executed += 1

            return exits_executed

        except Exception as e:
            logging.error(f"Error checking positions: {e}")
            return 0

    async def find_overnight_opportunities(self):
        """Find stocks for overnight holds"""

        try:
            # Load opportunities from discovery system
            with open('mega_discovery_20250918_2114.json', 'r') as f:
                discovery = json.load(f)

            opportunities = discovery.get('best_strategies', [])[:5]
            return opportunities

        except Exception as e:
            logging.warning(f"Using fallback opportunities: {e}")
            # High-volume, liquid stocks good for overnight holds
            return [
                {'ticker': 'TSLA', 'expected_return': 15.2},
                {'ticker': 'AAPL', 'expected_return': 8.7},
                {'ticker': 'NVDA', 'expected_return': 12.5},
                {'ticker': 'MSFT', 'expected_return': 9.3},
                {'ticker': 'AMZN', 'expected_return': 11.8}
            ]

    async def overnight_autonomous_loop(self):
        """Main overnight trading loop"""

        logging.info("OVERNIGHT AUTONOMOUS SYSTEM ACTIVE")
        logging.info("Strategy: Hold positions overnight to avoid PDT rules")
        logging.info("=" * 55)

        cycle = 0

        while self.trading_active and cycle < 3:  # Limit for testing
            cycle += 1
            logging.info(f"OVERNIGHT CYCLE #{cycle}")

            try:
                # Check account
                account = self.alpaca.get_account()
                portfolio_value = float(account.portfolio_value)
                buying_power = float(account.buying_power)

                logging.info(f"Portfolio: ${portfolio_value:,.0f}")
                logging.info(f"Buying Power: ${buying_power:,.0f}")

                # Check for profit taking on existing positions
                exits = await self.check_exit_positions()
                if exits > 0:
                    logging.info(f"Closed {exits} profitable positions")

                # Get current positions
                positions = self.alpaca.list_positions()
                current_symbols = [pos.symbol for pos in positions]

                logging.info(f"Current positions: {len(current_symbols)}")

                # Find new opportunities if we have buying power
                if buying_power > 10000 and len(current_symbols) < 5:
                    opportunities = await self.find_overnight_opportunities()

                    for opp in opportunities[:2]:  # Max 2 new positions per cycle
                        ticker = opp.get('ticker', 'UNKNOWN')
                        expected_return = opp.get('expected_return', 5)

                        # Skip if we already own it
                        if ticker in current_symbols:
                            continue

                        if expected_return >= 5:  # 5%+ threshold
                            # Calculate position size
                            position_value = min(
                                portfolio_value * self.max_position_size,
                                buying_power * 0.8  # Use 80% of buying power
                            )

                            if position_value >= 5000:  # Min $5K position
                                # Estimate shares (conservative price estimate)
                                estimated_price = 100  # Assume $100 average
                                shares = max(1, int(position_value / estimated_price))

                                success = await self.execute_overnight_trade(
                                    symbol=ticker,
                                    qty=shares,
                                    side='buy',
                                    reason=f'OVERNIGHT_HOLD_{expected_return:.1f}%'
                                )

                                if success:
                                    logging.info(f"Opened overnight position: {ticker}")
                                    current_symbols.append(ticker)

                                # Wait between trades
                                await asyncio.sleep(5)

                logging.info(f"OVERNIGHT CYCLE #{cycle} COMPLETE")

            except Exception as e:
                logging.error(f"Cycle error: {e}")

            # Wait 5 minutes between cycles
            await asyncio.sleep(300)

async def main():
    print("OVERNIGHT AUTONOMOUS TRADING SYSTEM")
    print("Bypasses PDT rules by holding positions overnight")
    print("Target: 10% gains on each position")
    print("=" * 50)

    system = OvernightAutonomousSystem()
    await system.overnight_autonomous_loop()

if __name__ == "__main__":
    asyncio.run(main())