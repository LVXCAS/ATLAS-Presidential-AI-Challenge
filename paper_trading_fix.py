"""
PAPER TRADING AUTONOMOUS SYSTEM
Fixed for paper account restrictions
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
        logging.FileHandler('paper_autonomous_trading.log'),
        logging.StreamHandler()
    ]
)

class PaperAutonomousSystem:
    """Fixed autonomous trading for paper accounts"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        self.max_position_size = 0.15  # 15% max per position
        self.min_position_value = 5000  # $5K minimum position
        self.trading_active = True

    async def get_account_info(self):
        """Get paper account status"""
        try:
            account = self.alpaca.get_account()
            return {
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'day_trading_bp': float(account.daytrading_buying_power)
            }
        except Exception as e:
            logging.error(f"Error getting account info: {e}")
            return None

    async def execute_paper_trade(self, symbol, qty, side, reason):
        """Execute trade using regular buying power (not day trading)"""

        try:
            logging.info(f"PAPER TRADE: {side.upper()} {qty} shares of {symbol}")
            logging.info(f"Reason: {reason}")

            # For paper trading, use GTC orders to avoid day trading restrictions
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'  # Good Till Cancelled instead of day
            )

            logging.info(f"SUCCESS: Paper order submitted - {order.id}")

            # Save execution
            execution = {
                'timestamp': datetime.now().isoformat(),
                'order_id': order.id,
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'reason': reason,
                'status': 'PAPER_EXECUTED'
            }

            with open('paper_executions.json', 'a') as f:
                f.write(json.dumps(execution) + '\n')

            return True

        except Exception as e:
            logging.error(f"Paper trade error: {e}")
            return False

    async def find_opportunities(self):
        """Get opportunities from discovery system"""
        try:
            with open('mega_discovery_20250918_2114.json', 'r') as f:
                discovery = json.load(f)

            opportunities = discovery.get('best_strategies', [])[:5]  # Top 5
            return opportunities

        except Exception as e:
            logging.warning(f"Could not load opportunities: {e}")
            return []

    async def autonomous_paper_trading(self):
        """Main paper trading loop"""

        logging.info("PAPER AUTONOMOUS TRADING ACTIVE")
        logging.info("=" * 45)

        cycle = 0

        while self.trading_active and cycle < 10:  # Limit cycles for testing
            cycle += 1
            logging.info(f"PAPER CYCLE #{cycle}")

            # Get account status
            account = await self.get_account_info()
            if not account:
                continue

            logging.info(f"Portfolio: ${account['portfolio_value']:,.0f}")
            logging.info(f"Buying Power: ${account['buying_power']:,.0f}")

            # Get opportunities
            opportunities = await self.find_opportunities()
            logging.info(f"Opportunities found: {len(opportunities)}")

            # Execute trades for top opportunities
            for opp in opportunities[:3]:  # Top 3
                ticker = opp.get('ticker', 'UNKNOWN')
                expected_return = opp.get('expected_return', 0)

                if expected_return >= 5:  # 5%+ return threshold
                    # Calculate position size
                    position_value = account['portfolio_value'] * self.max_position_size
                    position_value = min(position_value, account['buying_power'] * 0.9)

                    if position_value >= self.min_position_value:
                        # Assume $20 average stock price for calculation
                        shares = int(position_value / 20)

                        await self.execute_paper_trade(
                            symbol=ticker,
                            qty=shares,
                            side='buy',
                            reason=f'HIGH_RETURN_{expected_return:.1f}%'
                        )

                        # Small delay between trades
                        await asyncio.sleep(2)

            logging.info(f"PAPER CYCLE #{cycle} COMPLETE")

            # Wait before next cycle
            await asyncio.sleep(60)  # 1 minute between cycles

async def main():
    print("PAPER TRADING AUTONOMOUS SYSTEM")
    print("Fixed for day trading restrictions")
    print("=" * 40)

    system = PaperAutonomousSystem()
    await system.autonomous_paper_trading()

if __name__ == "__main__":
    asyncio.run(main())