"""
FULL AUTONOMOUS TRADING SYSTEM
Complete freedom to buy/sell whenever it wants based on opportunities
"""

import asyncio
import alpaca_trade_api as tradeapi
import yfinance as yf
import logging
import os
from datetime import datetime, timedelta
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('full_autonomous_trading.log'),
        logging.StreamHandler()
    ]
)

class FullAutonomousTradingSystem:
    """System with complete trading freedom - buy/sell anytime based on opportunities"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Trading parameters - AGGRESSIVE
        self.portfolio_value = 515000
        self.target_monthly_roi = 0.40
        self.max_position_size = 0.25      # 25% max per position
        self.profit_threshold = 0.15       # Take profits at 15%
        self.stop_loss = -0.20             # Stop loss at -20%
        self.opportunity_threshold = 0.50   # 50%+ expected return to trigger trade

        # State tracking
        self.last_trade_time = None
        self.trading_active = True

    async def get_current_opportunities(self):
        """Get current opportunities from discovery system"""

        try:
            # Load latest mega discovery
            with open('mega_discovery_20250918_2114.json', 'r') as f:
                discovery = json.load(f)

            strategies = discovery.get('best_strategies', [])

            # Filter for immediate opportunities
            immediate_opportunities = [
                s for s in strategies
                if s['expected_return'] >= self.opportunity_threshold
            ]

            return immediate_opportunities[:10]  # Top 10

        except Exception as e:
            logging.error(f"Error loading opportunities: {e}")
            return []

    async def get_current_positions(self):
        """Get current portfolio positions"""

        try:
            positions = self.alpaca.list_positions()
            position_data = []

            for pos in positions:
                position_data.append({
                    'symbol': pos.symbol,
                    'qty': int(pos.qty),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'avg_entry_price': float(pos.avg_entry_price)
                })

            return position_data

        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return []

    async def check_profit_taking_opportunities(self, positions):
        """Check if any positions should be closed for profits"""

        profit_actions = []

        for pos in positions:
            profit_pct = pos['unrealized_plpc']

            # Take profits if above threshold
            if profit_pct >= self.profit_threshold:
                profit_actions.append({
                    'action': 'SELL_ALL',
                    'symbol': pos['symbol'],
                    'qty': pos['qty'],
                    'reason': f'PROFIT_TAKING_{profit_pct:.1%}',
                    'expected_profit': pos['unrealized_pl']
                })

            # Stop loss if below threshold
            elif profit_pct <= self.stop_loss:
                profit_actions.append({
                    'action': 'SELL_ALL',
                    'symbol': pos['symbol'],
                    'qty': pos['qty'],
                    'reason': f'STOP_LOSS_{profit_pct:.1%}',
                    'expected_loss': pos['unrealized_pl']
                })

        return profit_actions

    async def check_new_buying_opportunities(self, opportunities, positions):
        """Check for new stocks to buy based on opportunities"""

        current_symbols = [pos['symbol'] for pos in positions]
        buying_actions = []

        # Get available buying power
        try:
            account = self.alpaca.get_account()
            buying_power = float(account.buying_power)
        except:
            buying_power = 0

        for opp in opportunities:
            symbol = opp['ticker']

            # Skip if we already own it
            if symbol in current_symbols:
                continue

            # Calculate position size
            if opp['strategy'] == 'covered_call':
                # Need to buy 100 shares per contract
                shares_needed = 100
                cost_per_position = opp['allocation_required'] * shares_needed
            else:
                # For other strategies, use allocation_required directly
                cost_per_position = opp['allocation_required']

            # Limit position size
            max_allocation = self.portfolio_value * self.max_position_size
            position_value = min(cost_per_position, max_allocation, buying_power * 0.9)

            if position_value >= 1000:  # Minimum $1000 position
                try:
                    # Get current stock price
                    stock = yf.Ticker(symbol)
                    current_price = stock.history(period='1d')['Close'].iloc[-1]

                    shares_to_buy = int(position_value / current_price)

                    if shares_to_buy > 0:
                        buying_actions.append({
                            'action': 'BUY',
                            'symbol': symbol,
                            'qty': shares_to_buy,
                            'estimated_cost': shares_to_buy * current_price,
                            'expected_return': opp['expected_return'],
                            'reason': f'HIGH_RETURN_OPPORTUNITY_{opp["expected_return"]:.1%}'
                        })
                except Exception as e:
                    logging.debug(f"Error calculating position for {symbol}: {e}")
                    continue

        return buying_actions

    async def execute_trade_action(self, action):
        """Execute a trade action"""

        try:
            symbol = action['symbol']
            qty = action['qty']
            side = 'buy' if action['action'] == 'BUY' else 'sell'

            logging.info(f"EXECUTING: {side.upper()} {qty} shares of {symbol}")
            logging.info(f"Reason: {action['reason']}")

            # Check if market is open
            clock = self.alpaca.get_clock()
            if not clock.is_open:
                logging.warning(f"Market closed - queuing {side} order for {symbol}")
                # Could queue for market open, but for now skip
                return False

            # Submit order
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )

            logging.info(f"Order submitted: {order.id}")

            # Track execution
            execution_record = {
                'timestamp': datetime.now().isoformat(),
                'order_id': order.id,
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'reason': action['reason'],
                'expected_return': action.get('expected_return', 0),
                'status': 'SUBMITTED'
            }

            # Save execution record
            with open('autonomous_executions.json', 'a') as f:
                f.write(json.dumps(execution_record) + '\n')

            self.last_trade_time = datetime.now()
            return True

        except Exception as e:
            logging.error(f"Trade execution error: {e}")
            return False

    async def autonomous_trading_loop(self):
        """Main autonomous trading loop - runs continuously"""

        logging.info("FULL AUTONOMOUS TRADING SYSTEM ACTIVE")
        logging.info("=" * 50)
        logging.info("System can buy/sell anytime based on opportunities")
        logging.info("Target: 40% monthly ROI through active management")
        logging.info("=" * 50)

        cycle_count = 0

        while self.trading_active:
            try:
                cycle_count += 1
                logging.info(f"AUTONOMOUS CYCLE #{cycle_count}")

                # Get current state
                positions = await self.get_current_positions()
                opportunities = await self.get_current_opportunities()

                logging.info(f"Current positions: {len(positions)}")
                logging.info(f"Available opportunities: {len(opportunities)}")

                # Check for profit taking / stop losses
                profit_actions = await self.check_profit_taking_opportunities(positions)

                for action in profit_actions:
                    logging.info(f"PROFIT/STOP ACTION: {action['reason']} for {action['symbol']}")
                    await self.execute_trade_action(action)
                    time.sleep(1)  # Space out orders

                # Check for new buying opportunities
                buying_actions = await self.check_new_buying_opportunities(opportunities, positions)

                for action in buying_actions[:3]:  # Limit to 3 new positions per cycle
                    logging.info(f"NEW OPPORTUNITY: {action['symbol']} - {action['reason']}")
                    await self.execute_trade_action(action)
                    time.sleep(1)  # Space out orders

                # Log cycle summary
                if profit_actions or buying_actions:
                    logging.info(f"CYCLE #{cycle_count} COMPLETE: {len(profit_actions)} exits, {len(buying_actions)} entries")
                else:
                    logging.debug(f"CYCLE #{cycle_count}: No actions taken")

                # Wait before next cycle (check every 5 minutes during market hours)
                await asyncio.sleep(300)

            except Exception as e:
                logging.error(f"Error in autonomous trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def start_autonomous_trading(self):
        """Start the autonomous trading system"""

        logging.info("STARTING FULL AUTONOMOUS TRADING")

        # Initial portfolio check
        try:
            account = self.alpaca.get_account()
            logging.info(f"Portfolio Value: ${float(account.portfolio_value):,.0f}")
            logging.info(f"Buying Power: ${float(account.buying_power):,.0f}")
        except Exception as e:
            logging.error(f"Error accessing account: {e}")

        # Start trading loop
        await self.autonomous_trading_loop()

def main():
    """Start full autonomous trading system"""

    print("FULL AUTONOMOUS TRADING SYSTEM")
    print("=" * 40)
    print("UNLIMITED TRADING FREEDOM")
    print("Buy/Sell anytime based on opportunities")
    print("Target: 40% monthly ROI")
    print("=" * 40)

    system = FullAutonomousTradingSystem()
    asyncio.run(system.start_autonomous_trading())

if __name__ == "__main__":
    main()