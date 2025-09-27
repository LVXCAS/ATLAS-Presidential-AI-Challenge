#!/usr/bin/env python3
"""
FULLY AUTONOMOUS REBALANCER - AGENTIC VERSION
This version has lowered thresholds to enable actual autonomous trading
It will execute trades automatically to free up buying power
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - AUTONOMOUS - %(message)s')

class AutonomousRebalancer:
    """FULLY AGENTIC REBALANCER - EXECUTES TRADES AUTOMATICALLY"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # AGGRESSIVE THRESHOLDS FOR AUTONOMOUS OPERATION
        self.rebalance_thresholds = {
            'strong_winner': 5.0,           # >5% = strong winner to scale up
            'weak_performer': 2.0,          # <2% = weak performer to trim
            'trim_percentage': 0.50,        # Trim 50% from weak positions (aggressive)
            'scale_percentage': 0.30,       # Scale winners by up to 30%
            'min_trade_size': 100           # Minimum $100 trades (lowered for action)
        }

        logging.info("AUTONOMOUS REBALANCER INITIALIZED")
        logging.info(f"AGGRESSIVE MODE: Min trade ${self.rebalance_thresholds['min_trade_size']}")
        logging.info("WILL EXECUTE TRADES AUTOMATICALLY")

    async def get_current_positions(self):
        """Get current portfolio positions with performance analysis"""
        try:
            positions = self.alpaca.list_positions()
            account = self.alpaca.get_account()

            portfolio_value = float(account.portfolio_value)
            position_analysis = []

            print(f"Portfolio Value: ${portfolio_value:,.0f}")
            print(f"Current Buying Power: ${float(account.buying_power):,.2f}")

            for pos in positions:
                unrealized_pl = float(pos.unrealized_pl)
                market_value = float(pos.market_value)
                pnl_pct = (unrealized_pl / (market_value - unrealized_pl)) * 100 if (market_value - unrealized_pl) != 0 else 0
                qty = float(pos.qty)

                position_analysis.append({
                    'symbol': pos.symbol,
                    'qty': qty,
                    'value': market_value,
                    'pnl': unrealized_pl,
                    'pnl_pct': pnl_pct
                })

                status = "STRONG" if pnl_pct >= self.rebalance_thresholds['strong_winner'] else "WEAK" if pnl_pct <= self.rebalance_thresholds['weak_performer'] else "NEUTRAL"
                print(f"{status}: {pos.symbol} {pnl_pct:+.1f}% (${market_value:,.0f})")

            return position_analysis

        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return []

    async def calculate_autonomous_trades(self, positions):
        """Calculate trades for autonomous execution"""

        weak_performers = [p for p in positions if p['pnl_pct'] <= self.rebalance_thresholds['weak_performer'] and p['value'] > 0]
        strong_winners = [p for p in positions if p['pnl_pct'] >= self.rebalance_thresholds['strong_winner']]

        print(f"\nAUTONOMOUS ANALYSIS:")
        print(f"Weak performers to trim: {len(weak_performers)}")
        print(f"Strong winners to scale: {len(strong_winners)}")

        autonomous_trades = []
        total_funding = 0

        # Create aggressive trim trades
        for weak_pos in weak_performers:
            trim_amount = abs(weak_pos['value'] * self.rebalance_thresholds['trim_percentage'])

            if trim_amount >= self.rebalance_thresholds['min_trade_size']:
                shares_to_sell = max(1, int(abs(weak_pos['qty']) * self.rebalance_thresholds['trim_percentage']))

                autonomous_trades.append({
                    'type': 'TRIM',
                    'symbol': weak_pos['symbol'],
                    'side': 'sell' if weak_pos['qty'] > 0 else 'buy',
                    'shares': shares_to_sell,
                    'estimated_proceeds': trim_amount,
                    'reason': f"Weak performer {weak_pos['pnl_pct']:+.1f}%"
                })

                total_funding += trim_amount
                print(f"TRIM: {weak_pos['symbol']} - ${trim_amount:,.0f} freed up")

        print(f"\nTOTAL FUNDING FROM TRIMMING: ${total_funding:,.0f}")

        # Create scale-up trades for winners
        if strong_winners and total_funding > 0:
            funding_per_winner = total_funding / len(strong_winners)

            for winner in strong_winners:
                scale_amount = min(
                    winner['value'] * self.rebalance_thresholds['scale_percentage'],
                    funding_per_winner
                )

                if scale_amount >= self.rebalance_thresholds['min_trade_size']:
                    try:
                        # Get current price
                        ticker_data = self.alpaca.get_latest_trade(winner['symbol'])
                        current_price = float(ticker_data.price)
                        shares_to_buy = max(1, int(scale_amount / current_price))

                        autonomous_trades.append({
                            'type': 'SCALE_UP',
                            'symbol': winner['symbol'],
                            'side': 'buy',
                            'shares': shares_to_buy,
                            'estimated_cost': scale_amount,
                            'reason': f"Strong winner {winner['pnl_pct']:+.1f}%"
                        })

                        print(f"SCALE: {winner['symbol']} - ${scale_amount:,.0f} invested")

                    except Exception as e:
                        logging.error(f"Error calculating scale for {winner['symbol']}: {e}")

        return autonomous_trades

    async def execute_autonomous_trades(self, trades):
        """EXECUTE TRADES AUTONOMOUSLY - THIS IS THE AGENTIC PART"""

        if not trades:
            print("NO AUTONOMOUS TRADES TO EXECUTE")
            return False

        print(f"\nEXECUTING {len(trades)} AUTONOMOUS TRADES...")
        successful_trades = 0

        for trade in trades:
            try:
                print(f"Executing: {trade['type']} {trade['shares']} {trade['symbol']}")

                # Execute the trade autonomously
                order = self.alpaca.submit_order(
                    symbol=trade['symbol'],
                    qty=trade['shares'],
                    side=trade['side'],
                    type='market',
                    time_in_force='day'
                )

                print(f"SUCCESS: Order {order.id} submitted")
                print(f"   {trade['side'].upper()} {trade['shares']} {trade['symbol']} - {trade['reason']}")

                successful_trades += 1

            except Exception as e:
                print(f"FAILED: {trade['symbol']} - {e}")
                logging.error(f"Autonomous trade failed: {e}")

        print(f"\nAUTONOMOUS EXECUTION COMPLETE")
        print(f"Successful: {successful_trades}/{len(trades)} trades")

        if successful_trades > 0:
            print("CHECKING NEW BUYING POWER...")
            try:
                new_account = self.alpaca.get_account()
                new_buying_power = float(new_account.buying_power)
                print(f"NEW BUYING POWER: ${new_buying_power:,.2f}")

                if new_buying_power > 1000:
                    print("SUCCESS: SUFFICIENT BUYING POWER FOR AUTONOMOUS TRADING!")
                    return True

            except Exception as e:
                logging.error(f"Error checking new buying power: {e}")

        return successful_trades > 0

    async def run_autonomous_rebalancing(self):
        """RUN FULLY AUTONOMOUS REBALANCING"""

        print("AUTONOMOUS REBALANCER - AGENTIC MODE")
        print("="*60)
        print("WILL EXECUTE TRADES AUTOMATICALLY")
        print("="*60)

        # Get current positions
        positions = await self.get_current_positions()

        if not positions:
            print("No positions found")
            return

        # Calculate autonomous trades
        trades = await self.calculate_autonomous_trades(positions)

        if not trades:
            print("NO TRADES MEET AUTONOMOUS CRITERIA")
            print(f"Current thresholds: Min trade ${self.rebalance_thresholds['min_trade_size']}")
            return

        # Show what will be executed
        print(f"\nAUTONOMOUS EXECUTION PLAN:")
        trim_value = sum(t['estimated_proceeds'] for t in trades if t['type'] == 'TRIM')
        scale_value = sum(t['estimated_cost'] for t in trades if t['type'] == 'SCALE_UP')

        print(f"Trim weak positions: ${trim_value:,.0f}")
        print(f"Scale strong positions: ${scale_value:,.0f}")

        # EXECUTE AUTONOMOUSLY
        print(f"\nAUTONOMOUS EXECUTION STARTING...")
        success = await self.execute_autonomous_trades(trades)

        if success:
            print("AUTONOMOUS REBALANCING COMPLETED SUCCESSFULLY!")
            print("Buying power increased - autonomous trading systems enabled!")
        else:
            print("AUTONOMOUS EXECUTION HAD ISSUES - Check logs")

async def main():
    """Run autonomous rebalancing"""
    rebalancer = AutonomousRebalancer()
    await rebalancer.run_autonomous_rebalancing()

if __name__ == "__main__":
    asyncio.run(main())