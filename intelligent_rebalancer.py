"""
INTELLIGENT AUTONOMOUS REBALANCER
Creates buying power by trimming underperformers to fund winners
This is what the system SHOULD have been doing automatically
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class IntelligentRebalancer:
    """Automatically rebalance by moving capital from weak to strong positions"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        self.rebalance_thresholds = {
            'strong_winner': 5.0,           # >5% = strong winner to scale up
            'weak_performer': 1.5,          # <1.5% = weak performer to trim
            'trim_percentage': 0.10,        # Trim 10% from weak positions
            'scale_percentage': 0.20,       # Scale winners by up to 20%
            'min_trade_size': 1000          # Minimum $1000 trades
        }

    async def get_current_positions(self):
        """Get current portfolio positions with performance analysis"""

        try:
            positions = self.alpaca.list_positions()
            account = self.alpaca.get_account()

            portfolio_value = float(account.portfolio_value)
            position_analysis = []

            for pos in positions:
                unrealized_pl = float(pos.unrealized_pl)
                market_value = float(pos.market_value)
                pnl_pct = (unrealized_pl / market_value) * 100 if market_value > 0 else 0

                position_analysis.append({
                    'symbol': pos.symbol,
                    'qty': int(pos.qty),
                    'value': market_value,
                    'pnl': unrealized_pl,
                    'pnl_pct': pnl_pct,
                    'weight': market_value / portfolio_value
                })

            return position_analysis, portfolio_value

        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return [], 0

    async def identify_rebalancing_opportunities(self, positions):
        """Identify which positions to trim and which to scale up"""

        strong_winners = []
        weak_performers = []

        print("INTELLIGENT REBALANCING ANALYSIS")
        print("=" * 50)

        for pos in positions:
            symbol = pos['symbol']
            pnl_pct = pos['pnl_pct']
            value = pos['value']

            if pnl_pct >= self.rebalance_thresholds['strong_winner']:
                strong_winners.append(pos)
                print(f"STRONG WINNER: {symbol} +{pnl_pct:.1f}% (${value:,.0f}) - SCALE UP TARGET")

            elif pnl_pct <= self.rebalance_thresholds['weak_performer']:
                weak_performers.append(pos)
                trim_amount = value * self.rebalance_thresholds['trim_percentage']
                print(f"WEAK PERFORMER: {symbol} +{pnl_pct:.1f}% (${value:,.0f}) - TRIM ${trim_amount:,.0f}")

        return strong_winners, weak_performers

    async def calculate_rebalancing_trades(self, strong_winners, weak_performers):
        """Calculate the optimal rebalancing trades"""

        rebalancing_trades = []

        # Calculate total funding available from trimming weak performers
        total_funding = 0
        for weak_pos in weak_performers:
            trim_amount = weak_pos['value'] * self.rebalance_thresholds['trim_percentage']
            if trim_amount >= self.rebalance_thresholds['min_trade_size']:
                total_funding += trim_amount

                # Create trim trade
                shares_to_sell = int(weak_pos['qty'] * self.rebalance_thresholds['trim_percentage'])
                if shares_to_sell > 0:
                    rebalancing_trades.append({
                        'type': 'TRIM',
                        'symbol': weak_pos['symbol'],
                        'side': 'sell',
                        'shares': shares_to_sell,
                        'estimated_proceeds': trim_amount,
                        'reason': f"Weak performer +{weak_pos['pnl_pct']:.1f}%"
                    })

        print(f"\nTOTAL FUNDING FROM TRIMMING: ${total_funding:,.0f}")

        # Allocate funding to strong winners
        if strong_winners and total_funding > 0:
            funding_per_winner = total_funding / len(strong_winners)

            for winner in strong_winners:
                scale_amount = min(
                    winner['value'] * self.rebalance_thresholds['scale_percentage'],  # Max 20% increase
                    funding_per_winner  # Available funding
                )

                if scale_amount >= self.rebalance_thresholds['min_trade_size']:
                    # Get current price to calculate shares
                    try:
                        ticker_data = self.alpaca.get_latest_trade(winner['symbol'])
                        current_price = float(ticker_data.price)
                        shares_to_buy = int(scale_amount / current_price)

                        if shares_to_buy > 0:
                            rebalancing_trades.append({
                                'type': 'SCALE_UP',
                                'symbol': winner['symbol'],
                                'side': 'buy',
                                'shares': shares_to_buy,
                                'estimated_cost': scale_amount,
                                'reason': f"Strong winner +{winner['pnl_pct']:.1f}%"
                            })

                    except Exception as e:
                        logging.error(f"Error getting price for {winner['symbol']}: {e}")

        return rebalancing_trades

    async def execute_intelligent_rebalancing(self, trades):
        """Execute the intelligent rebalancing trades"""

        if not trades:
            print("No rebalancing opportunities identified")
            return

        print(f"\nEXECUTING INTELLIGENT REBALANCING")
        print("-" * 40)

        # Execute trim trades first (to create buying power)
        trim_trades = [t for t in trades if t['type'] == 'TRIM']
        scale_trades = [t for t in trades if t['type'] == 'SCALE_UP']

        executed_trades = []

        # Phase 1: Execute trims
        for trade in trim_trades:
            try:
                order = self.alpaca.submit_order(
                    symbol=trade['symbol'],
                    qty=trade['shares'],
                    side=trade['side'],
                    type='market',
                    time_in_force='day'
                )

                executed_trades.append(f"TRIMMED {trade['symbol']}: -{trade['shares']} shares - {trade['reason']}")
                logging.info(f"TRIM EXECUTED: {trade['symbol']} -{trade['shares']} shares")

            except Exception as e:
                logging.error(f"Failed to trim {trade['symbol']}: {e}")

        # Small delay to let trim orders process
        await asyncio.sleep(2)

        # Phase 2: Execute scale-ups
        for trade in scale_trades:
            try:
                order = self.alpaca.submit_order(
                    symbol=trade['symbol'],
                    qty=trade['shares'],
                    side=trade['side'],
                    type='market',
                    time_in_force='day'
                )

                executed_trades.append(f"SCALED UP {trade['symbol']}: +{trade['shares']} shares - {trade['reason']}")
                logging.info(f"SCALE UP EXECUTED: {trade['symbol']} +{trade['shares']} shares")

            except Exception as e:
                logging.error(f"Failed to scale up {trade['symbol']}: {e}")

        # Report results
        print(f"\nINTELLIGENT REBALANCING COMPLETED")
        print(f"Executed {len(executed_trades)} trades:")
        for trade in executed_trades:
            print(f"  {trade}")

        return executed_trades

    async def run_intelligent_rebalancing(self):
        """Run the complete intelligent rebalancing process"""

        print("INTELLIGENT AUTONOMOUS REBALANCING")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%H:%M:%S PT')}")
        print("Strategy: Move capital from weak to strong performers")
        print("=" * 60)

        # Get current positions
        positions, portfolio_value = await self.get_current_positions()

        if not positions:
            print("No positions found")
            return

        print(f"Portfolio Value: ${portfolio_value:,.0f}")
        print(f"Positions: {len(positions)}")

        # Identify opportunities
        strong_winners, weak_performers = await self.identify_rebalancing_opportunities(positions)

        if not strong_winners:
            print("\nNo strong winners to scale up")
            return

        if not weak_performers:
            print("\nNo weak performers to trim")
            return

        # Calculate trades
        trades = await self.calculate_rebalancing_trades(strong_winners, weak_performers)

        if not trades:
            print("\nNo rebalancing trades calculated")
            return

        # Show proposed trades
        print(f"\nPROPOSED REBALANCING TRADES:")
        total_trim_value = sum(t.get('estimated_proceeds', 0) for t in trades if t['type'] == 'TRIM')
        total_scale_value = sum(t.get('estimated_cost', 0) for t in trades if t['type'] == 'SCALE_UP')

        print(f"Total Trim Value: ${total_trim_value:,.0f}")
        print(f"Total Scale Value: ${total_scale_value:,.0f}")

        for trade in trades:
            action = "SELL" if trade['side'] == 'sell' else "BUY"
            print(f"  {action} {trade['shares']} {trade['symbol']} - {trade['reason']}")

        # Execute trades automatically (autonomous mode)
        print(f"\nEXECUTING AUTOMATICALLY (AUTONOMOUS MODE)")

        executed = await self.execute_intelligent_rebalancing(trades)
        print(f"\nINTELLIGENT REBALANCING COMPLETED SUCCESSFULLY!")

async def main():
    """Run intelligent rebalancing now"""

    rebalancer = IntelligentRebalancer()
    await rebalancer.run_intelligent_rebalancing()

if __name__ == "__main__":
    asyncio.run(main())