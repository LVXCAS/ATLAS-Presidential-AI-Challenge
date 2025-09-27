#!/usr/bin/env python3
"""
QUALITY CONSTRAINED TRADER
Uses the approved institutional-quality asset universe
No more penny stocks, warrants, or illiquid garbage
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - QUALITY - %(message)s')

class QualityConstrainedTrader:
    """Autonomous trader using only institutional-quality assets"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Load approved universe
        self.approved_universe = self.load_approved_universe()

        logging.info("QUALITY CONSTRAINED TRADER INITIALIZED")
        logging.info(f"Using {len(self.approved_universe)} institutional-quality assets")

    def load_approved_universe(self):
        """Load the approved asset universe from filter"""
        try:
            with open('approved_asset_universe.json', 'r') as f:
                data = json.load(f)
                return data['approved_assets']
        except Exception as e:
            logging.error(f"Could not load approved universe: {e}")
            # Fallback to major stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'JPM', 'JNJ', 'PG', 'SPY', 'QQQ']

    async def get_quality_opportunities(self):
        """Get opportunities from approved universe only"""

        opportunities = []

        print("=== QUALITY OPPORTUNITY ANALYSIS ===")
        print(f"Scanning {len(self.approved_universe)} institutional-quality assets")
        print("-" * 50)

        # Focus on top performers from approved universe
        priority_assets = [
            'NVDA', 'META', 'GOOGL', 'AAPL',  # Tech leaders
            'SPY', 'QQQ', 'XLK', 'XLF',       # Liquid ETFs
            'JPM', 'JNJ', 'PG', 'HD',         # Blue chips
            'TSLA', 'INTC', 'MU', 'AMD'       # Growth/momentum
        ]

        # Filter to only approved assets
        priority_assets = [asset for asset in priority_assets if asset in self.approved_universe]

        for symbol in priority_assets:
            try:
                quote = self.alpaca.get_latest_quote(symbol)
                price = float(quote.bid_price) if quote.bid_price else 0

                if price > 0:
                    # Simple momentum scoring
                    score = price * 0.01  # Placeholder scoring

                    opportunities.append({
                        'symbol': symbol,
                        'price': price,
                        'score': score,
                        'conviction': 'HIGH' if score > 2.0 else 'MEDIUM'
                    })

                    print(f"{symbol:>6} | ${price:>8.2f} | Score: {score:>6.2f} | QUALITY APPROVED")

            except Exception as e:
                logging.error(f"Error analyzing {symbol}: {e}")

        print("-" * 50)
        print(f"Quality opportunities found: {len(opportunities)}")

        return sorted(opportunities, key=lambda x: x['score'], reverse=True)[:6]

    async def calculate_quality_positions(self, opportunities):
        """Calculate positions using quality assets only"""

        # Get buying power
        try:
            account = self.alpaca.get_account()
            buying_power = min(float(account.buying_power), float(account.cash) * 2)
        except:
            buying_power = 100000  # Fallback

        if buying_power < 10000:
            print("Insufficient buying power for quality trading")
            return []

        # Use 70% of buying power, larger positions
        usable_capital = buying_power * 0.70
        position_count = min(len(opportunities), 6)
        position_value = usable_capital / position_count

        print(f"\n=== QUALITY POSITION SIZING ===")
        print(f"Available Capital: ${buying_power:,.0f}")
        print(f"Deploying: ${usable_capital:,.0f} (70%)")
        print(f"Position Size: ${position_value:,.0f} each")
        print("-" * 50)

        positions = []

        for opp in opportunities:
            shares = max(1, int(position_value / opp['price']))
            actual_value = shares * opp['price']

            if actual_value >= 5000:  # Minimum $5K positions
                positions.append({
                    'symbol': opp['symbol'],
                    'shares': shares,
                    'price': opp['price'],
                    'value': actual_value,
                    'conviction': opp['conviction']
                })

                print(f"{opp['symbol']:>6} | {shares:>6} shares | ${opp['price']:>8.2f} | ${actual_value:>10,.0f} | {opp['conviction']}")

        total_deployment = sum(p['value'] for p in positions)
        print("-" * 50)
        print(f"TOTAL DEPLOYMENT: ${total_deployment:,.0f}")

        return positions

    async def execute_quality_trades(self, positions):
        """Execute trades with quality assets only"""

        if not positions:
            print("No quality positions to execute")
            return False

        print(f"\n=== EXECUTING QUALITY TRADES ===")
        print("Trading institutional-quality assets only")

        successful = 0
        total_deployed = 0

        for pos in positions:
            try:
                print(f"\nExecuting: {pos['symbol']} - {pos['shares']} shares (${pos['value']:,.0f})")

                order = self.alpaca.submit_order(
                    symbol=pos['symbol'],
                    qty=pos['shares'],
                    side='buy',
                    type='market',
                    time_in_force='day'
                )

                print(f"SUCCESS: Order {order.id}")
                print(f"  Institutional-quality asset: {pos['symbol']}")
                print(f"  Shares: {pos['shares']:,}")
                print(f"  Value: ${pos['value']:,.0f}")

                successful += 1
                total_deployed += pos['value']

                await asyncio.sleep(2)  # Brief delay

            except Exception as e:
                print(f"FAILED: {pos['symbol']} - {e}")

        print(f"\n=== QUALITY EXECUTION RESULTS ===")
        print(f"Successful: {successful}/{len(positions)}")
        print(f"Deployed: ${total_deployed:,.0f}")
        print(f"Asset Quality: INSTITUTIONAL GRADE")

        return successful > 0

    async def run_quality_trading(self):
        """Run quality-constrained autonomous trading"""

        print("QUALITY CONSTRAINED TRADER")
        print("=" * 50)
        print("Trading institutional-quality assets only")
        print("No penny stocks, warrants, or illiquid garbage")
        print("=" * 50)

        # Get quality opportunities
        opportunities = await self.get_quality_opportunities()

        if not opportunities:
            print("No quality opportunities found")
            return

        # Calculate positions
        positions = await self.calculate_quality_positions(opportunities)

        # Execute trades
        success = await self.execute_quality_trades(positions)

        if success:
            print("\nQUALITY TRADING COMPLETE!")
            print("Capital deployed in institutional-quality assets!")
            print("No more penny stock gambling!")

async def main():
    """Run quality-constrained trading"""
    trader = QualityConstrainedTrader()
    await trader.run_quality_trading()

if __name__ == "__main__":
    asyncio.run(main())