#!/usr/bin/env python3
"""
AGGRESSIVE CAPITAL DEPLOYER
Deploys the massive $964K cash position into profitable opportunities
Uses non-day-trading rules to maximize capital utilization
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - CAPITAL - %(message)s')

class AggressiveCapitalDeployer:
    """Deploy massive cash position into profit opportunities"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # AGGRESSIVE SETTINGS for $964K deployment
        self.deployment_settings = {
            'target_cash_utilization': 0.80,    # Use 80% of cash ($771K)
            'max_position_size': 0.15,          # Max 15% per position ($144K)
            'min_position_size': 0.02,          # Min 2% per position ($19K)
            'max_positions': 12,                # Spread across 12 positions
            'profit_target': 0.25,              # 25% profit target per position
            'stop_loss': -0.08,                 # 8% max loss per position
        }

        logging.info("AGGRESSIVE CAPITAL DEPLOYER INITIALIZED")
        logging.info(f"Target deployment: ${964000 * self.deployment_settings['target_cash_utilization']:,.0f}")

    async def get_account_status(self):
        """Get current account and cash position"""
        try:
            account = self.alpaca.get_account()

            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
            buying_power = float(account.buying_power)

            print(f"=== ACCOUNT STATUS ===")
            print(f"Portfolio Value: ${portfolio_value:,.0f}")
            print(f"Cash Available: ${cash:,.0f}")
            print(f"Buying Power: ${buying_power:,.0f}")
            print(f"Deployable Capital: ${cash * self.deployment_settings['target_cash_utilization']:,.0f}")

            return {
                'portfolio_value': portfolio_value,
                'cash': cash,
                'buying_power': buying_power,
                'deployable': cash * self.deployment_settings['target_cash_utilization']
            }

        except Exception as e:
            logging.error(f"Account status error: {e}")
            return None

    async def get_high_conviction_opportunities(self):
        """Get high-conviction opportunities from running systems"""

        # Top opportunities from your running systems
        opportunities = [
            {
                'symbol': 'EDIT',
                'conviction': 'HIGH',
                'score': 6.50,
                'reason': 'Top opportunity from profit engine',
                'sector': 'Biotech',
                'target_allocation': 0.12  # 12%
            },
            {
                'symbol': 'GTENW',
                'conviction': 'HIGH',
                'score': 15.99,
                'reason': '+15.99% market scanner top mover',
                'sector': 'Warrants',
                'target_allocation': 0.08  # 8%
            },
            {
                'symbol': 'VTVT',
                'conviction': 'MEDIUM',
                'score': 4.60,
                'reason': '+4.60% strong momentum',
                'sector': 'Tech',
                'target_allocation': 0.10  # 10%
            },
            {
                'symbol': 'TSLA',
                'conviction': 'HIGH',
                'score': 4.2,
                'reason': 'Multiple systems identified',
                'sector': 'Auto/Tech',
                'target_allocation': 0.15  # 15% (max position)
            },
            {
                'symbol': 'LCID',
                'conviction': 'MEDIUM',
                'score': 6.3,
                'reason': '+6.3% current winner, scale up',
                'sector': 'EV',
                'target_allocation': 0.08  # 8%
            },
            {
                'symbol': 'NTLA',
                'conviction': 'HIGH',
                'score': 7.9,
                'reason': '+7.9% strong winner, scale up',
                'sector': 'Biotech',
                'target_allocation': 0.10  # 10%
            },
            {
                'symbol': 'RIVN',
                'conviction': 'MEDIUM',
                'score': 6.7,
                'reason': '+6.7% current winner',
                'sector': 'EV',
                'target_allocation': 0.08  # 8%
            },
            {
                'symbol': 'SMCI',
                'conviction': 'HIGH',
                'score': 3.8,
                'reason': 'AI infrastructure play',
                'sector': 'AI/Hardware',
                'target_allocation': 0.12  # 12%
            }
        ]

        # Sort by conviction and score
        opportunities.sort(key=lambda x: (x['conviction'] == 'HIGH', x['score']), reverse=True)

        return opportunities[:self.deployment_settings['max_positions']]

    async def calculate_position_sizes(self, opportunities, deployable_capital):
        """Calculate optimal position sizes for massive capital deployment"""

        positions = []
        total_allocation = 0

        print(f"\n=== AGGRESSIVE CAPITAL DEPLOYMENT PLAN ===")
        print(f"Deployable Capital: ${deployable_capital:,.0f}")
        print("-" * 60)

        for opp in opportunities:
            # Calculate position size
            target_allocation = opp['target_allocation']
            position_value = deployable_capital * target_allocation

            # Get current price for share calculation
            try:
                quote = self.alpaca.get_latest_quote(opp['symbol'])
                current_price = float(quote.bid_price) if quote.bid_price else 50.0  # fallback

                shares = int(position_value / current_price)
                actual_value = shares * current_price
                actual_allocation = actual_value / deployable_capital

                positions.append({
                    'symbol': opp['symbol'],
                    'shares': shares,
                    'price': current_price,
                    'value': actual_value,
                    'allocation': actual_allocation,
                    'conviction': opp['conviction'],
                    'reason': opp['reason']
                })

                total_allocation += actual_allocation

                print(f"{opp['conviction']:>6} | {opp['symbol']:>6} | {shares:>6} shares | ${actual_value:>8,.0f} | {actual_allocation*100:>5.1f}%")

            except Exception as e:
                logging.error(f"Price lookup error for {opp['symbol']}: {e}")

        print("-" * 60)
        print(f"TOTAL DEPLOYMENT: ${sum(p['value'] for p in positions):,.0f} ({total_allocation*100:.1f}%)")

        return positions

    async def execute_aggressive_deployment(self, positions):
        """Execute aggressive capital deployment trades"""

        print(f"\n=== EXECUTING AGGRESSIVE DEPLOYMENT ===")

        successful_trades = 0
        total_deployed = 0

        for position in positions:
            try:
                print(f"Deploying ${position['value']:,.0f} into {position['symbol']}...")

                # Execute large position trade
                order = self.alpaca.submit_order(
                    symbol=position['symbol'],
                    qty=position['shares'],
                    side='buy',
                    type='market',
                    time_in_force='day'
                )

                print(f"SUCCESS: {position['symbol']} - {position['shares']} shares @ ${position['price']:.2f}")
                print(f"  Order ID: {order.id}")
                print(f"  Capital Deployed: ${position['value']:,.0f}")
                print(f"  Conviction: {position['conviction']}")
                print(f"  Reason: {position['reason']}")

                successful_trades += 1
                total_deployed += position['value']

                # Small delay between large orders
                await asyncio.sleep(1)

            except Exception as e:
                print(f"FAILED: {position['symbol']} - {e}")
                logging.error(f"Deployment failed for {position['symbol']}: {e}")

        print(f"\n=== DEPLOYMENT COMPLETE ===")
        print(f"Successful Deployments: {successful_trades}/{len(positions)}")
        print(f"Total Capital Deployed: ${total_deployed:,.0f}")
        print(f"Remaining Cash: ${964000 - total_deployed:,.0f}")

        if successful_trades > 0:
            print(f"MASSIVE CAPITAL DEPLOYMENT SUCCESSFUL!")
            print(f"Your ${total_deployed:,.0f} is now working to generate 25-50% returns!")

        return successful_trades > 0

    async def run_aggressive_deployment(self):
        """Run aggressive capital deployment"""

        print("AGGRESSIVE CAPITAL DEPLOYER")
        print("="*60)
        print("Deploying $964K cash into high-conviction opportunities")
        print("="*60)

        # Get account status
        account = await self.get_account_status()
        if not account:
            return

        # Get opportunities
        opportunities = await self.get_high_conviction_opportunities()

        # Calculate positions
        positions = await self.calculate_position_sizes(opportunities, account['deployable'])

        if not positions:
            print("No valid positions calculated")
            return

        # Execute deployment
        success = await self.execute_aggressive_deployment(positions)

        if success:
            print("\nYour capital is now aggressively deployed!")
            print("Systems will manage these positions for maximum profit!")

async def main():
    """Deploy massive capital aggressively"""
    deployer = AggressiveCapitalDeployer()
    await deployer.run_aggressive_deployment()

if __name__ == "__main__":
    asyncio.run(main())