"""
ACTIVATE AGGRESSIVE REBALANCING
Free up capital from ETFs and deploy to high-return options strategies
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('aggressive_rebalancing.log'),
        logging.StreamHandler()
    ]
)

class AggressiveRebalancer:
    """Rebalance ETFs to options strategies for 40% monthly ROI"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Rebalancing targets
        self.etf_trim_percentage = 0.30  # Sell 30% of ETFs initially
        self.target_options_allocation = 150000  # $150K to options strategies

    async def get_current_positions(self):
        """Get current ETF positions"""

        try:
            positions = self.alpaca.list_positions()

            etf_positions = []
            total_etf_value = 0

            for pos in positions:
                symbol = pos.symbol
                market_value = float(pos.market_value)
                qty = int(pos.qty)
                current_price = float(pos.current_price)
                unrealized_plpc = float(pos.unrealized_plpc)

                # Identify ETF positions
                if symbol in ['SPY', 'QQQ', 'IWM', 'TQQQ', 'SOXL', 'UPRO', 'VTI', 'VOO']:
                    etf_positions.append({
                        'symbol': symbol,
                        'qty': qty,
                        'market_value': market_value,
                        'current_price': current_price,
                        'unrealized_plpc': unrealized_plpc
                    })
                    total_etf_value += market_value

            logging.info(f"Total ETF value: ${total_etf_value:,.0f}")
            logging.info(f"ETF positions found: {len(etf_positions)}")

            return etf_positions, total_etf_value

        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return [], 0

    async def identify_trim_targets(self, etf_positions, total_etf_value):
        """Identify which ETFs to trim for rebalancing"""

        target_cash_needed = self.target_options_allocation
        trim_targets = []

        # Sort by performance (trim underperformers first)
        etf_positions.sort(key=lambda x: x['unrealized_plpc'])

        cash_to_generate = 0

        for pos in etf_positions:
            if cash_to_generate >= target_cash_needed:
                break

            symbol = pos['symbol']
            market_value = pos['market_value']

            # Calculate trim amount (30% of position or remaining need)
            trim_percentage = min(self.etf_trim_percentage,
                                (target_cash_needed - cash_to_generate) / market_value)

            trim_value = market_value * trim_percentage
            trim_shares = int(pos['qty'] * trim_percentage)

            if trim_shares > 0:
                trim_targets.append({
                    'symbol': symbol,
                    'current_qty': pos['qty'],
                    'trim_shares': trim_shares,
                    'trim_value': trim_value,
                    'trim_percentage': trim_percentage,
                    'reason': f'REBALANCE_TO_OPTIONS_{pos["unrealized_plpc"]:+.1%}'
                })

                cash_to_generate += trim_value

        logging.info(f"Trim targets identified: {len(trim_targets)}")
        logging.info(f"Cash to generate: ${cash_to_generate:,.0f}")

        return trim_targets

    async def execute_etf_trims(self, trim_targets):
        """Execute ETF sell orders to free up capital"""

        logging.info("EXECUTING ETF REBALANCING")
        logging.info("=" * 40)

        executed_sales = []
        total_freed_capital = 0

        for target in trim_targets:
            symbol = target['symbol']
            trim_shares = target['trim_shares']

            try:
                # Check if market is open
                clock = self.alpaca.get_clock()
                if not clock.is_open:
                    logging.warning("Market closed - would queue orders for market open")
                    # In real execution, would queue for market open
                    continue

                logging.info(f"SELLING {trim_shares} shares of {symbol}")
                logging.info(f"  Reason: {target['reason']}")
                logging.info(f"  Estimated value: ${target['trim_value']:,.0f}")

                # Submit sell order
                order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=trim_shares,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )

                executed_sales.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'shares_sold': trim_shares,
                    'estimated_value': target['trim_value'],
                    'order_id': order.id,
                    'reason': target['reason']
                })

                total_freed_capital += target['trim_value']
                logging.info(f"✅ SELL ORDER SUBMITTED: {order.id}")

            except Exception as e:
                logging.error(f"Error selling {symbol}: {e}")
                continue

        logging.info("=" * 40)
        logging.info(f"REBALANCING COMPLETE")
        logging.info(f"Orders submitted: {len(executed_sales)}")
        logging.info(f"Estimated freed capital: ${total_freed_capital:,.0f}")

        # Save rebalancing record
        rebalancing_record = {
            'timestamp': datetime.now().isoformat(),
            'rebalancing_type': 'ETF_TO_OPTIONS',
            'target_allocation': self.target_options_allocation,
            'executed_sales': executed_sales,
            'total_freed_capital': total_freed_capital
        }

        with open('rebalancing_execution.json', 'w') as f:
            json.dump(rebalancing_record, f, indent=2, default=str)

        return executed_sales, total_freed_capital

    async def deploy_to_options_strategies(self, freed_capital):
        """Deploy freed capital to discovered options strategies"""

        if freed_capital < 10000:  # Need at least $10K
            logging.warning(f"Insufficient freed capital: ${freed_capital:,.0f}")
            return False

        logging.info("DEPLOYING TO OPTIONS STRATEGIES")
        logging.info("=" * 40)

        try:
            # Load discovered strategies
            with open('mega_discovery_20250918_2114.json', 'r') as f:
                discovery = json.load(f)

            strategies = discovery.get('best_strategies', [])[:3]  # Top 3

            if not strategies:
                logging.error("No strategies found for deployment")
                return False

            deployment_plan = []
            capital_per_strategy = freed_capital / len(strategies)

            for strategy in strategies:
                ticker = strategy['ticker']
                strategy_type = strategy['strategy']
                expected_return = strategy['expected_return']

                deployment_plan.append({
                    'ticker': ticker,
                    'strategy': strategy_type,
                    'allocation': capital_per_strategy,
                    'expected_return': expected_return,
                    'execution_ready': True
                })

                logging.info(f"STRATEGY: {ticker} {strategy_type}")
                logging.info(f"  Allocation: ${capital_per_strategy:,.0f}")
                logging.info(f"  Expected Return: {expected_return:.1%}")

            # Save deployment plan
            deployment_record = {
                'timestamp': datetime.now().isoformat(),
                'freed_capital': freed_capital,
                'strategies_count': len(deployment_plan),
                'deployment_plan': deployment_plan,
                'status': 'READY_FOR_OPTIONS_EXECUTION'
            }

            with open('options_deployment_plan.json', 'w') as f:
                json.dump(deployment_record, f, indent=2, default=str)

            logging.info("=" * 40)
            logging.info("OPTIONS DEPLOYMENT PLAN READY")
            logging.info(f"Capital available: ${freed_capital:,.0f}")
            logging.info(f"Strategies ready: {len(deployment_plan)}")

            return True

        except Exception as e:
            logging.error(f"Error creating deployment plan: {e}")
            return False

    async def execute_aggressive_rebalancing(self):
        """Main rebalancing execution"""

        logging.info("AGGRESSIVE REBALANCING ACTIVATED")
        logging.info("=" * 50)
        logging.info("Trimming ETFs to deploy to options strategies")
        logging.info("Target: 40% monthly ROI")
        logging.info("=" * 50)

        # Get current positions
        etf_positions, total_etf_value = await self.get_current_positions()

        if not etf_positions:
            logging.error("No ETF positions found")
            return False

        # Identify trim targets
        trim_targets = await self.identify_trim_targets(etf_positions, total_etf_value)

        if not trim_targets:
            logging.error("No trim targets identified")
            return False

        # Execute ETF sales
        executed_sales, freed_capital = await self.execute_etf_trims(trim_targets)

        if freed_capital > 0:
            # Deploy to options strategies
            await self.deploy_to_options_strategies(freed_capital)

        logging.info("AGGRESSIVE REBALANCING EXECUTION COMPLETE")
        return True

async def main():
    """Activate aggressive rebalancing"""

    print("AGGRESSIVE REBALANCING ACTIVATION")
    print("=" * 50)
    print("Trimming ETFs → Deploying to Options")
    print("Target: 40% Monthly ROI")
    print("=" * 50)

    rebalancer = AggressiveRebalancer()
    await rebalancer.execute_aggressive_rebalancing()

if __name__ == "__main__":
    asyncio.run(main())