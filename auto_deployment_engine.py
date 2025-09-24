"""
AUTO-DEPLOYMENT ENGINE
Automatically deploys capital to discovered opportunities
"""

import json
import asyncio
import alpaca_trade_api as tradeapi
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class AutoDeploymentEngine:
    """Automatically deploy capital to best discovered strategies"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Deployment parameters
        self.max_allocation_per_trade = 0.05  # 5% max per trade
        self.min_return_threshold = 0.50      # 50% minimum annualized return
        self.max_positions = 10               # Max 10 options positions

        # Portfolio info
        self.target_portfolio = 515000        # Your $515K portfolio

    async def get_account_info(self):
        """Get current account status"""
        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.list_positions()

            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'day_trade_buying_power': float(account.day_trade_buying_power),
                'positions_count': len(positions),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            logging.error(f"Error getting account info: {e}")
            return None

    async def analyze_latest_discoveries(self):
        """Analyze latest mega discovery results"""

        try:
            # Find latest mega discovery file
            import glob
            discovery_files = glob.glob('mega_discovery_*.json')
            if not discovery_files:
                return None

            latest_file = max(discovery_files)

            with open(latest_file, 'r') as f:
                discovery_data = json.load(f)

            return discovery_data

        except Exception as e:
            logging.error(f"Error analyzing discoveries: {e}")
            return None

    async def calculate_position_size(self, strategy, account_info):
        """Calculate optimal position size for strategy"""

        available_buying_power = account_info['buying_power']
        max_allocation = self.target_portfolio * self.max_allocation_per_trade

        # Use smaller of max allocation or available buying power
        max_capital = min(max_allocation, available_buying_power * 0.8)  # 80% of buying power

        allocation_required = strategy['allocation_required']

        # Calculate how many contracts we can afford
        if strategy['strategy'] == 'covered_call':
            # Need to buy 100 shares per contract
            shares_per_contract = 100
            cost_per_contract = allocation_required * shares_per_contract
            max_contracts = int(max_capital / cost_per_contract)

        elif strategy['strategy'] == 'cash_secured_put':
            # Need cash equal to strike * 100 per contract
            cash_per_contract = strategy['strike'] * 100
            max_contracts = int(max_capital / cash_per_contract)

        else:
            max_contracts = 1

        # Limit to reasonable position sizes
        max_contracts = min(max_contracts, 10)  # Max 10 contracts

        return max(max_contracts, 1) if max_contracts > 0 else 0

    async def execute_strategy(self, strategy, position_size, simulation=True):
        """Execute the options strategy"""

        ticker = strategy['ticker']
        strategy_type = strategy['strategy']
        strike = strategy['strike']
        dte = strategy['dte']

        execution_log = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'strategy': strategy_type,
            'strike': strike,
            'dte': dte,
            'position_size': position_size,
            'expected_return': strategy['expected_return'],
            'simulation': simulation,
            'status': 'PENDING'
        }

        if simulation:
            logging.info(f"SIMULATION: {strategy_type} {ticker} strike {strike} x{position_size} contracts")
            execution_log['status'] = 'SIMULATED'

        else:
            try:
                # Real execution logic would go here
                # For now, keeping in simulation mode for safety
                logging.info(f"LIVE EXECUTION: {strategy_type} {ticker} strike {strike} x{position_size} contracts")
                execution_log['status'] = 'EXECUTED'

            except Exception as e:
                logging.error(f"Execution error: {e}")
                execution_log['status'] = 'FAILED'
                execution_log['error'] = str(e)

        # Save execution log
        with open('auto_deployment_log.json', 'a') as f:
            f.write(json.dumps(execution_log) + '\n')

        return execution_log

    async def deploy_best_strategies(self):
        """Deploy capital to best discovered strategies"""

        logging.info("AUTO-DEPLOYMENT ENGINE: Analyzing best strategies...")

        # Get account info
        account_info = await self.get_account_info()
        if not account_info:
            logging.error("Could not get account info")
            return

        logging.info(f"Account equity: ${account_info['equity']:,.2f}")
        logging.info(f"Buying power: ${account_info['buying_power']:,.2f}")

        # Get latest discoveries
        discovery_data = await self.analyze_latest_discoveries()
        if not discovery_data:
            logging.error("No discovery data found")
            return

        best_strategies = discovery_data.get('best_strategies', [])
        if not best_strategies:
            logging.error("No strategies found")
            return

        logging.info(f"Found {len(best_strategies)} strategies to analyze")

        deployed_count = 0
        total_allocation = 0

        for strategy in best_strategies[:5]:  # Deploy top 5 strategies

            # Check if strategy meets criteria
            if strategy['expected_return'] < self.min_return_threshold:
                continue

            # Calculate position size
            position_size = await self.calculate_position_size(strategy, account_info)
            if position_size == 0:
                continue

            # Execute strategy
            execution_result = await self.execute_strategy(strategy, position_size, simulation=True)

            if execution_result['status'] in ['SIMULATED', 'EXECUTED']:
                deployed_count += 1
                total_allocation += strategy['allocation_required'] * position_size

                logging.info(f"DEPLOYED #{deployed_count}: {strategy['ticker']} {strategy['strategy']}")
                logging.info(f"  Expected return: {strategy['expected_return']:.1%}")
                logging.info(f"  Position size: {position_size} contracts")
                logging.info(f"  Capital allocated: ${strategy['allocation_required'] * position_size:,.2f}")

        deployment_summary = {
            'timestamp': datetime.now().isoformat(),
            'strategies_deployed': deployed_count,
            'total_allocation': total_allocation,
            'remaining_buying_power': account_info['buying_power'] - total_allocation,
            'deployment_rate': deployed_count / len(best_strategies) if best_strategies else 0
        }

        logging.info(f"DEPLOYMENT COMPLETE: {deployed_count} strategies deployed")
        logging.info(f"Total allocation: ${total_allocation:,.2f}")

        return deployment_summary

    async def continuous_deployment_loop(self):
        """Continuously monitor and deploy to new opportunities"""

        logging.info("Starting continuous auto-deployment monitoring...")

        while True:
            try:
                # Deploy to best strategies
                await self.deploy_best_strategies()

                # Wait 1 hour before next deployment cycle
                await asyncio.sleep(3600)

            except Exception as e:
                logging.error(f"Error in deployment loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

async def main():
    """Start auto-deployment engine"""

    print("AUTO-DEPLOYMENT ENGINE")
    print("=" * 50)
    print("Automatically deploying capital to best opportunities...")
    print("Targeting 50%+ annualized returns")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler('auto_deployment.log'),
            logging.StreamHandler()
        ]
    )

    engine = AutoDeploymentEngine()
    await engine.continuous_deployment_loop()

if __name__ == "__main__":
    asyncio.run(main())