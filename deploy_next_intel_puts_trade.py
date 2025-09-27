#!/usr/bin/env python3
"""
DEPLOY NEXT INTEL-PUTS-STYLE TRADE
Execute the 4-trade strategy identified by hybrid conviction-genetic trader
Deploy $826K of $1.96M available buying power for 25-50% monthly returns
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import time
from datetime import datetime

class DeployIntelPutsStyle:
    """Deploy the next Intel-puts-style concentrated positions"""

    def __init__(self):
        load_dotenv()

        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL')
        )

        # The 4-trade strategy from hybrid conviction-genetic trader
        self.trades = [
            {
                'symbol': 'META',
                'type': 'STOCK',
                'action': 'BUY',
                'quantity': 432,
                'allocation_pct': 35.0,
                'target_amount': 311040,
                'source': 'genetic_algorithm',
                'score': 2.160,
                'reasoning': 'Highest genetic score (7.20) - AI-optimized quality stock'
            },
            {
                'symbol': 'SPY',
                'type': 'STOCK',
                'action': 'BUY',
                'quantity': 376,
                'allocation_pct': 28.0,
                'target_amount': 248634,
                'source': 'genetic_algorithm',
                'score': 1.984,
                'reasoning': 'Fed meeting catalyst - 17 days to rate cut decision'
            },
            {
                'symbol': 'AAPL250103C00265000',  # AAPL $265 CALL Jan 3 2025
                'type': 'OPTION',
                'action': 'BUY',
                'quantity': 235,
                'allocation_pct': 20.0,
                'target_amount': 177801,
                'source': 'conviction_scanner',
                'score': 1.884,
                'reasoning': 'Product launch catalyst - 269.2% potential in 13 days'
            },
            {
                'symbol': 'GOOGL250103C00252000',  # GOOGL $252 CALL Jan 3 2025
                'type': 'OPTION',
                'action': 'BUY',
                'quantity': 123,
                'allocation_pct': 10.0,
                'target_amount': 88560,
                'source': 'conviction_scanner',
                'score': 1.867,
                'reasoning': 'Q4 earnings catalyst - 266.7% potential in 20 days'
            }
        ]

        self.total_deployment = 826035

    def check_account_readiness(self):
        """Verify account has sufficient buying power"""
        try:
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            cash = float(account.cash)

            print("ACCOUNT READINESS CHECK")
            print("=" * 50)
            print(f"Available Buying Power: ${buying_power:,.2f}")
            print(f"Cash: ${cash:,.2f}")
            print(f"Required Capital: ${self.total_deployment:,.2f}")
            print(f"Deployment Ratio: {(self.total_deployment/buying_power)*100:.1f}% of buying power")

            if buying_power >= self.total_deployment:
                print("[OK] ACCOUNT READY for Intel-puts-style deployment")
                return True
            else:
                shortage = self.total_deployment - buying_power
                print(f"[ERROR] INSUFFICIENT CAPITAL: Need ${shortage:,.2f} more")
                return False

        except Exception as e:
            print(f"Error checking account: {e}")
            return False

    def execute_trade(self, trade):
        """Execute individual trade with Intel-puts-style precision"""
        symbol = trade['symbol']
        quantity = trade['quantity']
        trade_type = trade['type']
        allocation_pct = trade['allocation_pct']
        reasoning = trade['reasoning']

        print(f"\n[EXECUTING] {symbol} - {trade_type}")
        print(f"  Allocation: {allocation_pct}% (${trade['target_amount']:,.0f})")
        print(f"  Quantity: {quantity}")
        print(f"  Strategy: {reasoning}")

        try:
            if trade_type == 'STOCK':
                # Stock purchase - market order for immediate execution
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
            elif trade_type == 'OPTION':
                # Options purchase - market order for immediate fill
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )

            print(f"  [OK] ORDER SUBMITTED: {order.id}")
            print(f"  Status: {order.status}")

            return {
                'success': True,
                'order_id': order.id,
                'symbol': symbol,
                'quantity': quantity,
                'allocation': allocation_pct
            }

        except Exception as e:
            print(f"  [ERROR] EXECUTION FAILED: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }

    def deploy_intel_puts_strategy(self):
        """Execute complete Intel-puts-style 4-trade deployment"""
        print("DEPLOYING NEXT INTEL-PUTS-STYLE TRADE")
        print("=" * 80)
        print("Based on hybrid conviction-genetic analysis")
        print("Target: 25-50% monthly returns with quality concentrated positions")
        print("Total Deployment: ${:,.2f} of $1.96M available".format(self.total_deployment))
        print("=" * 80)

        # Verify account readiness
        if not self.check_account_readiness():
            return

        print(f"\n[PLAN] TRADE PLAN SUMMARY:")
        print(f"{'='*60}")
        for trade in self.trades:
            print(f"{trade['allocation_pct']:>5.1f}% {trade['symbol']:15} {trade['type']:6} {trade['source']:15}")
        print(f"{'='*60}")
        print(f"TOTAL: 93.0% deployment (7% cash reserve)")

        # Execute all trades
        executed_trades = []
        total_executed_value = 0

        for i, trade in enumerate(self.trades, 1):
            print(f"\n[TRADE] TRADE {i}/4:")
            result = self.execute_trade(trade)
            executed_trades.append(result)

            if result['success']:
                total_executed_value += trade['target_amount']

            # Brief pause between trades
            time.sleep(3)

        # Wait for fills
        print(f"\n[WAIT] Waiting for order fills...")
        time.sleep(15)

        # Summary
        successful_trades = [t for t in executed_trades if t['success']]
        failed_trades = [t for t in executed_trades if not t['success']]

        print(f"\n[RESULTS] DEPLOYMENT RESULTS:")
        print(f"{'='*60}")
        print(f"Successful Trades: {len(successful_trades)}/4")
        print(f"Capital Deployed: ${total_executed_value:,.2f}")

        if successful_trades:
            print(f"\n[SUCCESS] SUCCESSFUL EXECUTIONS:")
            for trade in successful_trades:
                matching_trade = next(t for t in self.trades if t['symbol'] == trade['symbol'])
                print(f"  {trade['symbol']:15} - {trade['allocation']:5.1f}% - Order: {trade['order_id']}")

        if failed_trades:
            print(f"\n[FAILED] FAILED EXECUTIONS:")
            for trade in failed_trades:
                print(f"  {trade['symbol']:15} - Error: {trade['error']}")

        if len(successful_trades) >= 2:
            print(f"\n[SUCCESS] INTEL-PUTS-STYLE DEPLOYMENT SUCCESSFUL!")
            print(f"Ready to target 25-50% monthly returns")
            print(f"Position concentration matches your winning Intel puts strategy")
        else:
            print(f"\n[WARNING] PARTIAL DEPLOYMENT - May need manual intervention")

        return executed_trades

def main():
    """Execute the Intel-puts-style deployment"""
    deployer = DeployIntelPutsStyle()
    deployer.deploy_intel_puts_strategy()

if __name__ == "__main__":
    main()