#!/usr/bin/env python3
"""
PORTFOLIO CLEANUP MANAGER
Cleans up failed penny stock positions and prepares for concentrated strategy
Gets rid of the -12% loss positions to deploy fresh capital
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - CLEANUP - %(message)s')

class PortfolioCleanupManager:
    """Manages portfolio cleanup and position consolidation"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Positions to avoid (quality assets we want to keep)
        self.keep_positions = {
            'META', 'AAPL', 'GOOGL', 'SPY', 'QQQ', 'NVDA', 'MSFT',
            'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG', 'HD'
        }

        # Penny stocks and warrants to definitely liquidate
        self.liquidate_immediately = {
            'UAVS', 'PHUN', 'DWAC', 'BBIG', 'SPRT', 'GREE', 'IRNT',
            'PROG', 'ATER', 'CEI', 'FAMI', 'RELI', 'ESSC', 'LGVN'
        }

        logging.info("PORTFOLIO CLEANUP MANAGER INITIALIZED")
        logging.info("Ready to clean up -12% loss positions")

    async def analyze_current_positions(self):
        """Analyze current portfolio positions"""

        print("PORTFOLIO POSITION ANALYSIS")
        print("=" * 60)
        print("Analyzing current positions for cleanup opportunities")
        print("=" * 60)

        try:
            # Get current positions
            positions = self.alpaca.list_positions()

            if not positions:
                print("No current positions found")
                return [], []

            # Get account info
            account = self.alpaca.get_account()
            total_equity = float(account.equity)
            total_pl = float(account.total_pl)

            print(f"Account Equity: ${total_equity:,.2f}")
            print(f"Total P&L: ${total_pl:,.2f} ({total_pl/total_equity*100:+.1f}%)")

            keep_positions = []
            liquidate_positions = []

            print(f"\n=== CURRENT POSITIONS ANALYSIS ===")
            print("Symbol | Qty | Market Value | Unrealized P&L | % P&L | Action")
            print("-" * 70)

            for position in positions:
                symbol = position.symbol
                qty = int(position.qty)
                market_value = float(position.market_value)
                unrealized_pl = float(position.unrealized_pl)
                unrealized_plpc = float(position.unrealized_plpc) * 100

                # Determine action
                if symbol in self.keep_positions and unrealized_plpc > -20:
                    action = "KEEP"
                    keep_positions.append(position)
                elif symbol in self.liquidate_immediately:
                    action = "LIQUIDATE"
                    liquidate_positions.append(position)
                elif unrealized_plpc < -15:  # More than 15% loss
                    action = "LIQUIDATE"
                    liquidate_positions.append(position)
                elif market_value < 1000:  # Small positions under $1K
                    action = "LIQUIDATE"
                    liquidate_positions.append(position)
                else:
                    action = "REVIEW"
                    keep_positions.append(position)

                print(f"{symbol:>6} | {qty:>3} | ${market_value:>8,.0f} | ${unrealized_pl:>8,.0f} | {unrealized_plpc:>+5.1f}% | {action}")

            print("-" * 70)
            print(f"KEEP: {len(keep_positions)} positions")
            print(f"LIQUIDATE: {len(liquidate_positions)} positions")

            return keep_positions, liquidate_positions

        except Exception as e:
            print(f"Error analyzing positions: {e}")
            return [], []

    async def calculate_cleanup_impact(self, liquidate_positions):
        """Calculate the impact of liquidating positions"""

        if not liquidate_positions:
            return {}

        print(f"\n=== CLEANUP IMPACT ANALYSIS ===")
        print("Calculating freed capital and realized losses")
        print("-" * 50)

        total_market_value = sum(float(pos.market_value) for pos in liquidate_positions)
        total_unrealized_pl = sum(float(pos.unrealized_pl) for pos in liquidate_positions)

        # Estimate freed capital (assuming 90% fill on liquidation)
        freed_capital = total_market_value * 0.90
        realized_loss = abs(total_unrealized_pl) if total_unrealized_pl < 0 else 0

        cleanup_impact = {
            'positions_to_liquidate': len(liquidate_positions),
            'current_market_value': total_market_value,
            'estimated_freed_capital': freed_capital,
            'realized_loss': realized_loss,
            'net_capital_gain': freed_capital - realized_loss
        }

        print(f"Positions to liquidate: {cleanup_impact['positions_to_liquidate']}")
        print(f"Current market value: ${cleanup_impact['current_market_value']:,.0f}")
        print(f"Estimated freed capital: ${cleanup_impact['estimated_freed_capital']:,.0f}")
        print(f"Realized loss: ${cleanup_impact['realized_loss']:,.0f}")
        print(f"Net capital for redeployment: ${cleanup_impact['net_capital_gain']:,.0f}")

        return cleanup_impact

    async def generate_liquidation_orders(self, liquidate_positions):
        """Generate liquidation orders for cleanup"""

        if not liquidate_positions:
            print("No positions to liquidate")
            return []

        print(f"\n=== LIQUIDATION ORDER GENERATION ===")
        print("Creating market orders to clean up losing positions")
        print("-" * 60)

        liquidation_orders = []

        for position in liquidate_positions:
            symbol = position.symbol
            qty = abs(int(position.qty))
            side = "sell" if int(position.qty) > 0 else "buy"

            order = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': 'market',
                'time_in_force': 'day',
                'reason': 'portfolio_cleanup'
            }

            liquidation_orders.append(order)

            print(f"{side.upper()} {qty} {symbol} - Market Order")

        print("-" * 60)
        print(f"Total liquidation orders: {len(liquidation_orders)}")

        return liquidation_orders

    async def execute_portfolio_cleanup(self, execute_trades=False):
        """Execute complete portfolio cleanup"""

        print("PORTFOLIO CLEANUP MANAGER")
        print("=" * 80)
        print("Cleaning up -12% loss positions for fresh capital deployment")
        print("Preparing for Intel-puts-style concentrated strategy")
        print("=" * 80)

        # Step 1: Analyze current positions
        keep_positions, liquidate_positions = await self.analyze_current_positions()

        if not liquidate_positions:
            print("No positions identified for cleanup")
            return {}

        # Step 2: Calculate cleanup impact
        cleanup_impact = await self.calculate_cleanup_impact(liquidate_positions)

        # Step 3: Generate liquidation orders
        liquidation_orders = await self.generate_liquidation_orders(liquidate_positions)

        # Step 4: Execute trades (if requested)
        executed_orders = []
        if execute_trades:
            print(f"\n=== EXECUTING LIQUIDATION ORDERS ===")
            print("EXECUTING REAL TRADES - LIQUIDATING POSITIONS")
            print("-" * 60)

            for order in liquidation_orders:
                try:
                    submitted_order = self.alpaca.submit_order(
                        symbol=order['symbol'],
                        qty=order['qty'],
                        side=order['side'],
                        type=order['type'],
                        time_in_force=order['time_in_force']
                    )

                    executed_orders.append({
                        'symbol': order['symbol'],
                        'qty': order['qty'],
                        'side': order['side'],
                        'order_id': submitted_order.id,
                        'status': 'submitted'
                    })

                    print(f"✅ {order['side'].upper()} {order['qty']} {order['symbol']} - Order ID: {submitted_order.id}")

                except Exception as e:
                    print(f"❌ Failed to submit {order['symbol']}: {e}")

            print(f"Executed {len(executed_orders)} liquidation orders")

        else:
            print(f"\n=== DRY RUN MODE ===")
            print("Set execute_trades=True to execute real liquidation orders")

        # Save cleanup report
        cleanup_report = {
            'cleanup_timestamp': datetime.now().isoformat(),
            'cleanup_impact': cleanup_impact,
            'positions_to_keep': [{'symbol': p.symbol, 'qty': p.qty, 'market_value': p.market_value} for p in keep_positions],
            'positions_to_liquidate': [{'symbol': p.symbol, 'qty': p.qty, 'market_value': p.market_value} for p in liquidate_positions],
            'liquidation_orders': liquidation_orders,
            'executed_orders': executed_orders,
            'execute_mode': execute_trades
        }

        filename = f'portfolio_cleanup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(cleanup_report, f, indent=2)

        print("=" * 80)
        print("PORTFOLIO CLEANUP ANALYSIS COMPLETE")
        print(f"Cleanup report saved to: {filename}")
        if cleanup_impact:
            print(f"Potential freed capital: ${cleanup_impact.get('estimated_freed_capital', 0):,.0f}")
        print("Ready to deploy concentrated Intel-puts-style strategy!")
        print("=" * 80)

        return cleanup_report

async def main():
    """Run portfolio cleanup analysis"""
    cleanup_manager = PortfolioCleanupManager()

    # First run in analysis mode
    print("Running cleanup analysis (dry run)...")
    report = await cleanup_manager.execute_portfolio_cleanup(execute_trades=False)

    # Ask user if they want to execute
    if report.get('cleanup_impact', {}).get('positions_to_liquidate', 0) > 0:
        print(f"\nFound {report['cleanup_impact']['positions_to_liquidate']} positions to liquidate")
        print("Run with execute_trades=True to execute real liquidation orders")

    return report

if __name__ == "__main__":
    asyncio.run(main())