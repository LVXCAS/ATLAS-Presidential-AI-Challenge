#!/usr/bin/env python3
"""
COMPREHENSIVE PORTFOLIO CLEANUP
Complete portfolio reset and consolidation for concentrated strategy
Get rid of the clutter, keep the winners, prepare for Intel-puts-style concentration
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

class ComprehensivePortfolioCleanup:
    """Complete portfolio cleanup and consolidation"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Positions we definitely want to keep (quality winners)
        self.keep_symbols = {
            'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA',
            'SPY', 'QQQ', 'JPM', 'JNJ'  # Quality blue chips
        }

        # Positions to definitely liquidate (penny stocks, warrants, major losers)
        self.liquidate_symbols = {
            'AXIL', 'BCTXZ', 'BLMZ', 'CDIOW', 'CPHC', 'CRDU', 'CTSO',
            'CVV', 'DSS', 'DWTX', 'ENPX', 'FAMI', 'GIBOW', 'GMHS',
            'GTENW', 'HCMAW', 'INNV', 'ISPO', 'ITP', 'KITTW', 'LGPS',
            'LIDRW', 'MGIH', 'MI', 'MVIS', 'NXGL', 'OPTXW', 'ORIQW',
            'PAVM', 'QH', 'RCG', 'ROLR', 'SGN', 'SSII', 'SUGP',
            'SUNE', 'TGE', 'TNFA', 'VTVT', 'WGRX', 'XLE', 'YGMZ'
        }

    def analyze_current_state(self):
        """Analyze current portfolio state"""

        print("COMPREHENSIVE PORTFOLIO CLEANUP")
        print("=" * 60)
        print("Complete portfolio reset for concentrated strategy")
        print("=" * 60)

        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.list_positions()

            print(f"Current Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"Available Cash: ${float(account.cash):,.2f}")
            print(f"Buying Power: ${float(account.buying_power):,.2f}")
            print(f"Total Positions: {len(positions)}")

            # Categorize positions
            keep_positions = []
            liquidate_positions = []
            option_positions = []
            review_positions = []

            for pos in positions:
                symbol = pos.symbol
                qty = int(pos.qty)
                market_value = float(pos.market_value)
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc) * 100

                position_data = {
                    'symbol': symbol,
                    'qty': qty,
                    'market_value': market_value,
                    'unrealized_pl': unrealized_pl,
                    'unrealized_plpc': unrealized_plpc,
                    'position_size': abs(market_value)
                }

                # Categorize position
                if any(opt in symbol for opt in ['C00', 'P00']):  # Options
                    option_positions.append(position_data)
                elif symbol in self.keep_symbols and unrealized_plpc > -10:
                    keep_positions.append(position_data)
                elif symbol in self.liquidate_symbols or unrealized_plpc < -15:
                    liquidate_positions.append(position_data)
                elif abs(market_value) < 5000:  # Small positions under $5K
                    liquidate_positions.append(position_data)
                else:
                    review_positions.append(position_data)

            print(f"\n=== PORTFOLIO CATEGORIZATION ===")
            print(f"Positions to KEEP: {len(keep_positions)} (quality winners)")
            print(f"Positions to LIQUIDATE: {len(liquidate_positions)} (losers/penny stocks)")
            print(f"OPTIONS positions: {len(option_positions)} (separate handling)")
            print(f"Positions to REVIEW: {len(review_positions)} (case-by-case)")

            return {
                'keep_positions': keep_positions,
                'liquidate_positions': liquidate_positions,
                'option_positions': option_positions,
                'review_positions': review_positions
            }

        except Exception as e:
            print(f"Error analyzing portfolio: {e}")
            return {}

    def show_cleanup_plan(self, categorized_positions):
        """Show the detailed cleanup plan"""

        if not categorized_positions:
            return

        print(f"\n=== DETAILED CLEANUP PLAN ===")

        # Show positions to keep
        keep_positions = categorized_positions['keep_positions']
        if keep_positions:
            print(f"\nKEEP ({len(keep_positions)} positions):")
            total_keep_value = 0
            for pos in keep_positions:
                total_keep_value += pos['market_value']
                print(f"  {pos['symbol']:>6} | ${pos['market_value']:>8,.0f} | {pos['unrealized_plpc']:>+5.1f}% | QUALITY")

            print(f"  TOTAL KEEP VALUE: ${total_keep_value:,.0f}")

        # Show positions to liquidate
        liquidate_positions = categorized_positions['liquidate_positions']
        if liquidate_positions:
            print(f"\nLIQUIDATE ({len(liquidate_positions)} positions):")
            total_liquidate_value = 0
            total_liquidate_pl = 0
            for pos in liquidate_positions:
                total_liquidate_value += pos['market_value']
                total_liquidate_pl += pos['unrealized_pl']
                reason = "LOSER" if pos['unrealized_plpc'] < -15 else "SMALL" if pos['position_size'] < 5000 else "JUNK"
                print(f"  {pos['symbol']:>6} | ${pos['market_value']:>8,.0f} | {pos['unrealized_plpc']:>+5.1f}% | {reason}")

            print(f"  TOTAL LIQUIDATE VALUE: ${total_liquidate_value:,.0f}")
            print(f"  TOTAL REALIZED P&L: ${total_liquidate_pl:,.0f}")

        # Show options positions (special handling)
        option_positions = categorized_positions['option_positions']
        if option_positions:
            print(f"\nOPTIONS ({len(option_positions)} positions):")
            winners = [pos for pos in option_positions if pos['unrealized_plpc'] > 20]
            losers = [pos for pos in option_positions if pos['unrealized_plpc'] < -50]

            for pos in option_positions:
                action = "TAKE_PROFITS" if pos['unrealized_plpc'] > 20 else "CLOSE_LOSER" if pos['unrealized_plpc'] < -50 else "MONITOR"
                print(f"  {pos['symbol']:>25} | ${pos['market_value']:>8,.0f} | {pos['unrealized_plpc']:>+5.1f}% | {action}")

        # Calculate cleanup impact
        liquidate_value = sum(pos['market_value'] for pos in liquidate_positions)
        liquidate_pl = sum(pos['unrealized_pl'] for pos in liquidate_positions)
        freed_capital = liquidate_value * 0.95  # Assume 5% slippage/fees

        print(f"\n=== CLEANUP IMPACT ===")
        print(f"Positions eliminated: {len(liquidate_positions)}")
        print(f"Capital freed: ${freed_capital:,.0f}")
        print(f"Realized P&L: ${liquidate_pl:,.0f}")
        print(f"Final position count: ~{len(keep_positions) + len(option_positions)} (clean portfolio)")

        return {
            'positions_eliminated': len(liquidate_positions),
            'capital_freed': freed_capital,
            'realized_pl': liquidate_pl
        }

    def execute_liquidation_orders(self, categorized_positions, execute_trades=False):
        """Execute liquidation orders for cleanup"""

        liquidate_positions = categorized_positions.get('liquidate_positions', [])

        if not liquidate_positions:
            print("No positions to liquidate")
            return []

        print(f"\n=== {'EXECUTING' if execute_trades else 'SIMULATING'} LIQUIDATION ORDERS ===")
        print(f"Closing {len(liquidate_positions)} positions for portfolio cleanup")
        print("-" * 60)

        executed_orders = []

        for pos in liquidate_positions:
            symbol = pos['symbol']
            qty = pos['qty']
            side = "sell" if qty > 0 else "buy"
            qty = abs(qty)

            try:
                if execute_trades:
                    order = self.alpaca.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        type='market',
                        time_in_force='day'
                    )

                    executed_orders.append({
                        'symbol': symbol,
                        'qty': qty,
                        'side': side,
                        'order_id': order.id,
                        'market_value': pos['market_value'],
                        'unrealized_pl': pos['unrealized_pl'],
                        'status': 'EXECUTED'
                    })

                    print(f"✓ {side.upper()} {qty:>6} {symbol:<6} | ${pos['market_value']:>8,.0f} | Order: {order.id}")
                else:
                    executed_orders.append({
                        'symbol': symbol,
                        'qty': qty,
                        'side': side,
                        'order_id': 'SIMULATED',
                        'market_value': pos['market_value'],
                        'unrealized_pl': pos['unrealized_pl'],
                        'status': 'SIMULATED'
                    })

                    print(f"○ {side.upper()} {qty:>6} {symbol:<6} | ${pos['market_value']:>8,.0f} | SIMULATED")

            except Exception as e:
                print(f"✗ ERROR {symbol}: {e}")

        total_value = sum(order['market_value'] for order in executed_orders)
        total_pl = sum(order['unrealized_pl'] for order in executed_orders)

        print("-" * 60)
        print(f"{'EXECUTED' if execute_trades else 'SIMULATED'}: {len(executed_orders)} liquidation orders")
        print(f"Total value: ${total_value:,.0f}")
        print(f"Realized P&L: ${total_pl:,.0f}")

        return executed_orders

    def run_comprehensive_cleanup(self, execute_trades=False):
        """Run complete portfolio cleanup process"""

        print("COMPREHENSIVE PORTFOLIO CLEANUP")
        print("=" * 80)
        print("Preparing clean slate for Intel-puts-style concentrated strategy")
        print("=" * 80)

        # Step 1: Analyze current state
        categorized = self.analyze_current_state()

        if not categorized:
            print("Unable to analyze portfolio - cannot proceed with cleanup")
            return {}

        # Step 2: Show cleanup plan
        cleanup_impact = self.show_cleanup_plan(categorized)

        # Step 3: Execute liquidation (if requested)
        liquidation_results = self.execute_liquidation_orders(categorized, execute_trades)

        print("\n" + "=" * 80)
        if execute_trades:
            print("PORTFOLIO CLEANUP EXECUTED")
            print("Portfolio cleaned and ready for concentrated strategy deployment")
        else:
            print("PORTFOLIO CLEANUP PLAN COMPLETE")
            print("Run with execute_trades=True to execute real cleanup")
        print("=" * 80)

        # Save cleanup report
        cleanup_report = {
            'cleanup_timestamp': datetime.now().isoformat(),
            'categorized_positions': categorized,
            'cleanup_impact': cleanup_impact,
            'liquidation_results': liquidation_results,
            'executed': execute_trades,
            'next_steps': [
                "Take profits on major options winners (RIVN +89.8%)",
                "Deploy concentrated positions with freed capital",
                "Focus on 5-10 quality positions maximum"
            ]
        }

        filename = f'comprehensive_cleanup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(cleanup_report, f, indent=2)

        print(f"Cleanup report saved: {filename}")

        return cleanup_report

def main():
    """Run comprehensive portfolio cleanup"""
    cleaner = ComprehensivePortfolioCleanup()

    # First run analysis
    report = cleaner.run_comprehensive_cleanup(execute_trades=False)

    print(f"\nTo execute the cleanup, run:")
    print(f"cleaner.run_comprehensive_cleanup(execute_trades=True)")

    return cleaner, report

if __name__ == "__main__":
    cleaner, report = main()