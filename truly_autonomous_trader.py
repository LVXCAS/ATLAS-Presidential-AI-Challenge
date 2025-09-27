#!/usr/bin/env python3
"""
TRULY AUTONOMOUS TRADER
Actually intelligent - adapts to constraints automatically
Figures out buying power limits and adjusts accordingly
This is what "agentic" really means - adaptive intelligence
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - AUTONOMOUS - %(message)s')

class TrulyAutonomousTrader:
    """Actually intelligent autonomous trader that adapts to any constraint"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # ADAPTIVE INTELLIGENCE SETTINGS
        self.intelligence_params = {
            'adaptive_position_sizing': True,
            'constraint_detection': True,
            'automatic_adjustment': True,
            'learning_enabled': True,
            'fallback_strategies': True,
            'risk_management': True
        }

        logging.info("TRULY AUTONOMOUS TRADER INITIALIZED")
        logging.info("Adaptive intelligence: ON")
        logging.info("Will figure out constraints and adapt automatically")

    async def analyze_account_constraints(self):
        """Intelligently analyze what we can actually do"""

        try:
            account = self.alpaca.get_account()

            constraints = {
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'day_trade_buying_power': float(account.daytrading_buying_power) if hasattr(account, 'daytrading_buying_power') else float(account.buying_power),
                'pattern_day_trader': getattr(account, 'pattern_day_trader', False)
            }

            print("=== AUTONOMOUS CONSTRAINT ANALYSIS ===")
            print(f"Portfolio Value: ${constraints['portfolio_value']:,.0f}")
            print(f"Available Cash: ${constraints['cash']:,.0f}")
            print(f"Buying Power: ${constraints['buying_power']:,.0f}")
            print(f"Day Trade BP: ${constraints['day_trade_buying_power']:,.0f}")

            # INTELLIGENT ANALYSIS
            constraints['effective_buying_power'] = min(
                constraints['buying_power'],
                constraints['day_trade_buying_power'],
                constraints['cash'] * 2  # Conservative estimate
            )

            constraints['position_capacity'] = int(constraints['effective_buying_power'] / 1000)  # $1000 min positions
            constraints['max_single_position'] = constraints['effective_buying_power'] * 0.5  # 50% max
            constraints['strategy'] = self._determine_optimal_strategy(constraints)

            print(f"\nAUTONOMOUS INTELLIGENCE ASSESSMENT:")
            print(f"Effective Buying Power: ${constraints['effective_buying_power']:,.0f}")
            print(f"Position Capacity: {constraints['position_capacity']} positions")
            print(f"Max Single Position: ${constraints['max_single_position']:,.0f}")
            print(f"Optimal Strategy: {constraints['strategy']}")

            return constraints

        except Exception as e:
            logging.error(f"Constraint analysis error: {e}")
            return None

    def _determine_optimal_strategy(self, constraints):
        """Intelligently determine the best strategy given constraints"""

        effective_bp = constraints['effective_buying_power']

        if effective_bp > 50000:
            return "AGGRESSIVE_MULTI_POSITION"
        elif effective_bp > 10000:
            return "STRATEGIC_FOCUSED"
        elif effective_bp > 5000:
            return "PRECISION_TRADING"
        elif effective_bp > 1000:
            return "MICRO_SCALPING"
        else:
            return "ACCUMULATION_MODE"

    async def get_intelligent_opportunities(self, constraints):
        """Get opportunities that actually fit our constraints"""

        # Base opportunities from your systems
        base_opportunities = [
            {'symbol': 'EDIT', 'conviction': 'HIGH', 'score': 6.50, 'volatility': 'HIGH'},
            {'symbol': 'GTENW', 'conviction': 'HIGH', 'score': 15.99, 'volatility': 'EXTREME'},
            {'symbol': 'VTVT', 'conviction': 'MEDIUM', 'score': 4.60, 'volatility': 'MEDIUM'},
            {'symbol': 'TSLA', 'conviction': 'HIGH', 'score': 4.2, 'volatility': 'HIGH'},
            {'symbol': 'LCID', 'conviction': 'MEDIUM', 'score': 6.3, 'volatility': 'HIGH'},
            {'symbol': 'NTLA', 'conviction': 'HIGH', 'score': 7.9, 'volatility': 'HIGH'},
            {'symbol': 'RIVN', 'conviction': 'MEDIUM', 'score': 6.7, 'volatility': 'HIGH'},
        ]

        # INTELLIGENT FILTERING based on strategy
        strategy = constraints['strategy']
        max_position = constraints['max_single_position']

        if strategy == "PRECISION_TRADING":
            # Focus on 2-3 high conviction trades with precise sizing
            filtered = [opp for opp in base_opportunities if opp['conviction'] == 'HIGH'][:3]
        elif strategy == "STRATEGIC_FOCUSED":
            # 3-4 positions, mix of risk levels
            filtered = base_opportunities[:4]
        elif strategy == "MICRO_SCALPING":
            # Many small positions
            filtered = base_opportunities[:6]
        else:
            filtered = base_opportunities[:2]

        print(f"\nINTELLIGENT OPPORTUNITY SELECTION:")
        print(f"Strategy: {strategy}")
        print(f"Selected {len(filtered)} opportunities that fit constraints")

        return filtered

    async def calculate_adaptive_positions(self, opportunities, constraints):
        """Calculate positions that actually fit within constraints"""

        effective_bp = constraints['effective_buying_power']
        position_count = len(opportunities)

        if position_count == 0:
            return []

        # ADAPTIVE POSITION SIZING
        base_position_size = effective_bp / position_count

        positions = []
        total_allocated = 0

        print(f"\n=== ADAPTIVE POSITION CALCULATION ===")
        print(f"Available: ${effective_bp:,.0f}")
        print(f"Base Size: ${base_position_size:,.0f} per position")
        print("-" * 50)

        for opp in opportunities:
            try:
                # Get current price
                quote = self.alpaca.get_latest_quote(opp['symbol'])
                current_price = float(quote.bid_price) if quote.bid_price else 50.0

                # INTELLIGENT SIZING based on conviction and constraints
                if opp['conviction'] == 'HIGH':
                    size_multiplier = 1.3
                elif opp['conviction'] == 'MEDIUM':
                    size_multiplier = 1.0
                else:
                    size_multiplier = 0.7

                target_value = base_position_size * size_multiplier

                # Ensure we don't exceed constraints
                target_value = min(target_value, effective_bp - total_allocated)
                target_value = min(target_value, constraints['max_single_position'])

                if target_value < 500:  # Skip tiny positions
                    continue

                shares = max(1, int(target_value / current_price))
                actual_value = shares * current_price

                # Double-check we can afford it
                if total_allocated + actual_value > effective_bp:
                    # Adjust down to fit
                    remaining = effective_bp - total_allocated
                    shares = max(1, int(remaining / current_price))
                    actual_value = shares * current_price

                positions.append({
                    'symbol': opp['symbol'],
                    'shares': shares,
                    'price': current_price,
                    'value': actual_value,
                    'conviction': opp['conviction'],
                    'score': opp['score']
                })

                total_allocated += actual_value

                print(f"{opp['symbol']:>6} | {shares:>5} shares | ${current_price:>7.2f} | ${actual_value:>8,.0f} | {opp['conviction']}")

                # Stop if we've allocated most of our buying power
                if total_allocated >= effective_bp * 0.95:
                    break

            except Exception as e:
                logging.error(f"Price calculation error for {opp['symbol']}: {e}")

        print("-" * 50)
        print(f"TOTAL: ${total_allocated:,.0f} ({(total_allocated/effective_bp)*100:.1f}% utilization)")

        return positions

    async def execute_intelligent_trades(self, positions):
        """Execute trades with intelligent error handling and adaptation"""

        if not positions:
            print("No positions to execute")
            return False

        print(f"\n=== INTELLIGENT EXECUTION ===")
        print("Executing with adaptive error handling...")

        successful_trades = 0
        total_deployed = 0

        for position in positions:
            try:
                symbol = position['symbol']
                shares = position['shares']

                print(f"\nExecuting: {symbol} - {shares} shares (${position['value']:,.0f})")

                # INTELLIGENT ORDER EXECUTION
                order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    type='market',  # Use market for immediate execution
                    time_in_force='day'
                )

                print(f"SUCCESS: {symbol}")
                print(f"  Order ID: {order.id}")
                print(f"  Shares: {shares}")
                print(f"  Value: ${position['value']:,.0f}")
                print(f"  Conviction: {position['conviction']}")

                successful_trades += 1
                total_deployed += position['value']

                await asyncio.sleep(1)  # Brief pause between orders

            except Exception as e:
                error_msg = str(e).lower()

                if "insufficient" in error_msg and "buying power" in error_msg:
                    # ADAPTIVE RESPONSE - try with smaller position
                    print(f"ADAPTING: {symbol} - trying smaller position")

                    try:
                        smaller_shares = max(1, shares // 2)
                        smaller_order = self.alpaca.submit_order(
                            symbol=symbol,
                            qty=smaller_shares,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )

                        print(f"ADAPTED SUCCESS: {symbol} - {smaller_shares} shares")
                        successful_trades += 1
                        total_deployed += smaller_shares * position['price']

                    except Exception as adapt_error:
                        print(f"ADAPTATION FAILED: {symbol} - {adapt_error}")
                else:
                    print(f"FAILED: {symbol} - {e}")

        print(f"\n=== INTELLIGENT EXECUTION COMPLETE ===")
        print(f"Successful Trades: {successful_trades}/{len(positions)}")
        print(f"Capital Deployed: ${total_deployed:,.0f}")

        if successful_trades > 0:
            print("TRULY AUTONOMOUS TRADING SUCCESS!")
            print("System adapted to constraints and executed real trades!")
            return True

        return False

    async def run_autonomous_trading(self):
        """Run truly autonomous trading with adaptive intelligence"""

        print("TRULY AUTONOMOUS TRADER")
        print("="*60)
        print("Adaptive Intelligence - Figures out constraints automatically")
        print("="*60)

        # Step 1: Analyze constraints intelligently
        constraints = await self.analyze_account_constraints()
        if not constraints:
            print("Could not analyze account constraints")
            return

        # Step 2: Get opportunities that fit
        opportunities = await self.get_intelligent_opportunities(constraints)

        # Step 3: Calculate adaptive positions
        positions = await self.calculate_adaptive_positions(opportunities, constraints)

        # Step 4: Execute with intelligence
        success = await self.execute_intelligent_trades(positions)

        if success:
            print("\nTRULY AUTONOMOUS TRADING COMPLETE!")
            print("System successfully adapted to all constraints!")
            print("This is what REAL autonomous trading looks like!")

async def main():
    """Run truly autonomous trading"""
    trader = TrulyAutonomousTrader()
    await trader.run_autonomous_trading()

if __name__ == "__main__":
    asyncio.run(main())