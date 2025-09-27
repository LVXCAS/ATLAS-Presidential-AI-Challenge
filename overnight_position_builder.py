#!/usr/bin/env python3
"""
OVERNIGHT POSITION BUILDER
Builds concentrated positions based on manually spotted momentum moves
Designed for RIVN, SNAP, INTC style explosive opportunities
"""

import asyncio
import logging
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - BUILDER - %(message)s')

class OvernightPositionBuilder:
    """Builds Intel-puts-style concentrated positions for momentum opportunities"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Current momentum opportunities you spotted
        self.current_momentum_plays = [
            {
                'symbol': 'RIVN',
                'move_type': 'explosive_up',
                'conviction': 'HIGH',
                'strategy': 'CALL',
                'rationale': 'User spotted explosive upward momentum'
            },
            {
                'symbol': 'SNAP',
                'move_type': 'explosive_up',
                'conviction': 'HIGH',
                'strategy': 'CALL',
                'rationale': 'User spotted explosive upward momentum'
            },
            {
                'symbol': 'INTC',
                'move_type': 'continued_strength',
                'conviction': 'HIGH',
                'strategy': 'CALL',
                'rationale': 'Following successful Intel puts - momentum continuation'
            }
        ]

        logging.info("OVERNIGHT POSITION BUILDER INITIALIZED")
        logging.info("Ready to build positions for manually spotted opportunities")

    async def analyze_momentum_opportunities(self):
        """Analyze the momentum opportunities you spotted"""

        print("MOMENTUM OPPORTUNITY ANALYSIS")
        print("=" * 60)
        print("Analyzing manually spotted RIVN, SNAP, INTC moves")
        print("Building Intel-puts-style concentrated positions")
        print("=" * 60)

        enhanced_opportunities = []

        for opp in self.current_momentum_plays:
            symbol = opp['symbol']

            # Get rough current pricing (simulated since we have data limits)
            try:
                # Try to get actual data, fallback to estimates
                latest_quote = self.alpaca.get_latest_quote(symbol)
                current_price = float(latest_quote.bid_price)
                print(f"Got real price for {symbol}: ${current_price:.2f}")
            except:
                # Fallback to estimated prices based on typical ranges
                price_estimates = {'RIVN': 15.50, 'SNAP': 11.25, 'INTC': 24.75}
                current_price = price_estimates.get(symbol, 25.00)
                print(f"Using estimated price for {symbol}: ${current_price:.2f}")

            # Calculate position parameters
            if opp['strategy'] == 'CALL':
                # OTM calls for momentum continuation
                target_strike = int(current_price * 1.10)  # 10% OTM
                estimated_premium = current_price * 0.03   # 3% estimate
                profit_potential = 2.5 if opp['conviction'] == 'HIGH' else 1.5  # 150-250%
            else:  # PUT
                target_strike = int(current_price * 0.90)  # 10% OTM
                estimated_premium = current_price * 0.025  # 2.5% estimate
                profit_potential = 2.0 if opp['conviction'] == 'HIGH' else 1.2

            enhanced_opp = {
                **opp,
                'current_price': current_price,
                'target_strike': target_strike,
                'estimated_premium': estimated_premium,
                'profit_potential': profit_potential,
                'expiry_days': 14,  # 2-week options for momentum
                'risk_level': 'HIGH',
                'position_type': 'OPTIONS'
            }

            enhanced_opportunities.append(enhanced_opp)

            print(f"\n{symbol} MOMENTUM ANALYSIS:")
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  Strategy: {opp['strategy']} ${target_strike}")
            print(f"  Premium Est: ${estimated_premium:.2f}")
            print(f"  Profit Pot: {profit_potential:.1%}")
            print(f"  Conviction: {opp['conviction']}")
            print(f"  Rationale: {opp['rationale']}")

        return enhanced_opportunities

    async def build_concentrated_positions(self, opportunities):
        """Build Intel-puts-style concentrated positions"""

        if not opportunities:
            return []

        print(f"\n=== CONCENTRATED POSITION BUILDING ===")
        print("Intel-puts-style allocation for momentum plays")
        print("-" * 60)

        # Get available capital (use 30% for momentum positions)
        try:
            account = self.alpaca.get_account()
            total_portfolio = float(account.portfolio_value)
            momentum_capital = total_portfolio * 0.30  # 30% for momentum
        except:
            total_portfolio = 879274  # Current estimated value
            momentum_capital = total_portfolio * 0.30

        print(f"Total Portfolio: ${total_portfolio:,.0f}")
        print(f"Momentum Capital: ${momentum_capital:,.0f} (30%)")

        positions = []

        # Sort by conviction and allocate accordingly
        high_conviction = [opp for opp in opportunities if opp['conviction'] == 'HIGH']

        # Intel-puts-style allocation: concentrated on high conviction
        if len(high_conviction) >= 2:
            # Split momentum capital between top 2-3 picks
            allocations = [0.15, 0.10, 0.05]  # 15%, 10%, 5% of total portfolio
        else:
            allocations = [0.20, 0.10]  # 20%, 10% for fewer opportunities

        for i, opp in enumerate(high_conviction[:3]):
            if i >= len(allocations):
                break

            allocation_pct = allocations[i]
            position_value = total_portfolio * allocation_pct

            # Calculate contracts
            premium_per_contract = opp['estimated_premium'] * 100
            contracts = max(1, int(position_value / premium_per_contract))
            actual_cost = contracts * premium_per_contract

            position = {
                **opp,
                'allocation_pct': allocation_pct,
                'position_value': position_value,
                'contracts': contracts,
                'total_cost': actual_cost,
                'cost_per_contract': premium_per_contract,
                'action_type': 'BUY_OPTIONS'
            }

            positions.append(position)

            print(f"{opp['symbol']:>6} | {opp['strategy']:>4} | ${opp['target_strike']:>3} | {allocation_pct:>6.1%} | {contracts:>4} contracts | ${actual_cost:>8,.0f}")

        total_allocation = sum(p['allocation_pct'] for p in positions)
        total_cost = sum(p['total_cost'] for p in positions)

        print("-" * 60)
        print(f"TOTAL MOMENTUM ALLOCATION: {total_allocation:.1%}")
        print(f"TOTAL MOMENTUM COST: ${total_cost:,.0f}")

        return positions

    async def generate_momentum_signals(self, positions):
        """Generate executable trading signals"""

        if not positions:
            print("No momentum positions to execute")
            return []

        print(f"\n=== MOMENTUM TRADING SIGNALS ===")
        print("Ready to execute Intel-puts-style momentum positions")
        print("-" * 70)

        signals = []

        for pos in positions:
            action = f"BUY {pos['contracts']} {pos['symbol']} ${pos['target_strike']} {pos['strategy']}"

            # Risk management
            stop_loss_pct = -0.50  # 50% stop loss on options
            take_profit_pct = pos['profit_potential']

            signal = {
                'action': action,
                'symbol': pos['symbol'],
                'strategy_type': pos['strategy'],
                'contracts': pos['contracts'],
                'strike': pos['target_strike'],
                'allocation': pos['allocation_pct'],
                'cost': pos['total_cost'],
                'profit_potential': pos['profit_potential'],
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'conviction': pos['conviction'],
                'rationale': pos['rationale'],
                'expiry_days': pos['expiry_days'],
                'source': 'manual_momentum_spotted'
            }

            signals.append(signal)

            print(f"{action}")
            print(f"  Allocation: {pos['allocation_pct']:.1%} (${pos['total_cost']:,.0f})")
            print(f"  Profit Target: {pos['profit_potential']:.1%}")
            print(f"  Stop Loss: {stop_loss_pct:.1%}")
            print(f"  Conviction: {pos['conviction']}")
            print(f"  Rationale: {pos['rationale']}")
            print()

        return signals

    async def build_momentum_positions(self):
        """Complete workflow to build momentum positions"""

        print("OVERNIGHT POSITION BUILDER")
        print("=" * 80)
        print("Building Intel-puts-style positions for manually spotted moves")
        print("Current opportunities: RIVN, SNAP, INTC momentum")
        print("=" * 80)

        # Step 1: Analyze opportunities
        opportunities = await self.analyze_momentum_opportunities()

        if not opportunities:
            print("No momentum opportunities to analyze")
            return []

        # Step 2: Build concentrated positions
        positions = await self.build_concentrated_positions(opportunities)

        if not positions:
            print("No viable positions created")
            return []

        # Step 3: Generate trading signals
        signals = await self.generate_momentum_signals(positions)

        print("=" * 80)
        print("MOMENTUM POSITION BUILDING COMPLETE")
        print(f"Generated {len(signals)} momentum trading signals")
        print("Ready to execute Intel-puts-style concentrated momentum positions!")
        print("=" * 80)

        # Save the results
        build_results = {
            'build_timestamp': datetime.now().isoformat(),
            'build_type': 'manual_momentum_spotted',
            'opportunities_analyzed': opportunities,
            'positions_built': positions,
            'trading_signals': signals,
            'total_signals': len(signals),
            'total_allocation': sum(s['allocation'] for s in signals),
            'user_spotted_symbols': ['RIVN', 'SNAP', 'INTC']
        }

        filename = f'momentum_positions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(build_results, f, indent=2)

        print(f"Results saved to: {filename}")

        return signals

async def main():
    """Build momentum positions for user-spotted opportunities"""
    builder = OvernightPositionBuilder()
    signals = await builder.build_momentum_positions()
    return signals

if __name__ == "__main__":
    asyncio.run(main())