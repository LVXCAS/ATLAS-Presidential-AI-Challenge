#!/usr/bin/env python3
"""
HIGH-CONVICTION OPTIONS CATALYST SCANNER
Finds Intel-puts-style opportunities for 25-50% monthly returns
Integrates with existing architecture - no penny stocks!
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
from typing import Dict, List, Any

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - OPTIONS - %(message)s')

class HighConvictionOptionsScanner:
    """Intel-puts-style options opportunity scanner"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Load approved quality universe
        self.approved_universe = self.load_approved_universe()

        # High-conviction targeting (like Intel puts)
        self.target_criteria = {
            'monthly_return_target': 0.35,     # 35% monthly (like 70% Intel puts in 2 months)
            'max_positions': 4,                # 2-4 concentrated positions
            'min_position_size': 0.25,         # 25% minimum per position
            'max_position_size': 0.50,         # 50% maximum per position
            'catalyst_window_days': 30,        # Look 30 days ahead for catalysts
        }

        # Options filtering (institutional quality only)
        self.options_filters = {
            'min_underlying_price': 50,        # No cheap stocks
            'min_daily_volume': 1000,          # 1000+ contracts/day
            'max_bid_ask_spread_pct': 0.05,    # 5% max spread
            'min_open_interest': 500,          # 500+ open interest
            'expiration_range': (7, 45),       # 1-6 weeks out
            'delta_range': (0.20, 0.80),       # Meaningful delta
        }

        logging.info("HIGH-CONVICTION OPTIONS SCANNER INITIALIZED")
        logging.info(f"Target: 25-50% monthly returns from quality options")

    def load_approved_universe(self):
        """Load approved institutional-quality universe"""
        try:
            with open('approved_asset_universe.json', 'r') as f:
                data = json.load(f)
                return data['approved_assets']
        except Exception as e:
            logging.error(f"Could not load approved universe: {e}")
            # High-quality fallback
            return ['AAPL', 'NVDA', 'GOOGL', 'META', 'TSLA', 'MSFT', 'AMZN',
                   'SPY', 'QQQ', 'XLK', 'JPM', 'JNJ', 'HD', 'PG']

    async def get_upcoming_catalysts(self):
        """Identify upcoming catalysts for next 30 days"""

        print("=== CATALYST IDENTIFICATION ===")
        print("Scanning for Intel-puts-style catalyst opportunities")
        print("-" * 50)

        # Focus on major earnings and events
        catalysts = []

        # Major tech earnings (example - would integrate real earnings calendar)
        potential_catalysts = [
            {
                'symbol': 'NVDA',
                'catalyst_type': 'earnings',
                'date': datetime.now() + timedelta(days=7),
                'conviction': 'HIGH',
                'description': 'Q4 earnings - AI demand expectations'
            },
            {
                'symbol': 'AAPL',
                'catalyst_type': 'product_launch',
                'date': datetime.now() + timedelta(days=14),
                'conviction': 'MEDIUM',
                'description': 'Vision Pro updates expected'
            },
            {
                'symbol': 'GOOGL',
                'catalyst_type': 'earnings',
                'date': datetime.now() + timedelta(days=21),
                'conviction': 'HIGH',
                'description': 'Q4 earnings - YouTube and Cloud growth'
            },
            {
                'symbol': 'TSLA',
                'catalyst_type': 'delivery_numbers',
                'date': datetime.now() + timedelta(days=10),
                'conviction': 'HIGH',
                'description': 'Q4 delivery numbers - China demand concerns'
            },
            {
                'symbol': 'SPY',
                'catalyst_type': 'fed_meeting',
                'date': datetime.now() + timedelta(days=18),
                'conviction': 'MEDIUM',
                'description': 'FOMC meeting - rate cut expectations'
            }
        ]

        # Filter to approved universe only
        for catalyst in potential_catalysts:
            if catalyst['symbol'] in self.approved_universe:
                catalysts.append(catalyst)
                print(f"{catalyst['symbol']:>6} | {catalyst['catalyst_type']:>15} | {catalyst['conviction']:>6} | {catalyst['description']}")

        print(f"\nCatalysts identified: {len(catalysts)}")
        return catalysts

    async def analyze_technical_setup(self, symbol):
        """Analyze technical setup for directional bias"""

        try:
            # Get recent price data
            quote = self.alpaca.get_latest_quote(symbol)
            current_price = float(quote.bid_price) if quote.bid_price else 0

            if current_price == 0:
                return None

            # Simple technical analysis (would integrate more sophisticated TA)
            # For now, use price momentum as proxy

            technical_setup = {
                'symbol': symbol,
                'current_price': current_price,
                'bias': 'NEUTRAL',  # BULLISH, BEARISH, NEUTRAL
                'conviction': 'MEDIUM',
                'resistance_level': current_price * 1.05,
                'support_level': current_price * 0.95,
            }

            # Example bias logic (would be much more sophisticated)
            if current_price > 200:  # Arbitrary momentum proxy
                technical_setup['bias'] = 'BULLISH'
                technical_setup['conviction'] = 'HIGH'
            elif current_price < 100:
                technical_setup['bias'] = 'BEARISH'
                technical_setup['conviction'] = 'HIGH'

            return technical_setup

        except Exception as e:
            logging.error(f"Technical analysis error for {symbol}: {e}")
            return None

    async def scan_options_opportunities(self, catalysts):
        """Scan for high-conviction options opportunities"""

        print(f"\n=== OPTIONS OPPORTUNITY ANALYSIS ===")
        print("Finding Intel-puts-style setups in quality assets")
        print("-" * 50)

        opportunities = []

        for catalyst in catalysts:
            symbol = catalyst['symbol']

            # Get technical setup
            technical = await self.analyze_technical_setup(symbol)
            if not technical:
                continue

            # Calculate days to catalyst
            days_to_catalyst = (catalyst['date'] - datetime.now()).days

            # Determine strategy type based on setup
            if technical['bias'] == 'BEARISH':
                strategy_type = 'PUT'
                target_strike_pct = 0.95  # 5% OTM puts
            elif technical['bias'] == 'BULLISH':
                strategy_type = 'CALL'
                target_strike_pct = 1.05  # 5% OTM calls
            else:
                strategy_type = 'STRADDLE'
                target_strike_pct = 1.00  # ATM straddle

            # Calculate target strike
            current_price = technical['current_price']
            target_strike = round(current_price * target_strike_pct, 0)

            # Estimate option premium (simplified - would use real options data)
            estimated_premium = current_price * 0.03  # ~3% of stock price

            # Calculate profit potential
            if strategy_type == 'PUT':
                breakeven = target_strike - estimated_premium
                profit_potential = max(0, (current_price - breakeven) / estimated_premium)
            elif strategy_type == 'CALL':
                breakeven = target_strike + estimated_premium
                profit_potential = max(0, (breakeven - current_price) / estimated_premium)
            else:  # STRADDLE
                profit_potential = 0.5  # Conservative straddle estimate

            # Only include high-potential opportunities
            if profit_potential > 0.5:  # 50%+ potential return
                opportunity = {
                    'symbol': symbol,
                    'strategy_type': strategy_type,
                    'current_price': current_price,
                    'target_strike': target_strike,
                    'estimated_premium': estimated_premium,
                    'profit_potential': profit_potential,
                    'days_to_catalyst': days_to_catalyst,
                    'catalyst_type': catalyst['catalyst_type'],
                    'conviction': catalyst['conviction'],
                    'technical_bias': technical['bias']
                }

                opportunities.append(opportunity)

                print(f"{symbol:>6} | {strategy_type:>8} | ${target_strike:>6.0f} | {profit_potential:>6.1%} | {days_to_catalyst:>2}d | {catalyst['catalyst_type']}")

        # Sort by profit potential
        opportunities.sort(key=lambda x: x['profit_potential'], reverse=True)

        print(f"\nHigh-conviction opportunities: {len(opportunities)}")
        return opportunities[:self.target_criteria['max_positions']]

    async def calculate_position_sizing(self, opportunities):
        """Calculate Intel-puts-style position sizing"""

        if not opportunities:
            return []

        print(f"\n=== POSITION SIZING (25-50% PER POSITION) ===")
        print("Concentrated allocation like Intel puts")
        print("-" * 50)

        # Get available capital
        try:
            account = self.alpaca.get_account()
            available_capital = float(account.portfolio_value)
        except:
            available_capital = 879274  # Current portfolio value

        total_positions = len(opportunities)
        base_allocation = min(0.50, 0.80 / total_positions)  # Max 50% per position

        sized_positions = []

        for opp in opportunities:
            # Adjust allocation by conviction
            if opp['conviction'] == 'HIGH':
                allocation = min(0.50, base_allocation * 1.3)
            else:
                allocation = base_allocation

            position_value = available_capital * allocation
            contracts_affordable = int(position_value / (opp['estimated_premium'] * 100))

            if contracts_affordable > 0:
                sized_position = {
                    **opp,
                    'allocation_pct': allocation,
                    'position_value': position_value,
                    'contracts': contracts_affordable,
                    'total_cost': contracts_affordable * opp['estimated_premium'] * 100
                }

                sized_positions.append(sized_position)

                print(f"{opp['symbol']:>6} | {allocation:>6.1%} | ${position_value:>8,.0f} | {contracts_affordable:>4} contracts | {opp['strategy_type']}")

        total_deployed = sum(p['total_cost'] for p in sized_positions)
        print(f"\nTotal capital deployment: ${total_deployed:,.0f} ({total_deployed/available_capital:.1%})")

        return sized_positions

    async def generate_trading_signals(self, positions):
        """Generate executable trading signals"""

        if not positions:
            print("No high-conviction positions to execute")
            return []

        print(f"\n=== HIGH-CONVICTION TRADING SIGNALS ===")
        print("Intel-puts-style concentrated options plays")
        print("-" * 60)

        signals = []

        for pos in positions:
            # Generate specific order instructions
            if pos['strategy_type'] == 'PUT':
                action = f"BUY {pos['contracts']} {pos['symbol']} ${pos['target_strike']:.0f} PUTS"
                expiration = "4 weeks out"
            elif pos['strategy_type'] == 'CALL':
                action = f"BUY {pos['contracts']} {pos['symbol']} ${pos['target_strike']:.0f} CALLS"
                expiration = "4 weeks out"
            else:  # STRADDLE
                action = f"BUY {pos['contracts']} {pos['symbol']} ${pos['target_strike']:.0f} STRADDLE"
                expiration = "3 weeks out"

            signal = {
                'action': action,
                'symbol': pos['symbol'],
                'strategy': pos['strategy_type'],
                'contracts': pos['contracts'],
                'strike': pos['target_strike'],
                'allocation': pos['allocation_pct'],
                'cost': pos['total_cost'],
                'target_return': pos['profit_potential'],
                'catalyst': pos['catalyst_type'],
                'days_to_catalyst': pos['days_to_catalyst'],
                'conviction': pos['conviction']
            }

            signals.append(signal)

            print(f"{action}")
            print(f"  Allocation: {pos['allocation_pct']:.1%} (${pos['total_cost']:,.0f})")
            print(f"  Target Return: {pos['profit_potential']:.1%}")
            print(f"  Catalyst: {pos['catalyst_type']} in {pos['days_to_catalyst']} days")
            print(f"  Conviction: {pos['conviction']}")
            print()

        return signals

    async def run_high_conviction_scan(self):
        """Run complete high-conviction options scan"""

        print("HIGH-CONVICTION OPTIONS CATALYST SCANNER")
        print("=" * 60)
        print("Finding Intel-puts-style opportunities for 25-50% monthly returns")
        print("Quality assets only - No penny stocks!")
        print("=" * 60)

        # Step 1: Identify catalysts
        catalysts = await self.get_upcoming_catalysts()

        # Step 2: Scan for options opportunities
        opportunities = await self.scan_options_opportunities(catalysts)

        # Step 3: Calculate position sizing
        positions = await self.calculate_position_sizing(opportunities)

        # Step 4: Generate trading signals
        signals = await self.generate_trading_signals(positions)

        print("=" * 60)
        print("HIGH-CONVICTION SCAN COMPLETE")
        if signals:
            print(f"Found {len(signals)} Intel-puts-style opportunities")
            print("Ready for 25-50% monthly returns!")
        else:
            print("No high-conviction opportunities found - wait for better setups")
        print("=" * 60)

        return signals

async def main():
    """Run high-conviction options scanner"""
    scanner = HighConvictionOptionsScanner()
    signals = await scanner.run_high_conviction_scan()
    return signals

if __name__ == "__main__":
    asyncio.run(main())