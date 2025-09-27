#!/usr/bin/env python3
"""
HYBRID CONVICTION-GENETIC TRADER
Integrates high-conviction options scanner with genetic algorithm optimization
Best of both worlds: Intel-puts-style conviction + AI-powered optimization
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Import our systems
from high_conviction_options_scanner import HighConvictionOptionsScanner
from quality_constrained_trader import QualityConstrainedTrader

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - HYBRID - %(message)s')

class HybridConvictionGeneticTrader:
    """Hybrid system combining conviction-based options with genetic optimization"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Initialize subsystems
        self.options_scanner = HighConvictionOptionsScanner()
        self.quality_trader = QualityConstrainedTrader()

        # Hybrid parameters (evolved by genetic algorithm)
        self.strategy_params = {
            'conviction_weight': 0.70,      # 70% conviction-based, 30% genetic
            'min_profit_potential': 0.50,   # 50%+ profit potential required
            'max_time_to_catalyst': 30,     # Max 30 days to catalyst
            'position_concentration': 0.35, # 35% max per position
            'stop_loss_threshold': -0.15,   # 15% stop loss
            'take_profit_threshold': 0.30,  # 30% take profit
        }

        logging.info("HYBRID CONVICTION-GENETIC TRADER INITIALIZED")
        logging.info("Intel-puts conviction + AI optimization")

    async def get_hybrid_opportunities(self):
        """Get opportunities from both conviction and genetic systems"""

        print("HYBRID OPPORTUNITY DISCOVERY")
        print("=" * 50)
        print("Combining Intel-puts conviction with AI optimization")
        print("=" * 50)

        # Get high-conviction options opportunities with ENHANCED CATALYST SCANNER
        print("\n=== ENHANCED HIGH-CONVICTION OPTIONS SCAN ===")
        conviction_opportunities = []

        try:
            # USE ENHANCED CATALYST SCANNER FOR 48+ OPPORTUNITIES
            from enhanced_catalyst_scanner import EnhancedCatalystScanner
            enhanced_scanner = EnhancedCatalystScanner()
            enhanced_catalysts = enhanced_scanner.get_comprehensive_catalysts()

            print(f"ENHANCED CATALYSTS: {len(enhanced_catalysts)} opportunities detected")

            # Convert enhanced catalysts to trading opportunities
            for catalyst in enhanced_catalysts:
                # Generate trading opportunity from catalyst
                trading_opp = {
                    'symbol': catalyst['symbol'],
                    'strategy_type': 'CALL' if catalyst['catalyst_type'] in ['earnings', 'fda_approval'] else 'PUT',
                    'profit_potential': 0.60 if catalyst['conviction'] == 'HIGH' else 0.40,
                    'days_to_catalyst': (catalyst['date'] - datetime.now()).days,
                    'catalyst_type': catalyst['catalyst_type'],
                    'description': catalyst['description'],
                    'conviction_level': catalyst['conviction']
                }

                # Filter by our hybrid criteria
                if (trading_opp['profit_potential'] >= self.strategy_params['min_profit_potential'] and
                    trading_opp['days_to_catalyst'] <= self.strategy_params['max_time_to_catalyst'] and
                    trading_opp['days_to_catalyst'] > 0):

                    conviction_opportunities.append({
                        **trading_opp,
                        'source': 'enhanced_catalyst_scanner',
                        'weight': self.strategy_params['conviction_weight']
                    })

                    print(f"ENHANCED: {trading_opp['symbol']} {trading_opp['strategy_type']} - {trading_opp['profit_potential']:.1%} potential ({trading_opp['conviction_level']})")

            # FALLBACK: Try original scanner if enhanced fails
            if not conviction_opportunities:
                print("Falling back to original conviction scanner...")
                catalysts = await self.options_scanner.get_upcoming_catalysts()
                options_opps = await self.options_scanner.scan_options_opportunities(catalysts)

                for opp in options_opps:
                    if (opp['profit_potential'] >= self.strategy_params['min_profit_potential'] and
                        opp['days_to_catalyst'] <= self.strategy_params['max_time_to_catalyst']):

                        conviction_opportunities.append({
                            **opp,
                            'source': 'conviction_scanner',
                            'weight': self.strategy_params['conviction_weight']
                        })

        except Exception as e:
            logging.error(f"Enhanced conviction scanner error: {e}")
            print(f"Scanner error: {e}")

        # Get quality genetic opportunities (stock positions)
        print("\n=== GENETIC QUALITY POSITIONS ===")
        genetic_opportunities = []

        try:
            quality_opps = await self.quality_trader.get_quality_opportunities()

            for opp in quality_opps[:3]:  # Top 3 genetic picks
                genetic_opportunities.append({
                    **opp,
                    'source': 'genetic_algorithm',
                    'weight': 1.0 - self.strategy_params['conviction_weight']
                })

                print(f"GENETIC: {opp['symbol']} - Score {opp['score']:.2f}")

        except Exception as e:
            logging.error(f"Genetic trader error: {e}")

        # DEDUPLICATE OPPORTUNITIES - No double exposure to same ticker!
        print(f"\n=== DEDUPLICATION & RISK MANAGEMENT ===")
        print("Removing duplicate symbols to prevent over-concentration")
        print("-" * 50)

        deduplicated_opportunities = []
        symbols_seen = set()

        # Process conviction opportunities first (higher priority)
        for opp in conviction_opportunities:
            symbol = opp['symbol']
            if symbol not in symbols_seen:
                deduplicated_opportunities.append(opp)
                symbols_seen.add(symbol)
                print(f"+ CONVICTION: {symbol} - {opp['profit_potential']:.1%} potential")
            else:
                print(f"x SKIPPED: {symbol} (already selected for conviction strategy)")

        # Process genetic opportunities, skip if symbol already used
        for opp in genetic_opportunities:
            symbol = opp['symbol']
            if symbol not in symbols_seen:
                deduplicated_opportunities.append(opp)
                symbols_seen.add(symbol)
                print(f"+ GENETIC: {symbol} - Score {opp['score']:.2f}")
            else:
                print(f"x SKIPPED: {symbol} (already selected for options strategy)")

        print(f"\nDEDUPLICATION COMPLETE:")
        print(f"Original opportunities: {len(conviction_opportunities + genetic_opportunities)}")
        print(f"Deduplicated opportunities: {len(deduplicated_opportunities)}")
        print(f"Unique symbols: {len(symbols_seen)}")

        return deduplicated_opportunities

    async def run_hybrid_analysis(self):
        """Main method called by master engine - runs full hybrid analysis"""
        try:
            print("\nRUNNING HYBRID CONVICTION-GENETIC ANALYSIS")
            print("=" * 60)

            # Get all opportunities
            opportunities = await self.get_hybrid_opportunities()

            if not opportunities:
                print("No hybrid opportunities found")
                return []

            # Score and rank opportunities
            scored_opportunities = []
            for opp in opportunities:
                # Calculate weighted score combining conviction and genetic factors
                base_score = opp.get('profit_potential', 0.3)
                weight = opp.get('weight', 0.5)

                weighted_score = base_score * weight

                scored_opportunities.append({
                    **opp,
                    'weighted_score': weighted_score
                })

            # Sort by weighted score
            scored_opportunities.sort(key=lambda x: x['weighted_score'], reverse=True)

            # Return top opportunities for execution
            top_opportunities = scored_opportunities[:4]  # Top 4 like your winning trades

            print(f"\nTOP HYBRID OPPORTUNITIES:")
            for i, opp in enumerate(top_opportunities):
                print(f"{i+1}. {opp['symbol']} - Score: {opp['weighted_score']:.3f} ({opp['source']})")

            return top_opportunities

        except Exception as e:
            logging.error(f"Hybrid analysis error: {e}")
            return []

        print(f"\nTotal hybrid opportunities: {len(all_opportunities)}")
        print(f"Conviction-based: {len(conviction_opportunities)}")
        print(f"Genetic-based: {len(genetic_opportunities)}")

        return all_opportunities

    async def optimize_position_sizing(self, opportunities):
        """Optimize position sizing using hybrid approach"""

        if not opportunities:
            return []

        print(f"\n=== HYBRID POSITION OPTIMIZATION ===")
        print("Optimizing Intel-puts-style concentrated positions")
        print("-" * 50)

        # Get available capital
        try:
            account = self.alpaca.get_account()
            available_capital = float(account.portfolio_value)
        except:
            available_capital = 879274  # Current portfolio value

        # Sort by weighted score (conviction * weight + genetic_score * weight)
        scored_opportunities = []

        for opp in opportunities:
            if opp['source'] == 'conviction_scanner':
                weighted_score = opp['profit_potential'] * opp['weight']
                position_type = 'OPTIONS'
                estimated_cost = opp['estimated_premium'] * 100  # Per contract
            else:  # genetic_algorithm
                weighted_score = opp['score'] * opp['weight']
                position_type = 'STOCK'
                estimated_cost = opp['price']

            scored_opportunities.append({
                **opp,
                'weighted_score': weighted_score,
                'position_type': position_type,
                'estimated_cost': estimated_cost
            })

        # Sort by weighted score
        scored_opportunities.sort(key=lambda x: x['weighted_score'], reverse=True)

        # Position sizing: Intel-puts-style concentration
        max_positions = 4
        selected_opportunities = scored_opportunities[:max_positions]

        positions = []
        total_allocation = 0

        for i, opp in enumerate(selected_opportunities):
            # ENHANCED RISK MANAGEMENT: Position concentration limits
            max_single_position = 0.30  # 30% max per ticker - HARD LIMIT

            # Decreasing allocation for lower-ranked opportunities
            if i == 0:
                allocation = min(0.40, max_single_position, self.strategy_params['position_concentration'])  # 30% max for top pick
            elif i == 1:
                allocation = min(0.30, max_single_position, self.strategy_params['position_concentration'] * 0.8)
            elif i == 2:
                allocation = min(0.20, max_single_position, self.strategy_params['position_concentration'] * 0.6)
            else:
                allocation = min(0.10, max_single_position, self.strategy_params['position_concentration'] * 0.4)

            # Apply additional safety check
            if allocation > max_single_position:
                print(f"⚠ WARNING: {opp['symbol']} allocation capped at {max_single_position:.0%} (was {allocation:.0%})")
                allocation = max_single_position

            position_value = available_capital * allocation

            if opp['position_type'] == 'OPTIONS':
                contracts = max(1, int(position_value / (opp['estimated_cost'])))
                actual_cost = contracts * opp['estimated_cost']

                position = {
                    **opp,
                    'allocation_pct': allocation,
                    'position_value': position_value,
                    'contracts': contracts,
                    'total_cost': actual_cost,
                    'action_type': 'BUY_OPTIONS'
                }
            else:  # STOCK
                shares = max(1, int(position_value / opp['estimated_cost']))
                actual_cost = shares * opp['estimated_cost']

                position = {
                    **opp,
                    'allocation_pct': allocation,
                    'position_value': position_value,
                    'shares': shares,
                    'total_cost': actual_cost,
                    'action_type': 'BUY_STOCK'
                }

            positions.append(position)
            total_allocation += allocation

            print(f"{opp['symbol']:>6} | {opp['source'][:10]:>10} | {allocation:>6.1%} | ${position_value:>10,.0f} | Score: {opp['weighted_score']:.3f}")

        print("-" * 50)
        print(f"TOTAL ALLOCATION: {total_allocation:.1%}")
        print(f"CAPITAL DEPLOYMENT: ${sum(p['total_cost'] for p in positions):,.0f}")

        return positions

    async def generate_hybrid_signals(self, positions):
        """Generate executable trading signals from hybrid system"""

        if not positions:
            print("No hybrid positions to execute")
            return []

        print(f"\n=== HYBRID TRADING SIGNALS ===")
        print("Intel-puts conviction + AI-optimized positions")
        print("-" * 60)

        signals = []

        for pos in positions:
            if pos['action_type'] == 'BUY_OPTIONS':
                action = f"BUY {pos['contracts']} {pos['symbol']} ${pos.get('target_strike', 0):.0f} {pos.get('strategy_type', 'OPTIONS')}"
                signal_type = 'OPTIONS'
            else:  # BUY_STOCK
                action = f"BUY {pos['shares']} {pos['symbol']} STOCK"
                signal_type = 'STOCK'

            # Risk management levels
            stop_loss = pos['total_cost'] * (1 + self.strategy_params['stop_loss_threshold'])
            take_profit = pos['total_cost'] * (1 + self.strategy_params['take_profit_threshold'])

            signal = {
                'action': action,
                'symbol': pos['symbol'],
                'signal_type': signal_type,
                'allocation': pos['allocation_pct'],
                'cost': pos['total_cost'],
                'source': pos['source'],
                'weighted_score': pos['weighted_score'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': 'hybrid_conviction_genetic'
            }

            signals.append(signal)

            print(f"{action}")
            print(f"  Source: {pos['source']}")
            print(f"  Allocation: {pos['allocation_pct']:.1%} (${pos['total_cost']:,.0f})")
            print(f"  Weighted Score: {pos['weighted_score']:.3f}")
            print(f"  Stop Loss: ${stop_loss:,.0f}")
            print(f"  Take Profit: ${take_profit:,.0f}")
            print()

        return signals

    async def execute_hybrid_strategy(self):
        """Execute complete hybrid conviction-genetic strategy"""

        print("HYBRID CONVICTION-GENETIC TRADER")
        print("=" * 60)
        print("Intel-puts conviction + AI-powered genetic optimization")
        print("Targeting 25-50% monthly returns with quality assets only")
        print("=" * 60)

        # Step 1: Discover hybrid opportunities
        opportunities = await self.get_hybrid_opportunities()

        if not opportunities:
            print("No hybrid opportunities found - waiting for better setups")
            return []

        # Step 2: Optimize position sizing
        positions = await self.optimize_position_sizing(opportunities)

        if not positions:
            print("No viable positions after optimization")
            return []

        # Step 3: Generate trading signals
        signals = await self.generate_hybrid_signals(positions)

        print("=" * 60)
        print("HYBRID STRATEGY COMPLETE")
        print(f"Generated {len(signals)} optimized trading signals")
        print("Ready to deploy Intel-puts-style conviction with AI optimization!")
        print("=" * 60)

        # Save signals for execution
        strategy_report = {
            'timestamp': datetime.now().isoformat(),
            'strategy_type': 'hybrid_conviction_genetic',
            'signals': signals,
            'total_positions': len(signals),
            'conviction_weight': self.strategy_params['conviction_weight'],
            'target_monthly_return': '25-50%'
        }

        with open(f'hybrid_strategy_signals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(strategy_report, f, indent=2)

        return signals

async def main():
    """Run hybrid conviction-genetic trading system"""
    trader = HybridConvictionGeneticTrader()
    signals = await trader.execute_hybrid_strategy()
    return signals

if __name__ == "__main__":
    asyncio.run(main())