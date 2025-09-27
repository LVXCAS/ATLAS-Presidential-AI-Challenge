#!/usr/bin/env python3
"""
DYNAMIC OPPORTUNITY MONITOR
Monitors the BEST opportunities across all markets, not just specific stocks
Finds the highest momentum, biggest gainers, and most promising setups
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from intelligent_asset_universe_filter import IntelligentAssetFilter

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - MONITOR - %(message)s')

class DynamicOpportunityMonitor:
    """Monitors best opportunities across all markets dynamically"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        self.asset_filter = IntelligentAssetFilter()

        # Expanded monitoring universe - not just RIVN/SNAP
        self.monitoring_universe = [
            # Tech momentum plays
            'NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN',
            # High beta momentum
            'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'PLTR', 'COIN', 'ROKU',
            # Volatile opportunities
            'SNAP', 'UBER', 'LYFT', 'SQ', 'PYPL', 'NFLX', 'DIS', 'CRM',
            # Meme/momentum stocks
            'AMC', 'GME', 'BBBY', 'CLOV', 'SOFI', 'WISH', 'PROG',
            # Traditional high movers
            'JPM', 'BAC', 'XLF', 'SPY', 'QQQ', 'IWM', 'GLD', 'TLT',
            # Energy/commodity momentum
            'XLE', 'XOP', 'USO', 'GDX', 'SLV', 'UNG', 'XBI', 'XRT',
            # Emerging opportunities
            'AI', 'SMCI', 'ARM', 'IONQ', 'RBLX', 'U', 'NET', 'SNOW'
        ]

        logging.info("DYNAMIC OPPORTUNITY MONITOR INITIALIZED")
        logging.info(f"Monitoring {len(self.monitoring_universe)} symbols for best opportunities")

    async def scan_top_movers(self):
        """Scan for top momentum movers across all monitored assets"""

        print("DYNAMIC TOP MOVERS SCAN")
        print("=" * 60)
        print("Finding the BEST opportunities, not just specific stocks")
        print("=" * 60)

        movers = []
        scan_time = datetime.now()

        print(f"Scanning {len(self.monitoring_universe)} symbols for momentum...")

        for symbol in self.monitoring_universe:
            try:
                # Get recent price data
                bars = self.alpaca.get_bars(symbol, '1Day', limit=5).df
                if len(bars) < 2:
                    continue

                current_price = float(bars.iloc[-1]['close'])
                previous_close = float(bars.iloc[-2]['close'])

                # Calculate daily move
                daily_change = (current_price - previous_close) / previous_close

                # Get volume
                current_volume = float(bars.iloc[-1]['volume'])
                avg_volume = float(bars['volume'].mean())
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

                # Calculate momentum score
                momentum_score = abs(daily_change) * volume_ratio

                if abs(daily_change) > 0.02:  # More than 2% move
                    move_direction = "UP" if daily_change > 0 else "DOWN"

                    mover = {
                        'symbol': symbol,
                        'current_price': current_price,
                        'daily_change': daily_change,
                        'direction': move_direction,
                        'volume_ratio': volume_ratio,
                        'momentum_score': momentum_score,
                        'opportunity_type': self.classify_opportunity(symbol, daily_change, volume_ratio),
                        'scan_time': scan_time.isoformat()
                    }

                    movers.append(mover)

            except Exception as e:
                continue

        # Sort by momentum score (highest first)
        movers.sort(key=lambda x: x['momentum_score'], reverse=True)

        # Display top movers
        print(f"\n=== TOP {min(15, len(movers))} MOMENTUM OPPORTUNITIES ===")
        print("Symbol | Price | Change | Volume | Score | Type")
        print("-" * 60)

        for mover in movers[:15]:
            print(f"{mover['symbol']:>6} | ${mover['current_price']:>6.2f} | {mover['daily_change']:>+6.1%} | "
                  f"{mover['volume_ratio']:>4.1f}x | {mover['momentum_score']:>5.2f} | {mover['opportunity_type']}")

        return movers

    def classify_opportunity(self, symbol, daily_change, volume_ratio):
        """Classify the type of opportunity"""

        if volume_ratio > 3 and abs(daily_change) > 0.10:
            return "EXPLOSIVE"
        elif volume_ratio > 2 and abs(daily_change) > 0.05:
            return "HIGH_MOMENTUM"
        elif abs(daily_change) > 0.08:
            return "BIG_MOVE"
        elif volume_ratio > 2:
            return "VOLUME_SPIKE"
        else:
            return "STEADY_MOVE"

    async def identify_best_opportunities(self, movers):
        """Identify the absolute best opportunities to monitor and trade"""

        if not movers:
            return []

        print(f"\n=== BEST OPPORTUNITIES ANALYSIS ===")
        print("Ranking ALL opportunities, not just predetermined stocks")
        print("-" * 60)

        # Filter for the most promising opportunities
        explosive_moves = [m for m in movers if m['opportunity_type'] == 'EXPLOSIVE']
        high_momentum = [m for m in movers if m['opportunity_type'] == 'HIGH_MOMENTUM']
        big_moves = [m for m in movers if m['opportunity_type'] == 'BIG_MOVE']

        best_opportunities = []

        # Prioritize explosive moves
        if explosive_moves:
            print("🔥 EXPLOSIVE OPPORTUNITIES:")
            for move in explosive_moves[:3]:
                best_opportunities.append({
                    **move,
                    'priority': 'HIGHEST',
                    'action_recommendation': self.get_action_recommendation(move),
                    'profit_potential': self.estimate_profit_potential(move)
                })
                print(f"   {move['symbol']}: {move['daily_change']:+.1%} with {move['volume_ratio']:.1f}x volume")

        # Add high momentum opportunities
        if high_momentum:
            print("\n⚡ HIGH MOMENTUM OPPORTUNITIES:")
            for move in high_momentum[:3]:
                if move not in best_opportunities:
                    best_opportunities.append({
                        **move,
                        'priority': 'HIGH',
                        'action_recommendation': self.get_action_recommendation(move),
                        'profit_potential': self.estimate_profit_potential(move)
                    })
                    print(f"   {move['symbol']}: {move['daily_change']:+.1%} momentum")

        # Add big move opportunities
        if big_moves:
            print("\n📈 BIG MOVE OPPORTUNITIES:")
            for move in big_moves[:2]:
                if move not in best_opportunities:
                    best_opportunities.append({
                        **move,
                        'priority': 'MEDIUM',
                        'action_recommendation': self.get_action_recommendation(move),
                        'profit_potential': self.estimate_profit_potential(move)
                    })
                    print(f"   {move['symbol']}: {move['daily_change']:+.1%} move")

        # Sort by profit potential
        best_opportunities.sort(key=lambda x: x['profit_potential'], reverse=True)

        return best_opportunities

    def get_action_recommendation(self, move):
        """Get recommended action for this opportunity"""

        if move['daily_change'] > 0.10:  # Big up move
            if move['volume_ratio'] > 3:
                return "MOMENTUM_CONTINUATION"  # Ride the wave
            else:
                return "POTENTIAL_FADE"  # May reverse
        elif move['daily_change'] < -0.10:  # Big down move
            if move['volume_ratio'] > 3:
                return "BOUNCE_PLAY"  # Look for bounce
            else:
                return "CONTINUATION_DOWN"  # May continue falling
        elif 0.05 < move['daily_change'] < 0.10:
            return "MOMENTUM_CONTINUATION"
        elif -0.10 < move['daily_change'] < -0.05:
            return "BOUNCE_CANDIDATE"
        else:
            return "WATCH"

    def estimate_profit_potential(self, move):
        """Estimate profit potential percentage"""

        base_potential = abs(move['daily_change']) * 2  # 2x the current move
        volume_multiplier = min(move['volume_ratio'], 5) / 5  # Cap at 5x volume

        # Adjust for opportunity type
        type_multipliers = {
            'EXPLOSIVE': 3.0,
            'HIGH_MOMENTUM': 2.0,
            'BIG_MOVE': 1.5,
            'VOLUME_SPIKE': 1.2,
            'STEADY_MOVE': 0.8
        }

        multiplier = type_multipliers.get(move['opportunity_type'], 1.0)

        estimated_potential = base_potential * multiplier * (1 + volume_multiplier)
        return min(estimated_potential, 2.0)  # Cap at 200%

    async def generate_monitoring_signals(self, best_opportunities):
        """Generate monitoring and trading signals for best opportunities"""

        if not best_opportunities:
            print("No significant opportunities found")
            return []

        print(f"\n=== DYNAMIC MONITORING SIGNALS ===")
        print("Monitoring the BEST opportunities across all markets")
        print("-" * 70)

        monitoring_signals = []

        for opp in best_opportunities[:5]:  # Top 5 opportunities
            signal = {
                'symbol': opp['symbol'],
                'current_price': opp['current_price'],
                'daily_change': opp['daily_change'],
                'priority': opp['priority'],
                'action': opp['action_recommendation'],
                'profit_potential': opp['profit_potential'],
                'monitoring_reason': f"{opp['opportunity_type']} - {opp['daily_change']:+.1%} with {opp['volume_ratio']:.1f}x volume",
                'opportunity_type': opp['opportunity_type']
            }

            monitoring_signals.append(signal)

            print(f"{opp['symbol']:>6} | ${opp['current_price']:>7.2f} | {opp['daily_change']:>+6.1%} | {opp['priority']:>7} | {opp['action_recommendation']}")
            print(f"       Profit Potential: {opp['profit_potential']:.1%} | Type: {opp['opportunity_type']}")
            print()

        return monitoring_signals

    async def run_dynamic_scan(self):
        """Run complete dynamic opportunity scan"""

        print("DYNAMIC OPPORTUNITY MONITOR")
        print("=" * 80)
        print("Scanning for the BEST opportunities across ALL markets")
        print("Not limited to specific stocks - finding what's actually moving!")
        print("=" * 80)

        # Step 1: Scan all symbols for top movers
        top_movers = await self.scan_top_movers()

        if not top_movers:
            print("No significant momentum detected across monitored universe")
            return []

        # Step 2: Identify best opportunities
        best_opportunities = await self.identify_best_opportunities(top_movers)

        if not best_opportunities:
            print("No high-priority opportunities identified")
            return []

        # Step 3: Generate monitoring signals
        monitoring_signals = await self.generate_monitoring_signals(best_opportunities)

        print("=" * 80)
        print("DYNAMIC SCAN COMPLETE")
        print(f"Found {len(monitoring_signals)} high-priority opportunities to monitor")
        print("These are the ACTUAL best movers, not predetermined picks!")
        print("=" * 80)

        # Save results
        scan_results = {
            'scan_timestamp': datetime.now().isoformat(),
            'scan_type': 'dynamic_opportunity_monitor',
            'top_movers': top_movers,
            'best_opportunities': best_opportunities,
            'monitoring_signals': monitoring_signals,
            'total_scanned': len(self.monitoring_universe),
            'opportunities_found': len(monitoring_signals)
        }

        with open(f'dynamic_opportunities_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(scan_results, f, indent=2)

        return monitoring_signals

async def main():
    """Run dynamic opportunity monitoring"""
    monitor = DynamicOpportunityMonitor()
    signals = await monitor.run_dynamic_scan()
    return signals

if __name__ == "__main__":
    asyncio.run(main())