#!/usr/bin/env python3
"""
SMART POSITION MONITOR
Monitors what actually matters: current positions, performance, and new opportunities
Focus on actionable intelligence, not random stock screening
"""

import os
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

class SmartPositionMonitor:
    """Smart monitoring of actual positions and actionable opportunities"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

    def analyze_portfolio_performance(self):
        """Analyze current portfolio performance and identify winners/losers"""

        print("SMART POSITION MONITORING")
        print("=" * 60)
        print("Focusing on what actually matters: performance and opportunities")
        print("=" * 60)

        try:
            # Get account status
            account = self.alpaca.get_account()
            positions = self.alpaca.list_positions()

            print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"Cash Available: ${float(account.cash):,.2f}")
            print(f"Buying Power: ${float(account.buying_power):,.2f}")

            if not positions:
                print("No current positions to monitor")
                return {}

            # Analyze position performance
            winners = []
            losers = []
            neutral = []

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

                if unrealized_plpc > 5:
                    winners.append(position_data)
                elif unrealized_plpc < -5:
                    losers.append(position_data)
                else:
                    neutral.append(position_data)

            # Sort by performance
            winners.sort(key=lambda x: x['unrealized_plpc'], reverse=True)
            losers.sort(key=lambda x: x['unrealized_plpc'])

            print(f"\n=== POSITION PERFORMANCE ANALYSIS ===")
            print(f"Total Positions: {len(positions)}")
            print(f"Winners (>5%): {len(winners)}")
            print(f"Losers (<-5%): {len(losers)}")
            print(f"Neutral: {len(neutral)}")

            # Show top winners
            if winners:
                print(f"\nTOP WINNING POSITIONS:")
                print("Symbol | Value | P&L | % Return")
                print("-" * 40)
                for pos in winners[:5]:
                    print(f"{pos['symbol']:>6} | ${pos['market_value']:>8,.0f} | ${pos['unrealized_pl']:>6,.0f} | {pos['unrealized_plpc']:>+5.1f}%")

            # Show worst losers
            if losers:
                print(f"\nWORST LOSING POSITIONS:")
                print("Symbol | Value | P&L | % Return")
                print("-" * 40)
                for pos in losers[:5]:
                    print(f"{pos['symbol']:>6} | ${pos['market_value']:>8,.0f} | ${pos['unrealized_pl']:>6,.0f} | {pos['unrealized_plpc']:>+5.1f}%")

            return {
                'winners': winners,
                'losers': losers,
                'neutral': neutral,
                'total_positions': len(positions)
            }

        except Exception as e:
            print(f"Error analyzing portfolio: {e}")
            return {}

    def identify_actionable_opportunities(self, performance_data):
        """Identify what actions to take based on current performance"""

        if not performance_data:
            return []

        print(f"\n=== ACTIONABLE OPPORTUNITIES ===")
        print("What you should actually DO, not just monitor")
        print("-" * 50)

        opportunities = []

        # Winners to consider taking profits
        strong_winners = [w for w in performance_data['winners'] if w['unrealized_plpc'] > 20]
        if strong_winners:
            print("PROFIT-TAKING OPPORTUNITIES:")
            for pos in strong_winners[:3]:
                opportunities.append({
                    'action': 'TAKE_PROFITS',
                    'symbol': pos['symbol'],
                    'current_gain': pos['unrealized_plpc'],
                    'rationale': f"Up {pos['unrealized_plpc']:+.1f}% - consider taking 50% profits",
                    'priority': 'HIGH'
                })
                print(f"   {pos['symbol']}: Up {pos['unrealized_plpc']:+.1f}% - Consider taking 50% profits")

        # Losers to consider cutting
        major_losers = [l for l in performance_data['losers'] if l['unrealized_plpc'] < -15 and l['position_size'] > 5000]
        if major_losers:
            print("\nLOSS-CUTTING OPPORTUNITIES:")
            for pos in major_losers[:3]:
                opportunities.append({
                    'action': 'CUT_LOSSES',
                    'symbol': pos['symbol'],
                    'current_loss': pos['unrealized_plpc'],
                    'rationale': f"Down {pos['unrealized_plpc']:+.1f}% - consider cutting losses",
                    'priority': 'MEDIUM'
                })
                print(f"   {pos['symbol']}: Down {pos['unrealized_plpc']:+.1f}% - Consider cutting losses")

        # Position sizing opportunities
        large_positions = [p for p in performance_data['winners'] + performance_data['neutral']
                          if p['position_size'] > 100000]
        if large_positions:
            print("\nPOSITION SIZING OPPORTUNITIES:")
            for pos in large_positions[:2]:
                opportunities.append({
                    'action': 'REBALANCE',
                    'symbol': pos['symbol'],
                    'position_size': pos['position_size'],
                    'rationale': f"${pos['position_size']:,.0f} position - consider rebalancing",
                    'priority': 'LOW'
                })
                print(f"   {pos['symbol']}: ${pos['position_size']:,.0f} - Large position to rebalance")

        return opportunities

    def suggest_new_opportunities(self):
        """Suggest new opportunities based on available buying power"""

        try:
            account = self.alpaca.get_account()
            buying_power = float(account.buying_power)

            print(f"\n=== NEW OPPORTUNITY POTENTIAL ===")
            print(f"Available Buying Power: ${buying_power:,.2f}")

            if buying_power > 50000:
                print("\nCAPITAL DEPLOYMENT OPPORTUNITIES:")
                print(f"   High conviction play: ${buying_power * 0.15:,.0f} (15% allocation)")
                print(f"   Medium conviction play: ${buying_power * 0.10:,.0f} (10% allocation)")
                print(f"   Diversified plays: ${buying_power * 0.05:,.0f} each (5% allocation)")

                return {
                    'available_capital': buying_power,
                    'high_conviction_size': buying_power * 0.15,
                    'medium_conviction_size': buying_power * 0.10,
                    'small_position_size': buying_power * 0.05
                }
            else:
                print("   Limited capital for new positions - focus on managing existing ones")
                return {'available_capital': buying_power}

        except Exception as e:
            print(f"Error checking buying power: {e}")
            return {}

    def run_smart_monitoring(self):
        """Run complete smart monitoring analysis"""

        print("SMART POSITION MONITOR")
        print("=" * 80)
        print("Monitoring what matters: Current performance and actionable opportunities")
        print("Focus on positions you own and decisions you can make")
        print("=" * 80)

        # Step 1: Analyze current portfolio
        performance = self.analyze_portfolio_performance()

        if not performance:
            print("Unable to analyze portfolio performance")
            return {}

        # Step 2: Identify actionable opportunities
        actions = self.identify_actionable_opportunities(performance)

        # Step 3: Suggest new opportunities
        capital_opportunities = self.suggest_new_opportunities()

        # Summary
        print("=" * 80)
        print("SMART MONITORING SUMMARY")
        print("=" * 80)

        if actions:
            print(f"Actionable opportunities: {len(actions)}")
            high_priority = [a for a in actions if a['priority'] == 'HIGH']
            if high_priority:
                print(f"HIGH PRIORITY ACTIONS: {len(high_priority)}")
                for action in high_priority:
                    print(f"  - {action['action']} {action['symbol']}: {action['rationale']}")
        else:
            print("No immediate actions required - positions are performing adequately")

        print(f"\nAvailable buying power: ${capital_opportunities.get('available_capital', 0):,.2f}")
        print("Focus on managing existing winners and cutting major losers")

        return {
            'performance_analysis': performance,
            'actionable_opportunities': actions,
            'capital_opportunities': capital_opportunities,
            'monitoring_timestamp': datetime.now().isoformat()
        }

def main():
    """Run smart position monitoring"""
    monitor = SmartPositionMonitor()
    results = monitor.run_smart_monitoring()
    return results

if __name__ == "__main__":
    main()