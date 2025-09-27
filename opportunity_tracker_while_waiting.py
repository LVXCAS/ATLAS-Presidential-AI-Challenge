#!/usr/bin/env python3
"""
OPPORTUNITY TRACKER WHILE WAITING
Track Intel-puts-style opportunities while account is restricted
Build a pipeline of high-conviction trades for immediate execution
"""

import yfinance as yf
import time
from datetime import datetime, timedelta
import json

class OpportunityTrackerWhileWaiting:
    """Track and rank opportunities while waiting for account access"""

    def __init__(self):
        # Track the current opportunities
        self.current_opportunities = {}

        # Key Intel-puts-style indicators to monitor
        self.watch_list = {
            'META': {
                'current_priority': 'HIGH',
                'catalyst': 'Genetic Score 7.20',
                'target_allocation': '35%',
                'reasoning': 'AI-optimized top pick'
            },
            'SPY': {
                'current_priority': 'HIGH',
                'catalyst': 'Fed Meeting',
                'target_allocation': '28%',
                'reasoning': 'Rate cut catalyst'
            },
            'AAPL': {
                'current_priority': 'MEDIUM',
                'catalyst': 'Product Launch',
                'target_allocation': '20%',
                'reasoning': '269% options potential'
            },
            'GOOGL': {
                'current_priority': 'MEDIUM',
                'catalyst': 'Q4 Earnings',
                'target_allocation': '10%',
                'reasoning': '267% options potential'
            },
            # Additional Intel-puts-style candidates
            'NVDA': {
                'current_priority': 'WATCH',
                'catalyst': 'AI Earnings',
                'target_allocation': 'TBD',
                'reasoning': 'High volatility potential'
            },
            'TSLA': {
                'current_priority': 'WATCH',
                'catalyst': 'Delivery Numbers',
                'target_allocation': 'TBD',
                'reasoning': 'Momentum potential'
            }
        }

    def get_real_time_data(self, symbol):
        """Get current price and momentum data"""
        try:
            ticker = yf.Ticker(symbol)

            # Get recent price data
            hist = ticker.history(period='5d', interval='1h')
            if hist.empty:
                return None

            current_price = float(hist['Close'].iloc[-1])
            yesterday_close = float(hist['Close'].iloc[-24])  # 24 hours ago
            daily_change = (current_price - yesterday_close) / yesterday_close * 100

            # Get volume momentum
            recent_volume = int(hist['Volume'].iloc[-1])
            avg_volume = int(hist['Volume'].mean())
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

            return {
                'symbol': symbol,
                'current_price': current_price,
                'daily_change_pct': daily_change,
                'volume_ratio': volume_ratio,
                'momentum_score': abs(daily_change) * volume_ratio,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }

        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None

    def calculate_intel_puts_score(self, symbol, data):
        """Calculate Intel-puts-style conviction score"""
        if not data:
            return 0

        watch_info = self.watch_list.get(symbol, {})

        score = 0

        # Base score from watch list priority
        priority_scores = {'HIGH': 100, 'MEDIUM': 50, 'WATCH': 25}
        score += priority_scores.get(watch_info.get('current_priority', 'WATCH'), 25)

        # Momentum component (like Intel puts explosive move)
        momentum_bonus = min(abs(data['daily_change_pct']) * 5, 50)
        score += momentum_bonus

        # Volume surge component (institutional interest)
        if data['volume_ratio'] > 1.5:
            score += 25
        elif data['volume_ratio'] > 2.0:
            score += 50

        # Volatility bonus (big moves = big opportunity)
        if abs(data['daily_change_pct']) > 3:
            score += 30
        elif abs(data['daily_change_pct']) > 5:
            score += 60

        return min(score, 200)  # Cap at 200

    def scan_current_opportunities(self):
        """Scan all watch list symbols for opportunities"""
        print(f"\n=== OPPORTUNITY SCAN - {datetime.now().strftime('%H:%M:%S')} ===")

        opportunities = []

        for symbol in self.watch_list.keys():
            data = self.get_real_time_data(symbol)
            if data:
                score = self.calculate_intel_puts_score(symbol, data)
                watch_info = self.watch_list[symbol]

                opportunity = {
                    'symbol': symbol,
                    'score': score,
                    'price': data['current_price'],
                    'daily_change': data['daily_change_pct'],
                    'volume_ratio': data['volume_ratio'],
                    'priority': watch_info['current_priority'],
                    'catalyst': watch_info['catalyst'],
                    'allocation': watch_info['target_allocation'],
                    'reasoning': watch_info['reasoning']
                }

                opportunities.append(opportunity)

        # Sort by Intel-puts conviction score
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        return opportunities

    def display_opportunity_rankings(self, opportunities):
        """Display ranked opportunities Intel-puts-style"""
        print("INTEL-PUTS-STYLE OPPORTUNITY RANKINGS")
        print("=" * 80)
        print("Score | Symbol | Price  | Change | Vol | Priority | Catalyst")
        print("-" * 80)

        for opp in opportunities:
            print(f"{opp['score']:>5} | {opp['symbol']:>6} | "
                  f"${opp['price']:>6.2f} | {opp['daily_change']:>5.1f}% | "
                  f"{opp['volume_ratio']:>3.1f}x | {opp['priority']:>8} | "
                  f"{opp['catalyst']}")

        print("-" * 80)

        # Highlight top 2 opportunities
        if len(opportunities) >= 2:
            top1, top2 = opportunities[0], opportunities[1]
            print(f"\n[TOP CONVICTION] {top1['symbol']} - Score: {top1['score']}")
            print(f"  Catalyst: {top1['catalyst']}")
            print(f"  Reasoning: {top1['reasoning']}")
            print(f"  Target Allocation: {top1['allocation']}")

            print(f"\n[SECOND PICK] {top2['symbol']} - Score: {top2['score']}")
            print(f"  Catalyst: {top2['catalyst']}")
            print(f"  Reasoning: {top2['reasoning']}")
            print(f"  Target Allocation: {top2['allocation']}")

    def save_opportunity_snapshot(self, opportunities):
        """Save current opportunities for execution when ready"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'opportunities': opportunities,
            'top_picks': opportunities[:2] if len(opportunities) >= 2 else opportunities,
            'ready_for_execution': True
        }

        try:
            with open('intel_puts_opportunities.json', 'w') as f:
                json.dump(snapshot, f, indent=2)
            print(f"[SAVED] Opportunity snapshot saved")
        except Exception as e:
            print(f"[ERROR] Failed to save snapshot: {e}")

    def run_continuous_opportunity_tracking(self):
        """Run continuous opportunity tracking"""
        print("INTEL-PUTS-STYLE OPPORTUNITY TRACKER")
        print("=" * 80)
        print("Tracking high-conviction opportunities while waiting for account access")
        print("Building pipeline for immediate execution when restrictions lift")
        print("=" * 80)

        scan_count = 0

        while True:
            try:
                scan_count += 1

                # Scan current opportunities
                opportunities = self.scan_current_opportunities()

                # Display rankings
                self.display_opportunity_rankings(opportunities)

                # Save snapshot for execution
                self.save_opportunity_snapshot(opportunities)

                # Summary
                if opportunities:
                    top_score = opportunities[0]['score']
                    top_symbol = opportunities[0]['symbol']
                    print(f"\n[SCAN #{scan_count}] Top opportunity: {top_symbol} (Score: {top_score})")

                    if top_score > 150:
                        print("[ALERT] HIGH CONVICTION OPPORTUNITY - Ready for immediate execution!")
                    elif top_score > 100:
                        print("[READY] Strong opportunity - Execute when account unlocks")

                print(f"\nNext scan in 2 minutes... (Scan #{scan_count} complete)")
                time.sleep(120)  # Scan every 2 minutes

            except KeyboardInterrupt:
                print(f"\n[STOPPED] Opportunity tracking stopped after {scan_count} scans")
                break
            except Exception as e:
                print(f"[ERROR] Scan error: {e}")
                time.sleep(60)

def main():
    """Run opportunity tracker"""
    tracker = OpportunityTrackerWhileWaiting()
    tracker.run_continuous_opportunity_tracking()

if __name__ == "__main__":
    main()