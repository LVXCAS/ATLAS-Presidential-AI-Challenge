#!/usr/bin/env python3
"""
INTEL-PUTS OPPORTUNITY WATCHDOG
Continuous monitoring system that tracks Intel-puts-style opportunities
Works without account API - just tracks market patterns for execution readiness
"""

import yfinance as yf
import time
from datetime import datetime, timedelta
import json
import logging
import sys

class IntelPutsOpportunityWatchdog:
    """Watchdog for Intel-puts-style opportunities"""

    def __init__(self):
        self.watch_list = {
            'META': {
                'priority': 'HIGH',
                'catalyst': 'AI Earnings Growth',
                'allocation': '35%',
                'reasoning': 'Top genetic score - AI leader'
            },
            'SPY': {
                'priority': 'HIGH',
                'catalyst': 'Fed Rate Decision',
                'allocation': '28%',
                'reasoning': 'Market catalyst timing'
            },
            'AAPL': {
                'priority': 'MEDIUM',
                'catalyst': 'Product Cycle',
                'allocation': '20%',
                'reasoning': 'Options potential 269%'
            },
            'GOOGL': {
                'priority': 'MEDIUM',
                'catalyst': 'Q4 Earnings',
                'allocation': '10%',
                'reasoning': 'Options potential 267%'
            },
            'TSLA': {
                'priority': 'WATCH',
                'catalyst': 'Delivery Numbers',
                'allocation': 'TBD',
                'reasoning': 'Volatility opportunity'
            },
            'NVDA': {
                'priority': 'WATCH',
                'catalyst': 'AI Revenue',
                'allocation': 'TBD',
                'reasoning': 'Sector momentum'
            }
        }

        self.scan_count = 0
        self.high_conviction_alerts = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - WATCHDOG - %(message)s',
            handlers=[
                logging.FileHandler('intel_puts_watchdog.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_market_data(self, symbol):
        """Get real-time market data - handles market hours and weekends"""
        try:
            ticker = yf.Ticker(symbol)

            # Try different data periods to handle market closure
            hist = None
            for period in ['5d', '1mo', '2mo']:
                hist = ticker.history(period=period, interval='1d')
                if not hist.empty and len(hist) >= 2:
                    break

            if hist is None or hist.empty:
                return None

            # Get current/latest price
            current_price = float(hist['Close'].iloc[-1])

            # Get comparison price (previous close)
            if len(hist) >= 2:
                previous_close = float(hist['Close'].iloc[-2])
                daily_change = ((current_price - previous_close) / previous_close * 100) if previous_close != 0 else 0
            else:
                daily_change = 0

            # Volume analysis
            recent_volume = int(hist['Volume'].iloc[-1])
            avg_volume = int(hist['Volume'].mean())
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

            # Weekly change (5 days back)
            if len(hist) >= 5:
                week_ago = float(hist['Close'].iloc[-5])
                weekly_change = ((current_price - week_ago) / week_ago * 100) if week_ago != 0 else 0
            else:
                weekly_change = daily_change

            return {
                'symbol': symbol,
                'price': current_price,
                'daily_change': daily_change,
                'weekly_change': weekly_change,
                'volume_ratio': volume_ratio,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }

        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {e}")
            return None

    def calculate_intel_puts_conviction_score(self, symbol, data):
        """Calculate Intel-puts-style conviction score"""
        if not data:
            return 0

        watch_info = self.watch_list.get(symbol, {})
        score = 0

        # Base score from priority
        priority_scores = {'HIGH': 100, 'MEDIUM': 60, 'WATCH': 30}
        score += priority_scores.get(watch_info.get('priority', 'WATCH'), 30)

        # Momentum scoring (Intel puts had explosive moves)
        momentum_score = min(abs(data['daily_change']) * 8, 80)
        score += momentum_score

        # Volume surge (institutional activity)
        if data['volume_ratio'] > 1.5:
            score += 30
        elif data['volume_ratio'] > 2.0:
            score += 60
        elif data['volume_ratio'] > 3.0:
            score += 100

        # Volatility explosion bonus (Intel puts key factor)
        if abs(data['daily_change']) > 2:
            score += 25
        elif abs(data['daily_change']) > 4:
            score += 50
        elif abs(data['daily_change']) > 6:
            score += 100

        # Weekly momentum component
        weekly_bonus = min(abs(data['weekly_change']) * 2, 40)
        score += weekly_bonus

        return min(int(score), 300)  # Cap at 300

    def scan_opportunities(self):
        """Scan all symbols for Intel-puts-style opportunities"""
        self.scan_count += 1
        scan_time = datetime.now().strftime('%H:%M:%S')

        print(f"\n=== INTEL-PUTS OPPORTUNITY SCAN #{self.scan_count} - {scan_time} ===")

        opportunities = []

        for symbol in self.watch_list.keys():
            data = self.get_market_data(symbol)
            if data:
                score = self.calculate_intel_puts_conviction_score(symbol, data)
                watch_info = self.watch_list[symbol]

                opp = {
                    'symbol': symbol,
                    'score': score,
                    'price': data['price'],
                    'daily_change': data['daily_change'],
                    'weekly_change': data['weekly_change'],
                    'volume_ratio': data['volume_ratio'],
                    'priority': watch_info['priority'],
                    'catalyst': watch_info['catalyst'],
                    'allocation': watch_info['allocation'],
                    'reasoning': watch_info['reasoning'],
                    'timestamp': data['timestamp']
                }

                opportunities.append(opp)

        # Sort by conviction score
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        return opportunities

    def display_opportunity_rankings(self, opportunities):
        """Display Intel-puts-style opportunity rankings"""
        print("INTEL-PUTS-STYLE CONVICTION RANKINGS")
        print("=" * 85)
        print("Score | Symbol | Price    | Daily% | Week% | Vol | Priority | Catalyst")
        print("-" * 85)

        for opp in opportunities:
            print(f"{opp['score']:>5} | {opp['symbol']:>6} | "
                  f"${opp['price']:>7.2f} | {opp['daily_change']:>5.1f}% | "
                  f"{opp['weekly_change']:>4.1f}% | {opp['volume_ratio']:>3.1f}x | "
                  f"{opp['priority']:>8} | {opp['catalyst']}")

        print("-" * 85)

        # Highlight top opportunities
        if opportunities:
            top = opportunities[0]
            print(f"\n[TOP CONVICTION] {top['symbol']} - Score: {top['score']}")
            print(f"  Price: ${top['price']:.2f} ({top['daily_change']:+.1f}%)")
            print(f"  Catalyst: {top['catalyst']}")
            print(f"  Target Allocation: {top['allocation']}")
            print(f"  Strategy: {top['reasoning']}")

            # Alert levels
            if top['score'] > 200:
                alert_msg = "[EXTREME ALERT] EXPLOSIVE INTEL-PUTS-STYLE OPPORTUNITY!"
                print(f"\n{alert_msg}")
                self.high_conviction_alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': top['symbol'],
                    'score': top['score'],
                    'alert_level': 'EXTREME'
                })
            elif top['score'] > 150:
                print(f"\n[HIGH ALERT] Strong Intel-puts-style opportunity detected")
            elif top['score'] > 100:
                print(f"\n[READY] Moderate opportunity - monitor closely")

    def save_opportunity_snapshot(self, opportunities):
        """Save opportunity snapshot for execution readiness"""
        snapshot = {
            'scan_number': self.scan_count,
            'timestamp': datetime.now().isoformat(),
            'opportunities': opportunities,
            'top_3_picks': opportunities[:3] if len(opportunities) >= 3 else opportunities,
            'high_conviction_count': sum(1 for opp in opportunities if opp['score'] > 150),
            'ready_for_execution': any(opp['score'] > 150 for opp in opportunities),
            'total_alerts': len(self.high_conviction_alerts)
        }

        try:
            with open('intel_puts_live_opportunities.json', 'w') as f:
                json.dump(snapshot, f, indent=2)

            # Also save alerts separately
            if self.high_conviction_alerts:
                with open('intel_puts_alerts.json', 'w') as f:
                    json.dump(self.high_conviction_alerts, f, indent=2)

            self.logger.info(f"Scan #{self.scan_count} snapshot saved")

        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")

    def run_continuous_watchdog(self):
        """Run continuous Intel-puts opportunity watchdog"""
        print("INTEL-PUTS OPPORTUNITY WATCHDOG - CONTINUOUS MONITORING")
        print("=" * 80)
        print("Tracking high-conviction Intel-puts-style opportunities 24/7")
        print("Building execution pipeline for account unlock")
        print("Monitoring for explosive volatility + catalyst convergence")
        print("=" * 80)

        while True:
            try:
                # Scan for opportunities
                opportunities = self.scan_opportunities()

                # Display rankings
                self.display_opportunity_rankings(opportunities)

                # Save snapshot
                self.save_opportunity_snapshot(opportunities)

                # Summary for this scan
                if opportunities:
                    top_score = opportunities[0]['score']
                    top_symbol = opportunities[0]['symbol']
                    high_conviction_count = sum(1 for opp in opportunities if opp['score'] > 150)

                    print(f"\n[SCAN SUMMARY] Top: {top_symbol} ({top_score}) | "
                          f"High Conviction: {high_conviction_count}/6 | "
                          f"Total Alerts: {len(self.high_conviction_alerts)}")

                # Next scan timing
                print(f"\nNext scan in 3 minutes... (Completed scan #{self.scan_count})")
                print("=" * 80)

                time.sleep(180)  # 3 minute intervals

            except KeyboardInterrupt:
                print(f"\n[STOPPED] Watchdog stopped after {self.scan_count} scans")
                print(f"Total high-conviction alerts generated: {len(self.high_conviction_alerts)}")
                break
            except Exception as e:
                self.logger.error(f"Watchdog error: {e}")
                print(f"Error in scan #{self.scan_count}: {e}")
                time.sleep(60)  # Wait 1 minute on error

def main():
    """Run Intel-puts opportunity watchdog"""
    print("Starting Intel-Puts Opportunity Watchdog...")
    watchdog = IntelPutsOpportunityWatchdog()
    watchdog.run_continuous_watchdog()

if __name__ == "__main__":
    main()