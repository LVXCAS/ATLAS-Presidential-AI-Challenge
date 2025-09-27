#!/usr/bin/env python3
"""
CATALYST NEWS MONITOR
Monitor news/catalysts for the 4 execution targets
Tracks AAPL product launch, GOOGL earnings, META developments, SPY Fed news
"""

import yfinance as yf
import time
from datetime import datetime
import logging

class CatalystNewsMonitor:
    """Monitor catalysts affecting execution targets"""

    def __init__(self):
        # Catalyst tracking for the 4 targets
        self.catalysts = {
            'AAPL': {
                'primary_catalyst': 'product_launch',
                'keywords': ['vision pro', 'iphone', 'mac', 'apple intelligence', 'product launch'],
                'impact': 'HIGH',
                'options_play': 'CALL',
                'timeframe': '13 days'
            },
            'GOOGL': {
                'primary_catalyst': 'q4_earnings',
                'keywords': ['earnings', 'youtube', 'cloud', 'advertising', 'alphabet'],
                'impact': 'HIGH',
                'options_play': 'CALL',
                'timeframe': '20 days'
            },
            'META': {
                'primary_catalyst': 'genetic_score',
                'keywords': ['meta', 'facebook', 'instagram', 'whatsapp', 'metaverse', 'ai'],
                'impact': 'MEDIUM',
                'options_play': 'STOCK',
                'timeframe': 'ongoing'
            },
            'SPY': {
                'primary_catalyst': 'fed_meeting',
                'keywords': ['fed', 'rate cut', 'powell', 'fomc', 'interest rates'],
                'impact': 'HIGH',
                'options_play': 'STOCK',
                'timeframe': '17 days'
            }
        }

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - CATALYST - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_recent_news(self, symbol):
        """Get recent news for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            if news and len(news) > 0:
                # Get last 3 news items
                recent_news = []
                for item in news[:3]:
                    recent_news.append({
                        'title': item.get('title', 'No title'),
                        'time': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%H:%M'),
                        'publisher': item.get('publisher', 'Unknown')
                    })
                return recent_news
            else:
                return [{'title': 'No recent news', 'time': 'N/A', 'publisher': 'N/A'}]

        except Exception as e:
            return [{'title': f'Error: {str(e)[:30]}', 'time': 'ERROR', 'publisher': 'ERROR'}]

    def analyze_catalyst_impact(self, symbol, news_items):
        """Analyze if news impacts our catalyst thesis"""
        catalyst_info = self.catalysts.get(symbol, {})
        keywords = catalyst_info.get('keywords', [])

        impact_score = 0
        relevant_news = []

        for news in news_items:
            title_lower = news['title'].lower()

            # Check for catalyst keywords
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in title_lower)

            if keyword_matches > 0:
                impact_score += keyword_matches * 10
                relevant_news.append({
                    'title': news['title'][:50] + '...' if len(news['title']) > 50 else news['title'],
                    'matches': keyword_matches,
                    'time': news['time']
                })

        # Determine impact level
        if impact_score >= 20:
            impact_level = "[HIGH] HIGH IMPACT"
        elif impact_score >= 10:
            impact_level = "[MED] MEDIUM IMPACT"
        elif impact_score > 0:
            impact_level = "[LOW] LOW IMPACT"
        else:
            impact_level = "[BASE] BASELINE"

        return {
            'impact_score': impact_score,
            'impact_level': impact_level,
            'relevant_news': relevant_news
        }

    def run_catalyst_monitor(self):
        """Run continuous catalyst monitoring"""
        print("CATALYST NEWS MONITOR - INTEL-PUTS-STYLE TRADES")
        print("=" * 80)
        print("Monitoring news/catalysts for 4 execution targets")
        print("AAPL: Product Launch | GOOGL: Earnings | META: General | SPY: Fed")
        print("=" * 80)

        while True:
            try:
                print(f"\n=== CATALYST UPDATE - {datetime.now().strftime('%H:%M:%S')} ===")

                for symbol, catalyst_info in self.catalysts.items():
                    print(f"\n{symbol} - {catalyst_info['primary_catalyst'].upper()} CATALYST:")
                    print(f"Impact: {catalyst_info['impact']} | Timeframe: {catalyst_info['timeframe']}")

                    # Get news
                    news = self.get_recent_news(symbol)

                    # Analyze catalyst impact
                    analysis = self.analyze_catalyst_impact(symbol, news)

                    print(f"News Impact: {analysis['impact_level']} (Score: {analysis['impact_score']})")

                    # Show relevant news
                    if analysis['relevant_news']:
                        print("[NEWS] Relevant News:")
                        for news_item in analysis['relevant_news'][:2]:  # Top 2
                            print(f"  - [{news_item['time']}] {news_item['title']}")
                    else:
                        print("[HEADLINES] Recent Headlines:")
                        for news_item in news[:1]:  # Just show 1 headline
                            print(f"  - [{news_item['time']}] {news_item['title'][:60]}...")

                # Overall catalyst status
                print(f"\n{'='*60}")
                print("CATALYST READINESS SUMMARY:")
                print("AAPL CALLS: Product launch momentum - 269.2% potential")
                print("GOOGL CALLS: Q4 earnings setup - 266.7% potential")
                print("META STOCK: High genetic score (7.20) - 35% allocation")
                print("SPY STOCK: Fed meeting catalyst - 28% allocation")
                print(f"{'='*60}")

                time.sleep(60)  # Update every minute

            except KeyboardInterrupt:
                print("\nCatalyst monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Catalyst monitor error: {e}")
                time.sleep(30)

def main():
    """Run catalyst news monitoring"""
    monitor = CatalystNewsMonitor()
    monitor.run_catalyst_monitor()

if __name__ == "__main__":
    main()