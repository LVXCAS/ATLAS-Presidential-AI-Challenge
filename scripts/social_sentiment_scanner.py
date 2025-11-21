"""
SOCIAL SENTIMENT SCANNER
Monitors Reddit/Twitter for trending stocks and analyzes sentiment

Key Insight:
Viral stocks often move 10-50% in 1-2 days when social media buzz peaks:
- GME: 400% in 2 weeks
- AMC: 300% in 1 week
- TSLA: Regular 5-10% moves on Elon tweets

This scanner catches them EARLY, before the main move.
"""
import os
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import Counter
import re

@dataclass
class SocialMentions:
    """Social media mentions for a stock"""
    symbol: str
    company_name: str

    # Mention counts
    total_mentions_24h: int
    reddit_mentions: int
    twitter_mentions: int  # Would require Twitter API

    # Sentiment
    bullish_mentions: int
    bearish_mentions: int
    neutral_mentions: int
    sentiment_score: float  # -1 (very bearish) to +1 (very bullish)

    # Volume indicators
    mention_velocity: float  # Mentions per hour
    vs_yesterday: float  # % change in mentions vs yesterday
    trending: bool  # Is this trending?

    # Price action
    current_price: float
    price_change_24h: float
    volume_vs_avg: float  # Volume vs 20-day average

    detected_at: str

@dataclass
class ViralStockAlert:
    """Alert for potentially viral stock"""
    symbol: str
    company_name: str
    current_price: float

    # Why it's viral
    mentions_24h: int
    sentiment_score: float  # 0-1
    mention_spike: float  # % increase in mentions
    volume_spike: float  # % increase in volume

    # Social drivers
    top_keywords: List[str]  # What people are saying
    catalyst: Optional[str]  # Identified catalyst if any

    # Technical
    price_momentum: float
    support_level: float
    resistance_level: float

    # Trade recommendation
    action: str  # 'BUY', 'WATCH', 'AVOID'
    confidence: float  # 0-1
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'

    detected_at: str

class SocialSentimentScanner:
    def __init__(self):
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Reddit API (free, no key needed for read-only)
        self.reddit_base_url = 'https://www.reddit.com'

        # Thresholds for viral detection
        self.min_mentions_24h = 50  # At least 50 mentions
        self.min_mention_spike = 2.0  # 200% increase vs yesterday
        self.min_sentiment_score = 0.5  # At least somewhat bullish
        self.min_confidence = 0.6

        # Track mention history
        self.mention_history = self._load_mention_history()

    def _load_mention_history(self) -> Dict:
        """Load historical mention data"""
        if os.path.exists('data/social_mention_history.json'):
            with open('data/social_mention_history.json') as f:
                return json.load(f)
        return {}

    def _save_mention_history(self):
        """Save mention history"""
        os.makedirs('data', exist_ok=True)
        with open('data/social_mention_history.json', 'w') as f:
            json.dump(self.mention_history, f, indent=2)

    def send_telegram_notification(self, message: str):
        """Send Telegram notification"""
        try:
            url = f'https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage'
            data = {
                'chat_id': self.telegram_chat_id,
                'text': f'VIRAL STOCK ALERT\n\n{message}'
            }
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"[SENTIMENT] Telegram notification failed: {e}")

    def scrape_reddit_mentions(self, subreddits: List[str] = ['wallstreetbets', 'stocks', 'investing']) -> Dict[str, int]:
        """Scrape Reddit for stock mentions"""
        print("[SENTIMENT] Scraping Reddit...")

        all_mentions = Counter()

        for subreddit in subreddits:
            try:
                # Get hot posts from subreddit
                url = f'{self.reddit_base_url}/r/{subreddit}/hot.json'
                headers = {'User-Agent': 'TradingBot/1.0'}
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code != 200:
                    continue

                data = response.json()
                posts = data.get('data', {}).get('children', [])

                # Extract stock tickers from titles and text
                ticker_pattern = r'\b[A-Z]{1,5}\b'  # 1-5 uppercase letters

                for post in posts:
                    post_data = post.get('data', {})
                    title = post_data.get('title', '')
                    selftext = post_data.get('selftext', '')

                    # Find tickers
                    text = f"{title} {selftext}"
                    tickers = re.findall(ticker_pattern, text)

                    # Filter out common words that aren't tickers (expanded list)
                    exclude = {
                        'A', 'I', 'THE', 'TO', 'FOR', 'AND', 'OR', 'NOT', 'BE', 'ON', 'AT', 'BY', 'IS', 'IT', 'OF', 'IN', 'AS',
                        'AI', 'US', 'CEO', 'ETF', 'IPO', 'DD', 'IMO', 'PM', 'AM', 'GO', 'ALL', 'SO', 'NOW', 'OUT', 'UP', 'DOWN',
                        'NEW', 'OLD', 'BIG', 'LOW', 'HIGH', 'TOP', 'HOT', 'FIRE', 'YOLO', 'WSB', 'GME', 'AMC', 'ARE', 'CAN',
                        'HAS', 'HAD', 'WAS', 'BUT', 'IF', 'WHEN', 'WHAT', 'WHERE', 'WHO', 'HOW', 'WHY', 'THIS', 'THAT', 'THESE',
                        'THEY', 'THEM', 'THEIR', 'YOUR', 'FROM', 'INTO', 'OVER', 'UNDER', 'AGAIN', 'ONCE', 'HERE', 'THERE'
                    }
                    tickers = [t for t in tickers if t not in exclude and len(t) <= 5]

                    # Count mentions
                    for ticker in tickers:
                        all_mentions[ticker] += 1

            except Exception as e:
                print(f"[SENTIMENT] Error scraping r/{subreddit}: {e}")

        print(f"[SENTIMENT] Found {len(all_mentions)} unique tickers on Reddit")
        return dict(all_mentions)

    def analyze_sentiment(self, symbol: str, mentions: int) -> SocialMentions:
        """Analyze sentiment for a stock"""
        try:
            # Get stock info
            stock = yf.Ticker(symbol)
            info = stock.info
            company_name = info.get('longName', symbol)

            # Get price data
            hist = stock.history(period='2d')
            if len(hist) < 2:
                return None

            current_price = float(hist['Close'].iloc[-1])
            yesterday_price = float(hist['Close'].iloc[-2])
            price_change_24h = ((current_price - yesterday_price) / yesterday_price) * 100

            # Volume analysis
            current_volume = float(hist['Volume'].iloc[-1])
            hist_60d = stock.history(period='60d')
            avg_volume = float(hist_60d['Volume'].mean())
            volume_vs_avg = (current_volume / avg_volume) if avg_volume > 0 else 1

            # Compare to yesterday's mentions
            today = datetime.now().date().isoformat()
            yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()

            yesterday_mentions = self.mention_history.get(symbol, {}).get(yesterday, 0)
            vs_yesterday = ((mentions - yesterday_mentions) / yesterday_mentions * 100) if yesterday_mentions > 0 else 100

            # Update mention history
            if symbol not in self.mention_history:
                self.mention_history[symbol] = {}
            self.mention_history[symbol][today] = mentions

            # Simple sentiment (positive if price and mentions both up)
            sentiment_score = 0.5  # Neutral default

            if price_change_24h > 0 and vs_yesterday > 0:
                sentiment_score = 0.7  # Bullish
            elif price_change_24h < 0 and vs_yesterday > 50:
                sentiment_score = 0.8  # Very bullish (buying the dip)
            elif price_change_24h > 5:
                sentiment_score = 0.9  # Very bullish

            # Estimate bullish/bearish mentions (simplified)
            if sentiment_score > 0.6:
                bullish = int(mentions * 0.7)
                bearish = int(mentions * 0.2)
                neutral = mentions - bullish - bearish
            else:
                bullish = int(mentions * 0.4)
                bearish = int(mentions * 0.3)
                neutral = mentions - bullish - bearish

            # Mention velocity (per hour)
            mention_velocity = mentions / 24

            # Trending?
            trending = mentions >= self.min_mentions_24h and vs_yesterday >= 100

            return SocialMentions(
                symbol=symbol,
                company_name=company_name,
                total_mentions_24h=mentions,
                reddit_mentions=mentions,
                twitter_mentions=0,  # Would need Twitter API
                bullish_mentions=bullish,
                bearish_mentions=bearish,
                neutral_mentions=neutral,
                sentiment_score=sentiment_score,
                mention_velocity=mention_velocity,
                vs_yesterday=vs_yesterday,
                trending=trending,
                current_price=current_price,
                price_change_24h=price_change_24h,
                volume_vs_avg=volume_vs_avg,
                detected_at=datetime.now().isoformat()
            )

        except Exception as e:
            print(f"[SENTIMENT] Error analyzing {symbol}: {e}")
            return None

    def generate_viral_alert(self, mentions: SocialMentions) -> Optional[ViralStockAlert]:
        """Generate viral stock alert if criteria met"""
        # Check if meets viral criteria
        if mentions.total_mentions_24h < self.min_mentions_24h:
            return None

        if mentions.vs_yesterday < self.min_mention_spike * 100:
            return None

        if mentions.sentiment_score < self.min_sentiment_score:
            return None

        # Get technical levels
        try:
            stock = yf.Ticker(mentions.symbol)
            hist = stock.history(period='1mo')

            support = float(hist['Low'].rolling(20).min().iloc[-1])
            resistance = float(hist['High'].rolling(20).max().iloc[-1])

            # Price momentum
            sma_20 = hist['Close'].rolling(20).mean()
            price_momentum = ((mentions.current_price - sma_20.iloc[-1]) / sma_20.iloc[-1] * 100)

        except:
            support = mentions.current_price * 0.95
            resistance = mentions.current_price * 1.05
            price_momentum = mentions.price_change_24h

        # Determine action
        if mentions.sentiment_score > 0.8 and mentions.vs_yesterday > 300:
            action = 'BUY'
            confidence = 0.8
            risk_level = 'HIGH'  # Viral stocks are risky
        elif mentions.sentiment_score > 0.6 and mentions.vs_yesterday > 200:
            action = 'WATCH'
            confidence = 0.7
            risk_level = 'MEDIUM'
        else:
            action = 'AVOID'
            confidence = 0.5
            risk_level = 'EXTREME'

        # Top keywords (simplified)
        top_keywords = ['social_media', 'trending', 'high_volume']

        # Identify catalyst
        catalyst = None
        if mentions.price_change_24h > 10:
            catalyst = 'Large price move'
        elif mentions.volume_vs_avg > 3:
            catalyst = 'Unusual volume'

        return ViralStockAlert(
            symbol=mentions.symbol,
            company_name=mentions.company_name,
            current_price=mentions.current_price,
            mentions_24h=mentions.total_mentions_24h,
            sentiment_score=mentions.sentiment_score,
            mention_spike=mentions.vs_yesterday,
            volume_spike=(mentions.volume_vs_avg - 1) * 100,
            top_keywords=top_keywords,
            catalyst=catalyst,
            price_momentum=price_momentum,
            support_level=support,
            resistance_level=resistance,
            action=action,
            confidence=confidence,
            risk_level=risk_level,
            detected_at=datetime.now().isoformat()
        )

    def scan_for_viral_stocks(self) -> List[ViralStockAlert]:
        """Scan for viral stock opportunities"""
        print("\n" + "="*70)
        print("SOCIAL SENTIMENT SCANNER - VIRAL STOCK DETECTION")
        print("="*70)

        # Scrape Reddit
        reddit_mentions = self.scrape_reddit_mentions()

        # Sort by mentions
        sorted_mentions = sorted(reddit_mentions.items(), key=lambda x: x[1], reverse=True)

        print(f"\n[SENTIMENT] Top 10 mentioned stocks:")
        for symbol, count in sorted_mentions[:10]:
            print(f"  {symbol}: {count} mentions")

        # Analyze top stocks
        alerts = []

        for symbol, count in sorted_mentions[:50]:  # Analyze top 50
            print(f"\n[SENTIMENT] Analyzing {symbol}...")

            mentions = self.analyze_sentiment(symbol, count)

            if not mentions:
                continue

            # Check if viral
            alert = self.generate_viral_alert(mentions)

            if alert:
                alerts.append(alert)
                print(f"[SENTIMENT] ✓ VIRAL: {symbol} - {alert.action} (Confidence: {alert.confidence:.0%})")

        # Save mention history
        self._save_mention_history()

        # Sort by confidence
        alerts.sort(key=lambda x: x.confidence, reverse=True)

        print(f"\n[SENTIMENT] Found {len(alerts)} viral stocks")

        return alerts

    def format_alert_report(self, alert: ViralStockAlert) -> str:
        """Format viral alert for display"""
        report = f"""
{alert.symbol} - {alert.action}
{alert.company_name}
Price: ${alert.current_price:.2f}

SOCIAL ACTIVITY:
  Mentions (24h): {alert.mentions_24h}
  Mention Spike: +{alert.mention_spike:.0f}%
  Sentiment: {alert.sentiment_score:.0%} bullish

PRICE ACTION:
  Momentum: {alert.price_momentum:+.1f}%
  Volume Spike: +{alert.volume_spike:.0f}%
  Support: ${alert.support_level:.2f}
  Resistance: ${alert.resistance_level:.2f}

CATALYST: {alert.catalyst or 'Unknown'}

RECOMMENDATION: {alert.action}
Risk Level: {alert.risk_level}
Confidence: {alert.confidence:.0%}
"""
        return report

def main():
    """Run social sentiment scanner"""
    scanner = SocialSentimentScanner()

    # Scan for viral stocks
    alerts = scanner.scan_for_viral_stocks()

    # Print top 5 alerts
    for alert in alerts[:5]:
        print(scanner.format_alert_report(alert))

if __name__ == '__main__':
    main()
