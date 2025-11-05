"""
NEWS-AWARE FOREX TRADING MODULE
================================
Integrates FRED, Polygon, OpenBB for fundamental analysis
Prevents trading against major news events (Fed decisions, BOJ policy, etc.)
"""
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class NewsForexIntegration:
    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')

        print("=" * 70)
        print("NEWS-AWARE FOREX MODULE INITIALIZED")
        print("=" * 70)
        print(f"FRED API: {'[OK]' if self.fred_api_key else '[MISSING]'}")
        print(f"Polygon API: {'[OK]' if self.polygon_api_key else '[MISSING]'}")
        print(f"Alpha Vantage API: {'[OK]' if self.alpha_vantage_key else '[MISSING]'}")
        print("=" * 70)

    def get_fed_policy_sentiment(self):
        """
        Get Federal Reserve policy sentiment from FRED
        Returns: 'hawkish' (bullish USD), 'dovish' (bearish USD), or 'neutral'
        """
        try:
            # Federal Funds Rate (current policy rate)
            fed_rate_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={self.fred_api_key}&file_type=json&sort_order=desc&limit=2"

            response = requests.get(fed_rate_url, timeout=10)
            data = response.json()

            if 'observations' in data and len(data['observations']) >= 2:
                current_rate = float(data['observations'][0]['value'])
                previous_rate = float(data['observations'][1]['value'])

                rate_change = current_rate - previous_rate

                # Determine sentiment
                if rate_change > 0:
                    sentiment = 'hawkish'  # Raising rates = strong USD
                    reason = f"Fed raised rates by {rate_change:.2f}%"
                elif rate_change < 0:
                    sentiment = 'dovish'  # Cutting rates = weak USD
                    reason = f"Fed cut rates by {abs(rate_change):.2f}%"
                else:
                    sentiment = 'neutral'
                    reason = f"Fed held rates at {current_rate}%"

                return {
                    'sentiment': sentiment,
                    'current_rate': current_rate,
                    'rate_change': rate_change,
                    'reason': reason
                }
        except Exception as e:
            print(f"  [WARN] FRED API error: {e}")
            return {'sentiment': 'neutral', 'reason': 'API unavailable'}

    def get_boj_policy_sentiment(self):
        """
        Get Bank of Japan policy sentiment
        Returns: 'hawkish' (bullish JPY), 'dovish' (bearish JPY), or 'neutral'
        """
        try:
            # Japan Policy Rate from FRED
            boj_rate_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=INTDSRJPM193N&api_key={self.fred_api_key}&file_type=json&sort_order=desc&limit=2"

            response = requests.get(boj_rate_url, timeout=10)
            data = response.json()

            if 'observations' in data and len(data['observations']) >= 1:
                current_rate = float(data['observations'][0]['value'])

                # BOJ has been ultra-dovish (negative rates)
                if current_rate < 0:
                    sentiment = 'dovish'  # Negative rates = weak JPY
                    reason = f"BOJ holding negative rates at {current_rate}%"
                elif current_rate == 0:
                    sentiment = 'neutral'
                    reason = f"BOJ at zero rates"
                else:
                    sentiment = 'hawkish'  # Positive rates = strong JPY
                    reason = f"BOJ raised rates to {current_rate}%"

                return {
                    'sentiment': sentiment,
                    'current_rate': current_rate,
                    'reason': reason
                }
        except Exception as e:
            print(f"  [WARN] BOJ data error: {e}")
            return {'sentiment': 'neutral', 'reason': 'Data unavailable'}

    def get_ecb_policy_sentiment(self):
        """
        Get European Central Bank policy sentiment
        Returns: 'hawkish' (bullish EUR), 'dovish' (bearish EUR), or 'neutral'
        """
        try:
            # ECB Deposit Facility Rate (main policy rate)
            ecb_rate_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=ECBDFR&api_key={self.fred_api_key}&file_type=json&sort_order=desc&limit=2"

            response = requests.get(ecb_rate_url, timeout=10)
            data = response.json()

            if 'observations' in data and len(data['observations']) >= 2:
                current_rate = float(data['observations'][0]['value'])
                previous_rate = float(data['observations'][1]['value'])

                rate_change = current_rate - previous_rate

                if rate_change > 0:
                    sentiment = 'hawkish'  # Raising rates = strong EUR
                    reason = f"ECB raised rates by {rate_change:.2f}%"
                elif rate_change < 0:
                    sentiment = 'dovish'  # Cutting rates = weak EUR
                    reason = f"ECB cut rates by {abs(rate_change):.2f}%"
                else:
                    sentiment = 'neutral'
                    reason = f"ECB holding rates at {current_rate}%"

                return {
                    'sentiment': sentiment,
                    'current_rate': current_rate,
                    'rate_change': rate_change,
                    'reason': reason
                }
        except Exception as e:
            print(f"  [WARN] ECB data error: {e}")
            return {'sentiment': 'neutral', 'reason': 'Data unavailable'}

    def get_boe_policy_sentiment(self):
        """
        Get Bank of England policy sentiment
        Returns: 'hawkish' (bullish GBP), 'dovish' (bearish GBP), or 'neutral'
        """
        try:
            # UK Bank Rate (BOE policy rate)
            boe_rate_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=BOERUKM&api_key={self.fred_api_key}&file_type=json&sort_order=desc&limit=2"

            response = requests.get(boe_rate_url, timeout=10)
            data = response.json()

            if 'observations' in data and len(data['observations']) >= 2:
                current_rate = float(data['observations'][0]['value'])
                previous_rate = float(data['observations'][1]['value'])

                rate_change = current_rate - previous_rate

                if rate_change > 0:
                    sentiment = 'hawkish'  # Raising rates = strong GBP
                    reason = f"BOE raised rates by {rate_change:.2f}%"
                elif rate_change < 0:
                    sentiment = 'dovish'  # Cutting rates = weak GBP
                    reason = f"BOE cut rates by {abs(rate_change):.2f}%"
                else:
                    sentiment = 'neutral'
                    reason = f"BOE holding rates at {current_rate}%"

                return {
                    'sentiment': sentiment,
                    'current_rate': current_rate,
                    'rate_change': rate_change,
                    'reason': reason
                }
        except Exception as e:
            print(f"  [WARN] BOE data error: {e}")
            return {'sentiment': 'neutral', 'reason': 'Data unavailable'}

    def get_forex_news_sentiment(self, pair):
        """
        Get recent news sentiment for forex pair from Polygon
        Returns: 'bullish', 'bearish', or 'neutral'
        """
        try:
            # Polygon ticker news endpoint
            # Convert OANDA format (EUR_USD) to Polygon format (C:EURUSD)
            polygon_ticker = f"C:{pair.replace('_', '')}"

            url = f"https://api.polygon.io/v2/reference/news?ticker={polygon_ticker}&limit=10&apiKey={self.polygon_api_key}"

            response = requests.get(url, timeout=10)
            data = response.json()

            if 'results' in data and len(data['results']) > 0:
                # Count sentiment keywords in recent news
                bullish_count = 0
                bearish_count = 0

                bullish_keywords = ['rally', 'surge', 'gain', 'rise', 'strong', 'bullish', 'climb']
                bearish_keywords = ['fall', 'drop', 'decline', 'weak', 'bearish', 'plunge', 'sink']

                for article in data['results'][:10]:  # Check last 10 articles
                    title = article.get('title', '').lower()
                    description = article.get('description', '').lower()
                    text = title + ' ' + description

                    for keyword in bullish_keywords:
                        if keyword in text:
                            bullish_count += 1

                    for keyword in bearish_keywords:
                        if keyword in text:
                            bearish_count += 1

                # Determine sentiment
                if bullish_count > bearish_count + 2:
                    sentiment = 'bullish'
                elif bearish_count > bullish_count + 2:
                    sentiment = 'bearish'
                else:
                    sentiment = 'neutral'

                return {
                    'sentiment': sentiment,
                    'bullish_count': bullish_count,
                    'bearish_count': bearish_count,
                    'news_count': len(data['results'])
                }
        except Exception as e:
            print(f"  [WARN] Polygon news error: {e}")
            return {'sentiment': 'neutral', 'news_count': 0}

    def get_economic_calendar_events(self):
        """
        Get upcoming high-impact economic events from Alpha Vantage
        Returns: List of events within next 7 days
        """
        try:
            url = f"https://www.alphavantage.co/query?function=ECONOMIC_CALENDAR&apikey={self.alpha_vantage_key}"

            response = requests.get(url, timeout=10)
            data = response.json()

            if 'data' in data:
                # Filter for high-impact events in next 7 days
                high_impact_events = []
                today = datetime.now()
                week_later = today + timedelta(days=7)

                for event in data['data']:
                    if event.get('importance') == 'High':
                        # Parse date and time if available
                        try:
                            event_date_str = event.get('date', '')
                            event_time_str = event.get('time', '00:00')

                            # Combine date and time
                            event_datetime_str = f"{event_date_str} {event_time_str}"
                            event_datetime = datetime.strptime(event_datetime_str, '%Y-%m-%d %H:%M')

                            if today <= event_datetime <= week_later:
                                high_impact_events.append({
                                    'datetime': event_datetime,
                                    'date': event_date_str,
                                    'time': event_time_str,
                                    'event': event.get('event', 'Unknown'),
                                    'currency': event.get('currency', 'USD'),
                                    'impact': event.get('importance', 'High')
                                })
                        except:
                            # If time parsing fails, still add event with date only
                            try:
                                event_date = datetime.strptime(event.get('date', ''), '%Y-%m-%d')
                                if today.date() <= event_date.date() <= week_later.date():
                                    high_impact_events.append({
                                        'datetime': event_date,
                                        'date': event.get('date', ''),
                                        'time': 'Unknown',
                                        'event': event.get('event', 'Unknown'),
                                        'currency': event.get('currency', 'USD'),
                                        'impact': event.get('importance', 'High')
                                    })
                            except:
                                pass

                return high_impact_events
        except Exception as e:
            print(f"  [WARN] Economic calendar error: {e}")
            return []

    def is_safe_to_trade_v2(self):
        """
        Practical time-based risk filter (no external API needed)
        Blocks trading during known high-volatility windows

        Returns:
            {
                'safe': bool,
                'reason': str
            }
        """
        try:
            now = datetime.now()
            current_hour = now.hour
            current_minute = now.minute
            day_of_week = now.weekday()  # 0=Monday, 4=Friday
            day_of_month = now.day

            # Rule 1: Block first Friday of month 8:00-9:00 AM (NFP)
            is_first_friday = (day_of_week == 4 and 1 <= day_of_month <= 7)
            is_nfp_window = (8 <= current_hour < 9)

            if is_first_friday and is_nfp_window:
                return {
                    'safe': False,
                    'reason': 'NFP (Non-Farm Payrolls) window - trading blocked'
                }

            # Rule 2: Block London open (3:00-3:30 AM EST) - high volatility
            if current_hour == 3 and current_minute < 30:
                return {
                    'safe': False,
                    'reason': 'London session open - high volatility window'
                }

            # Rule 3: Block New York open (8:00-8:30 AM EST) - high volatility
            if current_hour == 8 and current_minute < 30:
                return {
                    'safe': False,
                    'reason': 'New York session open - high volatility window'
                }

            # Rule 4: Block late night (11 PM - 2 AM EST) - low liquidity
            if current_hour >= 23 or current_hour < 2:
                return {
                    'safe': False,
                    'reason': 'Low liquidity window (after hours)'
                }

            # All checks passed - safe to trade
            return {
                'safe': True,
                'reason': 'No high-risk time windows detected'
            }

        except Exception as e:
            print(f"  [WARN] Safety check error: {e}")
            return {
                'safe': True,
                'reason': 'Safety check failed - assuming safe'
            }

    def is_safe_to_trade(self, currencies=['USD', 'EUR', 'GBP', 'JPY'], buffer_minutes=30):
        """
        Check if it's safe to trade based on upcoming economic events

        Args:
            currencies: List of currencies to check (default: USD, EUR, GBP, JPY)
            buffer_minutes: Minutes before/after event to block trading (default: 30)

        Returns:
            {
                'safe': bool,
                'reason': str,
                'next_event': dict or None
            }
        """
        try:
            events = self.get_economic_calendar_events()

            if not events:
                # No events found (or API error) - assume safe
                return {
                    'safe': True,
                    'reason': 'No high-impact events detected',
                    'next_event': None
                }

            now = datetime.now()
            buffer = timedelta(minutes=buffer_minutes)

            # Check each event
            for event in events:
                event_time = event.get('datetime')
                event_currency = event.get('currency', 'USD')

                # Only check events for currencies we trade
                if event_currency not in currencies:
                    continue

                # Check if event is within danger window
                time_until_event = event_time - now
                time_since_event = now - event_time

                # Block if event is within buffer window (before or after)
                if timedelta(0) <= time_until_event <= buffer:
                    # Event is coming up soon
                    minutes_until = int(time_until_event.total_seconds() / 60)
                    return {
                        'safe': False,
                        'reason': f"{event['event']} ({event_currency}) in {minutes_until} minutes",
                        'next_event': event
                    }
                elif timedelta(0) <= time_since_event <= buffer:
                    # Event just happened
                    minutes_since = int(time_since_event.total_seconds() / 60)
                    return {
                        'safe': False,
                        'reason': f"{event['event']} ({event_currency}) happened {minutes_since} minutes ago",
                        'next_event': event
                    }

            # No dangerous events in window
            return {
                'safe': True,
                'reason': 'No high-impact events in next 30 minutes',
                'next_event': None
            }

        except Exception as e:
            print(f"  [WARN] Safety check error: {e}")
            # On error, assume safe (don't want to block trading due to API issues)
            return {
                'safe': True,
                'reason': 'Safety check failed - assuming safe',
                'next_event': None
            }

    def should_trade_usdjpy(self):
        """
        Determine if USD/JPY should be traded based on fundamentals

        Returns: {
            'tradeable': bool,
            'direction': 'long' or 'short' or None,
            'confidence': 0-100,
            'reason': str
        }
        """
        print("\n" + "=" * 70)
        print("USD/JPY FUNDAMENTAL ANALYSIS")
        print("=" * 70)

        # Get Fed policy
        fed_policy = self.get_fed_policy_sentiment()
        print(f"\nFed Policy: {fed_policy['sentiment'].upper()}")
        print(f"  > {fed_policy['reason']}")

        # Get BOJ policy
        boj_policy = self.get_boj_policy_sentiment()
        print(f"\nBOJ Policy: {boj_policy['sentiment'].upper()}")
        print(f"  > {boj_policy['reason']}")

        # Get news sentiment
        news_sentiment = self.get_forex_news_sentiment('USD_JPY')
        if news_sentiment:
            print(f"\nNews Sentiment: {news_sentiment['sentiment'].upper()}")
            print(f"  > {news_sentiment['bullish_count']} bullish articles, {news_sentiment['bearish_count']} bearish articles")
        else:
            news_sentiment = {'sentiment': 'neutral', 'bullish_count': 0, 'bearish_count': 0}
            print(f"\nNews Sentiment: NEUTRAL (data unavailable)")

        # Scoring system
        score = 0
        reasons = []

        # Fed hawkish = USD strength = USD/JPY long
        if fed_policy['sentiment'] == 'hawkish':
            score += 3
            reasons.append("Fed hawkish (USD strong)")
        elif fed_policy['sentiment'] == 'dovish':
            score -= 3
            reasons.append("Fed dovish (USD weak)")

        # BOJ dovish = JPY weakness = USD/JPY long
        if boj_policy['sentiment'] == 'dovish':
            score += 2
            reasons.append("BOJ dovish (JPY weak)")
        elif boj_policy['sentiment'] == 'hawkish':
            score -= 2
            reasons.append("BOJ hawkish (JPY strong)")

        # News sentiment
        if news_sentiment['sentiment'] == 'bullish':
            score += 1
            reasons.append("News bullish")
        elif news_sentiment['sentiment'] == 'bearish':
            score -= 1
            reasons.append("News bearish")

        # Determine tradeable direction
        if score >= 2:
            direction = 'long'
            tradeable = True
            confidence = min(100, (score / 6) * 100)
        elif score <= -2:
            direction = 'short'
            tradeable = True
            confidence = min(100, (abs(score) / 6) * 100)
        else:
            direction = None
            tradeable = False
            confidence = 0

        print("\n" + "=" * 70)
        print(f"FUNDAMENTAL SCORE: {score}/6")
        print(f"RECOMMENDATION: {'TRADE ' + direction.upper() if tradeable else 'DO NOT TRADE'}")
        print(f"CONFIDENCE: {confidence:.0f}%")
        print(f"REASONING:")
        for reason in reasons:
            print(f"  - {reason}")
        print("=" * 70 + "\n")

        return {
            'tradeable': tradeable,
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'reasons': reasons
        }

    def should_trade_eurusd(self):
        """
        Determine if EUR/USD should be traded based on fundamentals

        Returns: {
            'tradeable': bool,
            'direction': 'long' or 'short' or None,
            'confidence': 0-100,
            'reason': str
        }
        """
        print("\n" + "=" * 70)
        print("EUR/USD FUNDAMENTAL ANALYSIS")
        print("=" * 70)

        # Get ECB policy
        ecb_policy = self.get_ecb_policy_sentiment()
        print(f"\nECB Policy: {ecb_policy['sentiment'].upper()}")
        print(f"  > {ecb_policy['reason']}")

        # Get Fed policy (for USD side)
        fed_policy = self.get_fed_policy_sentiment()
        print(f"\nFed Policy: {fed_policy['sentiment'].upper()}")
        print(f"  > {fed_policy['reason']}")

        # Get news sentiment
        news_sentiment = self.get_forex_news_sentiment('EUR_USD')
        if news_sentiment:
            print(f"\nNews Sentiment: {news_sentiment['sentiment'].upper()}")
            print(f"  > {news_sentiment['bullish_count']} bullish articles, {news_sentiment['bearish_count']} bearish articles")
        else:
            news_sentiment = {'sentiment': 'neutral', 'bullish_count': 0, 'bearish_count': 0}
            print(f"\nNews Sentiment: NEUTRAL (data unavailable)")

        # Scoring system
        score = 0
        reasons = []

        # ECB hawkish = EUR strength = EUR/USD long
        if ecb_policy['sentiment'] == 'hawkish':
            score += 3
            reasons.append("ECB hawkish (EUR strong)")
        elif ecb_policy['sentiment'] == 'dovish':
            score -= 3
            reasons.append("ECB dovish (EUR weak)")

        # Fed dovish = USD weakness = EUR/USD long
        if fed_policy['sentiment'] == 'dovish':
            score += 2
            reasons.append("Fed dovish (USD weak)")
        elif fed_policy['sentiment'] == 'hawkish':
            score -= 2
            reasons.append("Fed hawkish (USD strong)")

        # News sentiment
        if news_sentiment['sentiment'] == 'bullish':
            score += 1
            reasons.append("News bullish")
        elif news_sentiment['sentiment'] == 'bearish':
            score -= 1
            reasons.append("News bearish")

        # Determine tradeable direction
        if score >= 2:
            direction = 'long'
            tradeable = True
            confidence = min(100, (score / 6) * 100)
        elif score <= -2:
            direction = 'short'
            tradeable = True
            confidence = min(100, (abs(score) / 6) * 100)
        else:
            direction = None
            tradeable = False
            confidence = 0

        print("\n" + "=" * 70)
        print(f"FUNDAMENTAL SCORE: {score}/6")
        print(f"RECOMMENDATION: {'TRADE ' + direction.upper() if tradeable else 'DO NOT TRADE'}")
        print(f"CONFIDENCE: {confidence:.0f}%")
        print(f"REASONING:")
        for reason in reasons:
            print(f"  - {reason}")
        print("=" * 70 + "\n")

        return {
            'tradeable': tradeable,
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'reasons': reasons
        }

    def should_trade_gbpusd(self):
        """
        Determine if GBP/USD should be traded based on fundamentals

        Returns: {
            'tradeable': bool,
            'direction': 'long' or 'short' or None,
            'confidence': 0-100,
            'reason': str
        }
        """
        print("\n" + "=" * 70)
        print("GBP/USD FUNDAMENTAL ANALYSIS")
        print("=" * 70)

        # Get BOE policy
        boe_policy = self.get_boe_policy_sentiment()
        print(f"\nBOE Policy: {boe_policy['sentiment'].upper()}")
        print(f"  > {boe_policy['reason']}")

        # Get Fed policy (for USD side)
        fed_policy = self.get_fed_policy_sentiment()
        print(f"\nFed Policy: {fed_policy['sentiment'].upper()}")
        print(f"  > {fed_policy['reason']}")

        # Get news sentiment
        news_sentiment = self.get_forex_news_sentiment('GBP_USD')
        if news_sentiment:
            print(f"\nNews Sentiment: {news_sentiment['sentiment'].upper()}")
            print(f"  > {news_sentiment['bullish_count']} bullish articles, {news_sentiment['bearish_count']} bearish articles")
        else:
            news_sentiment = {'sentiment': 'neutral', 'bullish_count': 0, 'bearish_count': 0}
            print(f"\nNews Sentiment: NEUTRAL (data unavailable)")

        # Scoring system
        score = 0
        reasons = []

        # BOE hawkish = GBP strength = GBP/USD long
        if boe_policy['sentiment'] == 'hawkish':
            score += 3
            reasons.append("BOE hawkish (GBP strong)")
        elif boe_policy['sentiment'] == 'dovish':
            score -= 3
            reasons.append("BOE dovish (GBP weak)")

        # Fed dovish = USD weakness = GBP/USD long
        if fed_policy['sentiment'] == 'dovish':
            score += 2
            reasons.append("Fed dovish (USD weak)")
        elif fed_policy['sentiment'] == 'hawkish':
            score -= 2
            reasons.append("Fed hawkish (USD strong)")

        # News sentiment
        if news_sentiment['sentiment'] == 'bullish':
            score += 1
            reasons.append("News bullish")
        elif news_sentiment['sentiment'] == 'bearish':
            score -= 1
            reasons.append("News bearish")

        # Determine tradeable direction
        if score >= 2:
            direction = 'long'
            tradeable = True
            confidence = min(100, (score / 6) * 100)
        elif score <= -2:
            direction = 'short'
            tradeable = True
            confidence = min(100, (abs(score) / 6) * 100)
        else:
            direction = None
            tradeable = False
            confidence = 0

        print("\n" + "=" * 70)
        print(f"FUNDAMENTAL SCORE: {score}/6")
        print(f"RECOMMENDATION: {'TRADE ' + direction.upper() if tradeable else 'DO NOT TRADE'}")
        print(f"CONFIDENCE: {confidence:.0f}%")
        print(f"REASONING:")
        for reason in reasons:
            print(f"  - {reason}")
        print("=" * 70 + "\n")

        return {
            'tradeable': tradeable,
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'reasons': reasons
        }

    def should_trade_gbpjpy(self):
        """
        Determine if GBP/JPY should be traded based on fundamentals

        Returns: {
            'tradeable': bool,
            'direction': 'long' or 'short' or None,
            'confidence': 0-100,
            'reason': str
        }
        """
        print("\n" + "=" * 70)
        print("GBP/JPY FUNDAMENTAL ANALYSIS")
        print("=" * 70)

        # Get BOE policy
        boe_policy = self.get_boe_policy_sentiment()
        print(f"\nBOE Policy: {boe_policy['sentiment'].upper()}")
        print(f"  > {boe_policy['reason']}")

        # Get BOJ policy
        boj_policy = self.get_boj_policy_sentiment()
        print(f"\nBOJ Policy: {boj_policy['sentiment'].upper()}")
        print(f"  > {boj_policy['reason']}")

        # Get news sentiment
        news_sentiment = self.get_forex_news_sentiment('GBP_JPY')
        if news_sentiment:
            print(f"\nNews Sentiment: {news_sentiment['sentiment'].upper()}")
            print(f"  > {news_sentiment['bullish_count']} bullish articles, {news_sentiment['bearish_count']} bearish articles")
        else:
            news_sentiment = {'sentiment': 'neutral', 'bullish_count': 0, 'bearish_count': 0}
            print(f"\nNews Sentiment: NEUTRAL (data unavailable)")

        # Scoring system
        score = 0
        reasons = []

        # BOE hawkish = GBP strength = GBP/JPY long
        if boe_policy['sentiment'] == 'hawkish':
            score += 3
            reasons.append("BOE hawkish (GBP strong)")
        elif boe_policy['sentiment'] == 'dovish':
            score -= 3
            reasons.append("BOE dovish (GBP weak)")

        # BOJ dovish = JPY weakness = GBP/JPY long
        if boj_policy['sentiment'] == 'dovish':
            score += 2
            reasons.append("BOJ dovish (JPY weak)")
        elif boj_policy['sentiment'] == 'hawkish':
            score -= 2
            reasons.append("BOJ hawkish (JPY strong)")

        # News sentiment
        if news_sentiment['sentiment'] == 'bullish':
            score += 1
            reasons.append("News bullish")
        elif news_sentiment['sentiment'] == 'bearish':
            score -= 1
            reasons.append("News bearish")

        # Determine tradeable direction
        if score >= 2:
            direction = 'long'
            tradeable = True
            confidence = min(100, (score / 6) * 100)
        elif score <= -2:
            direction = 'short'
            tradeable = True
            confidence = min(100, (abs(score) / 6) * 100)
        else:
            direction = None
            tradeable = False
            confidence = 0

        print("\n" + "=" * 70)
        print(f"FUNDAMENTAL SCORE: {score}/6")
        print(f"RECOMMENDATION: {'TRADE ' + direction.upper() if tradeable else 'DO NOT TRADE'}")
        print(f"CONFIDENCE: {confidence:.0f}%")
        print(f"REASONING:")
        for reason in reasons:
            print(f"  - {reason}")
        print("=" * 70 + "\n")

        return {
            'tradeable': tradeable,
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'reasons': reasons
        }

    def check_all_pairs(self, pairs):
        """
        Check fundamental alignment for all forex pairs
        Returns dict of pair -> tradeable info
        """
        results = {}

        for pair in pairs:
            if pair == 'USD_JPY':
                results[pair] = self.should_trade_usdjpy()
            elif pair == 'EUR_USD':
                results[pair] = self.should_trade_eurusd()
            elif pair == 'GBP_USD':
                results[pair] = self.should_trade_gbpusd()
            elif pair == 'GBP_JPY':
                results[pair] = self.should_trade_gbpjpy()
            else:
                # Unknown pair - allow but with low confidence
                results[pair] = {
                    'tradeable': True,
                    'direction': None,
                    'confidence': 50,
                    'score': 0,
                    'reasons': ['No fundamental filter for this pair']
                }

        return results


if __name__ == "__main__":
    # Test the module
    news = NewsForexIntegration()

    print("\n" + "=" * 70)
    print("TESTING NEWS INTEGRATION")
    print("=" * 70)

    # Test USD/JPY fundamental analysis
    usdjpy_analysis = news.should_trade_usdjpy()

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    if usdjpy_analysis['tradeable']:
        print(f"[OK] USD/JPY is tradeable")
        print(f"[OK] Direction: {usdjpy_analysis['direction'].upper()}")
        print(f"[OK] Confidence: {usdjpy_analysis['confidence']:.0f}%")
    else:
        print(f"[NO] USD/JPY is NOT tradeable")
        print(f"[NO] Fundamentals are mixed or unclear")
    print("=" * 70)
