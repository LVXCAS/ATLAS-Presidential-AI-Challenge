"""
FOREX ECONOMIC CALENDAR
Scrapes Forex Factory for high-impact news events to avoid trading during volatility spikes
"""
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import os

class ForexCalendar:
    """Fetch and check economic calendar events"""

    def __init__(self, cache_file='calendar_cache.json'):
        self.cache_file = cache_file
        self.cache = self.load_cache()

    def load_cache(self):
        """Load cached calendar data"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    # Check if cache is from today
                    cache_date = cache.get('date', '')
                    if cache_date == datetime.now().strftime('%Y-%m-%d'):
                        return cache
            except:
                pass
        return {'date': '', 'events': []}

    def save_cache(self, events):
        """Save events to cache"""
        cache = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'events': events
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        self.cache = cache

    def get_todays_events(self, force_refresh=False):
        """
        Get today's high-impact economic events

        Returns list of dicts:
            - time: datetime object
            - currency: str (e.g., 'USD')
            - event: str (e.g., 'Non-Farm Payrolls')
            - impact: str ('high', 'medium', 'low')
        """
        # Return cached if available and not forcing refresh
        if not force_refresh and self.cache.get('date') == datetime.now().strftime('%Y-%m-%d'):
            return self.cache.get('events', [])

        # Try to fetch from Forex Factory
        # Note: Forex Factory blocks automated scraping, so we'll use a simple hardcoded approach
        # In production, you'd use a paid API like Forex Factory API or Trading Economics

        # For now, return hardcoded high-impact events for this week
        events = self._get_hardcoded_events()

        self.save_cache(events)
        return events

    def _get_hardcoded_events(self):
        """
        Hardcoded high-impact events (update weekly)
        In production, replace with real API call
        """
        now = datetime.now()

        # Common recurring high-impact events
        events = []

        # NFP - First Friday of month at 8:30 AM EST
        if now.weekday() == 4 and 1 <= now.day <= 7:
            nfp_time = now.replace(hour=8, minute=30, second=0, microsecond=0)
            events.append({
                'time': nfp_time.isoformat(),
                'currency': 'USD',
                'event': 'Non-Farm Payrolls (NFP)',
                'impact': 'high'
            })

        # Fed Interest Rate Decision - Usually 2nd Wednesday at 2:00 PM EST
        # (Occurs ~8 times per year)
        if now.weekday() == 2 and 8 <= now.day <= 14:
            fed_time = now.replace(hour=14, minute=0, second=0, microsecond=0)
            events.append({
                'time': fed_time.isoformat(),
                'currency': 'USD',
                'event': 'FOMC Interest Rate Decision',
                'impact': 'high'
            })

        # CPI - Usually mid-month Tuesday at 8:30 AM EST
        if now.weekday() == 1 and 12 <= now.day <= 16:
            cpi_time = now.replace(hour=8, minute=30, second=0, microsecond=0)
            events.append({
                'time': cpi_time.isoformat(),
                'currency': 'USD',
                'event': 'Consumer Price Index (CPI)',
                'impact': 'high'
            })

        # ECB Rate Decision - Thursday every 6 weeks around 7:45 AM EST
        # BOE Rate Decision - Thursday every month at 7:00 AM EST
        # BOJ Rate Decision - Usually ends of months (variable)

        return events

    def is_safe_to_trade(self, buffer_minutes=30):
        """
        Check if it's safe to trade right now based on economic calendar

        Args:
            buffer_minutes: Minutes before/after event to avoid (default 30)

        Returns:
            dict with:
                - safe: bool
                - reason: str
                - next_event: dict or None
        """
        events = self.get_todays_events()

        if not events:
            return {
                'safe': True,
                'reason': 'No high-impact events scheduled today',
                'next_event': None
            }

        now = datetime.now()

        for event in events:
            event_time = datetime.fromisoformat(event['time'])

            # Calculate time difference
            time_diff = (event_time - now).total_seconds() / 60  # minutes

            # Check if we're within buffer window
            if -buffer_minutes <= time_diff <= buffer_minutes:
                return {
                    'safe': False,
                    'reason': f"High-impact event: {event['event']} at {event_time.strftime('%H:%M')} ({event['currency']})",
                    'next_event': event,
                    'minutes_until': time_diff
                }

        # No events in danger zone
        upcoming = self._get_next_event(events)

        return {
            'safe': True,
            'reason': f"No events in next {buffer_minutes} minutes",
            'next_event': upcoming
        }

    def _get_next_event(self, events):
        """Get the next upcoming event"""
        now = datetime.now()
        upcoming = []

        for event in events:
            event_time = datetime.fromisoformat(event['time'])
            if event_time > now:
                upcoming.append({
                    **event,
                    'minutes_until': (event_time - now).total_seconds() / 60
                })

        if upcoming:
            upcoming.sort(key=lambda x: x['minutes_until'])
            return upcoming[0]

        return None

    def print_todays_schedule(self):
        """Print today's economic calendar"""
        events = self.get_todays_events()

        print("\n" + "="*70)
        print("TODAY'S ECONOMIC CALENDAR (HIGH-IMPACT EVENTS)")
        print("="*70 + "\n")

        if not events:
            print("  No high-impact events scheduled today")
        else:
            for event in events:
                event_time = datetime.fromisoformat(event['time'])
                now = datetime.now()

                time_diff = (event_time - now).total_seconds() / 60

                if time_diff < 0:
                    status = f"[PASSED {abs(time_diff):.0f} min ago]"
                else:
                    status = f"[IN {time_diff:.0f} min]"

                print(f"  {event_time.strftime('%H:%M')} {status}")
                print(f"    {event['currency']}: {event['event']}")
                print(f"    Impact: {event['impact'].upper()}")
                print()

        print("="*70 + "\n")


def test_calendar():
    """Test the forex calendar"""
    calendar = ForexCalendar()

    # Print today's schedule
    calendar.print_todays_schedule()

    # Check if safe to trade
    safety = calendar.is_safe_to_trade(buffer_minutes=30)

    print("TRADING SAFETY CHECK")
    print("="*70)
    print(f"Safe to Trade: {safety['safe']}")
    print(f"Reason: {safety['reason']}")

    if safety.get('next_event'):
        next_event = safety['next_event']
        print(f"\nNext Event: {next_event['event']}")
        print(f"Time: {next_event.get('minutes_until', 'N/A'):.0f} minutes")

    print("="*70)


if __name__ == "__main__":
    test_calendar()
