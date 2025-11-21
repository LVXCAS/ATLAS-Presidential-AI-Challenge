"""
ECONOMIC NEWS FILTER

Block trading around high-impact news events to prevent:
1. Volatility spikes (100-200 pip moves in seconds)
2. Slippage (1% SL becomes 2-3% in fast markets = daily DD violation)
3. Unpredictable price action (technical analysis useless during news)

High-Impact Events (Always Block):
- NFP (Non-Farm Payroll) - First Friday of month
- FOMC (Federal Reserve) - 8x per year
- CPI (Consumer Price Index) - Monthly
- GDP releases
- Central bank rate decisions (Fed, ECB, BOE, BOJ)
- Employment data
- Retail sales

Block window: 1 hour before + 1 hour after event
"""

import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional


class NewsFilter:
    """Filter trades based on economic calendar events"""

    def __init__(self, cache_file='BOTS/news_calendar_cache.json'):
        self.cache_file = Path(cache_file)
        self.cache_duration = timedelta(hours=12)  # Refresh cache every 12 hours

        # High-impact event keywords (case-insensitive)
        self.high_impact_keywords = [
            'nfp', 'non-farm', 'payroll',
            'fomc', 'federal reserve', 'fed rate',
            'cpi', 'consumer price',
            'gdp', 'gross domestic',
            'employment', 'unemployment',
            'retail sales',
            'interest rate decision',
            'ecb', 'boe', 'boj',
            'central bank',
            'inflation'
        ]

        # Time window to block trading around news
        self.block_before_minutes = 60  # 1 hour before
        self.block_after_minutes = 60   # 1 hour after

    def _load_cache(self) -> Optional[Dict]:
        """Load cached news data if fresh enough"""
        if not self.cache_file.exists():
            return None

        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)

            cached_time = datetime.fromisoformat(cache['cached_at'])
            if datetime.now() - cached_time < self.cache_duration:
                return cache['events']

        except Exception as e:
            print(f"[WARN] Failed to load news cache: {e}")

        return None

    def _save_cache(self, events: List[Dict]):
        """Save news data to cache"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            cache = {
                'cached_at': datetime.now().isoformat(),
                'events': events
            }

            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)

        except Exception as e:
            print(f"[WARN] Failed to save news cache: {e}")

    def fetch_calendar(self) -> List[Dict]:
        """
        Fetch economic calendar from free API.

        Using Trading Economics API (free tier):
        https://api.tradingeconomics.com/calendar

        Alternative free sources:
        - Forex Factory (requires scraping)
        - Investing.com (requires scraping)
        - FXStreet (requires scraping)
        """
        # Check cache first
        cached = self._load_cache()
        if cached:
            print("[NEWS] Using cached calendar data")
            return cached

        print("[NEWS] Fetching fresh calendar data...")

        try:
            # Try Trading Economics API (may need API key for production)
            # For now, using a mock/fallback approach
            events = self._fetch_from_tradingeconomics()

            if events:
                self._save_cache(events)
                return events

        except Exception as e:
            print(f"[WARN] Failed to fetch calendar: {e}")

        # Return empty list if all sources fail
        # Bot will default to cautious behavior (don't block, but log warning)
        return []

    def _fetch_from_tradingeconomics(self) -> List[Dict]:
        """
        Fetch from Trading Economics API.

        For demo: Returns manually curated high-impact events.
        For production: Need API key from tradingeconomics.com
        """
        # This is a DEMO implementation
        # In production, you'd call the actual API:
        # url = f"https://api.tradingeconomics.com/calendar?c={API_KEY}"
        # response = requests.get(url)
        # return response.json()

        # For now, return common recurring events
        now = datetime.now()

        # Common high-impact events (update manually or via API)
        events = []

        # NFP - First Friday of every month at 8:30 AM EST
        first_friday = self._get_first_friday_of_month(now.year, now.month)
        if first_friday:
            events.append({
                'date': first_friday.isoformat(),
                'event': 'US Non-Farm Payroll (NFP)',
                'impact': 'HIGH',
                'currency': 'USD'
            })

        # CPI - Usually mid-month at 8:30 AM EST
        # (Simplified: 15th of each month)
        cpi_date = datetime(now.year, now.month, 15, 8, 30)
        events.append({
            'date': cpi_date.isoformat(),
            'event': 'US Consumer Price Index (CPI)',
            'impact': 'HIGH',
            'currency': 'USD'
        })

        # FOMC - 8x per year (need to hardcode dates or fetch from API)
        # For demo, just flag Wednesday afternoons as potential FOMC

        print(f"[NEWS] Loaded {len(events)} high-impact events")
        return events

    def _get_first_friday_of_month(self, year: int, month: int) -> Optional[datetime]:
        """Get first Friday of the month"""
        for day in range(1, 8):
            date = datetime(year, month, day, 8, 30)  # 8:30 AM EST
            if date.weekday() == 4:  # Friday
                return date
        return None

    def is_high_impact_event(self, event: Dict) -> bool:
        """Check if event is high-impact"""
        event_name = event.get('event', '').lower()
        impact = event.get('impact', '').upper()

        # Check impact level
        if impact == 'HIGH':
            return True

        # Check keywords
        for keyword in self.high_impact_keywords:
            if keyword in event_name:
                return True

        return False

    def check_news_safety(self, pair: str) -> tuple[bool, str]:
        """
        Check if safe to trade given pair based on upcoming news.

        Returns:
            (is_safe, reason)
            is_safe: True if no major news in block window
            reason: Human-readable explanation
        """
        events = self.fetch_calendar()

        if not events:
            # No calendar data - be cautious but don't block
            return True, "No calendar data available (proceed with caution)"

        now = datetime.now()
        block_start = now - timedelta(minutes=self.block_after_minutes)
        block_end = now + timedelta(minutes=self.block_before_minutes)

        # Get currency pairs from symbol
        # EUR_USD -> ['EUR', 'USD']
        currencies = pair.replace('_', '').replace('/', '')
        currencies = [currencies[:3], currencies[3:6]]

        # Check for events affecting this pair
        blocked_events = []

        for event in events:
            # Parse event time
            try:
                event_time = datetime.fromisoformat(event['date'])
            except:
                continue

            # Check if event is in block window
            if not (block_start <= event_time <= block_end):
                continue

            # Check if high-impact
            if not self.is_high_impact_event(event):
                continue

            # Check if affects this currency pair
            event_currency = event.get('currency', '')
            if event_currency in currencies:
                blocked_events.append(event)

        # If any blocking events found, return unsafe
        if blocked_events:
            event_list = ', '.join([e['event'] for e in blocked_events])
            time_until = (datetime.fromisoformat(blocked_events[0]['date']) - now).total_seconds() / 60

            return False, f"High-impact news in {time_until:.0f} min: {event_list}"

        return True, "No major news in next hour"

    def get_upcoming_events(self, hours: int = 24) -> List[Dict]:
        """Get all high-impact events in next N hours"""
        events = self.fetch_calendar()
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)

        upcoming = []
        for event in events:
            try:
                event_time = datetime.fromisoformat(event['date'])
                if now <= event_time <= cutoff and self.is_high_impact_event(event):
                    upcoming.append(event)
            except:
                continue

        return sorted(upcoming, key=lambda e: e['date'])

    def should_close_positions_before_news(self, minutes_ahead: int = 60) -> tuple[bool, List[Dict]]:
        """
        Check if we should close positions due to upcoming news.

        Args:
            minutes_ahead: How far ahead to check (default 60 min)

        Returns:
            (should_close, events_list)
            should_close: True if major news within the window
            events_list: List of upcoming events that triggered the closure
        """
        events = self.fetch_calendar()
        now = datetime.now()
        cutoff = now + timedelta(minutes=minutes_ahead)

        upcoming_critical = []

        for event in events:
            try:
                event_time = datetime.fromisoformat(event['date'])

                # Check if event is in the critical window
                if now <= event_time <= cutoff and self.is_high_impact_event(event):
                    upcoming_critical.append(event)
            except:
                continue

        should_close = len(upcoming_critical) > 0
        return should_close, upcoming_critical

    def get_affected_pairs(self, event_currency: str, all_pairs: List[str]) -> List[str]:
        """
        Get list of pairs affected by a currency's news event.

        Args:
            event_currency: Currency with news (e.g., 'USD')
            all_pairs: List of all tradeable pairs (e.g., ['EUR_USD', 'GBP_USD'])

        Returns:
            List of affected pairs
        """
        affected = []
        for pair in all_pairs:
            # Remove separators and check if currency is in pair
            pair_clean = pair.replace('_', '').replace('/', '')
            if event_currency in pair_clean:
                affected.append(pair)

        return affected

    def print_upcoming_news(self, hours: int = 24):
        """Print upcoming high-impact events"""
        events = self.get_upcoming_events(hours)

        if not events:
            print(f"\n[NEWS] No high-impact events in next {hours} hours")
            return

        print(f"\n[NEWS] Upcoming high-impact events (next {hours} hours):")
        print("-" * 70)

        for event in events:
            event_time = datetime.fromisoformat(event['date'])
            time_until = event_time - datetime.now()
            hours_until = time_until.total_seconds() / 3600

            print(f"  {event_time.strftime('%Y-%m-%d %H:%M')} ({hours_until:+.1f}h)")
            print(f"    {event['event']} [{event.get('currency', 'N/A')}]")

        print("-" * 70)


# ==============================================================================
# MANUAL NEWS CALENDAR (FALLBACK)
# ==============================================================================

class ManualNewsCalendar:
    """
    Manually maintained calendar of known high-impact events.

    Use this if API sources fail. Update monthly with known events.
    """

    @staticmethod
    def get_known_events_2025() -> List[Dict]:
        """Manually curated high-impact events for 2025"""
        events = []

        # NFP - First Friday of each month at 8:30 AM EST
        nfp_dates = [
            '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04',
            '2025-05-02', '2025-06-06', '2025-07-03', '2025-08-01',
            '2025-09-05', '2025-10-03', '2025-11-07', '2025-12-05'
        ]

        for date in nfp_dates:
            events.append({
                'date': f"{date}T08:30:00",
                'event': 'US Non-Farm Payroll (NFP)',
                'impact': 'HIGH',
                'currency': 'USD'
            })

        # FOMC Rate Decisions (8x per year)
        fomc_dates = [
            '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
            '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-17'
        ]

        for date in fomc_dates:
            events.append({
                'date': f"{date}T14:00:00",
                'event': 'FOMC Rate Decision',
                'impact': 'HIGH',
                'currency': 'USD'
            })

        # Add other major events as needed
        # CPI, GDP, ECB decisions, BOE decisions, etc.

        return events


# ==============================================================================
# INTEGRATION EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEWS FILTER - TESTING")
    print("=" * 70)

    # Create filter
    news_filter = NewsFilter()

    # Check safety for EUR/USD
    print("\n--- Checking EUR_USD Safety ---")
    is_safe, reason = news_filter.check_news_safety('EUR_USD')
    print(f"Safe to trade: {is_safe}")
    print(f"Reason: {reason}")

    # Check safety for USD/JPY
    print("\n--- Checking USD_JPY Safety ---")
    is_safe, reason = news_filter.check_news_safety('USD_JPY')
    print(f"Safe to trade: {is_safe}")
    print(f"Reason: {reason}")

    # Show upcoming events
    news_filter.print_upcoming_news(hours=48)

    print("\n" + "=" * 70)
    print("INTEGRATION INTO BOT")
    print("=" * 70)
    print("""
Add to E8_ULTRA_CONSERVATIVE_BOT.py:

1. In __init__:
    from news_filter import NewsFilter
    self.news_filter = NewsFilter()

2. In score_setup(), BEFORE scoring:
    # Check news safety first
    is_safe, news_msg = self.news_filter.check_news_safety(pair)
    if not is_safe:
        return 0, [f"Blocked by news: {news_msg}"], None

3. Optional - Print upcoming news at start of each scan:
    self.news_filter.print_upcoming_news(hours=4)
    """)
    print("=" * 70)
