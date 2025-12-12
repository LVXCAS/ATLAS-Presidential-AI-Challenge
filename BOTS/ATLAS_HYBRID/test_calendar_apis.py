#!/usr/bin/env python3
"""
Test real economic calendar APIs for live news event data
"""

import requests
from datetime import datetime, timedelta

def test_forex_factory():
    """Forex Factory has free calendar but requires scraping"""
    print("\n[1/3] Forex Factory Calendar")
    print("-" * 70)
    print("Status: Free, but requires web scraping (no official API)")
    print("Data: NFP, FOMC, CPI, all major events with exact times")
    print("Implementation: Would need BeautifulSoup parsing")

def test_fcsapi():
    """FCS API - has free tier for economic calendar"""
    print("\n[2/3] FCS API (fcsapi.com)")
    print("-" * 70)

    # Free tier: 500 calls/month
    url = "https://fcsapi.com/api-v3/forex/economy_cal"

    params = {
        "from": datetime.now().strftime("%Y-%m-%d"),
        "to": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
        "access_key": "DEMO"  # Replace with real key
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get("status"):
                events = data.get("response", [])
                print(f"Found {len(events)} upcoming events")

                for event in events[:3]:
                    print(f"\n  {event.get('title')}")
                    print(f"  Time: {event.get('date')} {event.get('time')}")
                    print(f"  Impact: {event.get('impact')}")
                    print(f"  Currency: {event.get('country')}")
            else:
                print(f"API Error: {data}")
        else:
            print(f"HTTP {response.status_code}: {response.text[:200]}")

    except Exception as e:
        print(f"Error: {e}")

def test_alpha_vantage_calendar():
    """Alpha Vantage doesn't have calendar, but has earnings/IPO"""
    print("\n[3/3] Alpha Vantage Economic Calendar")
    print("-" * 70)
    print("Status: No forex economic calendar endpoint")
    print("Available: Earnings calendar, IPO calendar (stocks only)")
    print("Not useful for forex news protection")

def main():
    print("=" * 70)
    print("ECONOMIC CALENDAR API TESTING")
    print("=" * 70)

    test_forex_factory()
    test_fcsapi()
    test_alpha_vantage_calendar()

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
Best Option: Manual curation + Alpha Vantage news sentiment

Current approach is GOOD ENOUGH:
1. Manually add known events (NFP, FOMC, CPI) to cache
2. Use Alpha Vantage NEWS_SENTIMENT to detect breaking news
3. Update cache weekly with next month's schedule

Why this works:
- High-impact events are scheduled months in advance
- Only ~10-15 events per month worth blocking
- Takes 5 minutes to update cache monthly
- Free (no API costs)
- More reliable than web scraping

Alternative (if you want automation):
- FCS API: $10/month for 10,000 calls (overkill but cheap)
- Forex Factory scraper: Free but breaks if they change HTML
    """)

if __name__ == "__main__":
    main()
