#!/usr/bin/env python3
"""Visual timeline of upcoming high-impact news events"""

from datetime import datetime, timedelta
import json
from pathlib import Path

print("=" * 80)
print(" " * 20 + "UPCOMING HIGH-IMPACT NEWS EVENTS")
print(" " * 25 + "NewsFilter Protection Timeline")
print("=" * 80)

# Load calendar
cache_file = Path("BOTS/news_calendar_cache.json")
with open(cache_file) as f:
    cache = json.load(f)

events = cache.get('events', [])
now = datetime.now()

print(f"\nCurrent Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Events in Calendar: {len(events)}")

print("\n" + "=" * 80)
print("TIMELINE")
print("=" * 80 + "\n")

for event in events:
    event_time = datetime.fromisoformat(event['date'])
    time_until = event_time - now

    days = time_until.days
    hours = time_until.seconds // 3600

    # Calculate protection windows
    block_new_trades = event_time - timedelta(minutes=60)
    auto_close_positions = event_time - timedelta(minutes=30)

    print(f"Event: {event['event']}")
    print(f"  Date/Time: {event_time.strftime('%Y-%m-%d %H:%M')} ({event['currency']})")
    print(f"  Time Until: {days} days, {hours} hours")
    print(f"  Impact: {event['impact']}")
    print()
    print(f"  Protection Windows:")
    print(f"    Block New Trades:   {block_new_trades.strftime('%Y-%m-%d %H:%M')} (60 min before)")
    print(f"    Auto-Close Positions: {auto_close_positions.strftime('%Y-%m-%d %H:%M')} (30 min before)")
    print()

    # Status
    if now >= event_time:
        status = "[PAST EVENT]"
    elif now >= block_new_trades:
        status = "[BLOCKING TRADES NOW]"
    elif now >= auto_close_positions:
        status = "[CLOSING POSITIONS NOW]"
    else:
        status = "[SAFE - Trading Allowed]"

    print(f"  Current Status: {status}")
    print("=" * 80 + "\n")

print("\n" + "=" * 80)
print("NEWSFILTER AGENT BEHAVIOR")
print("=" * 80)

print("\nVoting Pattern:")
print("  SAFE Period (>60 min before event):")
print("    Vote: ALLOW")
print("    Confidence: 1.00")
print("    Action: Trades execute normally")
print()
print("  DANGER Period (0-60 min before event):")
print("    Vote: BLOCK")
print("    Confidence: 1.00")
print("    Action: All new trades rejected")
print()
print("  CRITICAL Period (0-30 min before event):")
print("    Vote: BLOCK + AUTO-CLOSE")
print("    Confidence: 1.00")
print("    Action: Reject trades + Close open positions")

print("\n" + "=" * 80)
print("KEY DATES TO WATCH")
print("=" * 80 + "\n")

print("  Dec 5  @ 08:30 - NFP (Non-Farm Payroll)")
print("             Expect: Major USD volatility, 100+ pip moves")
print()
print("  Dec 10 @ 14:00 - FOMC Rate Decision")
print("             Expect: Extreme volatility across all USD pairs")
print()
print("  Dec 10 @ 14:30 - Powell Press Conference")
print("             Expect: Whipsaw price action on every word")
print()
print("  Dec 15 @ 08:30 - CPI (Consumer Price Index)")
print("             Expect: High volatility, trend-setting moves")

print("\n" + "=" * 80)
print("PROTECTION STATUS: ACTIVE")
print("  NewsFilterAgent will automatically:")
print("  - Block all new trades 60 min before each event")
print("  - Close open positions 30 min before each event")
print("  - Protect account from news-driven volatility")
print("=" * 80 + "\n")
