#!/usr/bin/env python3
"""
Earnings Calendar Integration
Prevents trading options before earnings (IV crush protection)
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class EarningsCalendar:
    """Track earnings dates and avoid IV crush"""

    def __init__(self):
        self.earnings_cache = {}  # Cache earnings dates
        self.cache_expiry = {}

    def get_next_earnings(self, symbol: str) -> Optional[datetime]:
        """Get next earnings date for symbol"""
        try:
            # Check cache first (expires after 1 day)
            if symbol in self.earnings_cache:
                if datetime.now() < self.cache_expiry.get(symbol, datetime.min):
                    return self.earnings_cache[symbol]

            # Fetch from yfinance
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar

            if calendar is not None and 'Earnings Date' in calendar:
                earnings_dates = calendar['Earnings Date']

                # Get the next upcoming date
                if isinstance(earnings_dates, list) or hasattr(earnings_dates, '__iter__'):
                    for date in earnings_dates:
                        if isinstance(date, str):
                            date = datetime.strptime(date, '%Y-%m-%d')
                        if date > datetime.now():
                            # Cache it
                            self.earnings_cache[symbol] = date
                            self.cache_expiry[symbol] = datetime.now() + timedelta(days=1)
                            return date
                else:
                    # Single date
                    date = earnings_dates
                    if isinstance(date, str):
                        date = datetime.strptime(date, '%Y-%m-%d')
                    if date > datetime.now():
                        self.earnings_cache[symbol] = date
                        self.cache_expiry[symbol] = datetime.now() + timedelta(days=1)
                        return date

            return None

        except Exception as e:
            logger.debug(f"Could not get earnings for {symbol}: {e}")
            return None

    def is_safe_to_trade(self, symbol: str, days_before: int = 7, days_after: int = 1) -> Dict:
        """
        Check if it's safe to trade options (no earnings soon)

        Args:
            symbol: Stock symbol
            days_before: Don't trade this many days before earnings
            days_after: Don't trade this many days after earnings

        Returns:
            {
                'safe': bool,
                'reason': str,
                'days_until_earnings': int or None,
                'earnings_date': datetime or None
            }
        """
        earnings_date = self.get_next_earnings(symbol)

        if earnings_date is None:
            return {
                'safe': True,
                'reason': 'No upcoming earnings found',
                'days_until_earnings': None,
                'earnings_date': None
            }

        days_until = (earnings_date - datetime.now()).days

        # Check if too close to earnings
        if 0 <= days_until <= days_before:
            return {
                'safe': False,
                'reason': f'Earnings in {days_until} days - IV crush risk',
                'days_until_earnings': days_until,
                'earnings_date': earnings_date
            }

        # Check if just had earnings
        if -days_after <= days_until < 0:
            return {
                'safe': False,
                'reason': f'Earnings was {abs(days_until)} days ago - post-earnings volatility',
                'days_until_earnings': days_until,
                'earnings_date': earnings_date
            }

        # Safe to trade
        return {
            'safe': True,
            'reason': f'Earnings in {days_until} days - safe window',
            'days_until_earnings': days_until,
            'earnings_date': earnings_date
        }


# Global instance
_earnings_calendar = None

def get_earnings_calendar() -> EarningsCalendar:
    """Get singleton earnings calendar"""
    global _earnings_calendar
    if _earnings_calendar is None:
        _earnings_calendar = EarningsCalendar()
    return _earnings_calendar


if __name__ == "__main__":
    # Test
    calendar = EarningsCalendar()

    test_symbols = ['AAPL', 'MSFT', 'TSLA', 'SPY']

    print("EARNINGS CALENDAR TEST")
    print("="*60)

    for symbol in test_symbols:
        result = calendar.is_safe_to_trade(symbol)
        print(f"\n{symbol}:")
        print(f"  Safe to trade: {result['safe']}")
        print(f"  Reason: {result['reason']}")
        if result['earnings_date']:
            print(f"  Next earnings: {result['earnings_date'].strftime('%Y-%m-%d')}")
