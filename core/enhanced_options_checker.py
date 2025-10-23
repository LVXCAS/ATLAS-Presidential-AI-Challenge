#!/usr/bin/env python3
"""
ENHANCED OPTIONS AVAILABILITY CHECKER
Solves the "options unavailable" problem by checking multiple date/strike combinations
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

class EnhancedOptionsChecker:
    def __init__(self):
        load_dotenv()
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

    def find_available_options(self, symbol, target_strike, option_type='C'):
        """Find available options near target strike and expiration"""

        available_options = []

        # Try multiple expiration dates
        expirations = self.get_option_expirations()

        # Try multiple strikes around target
        strike_range = self.get_strike_range(target_strike)

        for exp_date in expirations:
            for strike in strike_range:
                try:
                    options_symbol = f"{symbol}{exp_date}{option_type}{int(strike * 1000):08d}"

                    # Test if this option exists
                    quote = self.api.get_latest_quote(options_symbol)

                    if quote and quote.bid_price and quote.ask_price:
                        available_options.append({
                            'symbol': options_symbol,
                            'underlying': symbol,
                            'strike': strike,
                            'expiration': exp_date,
                            'type': option_type,
                            'bid': float(quote.bid_price),
                            'ask': float(quote.ask_price),
                            'spread': float(quote.ask_price) - float(quote.bid_price)
                        })

                except Exception:
                    continue

        # Sort by proximity to target strike and spread
        if available_options:
            for opt in available_options:
                opt['strike_distance'] = abs(opt['strike'] - target_strike)
                opt['spread_pct'] = opt['spread'] / opt['ask'] if opt['ask'] > 0 else 1.0

            # Sort by strike distance, then spread
            available_options.sort(key=lambda x: (x['strike_distance'], x['spread_pct']))

        return available_options

    def get_option_expirations(self):
        """Get next few Friday expiration dates"""
        expirations = []
        today = datetime.now()

        # Get next 4 Fridays
        for weeks in range(1, 5):
            target_date = today + timedelta(weeks=weeks)
            # Find Friday of that week
            days_until_friday = (4 - target_date.weekday()) % 7
            if days_until_friday == 0 and target_date.weekday() != 4:
                days_until_friday = 7
            friday = target_date + timedelta(days=days_until_friday)
            expirations.append(friday.strftime('%y%m%d'))

        return expirations

    def get_strike_range(self, target_strike):
        """Get range of strikes around target"""
        strikes = []

        # Create range ±10% around target in $5 increments
        start_strike = max(5, int((target_strike * 0.9) / 5) * 5)
        end_strike = int((target_strike * 1.1) / 5) * 5 + 5

        for strike in range(start_strike, end_strike + 1, 5):
            strikes.append(float(strike))

        return strikes

    def get_best_option(self, symbol, target_strike, option_type='C', max_spread_pct=0.10):
        """Get the best available option for execution"""

        available = self.find_available_options(symbol, target_strike, option_type)

        if not available:
            return None

        # Filter by spread criteria
        good_options = [opt for opt in available if opt['spread_pct'] <= max_spread_pct]

        if good_options:
            best_option = good_options[0]  # Already sorted by proximity and spread
            return {
                'symbol': best_option['symbol'],
                'strike': best_option['strike'],
                'bid': best_option['bid'],
                'ask': best_option['ask'],
                'available': True,
                'reason': f"Found {option_type} ${best_option['strike']:.0f} with {best_option['spread_pct']:.1%} spread"
            }

        return {
            'symbol': None,
            'available': False,
            'reason': f"No options within spread criteria (max {max_spread_pct:.1%})"
        }

def test_options_availability():
    """Test options availability for common symbols"""
    checker = EnhancedOptionsChecker()

    test_cases = [
        ('SPY', 660, 'C'),
        ('SPY', 640, 'P'),
        ('AAPL', 255, 'C'),
        ('AAPL', 245, 'P'),
        ('QQQ', 595, 'C'),
        ('META', 740, 'C')
    ]

    print("ENHANCED OPTIONS AVAILABILITY TEST")
    print("=" * 60)

    for symbol, strike, opt_type in test_cases:
        result = checker.get_best_option(symbol, strike, opt_type)

        if result['available']:
            print(f"[OK] {symbol} {opt_type} ${strike}: {result['symbol']} (${result['strike']:.0f})")
            print(f"   Bid: ${result['bid']:.2f}, Ask: ${result['ask']:.2f}")
        else:
            print(f"[X] {symbol} {opt_type} ${strike}: {result['reason']}")
        print()

if __name__ == "__main__":
    test_options_availability()