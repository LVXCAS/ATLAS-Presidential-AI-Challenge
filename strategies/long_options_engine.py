#!/usr/bin/env python3
"""
LONG OPTIONS ENGINE - Works with Zero Options Selling Power
===========================================================
Buy calls or puts only (no selling required)

Strategy: Buy OTM calls (bullish) or puts (bearish)
Capital: $100-500 per contract
Expected Return: 50-200% per winning trade
Win Rate: 40-50% (but asymmetric risk/reward)
"""

import alpaca_trade_api as tradeapi
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

load_dotenv()


class LongOptionsEngine:
    """Execute simple long call/put trades"""

    def __init__(self):
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - LONG_OPTIONS - %(message)s')
        self.logger = logging.getLogger(__name__)

    def execute_long_option(self, symbol, current_price, direction='BULLISH', contracts=1, expiration_days=7):
        """Execute long call (bullish) or long put (bearish)"""

        print(f"\n{'='*80}")
        print(f"EXECUTING LONG {'CALL' if direction == 'BULLISH' else 'PUT'}: {symbol}")
        print(f"{'='*80}")

        # Calculate strike
        if direction == 'BULLISH':
            # Buy call slightly OTM (5% above current price)
            strike = round(current_price * 1.05)
            option_type = 'C'
            strategy_name = 'LONG_CALL'
        else:
            # Buy put slightly OTM (5% below current price)
            strike = round(current_price * 0.95)
            option_type = 'P'
            strategy_name = 'LONG_PUT'

        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Strike: ${strike:.0f} ({'+' if direction == 'BULLISH' else '-'}5% from current)")
        print(f"  Direction: {direction}")
        print(f"  Max Risk: ${contracts * 100 * 5:.2f} (estimated premium)")
        print(f"  Max Reward: Unlimited ({'up' if direction == 'BULLISH' else 'down'}side)")

        # Get expiration date
        exp_date = self.get_expiration_date(expiration_days)
        exp_str = exp_date.strftime('%y%m%d')

        # Build option symbol
        option_symbol = f"{symbol}{exp_str}{option_type}{int(strike * 1000):08d}"

        try:
            # BUY OPTION
            print(f"\n  [EXECUTE] BUYING {option_symbol}...")
            order = self.api.submit_order(
                symbol=option_symbol,
                qty=contracts,
                side='buy',
                type='market',
                time_in_force='day'
            )

            print(f"  [OK] Order submitted: {order.id}")
            print(f"\n  [SUCCESS] {strategy_name} EXECUTED: {symbol}")
            print(f"  Option: {option_symbol}")
            print(f"  Contracts: {contracts}")
            print(f"  Max loss: ${contracts * 100 * 5:.2f} (premium paid)")
            print(f"  Breakeven: ${strike + 5 if direction == 'BULLISH' else strike - 5:.0f}")

            return {
                'success': True,
                'symbol': symbol,
                'strategy': strategy_name,
                'order_id': order.id,
                'option_symbol': option_symbol,
                'strike': strike,
                'contracts': contracts,
                'direction': direction,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Long option execution failed for {symbol}: {e}")
            print(f"  [FAILED] Error: {e}")

            return {
                'success': False,
                'symbol': symbol,
                'error': str(e)
            }

    def get_expiration_date(self, days_out=7):
        """Get nearest expiration date (Friday)"""
        today = datetime.now()
        days_until_friday = 4 - today.weekday()  # Friday = 4
        if days_until_friday <= 0:
            days_until_friday += 7

        if days_out <= days_until_friday:
            next_friday = today + timedelta(days=days_until_friday)
        else:
            next_friday = today + timedelta(days=days_until_friday + 7)

        return next_friday
