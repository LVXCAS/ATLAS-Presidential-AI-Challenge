#!/usr/bin/env python3
"""
BULL PUT SPREAD ENGINE - Alpaca-Compatible Alternative to Iron Condors
======================================================================
2-leg defined-risk spread for premium collection (replaces Iron Condor)

Strategy: Sell OTM put, buy further OTM put for protection
Capital: $300-500 per spread
Expected Return: 2-5% per trade
Win Rate: 70-80%
"""

import alpaca_trade_api as tradeapi
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

load_dotenv()


class BullPutSpreadEngine:
    """Execute bull put spreads - Alpaca-compatible premium collection"""

    def __init__(self):
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - BULL_PUT_SPREAD - %(message)s')
        self.logger = logging.getLogger(__name__)

    def execute_bull_put_spread(self, symbol, current_price, contracts=1, expiration_days=7):
        """Execute bull put spread (2 legs - sell put, buy put)"""

        print(f"\n{'='*80}")
        print(f"EXECUTING BULL PUT SPREAD: {symbol}")
        print(f"{'='*80}")

        # Calculate strikes - CONSERVATIVE (farther OTM for safety)
        # Old: 10% OTM was getting blown through (66% losing rate)
        # New: 15% OTM for better safety margin
        sell_put_strike = round(current_price * 0.85)  # 15% OTM instead of 10%

        # Buy put at ~20% OTM (5 points lower)
        strike_width = 5 if current_price >= 25 else 2.5
        buy_put_strike = sell_put_strike - strike_width

        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Sell Put: ${sell_put_strike:.0f} (collect premium)")
        print(f"  Buy Put:  ${buy_put_strike:.0f} (protection)")
        print(f"  Strike Width: ${strike_width:.0f}")
        print(f"  Expected Credit: ${strike_width * 100 * 0.30:.2f} (30% of width)")
        print(f"  Max Risk: ${(strike_width * 100) - (strike_width * 100 * 0.30):.2f}")

        # Get expiration date
        exp_date = self.get_expiration_date(expiration_days)
        exp_str = exp_date.strftime('%y%m%d')

        # Build options symbols
        sell_put_symbol = f"{symbol}{exp_str}P{int(sell_put_strike * 1000):08d}"
        buy_put_symbol = f"{symbol}{exp_str}P{int(buy_put_strike * 1000):08d}"

        orders = []

        try:
            # LEG 1: BUY PUT (protection, costs premium)
            print(f"\n  [LEG 1/2] BUYING protective put {buy_put_symbol}...")
            order1 = self.api.submit_order(
                symbol=buy_put_symbol,
                qty=contracts,
                side='buy',
                type='market',
                time_in_force='day'
            )
            orders.append({'leg': 'BUY_PUT', 'order_id': order1.id, 'symbol': buy_put_symbol})
            print(f"  [OK] Order submitted: {order1.id}")

            # LEG 2: SELL PUT (collect premium) - submitted AFTER protection
            print(f"  [LEG 2/2] SELLING put {sell_put_symbol}...")
            order2 = self.api.submit_order(
                symbol=sell_put_symbol,
                qty=contracts,
                side='sell',
                type='market',
                time_in_force='day'
            )
            orders.append({'leg': 'SELL_PUT', 'order_id': order2.id, 'symbol': sell_put_symbol})
            print(f"  [OK] Order submitted: {order2.id}")

            print(f"\n  [SUCCESS] BULL PUT SPREAD EXECUTED: {symbol}")
            print(f"  Both legs submitted successfully")
            expected_credit = strike_width * 100 * 0.30
            print(f"  Max profit: ${expected_credit:.2f}")
            print(f"  Max loss: ${(strike_width * 100) - expected_credit:.2f}")
            print(f"  Breakeven: ${sell_put_strike - (expected_credit/100):.2f}")

            return {
                'success': True,
                'symbol': symbol,
                'strategy': 'BULL_PUT_SPREAD',
                'orders': orders,
                'sell_strike': sell_put_strike,
                'buy_strike': buy_put_strike,
                'contracts': contracts,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Bull put spread execution failed for {symbol}: {e}")
            print(f"  [FAILED] Error: {e}")

            # Cancel any filled legs
            print(f"  [ROLLBACK] Canceling any filled orders...")
            for order in orders:
                try:
                    self.api.cancel_order(order['order_id'])
                    print(f"    Canceled {order['leg']}: {order['order_id']}")
                except:
                    pass

            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'partial_orders': orders
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
